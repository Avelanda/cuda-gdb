/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2015-2023 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "defs.h"

#include "completer.h"
#include "gdbthread.h"
#include "inferior.h"
#include "process-stratum-target.h"
#include "readline/readline.h"
#include "regcache.h"
#include "target.h"

#include "cuda-api.h"
#include "cuda-context.h"
#include "cuda-corelow.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-iterator.h"
#include "cuda-linux-nat.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

#include "../libcudacore/libcudacore.h"

/* The CUDA core file target */

static const target_info cuda_core_target_info
    = { "cudacore", N_ ("Local CUDA core dump file"),
        N_ ("Use CUDA core file as a target.\n\
Specify the filename to the core file.") };

class cuda_core_target final : public process_stratum_target
{
public:
  /* public data members */
  static CudaCore *m_cuda_core;
  /* public methods */
  cuda_core_target () = delete;
  cuda_core_target (const char *);
  ~cuda_core_target () override = default;

  /* Return a reference to this target's unique target_info
     object.  */
  const target_info &
  info () const override
  {
    return cuda_core_target_info;
  }

  void close () override;
  void detach (inferior *inf, int from_tty) override;
  void fetch_registers (struct regcache *, int) override;

  bool
  thread_alive (ptid_t ptid) override
  {
    return true;
  }

  std::string pid_to_str (ptid_t) override;

  bool
  has_memory () override
  {
    return true;
  }
  bool
  has_stack () override
  {
    return true;
  }
  bool
  has_registers () override
  {
    return true;
  }
  bool
  has_execution (inferior *inf) override
  {
    return false;
  }
};

CudaCore *cuda_core_target::m_cuda_core = nullptr;

cuda_core_target::cuda_core_target (const char *filename)
    : process_stratum_target ()
{
  cuda_core_load_api (filename);
}

std::string
cuda_core_target::pid_to_str (ptid_t ptid)
{
  struct inferior *inf;
  int pid;

  /* Try the LWPID field first.  */
  pid = ptid.lwp ();
  if (pid != 0)
    return normal_pid_to_str (ptid_t (pid));

  /* Otherwise, this isn't a "threaded" core -- use the PID field, but
   * only if it isn't a fake PID.  */
  inf = find_inferior_ptid (this, ptid);
  if (inf != NULL && !inf->fake_pid_p)
    return normal_pid_to_str (ptid);

  /* No luck.  We simply don't have a valid PID to print.  */
  return "<main task>";
}

void
cuda_core_target::fetch_registers (struct regcache *regcache, int regno)
{
  cuda_core_fetch_registers (regcache, regno);
}

void
cuda_core_fetch_registers (struct regcache *regcache, int regno)
{
  unsigned reg_no, reg_value, num_regs;
  uint64_t pc;
  struct gdbarch *gdbarch = cuda_get_gdbarch ();
  uint32_t pc_regnum = gdbarch ? gdbarch_pc_regnum (gdbarch) : 256;

  if (!cuda_current_focus::isDevice ())
    return;

  const auto &p = cuda_current_focus::get ().physical ();

  num_regs = cuda_state::device_get_num_registers (p.dev ());
  for (reg_no = 0; reg_no < num_regs; ++reg_no)
    {
      reg_value = cuda_state::lane_get_register (p.dev (), p.sm (), p.wp (),
                                                 p.ln (), reg_no);
      regcache->raw_supply (reg_no, &reg_value);
    }

  /* Save PC as well */
  pc = cuda_state::lane_get_virtual_pc (p.dev (), p.sm (), p.wp (), p.ln ());
  regcache->raw_supply (pc_regnum, &pc);

  if (gdbarch)
    {
      int i;

      num_regs = cuda_state::device_get_num_uregisters (p.dev ());
      for (reg_no = 0; reg_no < num_regs; ++reg_no)
        {
          int reg = CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_FULL, reg_no);
          int regnum = cuda_reg_to_regnum (gdbarch, reg);

          reg_value = cuda_state::warp_get_uregister (p.dev (), p.sm (),
                                                      p.wp (), reg_no);
          regcache->raw_supply (regnum, &reg_value);
        }

      num_regs = cuda_state::device_get_num_upredicates (p.dev ());
      for (reg_no = 0; reg_no < num_regs; ++reg_no)
        {
          int reg = CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_PRED, reg_no);
          int regnum = cuda_reg_to_regnum (gdbarch, reg);

          reg_value = cuda_state::warp_get_upredicate (p.dev (), p.sm (),
                                                       p.wp (), reg_no);
          regcache->raw_supply (regnum, &reg_value);
        }

      /* Mark all registers not found in the core as unavailable.  */
      for (i = 0; i < gdbarch_num_regs (gdbarch); i++)
        if (regcache->get_register_status (i) == REG_UNKNOWN)
          regcache->raw_supply (i, NULL);
    }
}

#define CUDA_CORE_PID 966617

static void
cuda_core_register_tid (uint32_t tid)
{
  if (inferior_ptid != null_ptid)
    return;

  ptid_t ptid (CUDA_CORE_PID, tid, tid);
  struct thread_info *tp
      = add_thread (current_inferior ()->process_target (), ptid);
  switch_to_thread_no_regs (tp);
}

/*
 * This is called by both the cuda_core_target and the core_target.
 * For the latter, we don't want to install the entire cuda_core_target.
 */
void
cuda_core_load_api (const char *filename)
{
  CUDBGAPI api;

  printf_unfiltered (_ ("Opening GPU coredump: %s\n"), filename);

  gdb_assert (cuda_core_target::m_cuda_core == nullptr);

  cuda_core_target::m_cuda_core = cuCoreOpenByName (filename);
  if (cuda_core_target::m_cuda_core == nullptr)
    error ("Failed to read core file: %s", cuCoreErrorMsg ());
  api = cuCoreGetApi (cuda_core_target::m_cuda_core);
  if (api == NULL)
    error ("Failed to get debugger APIs: %s", cuCoreErrorMsg ());

  cuda_debugapi::set_api (api);
  cuda_debugapi::set_api_version (CUDBG_API_VERSION_MAJOR,
                                  CUDBG_API_VERSION_MINOR,
                                  CUDBG_API_VERSION_REVISION);

  /* Initialize the APIs */
  cuda_initialize ();
  if (!cuda_initialized)
    error ("Failed to initialize CUDA Core debugger API!");
}

void
cuda_core_free (void)
{
  if (cuda_core_target::m_cuda_core == nullptr)
    return;

  cuda_cleanup ();
  cuda_gdb_session_destroy ();
  cuCoreFree (cuda_core_target::m_cuda_core);
  cuda_core_target::m_cuda_core = nullptr;
}

void
cuda_core_initialize_events_exceptions (void)
{
  CUDBGEvent event;

  /* Flush registers cache */
  registers_changed ();

  /* Create session directory */
  if (cuda_gdb_session_create ())
    error ("Failed to create session directory");

  /* Drain the event queue */
  while (true)
    {
      cuda_debugapi::get_next_sync_event (&event);

      if (event.kind == CUDBG_EVENT_INVALID)
        break;

      if (event.kind == CUDBG_EVENT_CTX_CREATE)
        cuda_core_register_tid (event.cases.contextCreate.tid);

      cuda_process_event (&event);
    }

  // Read in all device state
  cuda_state::update_all_state (CUDBG_RESPONSE_TYPE_FULL);

  /* Figure out, where exception happened */
  cuda_exception ex;
  if (ex.valid ())
    {
      cuda_coords c{ ex.coords () };
      cuda_current_focus::set (c);

      /* Set the current coordinates context to current */
      kernel_t kernel
          = kernels_find_kernel_by_kernel_id (c.logical ().kernelId ());
      context_t ctx
          = kernel ? kernel_get_context (kernel) : get_current_context ();
      if (ctx != NULL)
        set_current_context (ctx);

      ex.printMessage ();
    }
  else
    {
      /* No exception detected, check for fatal signals (SIGTRAP) */
      cuda_iterator<cuda_iterator_type::threads, select_valid | select_trap
                                                     | select_current_clock
                                                     | select_sngl>
          coord{ cuda_coords::wild () };
      if (coord.size ())
        {
          /* This is the first lane in the warp at a trap */
          auto it = coord.begin ();
          cuda_current_focus::set (const_cast<cuda_coords &> (*it));

          /* Set the current coordinates context to current */
          kernel_t kernel
              = kernels_find_kernel_by_kernel_id (it->logical ().kernelId ());
          context_t ctx
              = kernel ? kernel_get_context (kernel) : get_current_context ();
          if (ctx != NULL)
            set_current_context (ctx);

          cuda_set_signo (GDB_SIGNAL_TRAP);
          gdb_printf (_ ("Program terminated with signal %s, %s.\n"),
                           gdb_signal_to_name (GDB_SIGNAL_TRAP),
                           gdb_signal_to_string (GDB_SIGNAL_TRAP));
        }
    }

  /* Fetch latest information about coredump grids */
  kernels_update_args ();
}

static void
cuda_find_first_valid_lane (void)
{
  cuda_iterator<cuda_iterator_type::threads, select_valid | select_sngl> coord{
    cuda_coords::wild ()
  };
  if (!coord.size ())
    {
      /* No valid coords found! */
      cuda_current_focus::invalidate ();
      return;
    }
  cuda_current_focus::set (const_cast<cuda_coords &> (*coord.begin ()));
}

static void
cuda_core_target_open (const char *filename, int from_tty)
{
  struct inferior *inf;
  gdbarch *old_gdbarch = nullptr;

  target_preopen (from_tty);

  if (filename == NULL)
    error (_ ("No core file specified."));

  gdb::unique_xmalloc_ptr<char> expanded_filename (tilde_expand (filename));

  cuda_core_target *target = new cuda_core_target (expanded_filename.get ());

  /* Own the target until it is sucessfully pushed. */
  target_ops_up target_holder (target);

  try
    {
      /* Push the target */
      current_inferior ()->push_target (std::move (target_holder));

      switch_to_no_thread ();

      /* flush register cache from a previous debug session. */
      registers_changed ();

      /* A CUDA corefile does not contain host process pid information.
       * We need to fake it here since we are only examining CUDA state.
       * Add the fake PID for the host thread. */
      inf = current_inferior ();
      inferior_appeared (inf, CUDA_CORE_PID);
      inf->fake_pid_p = true;
      thread_info *thread = add_thread_silent (target, ptid_t (CUDA_CORE_PID));
      switch_to_thread_no_regs (thread);

      /* Set debuggers architecture to CUDA */
      old_gdbarch = target_gdbarch ();
      set_target_gdbarch (cuda_get_gdbarch ());

      cuda_core_initialize_events_exceptions ();

      post_create_inferior (from_tty);

      /* If no exception found try to set focus to first valid thread */
      if (!cuda_current_focus::isDevice ())
	{
	  warning ("No exception was found on any device");
	  cuda_find_first_valid_lane ();

	  // If we still are not focused on the device, give up but
	  // allow the user to debug global memory contents
	  if (!cuda_current_focus::isDevice ())
	    warning ("No focus could be set on any device");
	}

      // Print the CUDA focus if valid
      // Switch back to the old arch if not
      if (cuda_current_focus::isDevice ())
	cuda_current_focus::printFocus (false);
      else if (old_gdbarch != nullptr)
	set_target_gdbarch (old_gdbarch);

      switch_to_thread_keep_cuda_focus (thread);

      /* Fetch all registers from core file.  */
      target_fetch_registers (get_current_regcache (), -1);

      // Set up the frame cache
      reinit_frame_cache ();

      // Print the backtrace if we found something on the GPU to focus on
      if (cuda_current_focus::isDevice ())
	print_stack_frame (get_selected_frame (NULL), 1, SRC_AND_LOC, 1);
    }
  catch (const gdb_exception_error &e)
    {
      if (e.reason < 0)
        {
          inf->pop_all_targets_at_and_above (process_stratum);

	  if (old_gdbarch != nullptr)
	    set_target_gdbarch (old_gdbarch);

	  registers_changed ();
	  reinit_frame_cache ();
	  cuda_cleanup ();

	  error (_ ("Could not open CUDA core file: %s"), e.what ());
	}
    }
}

void
cuda_core_target::close ()
{
  switch_to_no_thread ();
  exit_inferior_silent (current_inferior ());
  cuda_core_free ();
}

void
cuda_core_target::detach (inferior *inf, int from_tty)
{
  inf->unpush_target (this);
  registers_changed ();
  reinit_frame_cache ();

  if (from_tty)
    gdb_printf (_ ("No core file now.\n"));
}

void _initialize_cuda_corelow ();
void
_initialize_cuda_corelow ()
{
  add_target (cuda_core_target_info, cuda_core_target_open,
              filename_completer);
}
