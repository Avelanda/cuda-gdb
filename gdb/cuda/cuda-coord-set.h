/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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

#ifndef _CUDA_COORD_SET_H
#define _CUDA_COORD_SET_H 1

#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "gdbsupport/gdb_optional.h"

#include "breakpoint.h"
#include "inferior.h"
#include "target.h"

#include "cuda-coords.h"
#include "cuda-defs.h"
#include "cuda-kernel.h"
#include "cuda-state.h"
#include "cuda-utils.h"

enum class cuda_coord_compare_type
{
  logical,
  physical
};

enum class cuda_coord_set_type
{
  devices,
  sms,
  warps,
  lanes,
  kernels,
  blocks,
  threads
};

typedef enum : uint32_t
{
  select_all = 0x0,
  select_valid = 0x1 << 0,
  select_bkpt = 0x1 << 1,
  select_excpt = 0x1 << 2,
  select_sm_at_excpt = 0x1 << 3,
  select_sngl = 0x1 << 4,
  select_trap = 0x1 << 5,
  select_current_clock = 0x1 << 6,
  select_active = 0x1 << 7
} cuda_coord_set_mask_t;

template <cuda_coord_compare_type compare_type> class cuda_coord_compare
{
private:
  bool m_sequential_order; // order sequentially?
  cuda_coords m_origin;	   // order nearest to origin?

public:
  // Store in sorted order from zero origin
  cuda_coord_compare () : m_sequential_order{ true }, m_origin{} {}

  // Store in nearest neighbor order from provided origin
  cuda_coord_compare (const cuda_coords &origin)
      : m_sequential_order{ false }, m_origin{ origin }
  {
  }

  ~cuda_coord_compare () = default;

  // Reset the origin used in distance calculations
  void
  resetOrigin (const cuda_coords &origin)
  {
    m_sequential_order = false;
    m_origin = origin;
  }

  // This comparison operator will either sort in sequential order from lowest
  // to highest or it will sort closest to the provided origin.
  bool
  operator() (const cuda_coords &lhs, const cuda_coords &rhs) const
  {
    // If we are sorting without a user provided origin, directly compare lhs
    // and rhs
    if (m_sequential_order)
      {
	switch (compare_type)
	  {
	  case cuda_coord_compare_type::logical:
	    return (lhs.logical () < rhs.logical ());
	  case cuda_coord_compare_type::physical:
	    return (lhs.physical () < rhs.physical ());
	  }
      }
    else
      {
	switch (compare_type)
	  {
	  case cuda_coord_compare_type::logical:
	    {
	      // Operate on logical coordinates
	      const auto &origin = m_origin.logical ();
	      const auto &lhl = lhs.logical ();
	      const auto &rhl = rhs.logical ();

	      // lhs and rhs should be fully defined
	      gdb_assert (lhl.isFullyDefined () && rhl.isFullyDefined ());

	      bool res;

	      // Check kernelID
	      if (cuda_coord_distance (res, origin.kernelId (),
				       lhl.kernelId (), rhl.kernelId ()))
		return res;

	      // Check gridId
	      if (cuda_coord_distance (res, origin.gridId (), lhl.gridId (),
				       rhl.gridId ()))
		return res;

	      // Check blockIdx
	      if (cuda_coord_distance (res, origin.blockIdx (),
				       lhl.blockIdx (), rhl.blockIdx ()))
		return res;

	      // Check threadIdx
	      if (cuda_coord_distance (res, origin.threadIdx (),
				       lhl.threadIdx (), rhl.threadIdx ()))
		return res;

	      // If we get here - logical coords match or origin is wildcard.
	      // Default to lessthan ignoring origin.
	      return lhl < rhl;
	    }
	  case cuda_coord_compare_type::physical:
	    {
	      // Operate on physical coordinates
	      const auto &origin = m_origin.physical ();
	      const auto &lhp = lhs.physical ();
	      const auto &rhp = rhs.physical ();

	      // lhs and rhs should be fully defined
	      gdb_assert (lhp.isFullyDefined () && rhp.isFullyDefined ());

	      bool res;

	      // Check device
	      if (cuda_coord_distance (res, origin.dev (), lhp.dev (),
				       rhp.dev ()))
		return res;

	      // Check sm
	      if (cuda_coord_distance (res, origin.sm (), lhp.sm (),
				       rhp.sm ()))
		return res;

	      // Check warp
	      if (cuda_coord_distance (res, origin.wp (), lhp.wp (),
				       rhp.wp ()))
		return res;

	      // Check lane
	      if (cuda_coord_distance (res, origin.ln (), lhp.ln (),
				       rhp.ln ()))
		return res;

	      // If we get here - physical coords match or origin is wildcard.
	      // Default to lessthan ignoring origin.
	      return (lhp < rhp);
	    }
	  }
      }
  }
};

template <cuda_coord_set_type type, uint32_t mask,
	  cuda_coord_compare_type order = cuda_coord_compare_type::logical>
class cuda_coord_set
{
private:
  cuda_coord_compare<order> m_compare;
  std::set<cuda_coords,
	   std::function<bool (const cuda_coords &, const cuda_coords &)> >
      m_coord_set;

  constexpr bool
  physical_type () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
	return true;
      case cuda_coord_set_type::kernels:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return false;
      }
  }

  constexpr bool
  logical_type () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
	return false;
      case cuda_coord_set_type::kernels:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeSm () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::kernels:
	return false;
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeWarp () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::kernels:
      case cuda_coord_set_type::blocks:
	return false;
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeLane () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::kernels:
      case cuda_coord_set_type::blocks:
	return false;
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeKernel () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
	return false;
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::kernels:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeBlock () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::kernels:
	return false;
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

  constexpr bool
  storeThread () const
  {
    switch (type)
      {
      case cuda_coord_set_type::devices:
      case cuda_coord_set_type::sms:
      case cuda_coord_set_type::kernels:
	return false;
      case cuda_coord_set_type::warps:
      case cuda_coord_set_type::lanes:
      case cuda_coord_set_type::blocks:
      case cuda_coord_set_type::threads:
	return true;
      }
  }

public:
  cuda_coord_set (const cuda_coords &filter, gdb::optional<cuda_coords> origin
					     = gdb::optional<cuda_coords> ())
      : m_compare{},
	m_coord_set{ [&] (const cuda_coords &lhs, const cuda_coords &rhs) {
	  return this->m_compare (lhs, rhs);
	} }
  {
    // Set the optional origin
    if (origin.has_value ())
      m_compare.resetOrigin (*origin);

    // For logical coord sets, we only want to store unique entries
    std::unordered_set<uint64_t> foundKernels;
    std::unordered_map<uint64_t, std::unordered_set<CuDim3, cudim3_hash> >
	foundBlocks;

    // Check select mask options
    constexpr bool valid = mask & static_cast<decltype (mask)> (select_valid);
    constexpr bool atBreakpoint
	= mask & static_cast<decltype (mask)> (select_bkpt);
    constexpr bool atException
	= mask & static_cast<decltype (mask)> (select_excpt);
    constexpr bool atAnyException
	= mask & static_cast<decltype (mask)> (select_sm_at_excpt);
    constexpr bool single = mask & static_cast<decltype (mask)> (select_sngl);
    constexpr bool atTrap = mask & static_cast<decltype (mask)> (select_trap);
    constexpr bool atClock
	= mask & static_cast<decltype (mask)> (select_current_clock);
    constexpr bool active
	= mask & static_cast<decltype (mask)> (select_active);

    // We need the aspace when reading lane pc for checking breakpoints
    struct address_space *aspace = nullptr;

    // Iterate over devices
    for (uint32_t dev = 0; dev < cuda_state::get_num_devices (); ++dev)
      {
	// Skip if this dev doesn't match filter
	if (!cuda_coord_equals (filter.physical ().dev (), dev))
	  continue;

	// Iterate over sms
	for (uint32_t sm = 0; sm < cuda_state::device_get_num_sms (dev); ++sm)
	  {
	    // Skip if this sm doesn't match the filter
	    if (!cuda_coord_equals (filter.physical ().sm (), sm))
	      continue;

	    // Is this sm at an exception?
	    if ((atException || atAnyException)
		&& !cuda_state::sm_has_exception (dev, sm))
	      continue;

	    // Is this sm valid?
	    if (valid && !cuda_state::sm_valid (dev, sm))
	      continue;

	    // Save current sm epoch
	    const auto smCnt = m_coord_set.size ();

	    // Iterate over warps
	    for (uint32_t wp = 0; wp < cuda_state::device_get_num_warps (dev);
		 ++wp)
	      {
		// Skip if this warp doesn't match the filter warp
		if (!cuda_coord_equals (filter.physical ().wp (), wp))
		  continue;

		// Is this Warp valid?
		const bool validWarp = cuda_state::sm_valid (dev, sm)
				       && cuda_state::warp_valid (dev, sm, wp);

		// Skip if masking for valid or if this warp is invalid and we
		// are iterating over logical types
		if (!validWarp && (valid || logical_type ()))
		  continue;

		// Skip out-of-date warps
		if (atClock && cuda_state::warp_timestamp_valid (dev, sm, wp)
		    && (cuda_state::warp_timestamp (dev, sm, wp)
			< cuda_clock ()))
		  continue;

		// If looking for traps, skip non-broken warps
		if (atTrap && !cuda_state::warp_broken (dev, sm, wp))
		  continue;

		// Get the coord info
		uint64_t kernelId = CUDA_INVALID;
		CuDim3 clusterIdx = CUDA_INVALID_DIM;
		if (validWarp)
		  {
		    const auto kernel = cuda_state::warp_get_kernel (dev, sm, wp);
		    gdb_assert (kernel);
		    kernelId = kernel->id ();
		    const auto& clusterDim = kernel->cluster_dim ();
		    if ((clusterDim.x != 0) && (clusterDim.y != 0)
			&& (clusterDim.z != 0))
		      {
			clusterIdx
			    = cuda_state::warp_get_cluster_idx (dev, sm, wp);
		      }
		    else
		      {
			clusterIdx = CUDA_IGNORE_DIM;
		      }
		  }

		const uint64_t gridId
		    = validWarp ? cuda_state::warp_get_grid_id (dev, sm, wp)
				: CUDA_INVALID;
		const CuDim3 blockIdx
		    = validWarp ? cuda_state::warp_get_block_idx (dev, sm, wp)
				: CUDA_INVALID_DIM;

		// Skip if the logical coords don't match the filter logical
		// coords
		if (!cuda_coord_equals (filter.logical ().kernelId (),
					kernelId)
		    || !cuda_coord_equals (filter.logical ().gridId (), gridId)
		    || !cuda_coord_equals (filter.logical ().blockIdx (),
					   blockIdx))
		  continue;

		// The follow are used for kernel and block coord sets only
		if (type == cuda_coord_set_type::kernels)
		  {
		    // Skip if we have already seen this kernel
		    auto it = foundKernels.find (kernelId);
		    if (it != foundKernels.end ())
		      continue;
		    // Mark this kernel as seen
		    foundKernels.insert (kernelId);
		  }
		else if (type == cuda_coord_set_type::blocks)
		  {
		    // Check to see if we have seen this kernel before
		    auto it = foundBlocks.find (kernelId);
		    if (it != foundBlocks.end ())
		      {
			// Skip if we have already seen this block in the
			// kernel
			auto it2 = it->second.find (blockIdx);
			if (it2 != it->second.end ())
			  continue;
			// Mark this block as seen
			it->second.insert (blockIdx);
		      }
		    else
		      {
			// Mark this kernel and block as seen
			auto res = foundBlocks.emplace (
			    std::piecewise_construct,
			    std::forward_as_tuple (kernelId),
			    std::forward_as_tuple ());
			res.first->second.insert (blockIdx);
		      }
		  }

		// Save current warp epoch
		const auto wpCnt = m_coord_set.size ();

		// Iterate over lanes
		for (uint32_t ln = 0;
		     ln < cuda_state::device_get_num_lanes (dev); ++ln)
		  {
		    // Skip if this lane doesn't match the filter lane
		    if (!cuda_coord_equals (filter.physical ().ln (), ln))
		      continue;

		    // Skip if this lane is invalid
		    if (valid && !cuda_state::lane_valid (dev, sm, wp, ln))
		      continue;

		    // Skip if this lane is not active
		    if (active && !cuda_state::lane_active (dev, sm, wp, ln))
		      continue;

		    // If looking for current clock, ignore out of date lanes
		    if (atClock
			&& cuda_state::lane_timestamp_valid (dev, sm, wp, ln)
			&& (cuda_state::lane_timestamp (dev, sm, wp, ln)
			    < cuda_clock ()))
		      continue;

		    // Skip if not at a breakpoint
		    if (atBreakpoint)
		      {
			// Obtain the aspace if we haven't already.
			if ((aspace == nullptr)
			    && (inferior_ptid != null_ptid))
			  aspace = target_thread_address_space (inferior_ptid);
			// Skip non-broken kernels
			if (!cuda_state::sm_valid (dev, sm)
			    || !cuda_state::warp_valid (dev, sm, wp)
			    || !cuda_state::lane_valid (dev, sm, wp, ln)
			    || !cuda_state::lane_active (dev, sm, wp, ln)
			    || !breakpoint_here_p (
				aspace,
				cuda_state::lane_get_pc (dev, sm, wp, ln)))
			  continue;
		      }

		    // Skip if kernel is healthy
		    if (atException
			&& (!cuda_state::sm_valid (dev, sm)
			    || !cuda_state::warp_valid (dev, sm, wp)
			    || !cuda_state::lane_valid (dev, sm, wp, ln)
			    || !cuda_state::lane_active (dev, sm, wp, ln)
			    || !cuda_state::lane_get_exception (dev, sm, wp,
								ln)))
		      continue;

		    // Skip if this lane is invalid for traps. We already
		    // verified the warp is broken.
		    if (atTrap
			&& (!cuda_state::sm_valid (dev, sm)
			    || !cuda_state::warp_valid (dev, sm, wp)
			    || !cuda_state::lane_valid (dev, sm, wp, ln)
			    || !cuda_state::lane_active (dev, sm, wp, ln)))
		      continue;

		    const CuDim3 threadIdx
			= ((cuda_state::sm_valid (dev, sm)
			    && cuda_state::warp_valid (dev, sm, wp)
			    && cuda_state::lane_valid (dev, sm, wp, ln))
			       ? cuda_state::lane_get_thread_idx (dev, sm, wp,
								  ln)
			       : CUDA_INVALID_DIM);

		    // Skip if thread doesn't match the filter thread
		    if (!cuda_coord_equals (filter.logical ().threadIdx (),
					    threadIdx))
		      continue;

		    // We found a valid coordinate! We still may need to apply
		    // wildcards based on type.

		    const uint32_t c_dev = dev;
		    const uint32_t c_sm = storeSm () ? sm : CUDA_WILDCARD;
		    const uint32_t c_wp = storeWarp () ? wp : CUDA_WILDCARD;
		    const uint32_t c_ln = storeLane () ? ln : CUDA_WILDCARD;
		    const uint64_t c_kernelId
			= storeKernel () ? kernelId : CUDA_WILDCARD;
		    const uint64_t c_gridId
			= storeKernel () ? gridId : CUDA_WILDCARD;
		    const CuDim3 c_clusterIdx
			= storeBlock () ? clusterIdx : CUDA_WILDCARD_DIM;
		    const CuDim3 c_blockIdx
			= storeBlock () ? blockIdx : CUDA_WILDCARD_DIM;
		    const CuDim3 c_threadIdx
			= storeThread () ? threadIdx : CUDA_WILDCARD_DIM;

		    // Add the coord to the set
		    m_coord_set.emplace (c_dev, c_sm, c_wp, c_ln, c_kernelId,
					 c_gridId, c_clusterIdx, c_blockIdx,
					 c_threadIdx);

		    // Skip if only storing a single entry
		    if (single)
		      break;

		    // If we are not at lane level granularity
		    // we only want to store one entry for the warp
		    // Note here we can skip for logical coordinates as well
		    // since the entire warp belongs to the same kernel/block
		    if ((type == cuda_coord_set_type::devices)
			|| (type == cuda_coord_set_type::sms)
			|| (type == cuda_coord_set_type::warps)
			|| (type == cuda_coord_set_type::kernels)
			|| (type == cuda_coord_set_type::blocks))
		      break;
		  }

		// Skip if only storing a single entry
		if (single && m_coord_set.size ())
		  break;

		// If we are not at warp level granularity
		// we only want to store one entry for the sm
		if (((type == cuda_coord_set_type::devices)
		     || (type == cuda_coord_set_type::sms))
		    && (m_coord_set.size () > wpCnt))
		  break;
	      }

	    // Skip if only storing a single entry
	    if (single && m_coord_set.size ())
	      break;

	    // If we are not at sm level granularity
	    // we only want to store one entry for the device
	    if ((type == cuda_coord_set_type::devices)
		&& (m_coord_set.size () > smCnt))
	      break;
	  }

	// Skip if only storing a single entry
	if (single && m_coord_set.size ())
	  break;
      }
  }

  // This default constructor is kludge to get around the struct initializer in
  // cuda-autostep.c.
  cuda_coord_set ()
      : m_compare{},
	m_coord_set{ [&] (const cuda_coords &lhs, const cuda_coords &rhs) {
	  return this->m_compare (lhs, rhs);
	} }
  {
  }
  cuda_coord_set (const cuda_coord_set &) = default;
  cuda_coord_set (cuda_coord_set &&) = default;
  ~cuda_coord_set () = default;
  cuda_coord_set &operator= (const cuda_coord_set &) = default;
  cuda_coord_set &operator= (cuda_coord_set &&) = default;

  /* Methods */
  size_t
  size () const
  {
    return m_coord_set.size ();
  }

  /* Iterators */
  using iterator = typename decltype (m_coord_set)::iterator;
  iterator
  begin ()
  {
    return m_coord_set.begin ();
  }
  iterator
  end ()
  {
    return m_coord_set.end ();
  }
  using const_iterator = typename decltype (m_coord_set)::const_iterator;
  const_iterator
  cbegin ()
  {
    return m_coord_set.cbegin ();
  }
  const_iterator
  cend ()
  {
    return m_coord_set.cend ();
  }
};

#endif
