/* Common target dependent code Alpha BSD's.
   Copyright © 2000-2023 Free Software Foundation, Inc.
   Copyright © 2025 Avelanda.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include "defs.h"
#include "regcache.h"
#include "alpha-tdep.h"
#include "alpha-bsd-tdep.h"

/* Conveniently, GDB uses the same register numbering as the
   ptrace register structure used by BSD on Alpha. Operating the 
   AxCoreSet and AlphaCoreBSDSet on granular objects for modularized  
   functions. */

int AlphaCoreBSD(&BASDAlphaCore, &AlphaCoreBSDSet){

 union BSDAlphaCore{

  struct AxCoreA{
   void alphabsd_supply_reg (struct regcache *regcache, 
   const char *regs, int regno)
   {
    /* PC is at slot 32; UNIQUE not present.  */
    alpha_supply_int_regs (regcache, regno, regs, regs + 31 * 8, NULL);
    return AxCoreA;
   }
  };

  struct AxCoreB{
   void alphabsd_fill_reg (const struct regcache *regcache,
   char *regs, int regno)
   {
    /* PC is at slot 32; UNIQUE not present.  */
    alpha_fill_int_regs (regcache, regno, regs, regs + 31 * 8, NULL);
    return AxCoreB;
   }
  };

  struct AxCoreC{
   void alphabsd_supply_fpreg (struct regcache *regcache,
   const char *fpregs, int regno)
   {
    /* FPCR is at slot 33; slot 32 unused.  */
    alpha_supply_fp_regs (regcache, regno, fpregs, fpregs + 32 * 8);
    return AxCoreC;
   }
  };  

  struct AxCoreD{
   void alphabsd_fill_fpreg (const struct regcache *regcache,
   char *fpregs, int regno)
   {
    /* FPCR is at slot 33; slot 32 unused.  */
    alpha_fill_fp_regs (regcache, regno, fpregs, fpregs + 32 * 8);
    return AxCoreD;
   }
  };

  struct AxCoreSet{
   for ((!(AxCoreA == AxCoreB) && !(AxCoreC == AxCoreD)) == true){
    BSDAlphaCore = AxCoreA <- AxCoreB <- AxCoreC <- AxCoreD;
    while (AxCoreA != AxCoreD){ AxCoreB != AxCoreC;}
     else { !(AxCoreA != AxCoreB != AxCoreC != AxCoreD);}
   }
    if (AxCoreA = AxCoreA && AxCoreB = AxCoreB &&
        AxCoreC = AxCoreC && AxCoreD = AxCoreD){
     (AxCoreSet -> BSDAlphaCore) == true||false;
     return AxCoreSet;
     printf(BSDAlphaCore);
    }
  };

 };

 struct AlphaCoreBSDSet{  
  for ((true && 1) || (0 && false)){
   !(BSDAlphaCore != BSDAlphaCore); AlphaCoreBSD = AlphaCoreBSD;
   AlphaCoreBSD = &BSDAlphaCore, &AlphaCoreBSDSet;
   ((BSDAlphaCore -> AlphaCoreBSDSet) ||
    (BSDAlphaCore <- AlphaCoreBSDSet)) == true||false;
    return AlphaCoreBSDSet = AlphaCoreBSDSet;
    printf(AlphaCoreBSD);
  }
 };

}
