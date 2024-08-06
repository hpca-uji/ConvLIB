#!/usr/bin/python3

import os
import sys
import argparse

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def generate_selector(arch, micros, maxMR, maxNR, dtype, cfile, hfile):
    #--------------------------------------
    # Headers
    #--------------------------------------
    gen  =  "\n"
    if arch == "armv8":
        gen += f"#include <arm_neon.h>\n"
    else:
        gen += f"#include <riscv_vector.h>\n"

    gen += f"#include <stdio.h>\n"
    gen += f"#include <stdint.h>\n"
    gen += f"#include <stdlib.h>\n\n"

    if dtype == "fp32":
        gen += f"typedef void (*uk_intrinsic_{dtype})(int, int, int, float *, float *, float *, float, int);\n\n"
    elif dtype == "fp16":
        gen += f"typedef void (*uk_intrinsic_{dtype})(int, int, int, float16_t *, float16_t *, float16_t *, float16_t, int);\n\n"
    elif dtype == "int32":
        gen += f"typedef void (*uk_intrinsic_{dtype})(int, int, int, int32_t *, int32_t *, int32_t *, int32_t, int);\n\n"
    else:
        gen += f"typedef void (*uk_intrinsic_{dtype})(int, int, int, int8_t *, int8_t *, int32_t *, int32_t, int);\n\n"

    gen += f"typedef struct uk_config_{dtype} {{\n"
    gen += f"  int mr_pool[128];\n"
    gen += f"  int nr_pool[128];\n"
    gen += f"  int uk_num;\n"
    gen += f"}} uk_config_{dtype}_t;\n\n"

    gen += f"\nuk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}();\n"

    gen += f"\nuk_config_{dtype}_t *new_uk_intrinsic_config_{dtype}();\n"
    
    gen += f"\nvoid uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr);\n"

    hfile.write(gen)

    gen  =  "\n"
    gen += f"uk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}() {{ \n"
    gen += f"  uk_intrinsic_{dtype} *uk_vec = (uk_intrinsic_{dtype} *)malloc(sizeof(uk_intrinsic_{dtype}) * {maxMR} * {maxNR});\n"
    for mr, nr in micros:
        gen += f"  uk_vec[{nr}*{maxMR} + {mr}] = ukernel_intrinsic_{mr}x{nr}_{dtype};\n"
    gen +=  "  return uk_vec;\n"
    gen +=  "}\n\n"

    gen += f"void uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr) {{\n"
    gen += f"  (*ukr) = uk_vec[nr*{maxMR} + mr];\n"
    gen +=  "}\n\n"
    
    gen += f"uk_config_{dtype}_t *new_uk_intrinsic_config_{dtype}() {{\n"
    gen += f"  uk_config_{dtype}_t *uk_config = (uk_config_{dtype}_t *)malloc(sizeof(uk_config_{dtype}_t));\n"
    gen += f"  uk_config->uk_num = 0;\n"
    for mr, nr in micros:
        gen += f"  uk_config->mr_pool[uk_config->uk_num] = {mr};\n" 
        gen += f"  uk_config->nr_pool[uk_config->uk_num] = {nr};\n"
        gen += f"  uk_config->uk_num++;\n"
    gen += f"  return uk_config;\n"
    gen +=  "}\n\n"
    cfile.write(gen)
    
    return True


#--------------------------------------------------------------------------------------------
# MICRO-KERNEL GENERATOR FOR ARMv8
#--------------------------------------------------------------------------------------------

def micro_kernel_int8_int32_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen


    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*ldC+i]\n"
        micro += f"#define vinit_{dtype}(vreg, value)   vreg = vmovq_n_s32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_s32(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_s32(mem, vreg)\n"
        micro += f"#define _vload_{dtype}(vreg, mem) vreg=vld1_s8(mem)\n"
        micro += f"#define vloadx2_{dtype}(vreg0, vreg1, vtmp, mem) \\\n"
        micro += f"_vload_{dtype}(vtmp, mem); \\\n"
        micro += f"vreg0=vget_low_s16(vmovl_s8(vtmp)); \\\n"
        micro += f"vreg1=vget_high_s16(vmovl_s8(vtmp))\n"
        micro += f"#define vload_{dtype}(vreg, vtmp, mem) \\\n"
        micro += f"_vload_{dtype}(vtmp, mem); \\\n"
        micro += f"vreg=vget_low_s16(vmovl_s8(vtmp)); \n"
        micro += f"#define vdup_{dtype}(vreg, mem)         vreg=vdup_n_s16((int16_t) mem)\n"
        micro += f"#define vupdate_{dtype}(vreg1, vreg2, vreg3) vreg1=vmlal_s16(vreg1, vreg2, vreg3)\n\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1=vmlal_lane_s16(vreg1, vreg2, vreg3, lane)\n\n"
    
    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){\n"
    micro += f"  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    
    micro += f"  int8x8_t  vtmp;\n"
    micro += f"  int16x4_t "

    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "
    blim = NR
    if lane: 
        blim =vNR
        if (NR % vlen != 0):
            blim += 1
        
    for nr in range(0, blim):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"

    micro += "  int32x4_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"

    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"
    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    ar = 0
    for mr_x2 in range(0, MR // 8):
        micro += f"    vloadx2_{dtype}(A{ar}, A{ar + 1}, vtmp, &Ar[bA + {mr_x2 * 8}]);\n"
        ar += 2
    if MR % 8 != 0:
        micro += f"    vload_{dtype}(A{ar}, vtmp, &Ar[bA + {MR // 8 * 8}]);\n"

    if lane:
        br = 0
        for nr_x2 in range(0, NR // 8):
            micro += f"    vloadx2_{dtype}(B{br}, B{br + 1}, vtmp, &Br[bB + {nr_x2 * 8}]);\n"
            br += 2
        if NR % 8 != 0:
            micro += f"    vload_{dtype}(B{br}, vtmp, &Br[bB + {NR // 8 * 8}]);\n"
    else:
        for nr in range(0, NR):
            micro += f"    vdup_{dtype}(B{nr}, Br[bB+{nr}]);\n"


    if lane:
        for mr in range(0, vMR):
            for nr in range(0, vNR):
                for lane in range(0, vlen):
                    micro += f"    vupdate_lane_{dtype}(C{mr}{nr*vlen+lane}, A{mr}, B{nr}, {lane}); \n"
            if (NR % vlen != 0): 
                for lane in range(0, vlen % NR):
                    micro += f"    vupdate_lane_{dtype}(C{mr}{nr*vlen+lane}, A{mr}, B{nr+1}, {lane}); \n"
    else:
        for mr in range(0, vMR):
            for nr in range(0, NR):
                micro += f"    vupdate_{dtype}(C{mr}{nr}, A{mr}, B{nr}); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"

    micro += "}\n"
    
    cfile.write(micro)

    return True

def micro_kernel_int8_int32_u8_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
   
    if not lane:
        print("ERROR: Gerenator for int8-int32 only compatible with lane")
        sys.exit(-1)

    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*ldC+i]\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)                           vst1q_s32(mem, vreg)\n"
        micro += f"#define vinit_{dtype}(vreg, value)                           vreg  = vmovq_n_s32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)                            vreg  = vld1q_s32(mem)\n"
        micro += f"#define vload_{dtype}(vreg, mem)                             vreg  = vmovl_s8(vld1_s8(mem))\n"
        micro += f"#define vgetlow_{dtype}(vreg1, vreg2)                        vreg1 = vget_low_s16(vreg2)\n"
        micro += f"#define vdup_{dtype}(vreg, mem)                              vreg  = vdup_n_s16((int16_t) mem)\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane)      vreg1 = vmlal_laneq_s16     (vreg1, vreg2, vreg3, lane)\n"
        micro += f"#define vupdate_high_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1 = vmlal_high_laneq_s16(vreg1, vreg2, vreg3, lane)\n"

    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){\n"
    micro += f"  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    
    micro += "  int16x8_t "
    abregs = ""
    for mr in range(0, MR // 8):
        abregs += f" A{mr}, "
    if MR % 8 != 0:
        abregs += f" A{MR // 8}, "

    for nr in range(0, NR // 8):
        abregs += f" B{nr}, "
    if NR % 8 != 0:
        abregs += f" B{NR // 8}, "
    micro += abregs[:-2] + ";\n"
   

    micro += "  int16x4_t "
    abregs = ""
    for mr in range(0, MR // 8):
        abregs += f" Alow{mr}, "
    if MR % 8 != 0:
        abregs += f" Alow{MR // 8}, "
    micro += abregs[:-2] + ";\n"


    micro += "  int32x4_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"


    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"


    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    for mr in range(0, MR // 8):
        micro += f"    vload_{dtype}(A{mr}, &Ar[bA + {mr * 8}]);\n"
        micro += f"    vgetlow_{dtype}(Alow{mr}, A{mr});\n"
    if MR % 8 != 0:
        micro += f"    vload_{dtype}(A{MR // 8}, &Ar[bA + {MR // 8 * 8}]);\n"
        micro += f"    vgetlow_{dtype}(Alow{MR // 8}, A{MR // 8});\n"

    for nr in range(0, NR // 8):
        micro += f"    vload_{dtype}(B{nr}, &Br[bB + {nr * 8}]);\n"
    if NR % 8 != 0:
        micro += f"    vload_{dtype}(B{NR // 8}, &Br[bB + {NR // 8 * 8}]);\n"


    for nr in range(0, NR):
        for mr in range(0, MR // 8):
            micro += f"    vupdate_lane_{dtype}(C{mr*2}{nr},   Alow{mr}, B{nr // 8}, {nr % 8}); \n"
            micro += f"    vupdate_high_lane_{dtype}(C{mr*2 + 1}{nr}, A{mr}, B{nr // 8}, {nr % 8}); \n"
        if (MR % 8 != 0): 
            micro += f"    vupdate_lane_{dtype}(C{mr}{nr}, A{mr}, B{nr // 8}, {nr % 8}); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"
    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"
    micro += "}\n"
    
    cfile.write(micro)

    return True

def micro_kernel_int8_int32_s8_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
   
    if not lane:
        print("ERROR: Gerenator for int8-int32 only compatible with lane")
        sys.exit(-1)

    vMR = MR // vlen
    vNR = NR // vlen


    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*Clda+i]\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)                    vst1q_s32(mem, vreg)\n"
        micro += f"#define vinit_{dtype}(vreg, value)                    vreg  = vmovq_n_s32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)                     vreg  = vld1q_s32(mem)\n"
        micro += f"#define vload_{dtype}(vreg, mem)                      vreg  = vld1q_s8(mem)\n"
        micro += f"#define vload_s_{dtype}(vreg, mem)                     vreg  = vld1_s8(mem)\n"
        micro += f"#define vgetlow_{dtype}(vreg1, vreg2)                 vreg1 = vget_low_s8(vreg2)\n"
        micro += f"#define vgethigh_{dtype}(vreg1, vreg2)                vreg1 = vget_high_s8(vreg2)\n"
        micro += f"#define vdup_{dtype}(vreg, mem)                       vreg  = vdup_n_s8(mem)\n"
        micro += f"#define vmull_{dtype}(vreg1, vreg2, vreg3)            vreg1 = vmull_s8(vreg2, vreg3)\n"
        micro += f"#define vmlal_{dtype}(vreg1, vreg2, vreg3)            vreg1 = vmlal_s8(vreg1, vreg2, vreg3)\n"
        micro += f"#define vaddq_low_{dtype}(vreg1, vreg2)               vreg1 = vaddq_s32(vreg1, vmovl_s16(vget_low_s16(vreg2)))\n"
        micro += f"#define vaddq_high_{dtype}(vreg1, vreg2)              vreg1 = vaddq_s32(vreg1, vmovl_s16(vget_high_s16(vreg2)))\n"


    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro += "  int8x16_t _A, _An;\n"
    micro += "  int16x8_t VM;\n "
    micro += "  int8x8_t B, Bn;\n"
    
    micro += "  int8x8_t "
    abregs = ""
    for mr in range(0, MR // 8):
        abregs += f" A{mr}, An{mr}, "
    if MR % 8 != 0:
        abregs += f" A{MR // 8}, An{MR // 8}, "
    micro += abregs[:-2] + ";\n"
   
    micro += "  int32x4_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"


    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"


    micro += "  for (pr=0; pr<kc-1; pr+=2) { // Loop L6\n"
    mr_id = 0
    desp  = 0
    for mr in range(0, MR // 16):
        micro += f"    vload_{dtype}(_A, &Ar[bA + {desp}]);\n"
        micro += f"    vgetlow_{dtype}(A{mr_id}, _A);\n" 
        mr_id += 1
        micro += f"    vgethigh_{dtype}(A{mr_id}, _A);\n\n" 
        mr_id += 1
        desp  += 16
    if MR % 16 != 0:
        micro += f"    vload_s_{dtype}(A{mr_id}, &Ar[bA + {desp}]);\n"
        desp += 8
    

    mr_id = 0
    for mr in range(0, MR // 16):
        micro += f"    vload_{dtype}(_A, &Ar[bA + {desp}]);\n"
        micro += f"    vgetlow_{dtype}(An{mr_id}, _A);\n" 
        mr_id += 1
        micro += f"    vgethigh_{dtype}(An{mr_id}, _A);\n\n" 
        mr_id += 1
        desp  += 16

    if MR % 16 != 0:
        micro += f"    vload_s_{dtype}(An{mr_id}, &Ar[bA + {desp}]);\n\n"


    for nr in range(0, NR):
        micro += f"    vdup_{dtype}(B, Br[bB+{nr}]);\n"
        micro += f"    vdup_{dtype}(Bn, Br[bB+{nr+NR}]);\n"
        mr_id = 0
        for mr in range(0, MR // 4, 2):
            micro += f"    vmull_{dtype}(VM, A{mr_id}, B);\n"
            micro += f"    vmlal_{dtype}(VM, An{mr_id}, Bn);\n"
            micro += f"    vaddq_low_{dtype}(C{mr}{nr}, VM);\n"
            micro += f"    vaddq_high_{dtype}(C{mr+1}{nr}, VM);\n\n"
            mr_id += 1


    micro += f"    bA+={MR*2};\n"
    micro += f"    bB+={NR*2};\n"
    micro += "  }\n\n"

    micro += "  if ((kc%2) != 0) {\n"
    mr_id = 0
    desp  = 0
    for mr in range(0, MR // 16):
        micro += f"    vload_{dtype}(_A, &Ar[bA + {desp}]);\n"
        micro += f"    vgetlow_{dtype}(A{mr_id}, _A);\n" 
        mr_id += 1
        micro += f"    vgethigh_{dtype}(A{mr_id}, _A);\n" 
        mr_id += 1
        desp  += 16

    if MR % 16 != 0:
        micro += f"    vload_s_{dtype}(A{mr_id}, &Ar[bA + {desp}]);\n"

    for nr in range(0, NR):
        micro += f"    vdup_{dtype}(B, Br[bB+{nr}]);\n"
        mr_id = 0
        for mr in range(0, MR // 4, 2):
            micro += f"    vmull_{dtype}(VM, A{mr_id}, B);\n"
            micro += f"    vaddq_low_{dtype}(C{mr}{nr}, VM);\n"
            micro += f"    vaddq_high_{dtype}(C{mr+1}{nr}, VM);\n"
            mr_id += 1

    micro += "  }\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"
    micro += "}\n"
    
    cfile.write(micro)

    return True


def micro_kernel_int8_int16_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vlen = 8

    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*ldC+i]\n"
        micro += f"#define vinit_{dtype}(vreg, value)   vreg=vmovq_n_s16(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_s16(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_s16(mem, vreg)\n"
        
        micro += f"#define _vload_{dtype}(vreg, mem) vreg=vld1q_s8(mem)\n"
        
        micro += f"#define vloadx2_{dtype}(vreg0, vreg1, vtmp, mem) \\\n"
        micro += f"_vload_{dtype}(vtmp, mem); \\\n"
        micro += f"vreg0=vget_low_s8(vtmp); \\\n"
        micro += f"vreg1=vget_high_s8(vtmp)\n"

        micro += f"#define vload_{dtype}(vreg, vtmp, mem) \\\n"
        micro += f"_vload_{dtype}(vtmp, mem); \\\n"
        micro += f"vreg=vget_low_s8(vtmp); \n"
        
        #micro += f"#define vload_{dtype}(vreg, mem)     vreg=vld1_s8(mem)\n"
        micro += f"#define vdup_{dtype}(vreg, mem)      vreg=vdup_n_s8((int8_t) mem)\n"
        micro += f"#define vupdate_{dtype}(vreg1, vreg2, vreg3) vreg1=vmlal_s8(vreg1, vreg2, vreg3)\n\n"
    
    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int16_t *Cor, int16_t betaI, int Clda){\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int16_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int16_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro += "  int8x16_t  vtmp;\n"
    micro += "  int8x8_t "

    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "
        
    for nr in range(0, NR):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"

    micro += "  int16x8_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"

    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"
    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    ar = 0
    for mr_x2 in range(0, MR // 16):
        micro += f"    vloadx2_{dtype}(A{ar}, A{ar + 1}, vtmp, &Ar[bA + {mr_x2 * 16}]);\n"
        ar += 2

    if MR % 16 != 0:
        micro += f"    vload_{dtype}(A{ar}, vtmp, &Ar[bA + {MR // 16 * 16}]);\n"

    for nr in range(0, NR):
        micro += f"    vdup_{dtype}(B{nr}, Br[bB+{nr}]);\n"


    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vupdate_{dtype}(C{mr}{nr}, A{mr}, B{nr}); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"
    micro += "}\n"
    
    cfile.write(micro)

    return True


def micro_kernel_fp32_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*ldC+i]\n"

        micro += f"#define vinit_{dtype}(vreg, value)   vreg = vmovq_n_f32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_f32(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_f32(mem, vreg)\n"
        micro += f"#define vload_{dtype}(vreg, mem) vreg=vld1q_f32(mem)\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1=vfmaq_laneq_f32(vreg1, vreg2, vreg3, lane)\n\n"

    micro += f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  float Ctmp[{MR}*{NR}];\n"
   
    micro += f"  float beta;\n"
    micro += f"  float *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro += "  float32x4_t  vtmp;\n"
    micro += "  float32x4_t "
    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "
    blim =vNR
    if (NR % vlen != 0):
        blim += 1
    for nr in range(0, blim):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"
    micro += "  float32x4_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"
    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"

    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"
    for ar in range(0, vMR):
        micro += f"    vload_{dtype}(A{ar}, &Ar[bA + {ar * vlen}]);\n"

    for br in range(0, vNR):
        micro += f"    vload_{dtype}(B{br}, &Br[bB + {br * vlen}]);\n"


    for mr in range(0, vMR):
        for nr in range(0, vNR):
            for lane in range(0, vlen):
                micro += f"    vupdate_lane_{dtype}(C{mr}{nr*vlen+lane}, A{mr}, B{nr}, {lane}); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"
    
    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"
   
    micro += "}\n"

    cfile.write(micro)
    
    return True

def micro_kernel_int32_armv8_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen
    
    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*ldC+i]\n"

        micro += f"#define vinit_{dtype}(vreg, value)   vreg = vmovq_n_s32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_s32(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_s32(mem, vreg)\n"
        micro += f"#define vload_{dtype}(vreg, mem) vreg=vld1q_s32(mem)\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1=vmlaq_laneq_s32(vreg1, vreg2, vreg3, lane)\n\n"

    micro += f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int mr, int nr, int kc, int32_t  *Ar, int32_t *Br, int32_t *Cor, int32_t betaI, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro += "  int32x4_t  vtmp;\n"
    micro += "  int32x4_t "
    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "
    blim =vNR
    if (NR % vlen != 0):
        blim += 1
    for nr in range(0, blim):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"
    micro += "  int32x4_t "
    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"
    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit_{dtype}(C{mr}{nr}, 0);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC_{dtype}(C{mr}{nr}, &Crref({mr*vlen}, {nr}));\n"
    micro += "  }\n\n"

    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"
    for ar in range(0, vMR):
        micro += f"    vload_{dtype}(A{ar}, &Ar[bA + {ar * vlen}]);\n"

    for br in range(0, vNR):
        micro += f"    vload_{dtype}(B{br}, &Br[bB + {br * vlen}]);\n"


    for mr in range(0, vMR):
        for nr in range(0, vNR):
            for lane in range(0, vlen):
                micro += f"    vupdate_lane_{dtype}(C{mr}{nr*vlen+lane}, A{mr}, B{nr}, {lane}); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC_{dtype}(&Crref({mr*vlen},{nr}), C{mr}{nr}); \n"
    
    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*ldC + i] = (betaI) * Cor[j*ldC + i] + Ctmp[j * MR + i];\n"
    
    micro += "}\n"

    cfile.write(micro)
    
    return True


#--------------------------------------------------------------------------------------------
# MICRO-KERNEL GENERATOR FOR RISCV
#--------------------------------------------------------------------------------------------

def micro_kernel_fp32_riscv_generator(rvv, rv_mod, arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*ldC+i]\n"
        micro += f"#define min(a,b) a >= b ? a : b\n"

        if rvv == "1.0":
            micro += "#define vinit(vreg, value, vlength)        vreg = __riscv_vfmv_v_f_f32m1(value, vlength)\n"
            micro += "#define vloadC(vreg, mem,  vlength)        vreg = __riscv_vle32_v_f32m1 (mem, vlength)\n"
            micro += "#define vstoreC(mem, vreg, vlength)               __riscv_vse32_v_f32m1 (mem, vreg, vlength)\n"
            micro += "#define vload(vreg, vtmp, mem, vlength)    vreg = __riscv_vle32_v_f32m1 (mem, vlength); \n"
            micro += "#define vdup(vreg, mem, vlength)           vreg = __riscv_vfmv_v_f_f32m1(mem, vlength)\n"
            if rv_mod == "GATHER":
                micro += "#define gather(vreg1, vreg2, index, vlength)   vreg1 = __riscv_vrgather_vx_f32m1(vreg2, index, vlength)\n"

            if rv_mod == "BROADCAST":
                micro += "#define vupdate(vreg1, vreg2, value, vlength) vreg1=__riscv_vfmacc_vf_f32m1(vreg1, value, vreg2, vlength)\n\n"
            else:
                micro += "#define vupdate(vreg1, vreg2, vreg3, vlength) vreg1=__riscv_vfmacc_vv_f32m1(vreg1, vreg2, vreg3, vlength)\n\n"
        else:
            #TODO: Implementation for rvv0.7
            return
#
    micro +=f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  float Ctmp[{MR}*{NR}];\n"
   
    micro += f"  float beta;\n"
    micro += f"  float *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro +=f"  int vlength = {vlen};\n"
    if rv_mod == "GATHER":
        micro +=f"  int nr_vl = min(vlength, nr);\n"
    micro += "  vfloat32m1_t vtmp;\n"
    micro += "  vfloat32m1_t "

    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "

    for nr in range(0, NR):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"

    micro += "  vfloat32m1_t "

    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"

    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit(C{mr}{nr}, 0, vlength);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC(C{mr}{nr}, &Crref({mr*vlen}, {nr}), vlength);\n"
    micro += "  }\n\n"
    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    for mr in range(0, vMR):
        micro += f"    vload(A{mr}, vtmp, &Ar[bA + {mr * vlen}], vlength);\n"

   
    if rv_mod == "NORMAL":
        for nr in range(0, NR):
            micro += f"    vdup(B{nr}, Br[bB+{nr}], vlength);\n"
    elif rv_mod == "GATHER":
        micro += f"    vload(vtmp, vtmp, &Br[bB], nr_vl);\n"
        for nr in range(0, NR):
            micro += f"    gather(B{nr}, vtmp, {nr}, nr_vl);\n"


    for mr in range(0, vMR):
        for nr in range(0, NR):
            if rv_mod == "BROADCAST":
                micro += f"    vupdate(C{mr}{nr}, A{mr}, Br[bB + {nr}], vlength); \n"
            else:
                micro += f"    vupdate(C{mr}{nr}, A{mr}, B{nr}, vlength); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC(&Crref({mr*vlen},{nr}), C{mr}{nr}, vlength); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];\n"

    micro += "}\n"

    cfile.write(micro)
    
    return True


def micro_kernel_int32_riscv_generator(rvv, rv_mod, arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen
    
    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*ldC+i]\n"
        micro += f"#define min(a,b) a >= b ? a : b\n"

        if rvv == "1.0":
            micro += "#define vinit(vreg, value, vlength)        vreg = __riscv_vmv_v_x_i32m1(value, vlength)\n"
            micro += "#define vloadC(vreg, mem,  vlength)        vreg = __riscv_vle32_v_i32m1 (mem, vlength)\n"
            micro += "#define vstoreC(mem, vreg, vlength)               __riscv_vse32_v_i32m1 (mem, vreg, vlength)\n"
            micro += "#define vload(vreg, vtmp, mem, vlength)    vreg = __riscv_vle32_v_i32m1 (mem, vlength); \n"
            micro += "#define vdup(vreg, mem, vlength)           vreg = __riscv_vmv_v_x_i32m1(mem, vlength)\n"
            if rv_mod == "GATHER":
                micro += "#define gather(vreg1, vreg2, index, vlength)   vreg1 = __riscv_vrgather_vx_i32m1(vreg2, index, vlength)\n"

            if rv_mod == "BROADCAST":
                micro += "#define vupdate(vreg1, vreg2, value, vlength) vreg1=__riscv_vmacc_vx_i32m1(vreg1, value, vreg2, vlength)\n\n"
            else:
                micro += "#define vupdate(vreg1, vreg2, vreg3, vlength) vreg1=__riscv_vmacc_vv_i32m1(vreg1, vreg2, vreg3, vlength)\n\n"
        else:
            #TODO: Implementation for rvv0.7
            return
#
    micro +=f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int mr, int nr, int kc, int32_t  *Ar, int32_t *Br, int32_t *Cor, int32_t betaI, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
   
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro +=f"  int vlength = {vlen};\n"
    if rv_mod == "GATHER":
        micro +=f"  int nr_vl = min(vlength, nr);\n"
    micro += "  vint32m1_t vtmp;\n"
    micro += "  vint32m1_t "

    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "

    for nr in range(0, NR):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"

    micro += "  vint32m1_t "

    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"

    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit(C{mr}{nr}, 0, vlength);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC(C{mr}{nr}, &Crref({mr*vlen}, {nr}), vlength);\n"
    micro += "  }\n\n"
    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    for mr in range(0, vMR):
        micro += f"    vload(A{mr}, vtmp, &Ar[bA + {mr * vlen}], vlength);\n"

   
    if rv_mod == "NORMAL":
        for nr in range(0, NR):
            micro += f"    vdup(B{nr}, Br[bB+{nr}], vlength);\n"
    elif rv_mod == "GATHER":
        micro += f"    vload(vtmp, vtmp, &Br[bB], nr_vl);\n"
        for nr in range(0, NR):
            micro += f"    gather(B{nr}, vtmp, {nr}, nr_vl);\n"


    for mr in range(0, vMR):
        for nr in range(0, NR):
            if rv_mod == "BROADCAST":
                micro += f"    vupdate(C{mr}{nr}, A{mr}, Br[bB + {nr}], vlength); \n"
            else:
                micro += f"    vupdate(C{mr}{nr}, A{mr}, B{nr}, vlength); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC(&Crref({mr*vlen},{nr}), C{mr}{nr}, vlength); \n"

    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];\n"

    micro += "}\n"

    cfile.write(micro)
    
    return True


def micro_kernel_int8_int32_riscv_generator(rvv, rv_mod, arch, MR, NR, lane, dtype, vlen, macros, cfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*ldC+i]\n"
        micro += f"#define min(a,b) a >= b ? a : b\n"
        micro += "#include <riscv_vector.h>\n"

        if rvv == "1.0":
            micro += "#define vinit(vreg, value, vlength) vreg = __riscv_vmv_v_x_i32m1(value, vlength)\n"
            micro += "#define vloadC(vreg, mem,  vlength) vreg = __riscv_vle32_v_i32m1(mem, vlength)\n"
            micro += "#define vstoreC(mem, vreg, vlength)        __riscv_vse32_v_i32m1(mem, vreg, vlength)\n"
            micro += "#define vload(vreg, vtmp, mem, vlength, vlength2) vtmp = __riscv_vle8_v_i8mf4(mem, vlength); \\\n\
                                                    vreg = __riscv_vwadd_vx_i16mf2(vtmp, 0, vlength2)\n"
            micro += "#define vdup(vreg, mem, vlength) vreg = __riscv_vmv_v_x_i16mf2(mem, vlength)\n"
            if rv_mod == "GATHER":
                micro += "#define gather(vreg1, vreg2, index, vlength)   vreg1 = __riscv_vrgather_vx_i16mf2(vreg2, index, vlength)\n"

            if rv_mod == "BROADCAST":
                micro += "#define vupdate(vreg1, vreg2, value, vlength) vreg1=__riscv_vwmacc_vx_i32m1(vreg1, value, vreg2, vlength)\n\n"
            else:
                micro += "#define vupdate(vreg1, vreg2, vreg3, vlength) vreg1=__riscv_vwmacc_vv_i32m1(vreg1, vreg2, vreg3, vlength)\n\n"
        else:
            #TODO: Implementation for rvv0.7
            return

    micro += f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){{\n"

    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int ldC;\n"
    micro += f"  const int MR={MR};\n"
    micro += f"  const int NR={NR};\n"
    micro += f"  int32_t Ctmp[{MR}*{NR}];\n"
    micro += f"  int32_t beta;\n"
    micro += f"  int32_t *Cr;\n"
    micro += "  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}\n"
    micro +=f"  int vlength = {vlen};\n"
    micro +=f"  int vlengthx2 = {2*vlen};\n"
    if rv_mod == "GATHER":
        micro +=f"  int nr_vl = min(vlength, nr);\n"
        micro += "  vint16mf2_t vtmp2;\n"

    micro += "  vint8mf4_t vtmp;\n"
    micro += "  vint16mf2_t "

    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "

    for nr in range(0, NR):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"

    micro += "  vint32m1_t "

    cregs  = ""
    for mr in range(0, vMR):
        for nr in range(0, NR):
            cregs += f" C{mr}{nr}, "
    micro += cregs[:-2] + ";\n\n"

    micro += "  if (beta == 0) {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vinit(C{mr}{nr}, 0, vlength);\n"
    micro += "  } else {\n"
    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"    vloadC(C{mr}{nr}, &Crref({mr*vlen}, {nr}), vlength);\n"
    micro += "  }\n\n"
    
    micro += "  for (pr=0; pr<kc; pr++) { // Loop L6\n"

    for mr in range(0, vMR):
        micro += f"    vload(A{mr}, vtmp, &Ar[bA + {mr * vlen}], vlengthx2, vlength);\n"

    if rv_mod == "NORMAL":
        for nr in range(0, NR):
            micro += f"    vdup(B{nr}, Br[bB+{nr}], vlength);\n"
    elif rv_mod == "GATHER":
        micro += f"    vload(vtmp2, vtmp, &Br[bB], nr_vl, nr_vl);\n"
        for nr in range(0, NR):
            micro += f"    gather(B{nr}, vtmp2, {nr}, nr_vl);\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            if rv_mod == "BROADCAST":
                micro += f"    vupdate(C{mr}{nr}, A{mr}, Br[bB + {nr}], vlength); \n"
            else:
                micro += f"    vupdate(C{mr}{nr}, A{mr}, B{nr}, vlength); \n"

    micro += f"    bA+={MR};\n"
    micro += f"    bB+={NR};\n"

    micro += "  }\n\n"

    for mr in range(0, vMR):
        for nr in range(0, NR):
            micro += f"  vstoreC(&Crref({mr*vlen},{nr}), C{mr}{nr}, vlength); \n"

    micro += f"\n"
    micro += f"  if (mr != MR || nr != NR)\n"
    micro += f"    for (int j = 0; j < nr; j++)\n"
    micro += f"      for (int i = 0; i < mr; i++)\n"
    micro += f"        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];\n"

    micro += "}\n\n"

    cfile.write(micro)
    
    return True



def main() -> int:
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', '-a', required=True,  type=str, action='store', help="Architecture selection [armv8|riscv]")
    parser.add_argument('--dup',  '-p', required=False, action='store_true', help="Intrinsics Lane variation")
    parser.add_argument('--cpu',  '-c', required=False, type=str, action='store', help="CPU model")
    parser.add_argument('--op_b', '-o', required=False, type=str, action='store', help="RISCV Mod")
    parser.add_argument('--vlen', '-v', required=False, type=int, help="Vector length")
    
    args   = parser.parse_args()
    arch   = args.arch
    lane   = not args.dup
    vlen   = int(args.vlen)
    cpu    = args.cpu
    rvv    = "0.7"

    rv_mod = "NORMAL"

    if args.op_b == "GATHER" or args.op_b == "gather": 
        rv_mod = "GATHER"
    elif args.op_b == "BROADCAST" or args.op_b == "broadcast": 
        rv_mod = "BROADCAST"
    
    if arch == "riscv":
        if cpu == "K1" or cpu == "XUANTIE_908": rvv = "1.0"

    for dtype in ["int8_int32", "fp32", "int32"]:

        cpath = os.path.dirname(os.path.abspath(__file__)) + "/ukernels/uKernels_intrinsic_"+dtype+".c"
        hpath = os.path.dirname(os.path.abspath(__file__)) + "/ukernels/uKernels_intrinsic_"+dtype+".h"
    
        hfile = open(hpath, "w")
        cfile = open(cpath, "w")

        macros = True
        micro  = []

        if dtype == "int8_int16" or dtype == "fp16":
            dvlen = vlen // 16
        else:
            dvlen = vlen // 32

        maxMR   = dvlen // 2 * 10 + 1
        maxNR   = dvlen // 2 * 10 + 1
        stepMR  = dvlen
        stepNR  = dvlen

        if arch == "armv8":
            if "int8_int32" in dtype: stepNR = 4

            for mr in range(stepMR, maxMR, stepMR):
                for nr in range(stepNR, maxNR, stepNR):
                    if dtype == "int8_int32":
                        micro_kernel_int8_int32_s8_armv8_generator(arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                        #micro_kernel_int8_int32_u8_generator(arch, mr, nr, lane, dtype, vlen, macros, cfile, hfile)
                    elif dtype == "int8_int16":
                        if mr % 8 != 0:
                            continue
                        micro_kernel_int8_int16_armv8_generator(arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    elif dtype == "fp32":
                        micro_kernel_fp32_armv8_generator(arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    elif dtype == "int32":
                        micro_kernel_int32_armv8_generator(arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    else:
                        micro_kernel_fp16_armv8_generator(arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    
                    micro.append((mr, nr))
                    macros = False
        else:
            stepNR = 2;
            maxNR  = 9;
            for mr in range(stepMR, maxMR, stepMR):
                for nr in range(stepNR, maxNR, stepNR):
                    if dtype == "int8_int32":
                        micro_kernel_int8_int32_riscv_generator(rvv, rv_mod, arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    elif dtype == "fp32":
                        micro_kernel_fp32_riscv_generator(rvv, rv_mod, arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    elif dtype == "int32":
                        micro_kernel_int32_riscv_generator(rvv, rv_mod, arch, mr, nr, lane, dtype, dvlen, macros, cfile)
                    
                    micro.append((mr, nr))
                    macros = False
    
        generate_selector(arch, micro, maxMR, maxNR, dtype, cfile, hfile)

        cfile.close()
        hfile.close()

        print("")
        print("+============================================================+")
        if dtype == "int8_int32":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sINT8-INT32%s              |" % (bcolor.WARNING, bcolor.ENDC))
        elif dtype == "fp32":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sFP32%s                    |" % (bcolor.WARNING, bcolor.ENDC))
        elif dtype == "int32":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sINT32%s                    |" % (bcolor.WARNING, bcolor.ENDC))
        elif dtype == "int8_int16":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sINT8-INT16%s              |" % (bcolor.WARNING, bcolor.ENDC))
        else:
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sFP16%s                    |" % (bcolor.WARNING, bcolor.ENDC))
        print("+=======================================+====================+")
        print("|  [*] ARCHITECTURE                     | {:<18} |".format(arch)) 
        print("|  [*] LANE                             | {:<18} |".format("True" if lane else "False")) 
        print("|  [*] DATA TYPE                        | {:<18} |".format(dtype))
        print("|  [*] VLEN                             | {:<18} |".format(vlen))
        print("+=======================================+==========+=========+")

        nmicro = 1
        for mr, nr in micro:
            print(f"|    [%2d] Micro-kernel                  | %s%-8d%s | %s%-8d%s|" % (nmicro, bcolor.OKCYAN, mr, bcolor.ENDC, bcolor.OKCYAN, nr, bcolor.ENDC))
            nmicro += 1
        print("+============================================================+")

    return 0;


if __name__ == '__main__':
    sys.exit(main())
