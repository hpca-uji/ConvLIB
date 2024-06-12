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

def generate_selector(micros, maxMR, maxNR, dtype, cfile, hfile):
    gen  =  "\n"
    gen += f"uk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}() {{ \n"
    gen += f"  uk_intrinsic_{dtype} *uk_vec = (uk_intrinsic_{dtype} *)malloc(sizeof(uk_intrinsic_{dtype}) * {maxMR} * {maxNR});\n"
    for mr, nr in micros:
        gen += f"  uk_vec[{nr}*{maxMR} + {mr}] = ukernel_intrinsic_{mr}x{nr}_{dtype};\n"
    gen +=  "  return uk_vec;\n"
    gen +=  "}\n\n"

    gen += f"void uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr) {{\n"
    gen += f"  (*ukr) = uk_vec[nr*{maxMR} + mr];\n"
    gen +=  "}\n"
    
    cfile.write(gen)
    
    return True

def micro_kernel_int8_int32_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile, hfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Headers
    #--------------------------------------
    if macros:
        hfile.write(f"#include <arm_neon.h>\n")
        hfile.write(f"#include <stdio.h>\n")
        hfile.write(f"#include <stdint.h>\n")
        hfile.write(f"#include <stdlib.h>\n\n")
        hfile.write(f"\ntypedef void (*uk_intrinsic_{dtype})(int, int8_t *, int8_t *, int32_t *, int32_t, int);\n")
        hfile.write(f"\nuk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}();\n")
        hfile.write(f"\nvoid uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr);\n")
        #hfile.write(f"\nvoid ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int Clda);\n")

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*Clda+i]\n"
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
    
    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int Clda){\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  int8x8_t  vtmp;\n"
    micro += "  int16x4_t "

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

    micro += "}\n"
    
    cfile.write(micro)

    return True


def micro_kernel_int8_int16_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile, hfile):
    
    vlen = 8

    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Headers
    #--------------------------------------
    if macros:
        hfile.write(f"#include <arm_neon.h>\n")
        hfile.write(f"#include <stdio.h>\n")
        hfile.write(f"#include <stdint.h>\n")
        hfile.write(f"#include <stdlib.h>\n\n")
        hfile.write(f"\ntypedef void (*uk_intrinsic_{dtype})(int, int8_t *, int8_t *, int16_t *, int16_t, int);\n")
        hfile.write(f"\nuk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}();\n")
        hfile.write(f"\nvoid uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr);\n")
        #hfile.write(f"\nvoid ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, int8_t  *Ar, int8_t *Br, int16_t *Cr, int16_t beta, int Clda);\n")

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n\n"

        micro += f"#define Crref(i,j) Cr[j*Clda+i]\n"
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
    
    micro += "void ukernel_intrinsic_"+str(MR)+"x"+str(NR)+"_"+dtype+"(int kc, int8_t  *Ar, int8_t *Br, int16_t *Cr, int16_t beta, int Clda){\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
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

    micro += "}\n"
    
    cfile.write(micro)

    return True


def micro_kernel_fp32_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile, hfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Headers
    #--------------------------------------
    if macros:
        hfile.write(f"#include <arm_neon.h>\n")
        hfile.write(f"#include <stdio.h>\n")
        hfile.write(f"#include <stdint.h>\n")
        hfile.write(f"#include <stdlib.h>\n\n")
        hfile.write(f"\ntypedef void (*uk_intrinsic_{dtype})(int, float *, float *, float *, float, int );\n")
        hfile.write(f"\nuk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}();\n")
        hfile.write(f"\nvoid uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr);\n")
        #hfile.write(f"\nvoid ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, float  *Ar, float *Br, float *Cr, float beta, int Clda);\n") 

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*Clda+i]\n"

        micro += f"#define vinit_{dtype}(vreg, value)   vreg = vmovq_n_f32(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_f32(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_f32(mem, vreg)\n"
        micro += f"#define vload_{dtype}(vreg, mem) vreg=vld1q_f32(mem)\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1=vfmaq_laneq_f32(vreg1, vreg2, vreg3, lane)\n\n"

    micro += f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, float  *Ar, float *Br, float *Cr, float beta, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
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
    micro += "}\n"

    cfile.write(micro)
    
    return True

def micro_kernel_fp16_generator(arch, MR, NR, lane, dtype, vlen, macros, cfile, hfile):
    
    vMR = MR // vlen
    vNR = NR // vlen

    #--------------------------------------
    # Headers
    #--------------------------------------
    if macros:
        hfile.write(f"#include <arm_neon.h>\n")
        hfile.write(f"#include <stdio.h>\n")
        hfile.write(f"#include <stdint.h>\n")
        hfile.write(f"#include <stdlib.h>\n\n")
        hfile.write(f"\ntypedef void (*uk_intrinsic_{dtype})(int, float16_t *, float16_t *, float16_t *, float16_t, int );\n")
        hfile.write(f"\nuk_intrinsic_{dtype} *new_uk_intrinsic_selector_{dtype}();\n")
        hfile.write(f"\nvoid uk_intrinsic_selector_{dtype}(int mr, int nr, uk_intrinsic_{dtype} *uk_vec, uk_intrinsic_{dtype} *ukr);\n")
        #hfile.write(f"\nvoid ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, float16_t  *Ar, float16_t *Br, float16_t *Cr, float16_t beta, int Clda);\n") 

    #--------------------------------------
    # Micro-kernel implementation
    #--------------------------------------
    micro = ""
    
    if macros:
        micro += f"#include \"uKernels_intrinsic_{dtype}.h\"\n"
        micro += f"#define Crref(i,j)  Cr[j*Clda+i]\n"

        micro += f"#define vinit_{dtype}(vreg, value)   vreg = vmovq_n_f16(value)\n"
        micro += f"#define vloadC_{dtype}(vreg, mem)    vreg=vld1q_f16(mem)\n"
        micro += f"#define vstoreC_{dtype}(mem, vreg)   vst1q_f16(mem, vreg)\n"
        micro += f"#define vload_{dtype}(vreg, mem) vreg=vld1q_f16(mem)\n"
        micro += f"#define vupdate_lane_{dtype}(vreg1, vreg2, vreg3, lane) vreg1=vfmaq_laneq_f16(vreg1, vreg2, vreg3, lane)\n\n"

    micro += f"void ukernel_intrinsic_{MR}x{NR}_{dtype}(int kc, float16_t *Ar, float16_t *Br, float16_t *Cr, float16_t beta, int Clda){{\n"
    micro += "  int pr, bA = 0, bB = 0;\n"
    micro += "  float16x8_t  vtmp;\n"
    micro += "  float16x8_t "
    abregs = ""
    for mr in range(0, vMR):
        abregs += f" A{mr}, "
    blim =vNR
    if (NR % vlen != 0):
        blim += 1

    for nr in range(0, blim):
        abregs += f" B{nr}, "
    micro += abregs[:-2] + ";\n"
    micro += "  float16x8_t "
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
    micro += "}\n"

    cfile.write(micro)
    
    return True

def main() -> int:
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch',  '-a', required=True,  type=str, action='store', help="Architecture selection [armv8|riscv]")
    parser.add_argument('--dup',  '-p', required=False, action='store_true', help="Intrinsics Lane variation")
    
    args   = parser.parse_args()
    arch  = args.arch
    lane  = not args.dup

   
    for dtype in ["int8_int16", "int8_int32", "fp32", "fp16"]:

        cpath = os.path.dirname(os.path.abspath(__file__)) + "/ukernels/uKernels_intrinsic_"+dtype+".c"
        hpath = os.path.dirname(os.path.abspath(__file__)) + "/ukernels/uKernels_intrinsic_"+dtype+".h"
    
        hfile = open(hpath, "w")
        cfile = open(cpath, "w")

        macros = True
        micro  = []

        maxMR   = 28
        maxNR   = 28
        stepMR  = 4
        stepNR  = 4
        #Generating Micro-kernels
        if not lane and dtype == "int8_int32" or dtype == "int8_int16":
            maxNR = 13
            stepNR = 1

        if dtype == "int8_int16" or dtype == "fp16":
            maxMR  = 41
            maxNR  = 41 
            stepMR = 8
            stepNR = 8

        for mr in range(stepMR, maxMR, stepMR):
            for nr in range(stepNR, maxNR, stepNR):
                if dtype == "int8_int32":
                    vlen = 4
                    micro_kernel_int8_int32_generator(arch, mr, nr, lane, dtype, vlen, macros, cfile, hfile)
                elif dtype == "int8_int16":
                    vlen = 8
                    if mr % 8 != 0:
                        continue
                    micro_kernel_int8_int16_generator(arch, mr, nr, lane, dtype, vlen, macros, cfile, hfile)
                elif dtype == "fp32":
                    vlen = 4
                    micro_kernel_fp32_generator(arch, mr, nr, lane, dtype, vlen, macros, cfile, hfile)
                else:
                    vlen = 8
                    micro_kernel_fp16_generator(arch, mr, nr, lane, dtype, vlen, macros, cfile, hfile)

                micro.append((mr, nr))
                macros = False
    
        generate_selector(micro, maxMR, maxNR, dtype, cfile, hfile)

        cfile.close()
        hfile.close()

        print("")
        print("+============================================================+")
        if dtype == "int8_int32":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sINT8-INT32%s              |" % (bcolor.WARNING, bcolor.ENDC))
        elif dtype == "int8_int16":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sINT8-INT16%s              |" % (bcolor.WARNING, bcolor.ENDC))
        elif dtype == "fp32":
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sFP32%s                    |" % (bcolor.WARNING, bcolor.ENDC))
        else:
            print(f"| INTRINSICS MICRO-KERNEL GENERATOR: %sFP16%s                    |" % (bcolor.WARNING, bcolor.ENDC))
        print("+=======================================+====================+")
        print("|  [*] ARCHITECTURE                     | {:<18} |".format(arch)) 
        print("|  [*] LANE                             | {:<18} |".format("True" if lane else "False")) 
        print("|  [*] DATA TYPE                        | {:<18} |".format(dtype))
        print("+=======================================+==========+=========+")

        nmicro = 1
        for mr, nr in micro:
            print(f"|    [%2d] Micro-kernel                  | %s%-8d%s | %s%-8d%s|" % (nmicro, bcolor.OKCYAN, mr, bcolor.ENDC, bcolor.OKCYAN, nr, bcolor.ENDC))
            nmicro += 1
        print("+============================================================+")

    return 0;


if __name__ == '__main__':
    sys.exit(main())
