#ifndef UKERNELS_H
#define UKERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "asm_generator/ukernels/gemm_ukernel_headers.h"
#include "intrinsic_generator/ukernels/uKernels_intrinsic_int8_int32.h"

#define CONVDIRECT 0
#define LOWERING   1
#define CONVGEMM   2

#ifdef FP32
  #define AB_TYPE        float
  #define C_TYPE         float
  #define UK_TYPE        uk_asm_fp32
  #define UK_EDGE_TYPE   uk_asm_edge_fp32
#elif INT8_INT32
  #define AB_TYPE       int8_t
  #define C_TYPE        int32_t
  #define UK_TYPE       uk_intrinsic_int8_int32
  #define UK_EDGE_TYPE  uk_intrinsic_int8_int32
#endif

void generic_microkernel(int mr, int nr, int MR, int NR, AB_TYPE *A, AB_TYPE *B, 
		         C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
			 C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge);

#endif
