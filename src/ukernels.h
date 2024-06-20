#ifndef UKERNELS_H
#define UKERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#include "asm_generator/ukernels/gemm_ukernel_headers.h"

#if FP16
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_fp16.h"
#else
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_int8_int32_s8.h"
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_int8_int32_u8.h"
#endif

#define CONVDIRECT 0
#define LOWERING   1
#define CONVGEMM   2

#ifdef FP32
  #define AB_TYPE        float
  #define C_TYPE         float
  #define UK_TYPE        uk_asm_fp32
  #define UK_EDGE_TYPE   uk_asm_edge_fp32
#elif FP16
  #define AB_TYPE        float16_t
  #define C_TYPE         float16_t
  #define UK_TYPE        uk_intrinsic_fp16
  #define UK_EDGE_TYPE   uk_intrinsic_fp16
#elif INT8_INT32_U8
  #define AB_TYPE       int8_t
  #define C_TYPE        int32_t
  #define UK_TYPE       uk_intrinsic_int8_int32_u8
  #define UK_EDGE_TYPE  uk_intrinsic_int8_int32_u8
#elif INT8_INT32_S8
  #define AB_TYPE       int8_t
  #define C_TYPE        int32_t
  #define UK_TYPE       uk_intrinsic_int8_int32_s8
  #define UK_EDGE_TYPE  uk_intrinsic_int8_int32_s8
#endif

void fselector(int MR, int NR, UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec, 
	       UK_TYPE *uk, UK_EDGE_TYPE *uk_edge);

void generic_microkernel(int mr, int nr, int MR, int NR, AB_TYPE *A, AB_TYPE *B, 
		         C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
			 C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge);

#endif
