#ifndef UKERNELS_H
#define UKERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#if defined(NQ_FP32) || defined(FQ_FP32)
  #if defined(ARMV8)
    #include "asm_generator/ukernels/gemm_ukernel_headers.h"
  #else 
    #include "intrinsic_generator/ukernels/uKernels_intrinsic_fp32.h"
  #endif
#elif FP16
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_fp16.h"
#else
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_int8_int32.h"
  #include "intrinsic_generator/ukernels/uKernels_intrinsic_int32.h"
#endif

//Algorithms
#define CONVDIRECT 0
#define LOWERING   1
#define CONVGEMM   2
#define GEMM       3

//GEMMs for Lowering aproach
#define B3A2C0     4
#define A3B2C0     5
#define BLIS       6
#define OPENBLAS   7
#define SDOT_GEMM  8

#define UNKNOWN    101

#ifdef NQ_FP32
  #define AB_TYPE        float
  #define AB_PACK_TYPE   float
  #define C_TYPE         float
  #ifdef ARMV8
    #define UK_TYPE        uk_asm_fp32
    #define UK_EDGE_TYPE   uk_asm_edge_fp32
  #else
    #define UK_TYPE        uk_intrinsic_fp32
    #define UK_EDGE_TYPE   uk_intrinsic_fp32
  #endif
  #define UK_CONFIG        uk_config_fp32_t
#elif FQ_FP32
  #define AB_TYPE        int8_t
  #define AB_PACK_TYPE   float
  #define C_TYPE         float
  #ifdef ARMV8
    #define UK_TYPE        uk_asm_fp32
    #define UK_EDGE_TYPE   uk_asm_edge_fp32
  #else
    #define UK_TYPE        uk_intrinsic_fp32
    #define UK_EDGE_TYPE   uk_intrinsic_fp32
  #endif
  #define UK_CONFIG        uk_config_fp32_t
#elif NQ_INT32
  #define AB_TYPE        int32_t
  #define AB_PACK_TYPE   int32_t
  #define C_TYPE         int32_t
  #define UK_TYPE        uk_intrinsic_int32
  #define UK_EDGE_TYPE   uk_intrinsic_int32
  #define UK_CONFIG      uk_config_int32_t
#elif FQ_INT32
  #define AB_TYPE        int8_t
  #define AB_PACK_TYPE   int32_t
  #define C_TYPE         int32_t
  #define UK_TYPE        uk_intrinsic_int32
  #define UK_EDGE_TYPE   uk_intrinsic_int32
  #define UK_CONFIG      uk_config_int32_t
#elif Q_INT8_INT32
  #define AB_TYPE        int8_t
  #define AB_PACK_TYPE   int8_t
  #define C_TYPE         int32_t
  #define UK_TYPE        uk_intrinsic_int8_int32
  #define UK_EDGE_TYPE   uk_intrinsic_int8_int32
  #define UK_CONFIG      uk_config_int8_int32_t
#elif FP16
  #define AB_TYPE        float16_t
  #define AB_PACK_TYPE   float16_t
  #define C_TYPE         float16_t
  #define UK_TYPE        uk_intrinsic_fp16
  #define UK_EDGE_TYPE   uk_intrinsic_fp16
  #define UK_CONFIG      uk_config_fp16_t
#endif

void fselector(int MR, int NR, int algorithm, int gemm, UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec, 
	       UK_TYPE *uk, UK_EDGE_TYPE *uk_edge);

void generic_microkernel(int mr, int nr, int MR, int NR, AB_PACK_TYPE *A, AB_PACK_TYPE *B, 
		         C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
			 C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge);

void sdot_microkernel(int mr, int nr, int MR, int NR, AB_PACK_TYPE *A, AB_PACK_TYPE *B, 
		         C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
			 C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge);

//Special micro-kernels headers
void uk_intrinsic_quantize_int8_4x4_sdot(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC);
void uk_intrinsic_quantize_int8_4x16_sdot(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC);
void uk_intrinsic_quantize_int8_6x16_sdot(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC);

#ifdef A78AE
  void ukernel_intrinsic_16x8_A78_fp16(int kc, float16_t *Ar, float16_t *Br, float16_t *Cr, float16_t beta, int Clda);
#endif

#endif
