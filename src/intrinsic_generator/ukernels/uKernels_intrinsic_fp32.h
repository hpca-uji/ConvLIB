#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_fp32)(int, float *, float *, float *, float, int );

uk_intrinsic_fp32 *new_uk_intrinsic_selector_fp32();

void uk_intrinsic_selector_fp32(int mr, int nr, uk_intrinsic_fp32 *uk_vec, uk_intrinsic_fp32 *ukr);

void ukernel_intrinsic_4x4_fp32(int kc, float  *Ar, float *Br, float *Cr, float beta, int Clda);
