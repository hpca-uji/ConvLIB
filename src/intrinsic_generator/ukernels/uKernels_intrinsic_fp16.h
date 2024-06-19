#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_fp16)(int, float16_t *, float16_t *, float16_t *, float16_t, int );

uk_intrinsic_fp16 *new_uk_intrinsic_selector_fp16();

void uk_intrinsic_selector_fp16(int mr, int nr, uk_intrinsic_fp16 *uk_vec, uk_intrinsic_fp16 *ukr);