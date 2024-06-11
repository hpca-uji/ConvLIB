#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_int8_int32)(int, int8_t *, int8_t *, int32_t *, int32_t, int);

uk_intrinsic_int8_int32 *new_uk_intrinsic_selector_int8_int32();

void uk_intrinsic_selector_int8_int32(int mr, int nr, uk_intrinsic_int8_int32 *uk_vec, uk_intrinsic_int8_int32 *ukr);

void ukernel_intrinsic_4x4_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int Clda);
