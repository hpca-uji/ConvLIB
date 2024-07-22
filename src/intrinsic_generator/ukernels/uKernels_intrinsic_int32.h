#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_int32)(int, int32_t *, int32_t *, int32_t *, int32_t, int );

uk_intrinsic_int32 *new_uk_intrinsic_selector_int32();

void uk_intrinsic_selector_int32(int mr, int nr, uk_intrinsic_int32 *uk_vec, uk_intrinsic_int32 *ukr);
