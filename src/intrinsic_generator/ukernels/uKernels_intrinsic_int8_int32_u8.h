#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_int8_int32_u8)(int, int8_t *, int8_t *, int32_t *, int32_t, int);

uk_intrinsic_int8_int32_u8 *new_uk_intrinsic_selector_int8_int32_u8();

void uk_intrinsic_selector_int8_int32_u8(int mr, int nr, uk_intrinsic_int8_int32_u8 *uk_vec, uk_intrinsic_int8_int32_u8 *ukr);
