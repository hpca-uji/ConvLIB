#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


typedef void (*uk_intrinsic_int8_int16)(int, int8_t *, int8_t *, int16_t *, int16_t, int);

uk_intrinsic_int8_int16 *new_uk_intrinsic_selector_int8_int16();

void uk_intrinsic_selector_int8_int16(int mr, int nr, uk_intrinsic_int8_int16 *uk_vec, uk_intrinsic_int8_int16 *ukr);
