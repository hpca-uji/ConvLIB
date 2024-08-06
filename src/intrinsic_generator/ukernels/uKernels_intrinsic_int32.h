
#include <riscv_vector.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*uk_intrinsic_int32)(int, int, int, int32_t *, int32_t *, int32_t *, int32_t, int);

typedef struct uk_config_int32 {
  int mr_pool[128];
  int nr_pool[128];
  int uk_num;
} uk_config_int32_t;


uk_intrinsic_int32 *new_uk_intrinsic_selector_int32();

uk_config_int32_t *new_uk_intrinsic_config_int32();

void uk_intrinsic_selector_int32(int mr, int nr, uk_intrinsic_int32 *uk_vec, uk_intrinsic_int32 *ukr);
