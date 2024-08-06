
#include <arm_neon.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*uk_intrinsic_fp32)(int, int, int, float *, float *, float *, float, int);

typedef struct uk_config_fp32 {
  int mr_pool[128];
  int nr_pool[128];
  int uk_num;
} uk_config_fp32_t;


uk_intrinsic_fp32 *new_uk_intrinsic_selector_fp32();

uk_config_fp32_t *new_uk_intrinsic_config_fp32();

void uk_intrinsic_selector_fp32(int mr, int nr, uk_intrinsic_fp32 *uk_vec, uk_intrinsic_fp32 *ukr);
