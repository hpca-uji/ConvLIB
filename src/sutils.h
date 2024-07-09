#include "ukernels.h"

void print_tensor4D_fp32( char *name, int m1, int m2, int m3, int m4, float *T, int ldT1, int ldT2, int ldT3 );
void print_tensor4D_int32( char *name, int m1, int m2, int m3, int m4, int32_t *T, int ldT1, int ldT2, int ldT3 );
void print_tensor4D_int8( char *name, int m1, int m2, int m3, int m4, int8_t *T, int ldT1, int ldT2, int ldT3 );

#ifdef FP16
  void print_tensor4D_fp16( char *name, int m1, int m2, int m3, int m4, float16_t *T, int ldT1, int ldT2, int ldT3 );
#endif

int    print_matrix( char *, char, int, int, AB_TYPE *, int );
int    generate_tensor4D( int, int, int, int, AB_TYPE *, int, int, int );

double dclock();

void convert_row2col(int32_t *Ai, int32_t *Ao, int m, int n);
void convert_col2row(int32_t *Ai, int32_t *Ao, int m, int n);

