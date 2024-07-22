#include "ukernels.h"

void generate_matrix_fp32( int orderM, size_t m, size_t n, float *M, size_t ldM );
void generate_matrix_int8( int orderM, size_t m, size_t n, int8_t *M, size_t ldM );
void generate_matrix_int16( int orderM, size_t m, size_t n, int16_t *M, size_t ldM );
void generate_matrix_int32( int orderM, size_t m, size_t n, int32_t *M, size_t ldM );
  
void print_tensor4D_fp32( char *name, int m1, int m2, int m3, int m4, float *T, int ldT1, int ldT2, int ldT3 );
void print_tensor4D_int32( char *name, int m1, int m2, int m3, int m4, int32_t *T, int ldT1, int ldT2, int ldT3 );
void print_tensor4D_int8( char *name, int m1, int m2, int m3, int m4, int8_t *T, int ldT1, int ldT2, int ldT3 );

#ifdef FP16
  void generate_matrix_fp16( int orderM, size_t m, size_t n, float16_t *M, size_t ldM );
  void print_tensor4D_fp16( char *name, int m1, int m2, int m3, int m4, float16_t *T, int ldT1, int ldT2, int ldT3 );
#endif

int    print_matrix( char *, char, int, int, AB_TYPE *, int );
int    generate_tensor4D( int, int, int, int, AB_TYPE *, int, int, int );

double dclock();

void convert_row2col(C_TYPE *Ai, C_TYPE *Ao, int m, int n);
void convert_col2row(C_TYPE *Ai, C_TYPE *Ao, int m, int n);

