#include <stdlib.h>
#include <sys/time.h>

#include "ukernels.h"

#define Trow4D(a1,a2,a3,a4)       T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4) ]
#define Trow5D(a1,a2,a3,a4,a5)    T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4)*(ldT4) + (a5) ]
#define Trow6D(a1,a2,a3,a4,a5,a6) T[ (a1)*(ldT1) + (a2)*(ldT2) + (a3)*(ldT3) + (a4)*(ldT4) + (a5)*(ldT5) + (a6) ]

#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

void convert_row2col(C_TYPE *Ai, C_TYPE *Ao, int m, int n) {
  //Arow: Input  Matrix in row-major
  //Acol: Output Matrix in col-major

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      Ao[j * m + i] = Ai[i * n + j];
}

void convert_col2row(C_TYPE *Ai, C_TYPE *Ao, int m, int n) {
  //Arow: Input  Matrix in col-major
  //Acol: Output Matrix in row-major

  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++)
      Ao[i * n + j] = Ai[j * m + i];
}

double dclock() {
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}

/*===========================================================================*/
void generate_tensor4D( int m1, int m2, int m3, int m4, AB_TYPE *T, int ldT1, int ldT2, int ldT3 ) {
 //Generate a 4D tensor with random entries. 
  int i1, i2, i3, i4;
  float tmp;
  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ ) {
  #if defined(INT8_INT32_S8) || defined(INT8_INT32_U8)
    Trow4D(i1,i2,i3,i4) = (AB_TYPE)rand() % (12) + 1;
  #elif FP16
    tmp = rand() / RAND_MAX;
    Trow4D(i1,i2,i3,i4) = (AB_TYPE)(tmp + 1);
  #else
    Trow4D(i1,i2,i3,i4) = (AB_TYPE)rand() / RAND_MAX;
  #endif
  }

}

/*===========================================================================*/
void print_tensor4D_fp32( char *name, int m1, int m2, int m3, int m4, float *T, int ldT1, int ldT2, int ldT3 ) {
  int i1, i2, i3, i4;

  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
    printf( "%s[%d,%d,%d,%d] = %8.4e;\n", name, i1, i2, i3, i4, ((double) Trow4D(i1, i2, i3, i4)) );
}

#ifdef FP16
void print_tensor4D_fp16( char *name, int m1, int m2, int m3, int m4, float16_t *T, int ldT1, int ldT2, int ldT3 ) {
  int i1, i2, i3, i4;

  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
    printf( "%s[%d,%d,%d,%d] = %8.4e;\n", name, i1, i2, i3, i4, ((double) Trow4D(i1, i2, i3, i4)) );
}
#endif

void print_tensor4D_int32( char *name, int m1, int m2, int m3, int m4, int32_t *T, int ldT1, int ldT2, int ldT3 ) {
  int i1, i2, i3, i4;
  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
    printf( "%s[%d,%d,%d,%d] = %8d;\n", name, i1, i2, i3, i4, ((int) Trow4D(i1, i2, i3, i4)) );
}

void print_tensor4D_int8( char *name, int m1, int m2, int m3, int m4, int8_t *T, int ldT1, int ldT2, int ldT3 ) {
  int i1, i2, i3, i4;
  for ( i1=0; i1<m1; i1++ )
  for ( i2=0; i2<m2; i2++ )
  for ( i3=0; i3<m3; i3++ )
  for ( i4=0; i4<m4; i4++ )
    printf( "%s[%d,%d,%d,%d] = %8d;\n", name, i1, i2, i3, i4, ((int) Trow4D(i1, i2, i3, i4)) );
}

/*===========================================================================*/
/*===========================================================================*/
void print_matrix( char *name, char orderM, int m, int n, AB_TYPE *M, int ldM )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 *
 */
  int i, j;
  
  if ( orderM=='C' )
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP32)
        printf( "%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(FP64)
        printf( "%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mcol(i,j)) );
#elif defined(INT8_INT32_S8) || defined(INT8_INT32_U8)
        printf( "%s[%d,%d] = %d;\n", name, i, j, ((int) Mcol(i,j)) );
#endif
  else
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ )
#if defined(FP16)
        printf( "%s[%d,%d] = %8.2e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP32)
        printf( "%s[%d,%d] = %14.8e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(FP64)
        printf( "%s[%d,%d] = %22.16e;\n", name, i, j, ((double) Mrow(i,j)) );
#elif defined(INT8_INT32_S8) || defined(INT8_INT32_U8)
	printf( "%s[%d,%d] = %d;\n", name, i, j, ((int) Mrow(i,j)) );
#endif
}
