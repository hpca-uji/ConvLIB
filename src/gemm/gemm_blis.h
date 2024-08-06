#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../ukernels.h"
//#include "../asm_generator/ukernels/gemm_ukernel_headers.h"
//#include "../intrinsic_generator/ukernels/uKernels_intrinsic_int8_int32.h"

#ifdef OMP_ENABLE
  #include <omp.h>
#endif

#ifdef ARMV8
  #include <arm_neon.h>
#endif

#ifndef max
  #define max(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef min 
  #define min(a,b) ((a) > (b)?(b):(a))
#endif

#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Cgcol(a1,a2) Cg[ (a2)*(ldC)+(a1) ]
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Cgrow(a1,a2) Cg[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]

#define dabs(a) ( (a) > 0.0 ? (a) : -(a) )

#define ROW_MAJOR    101
#define COLUMN_MAJOR 102

#define TRANSPOSE    112
#define NO_TRANSPOSE 111

double gemm_validation(int order, int transA, int transB, AB_TYPE *A, AB_TYPE *B, 
		       int m, int n, int k, C_TYPE alpha, C_TYPE beta, int ldA, int ldB, 
		       int ldC, C_TYPE *C, C_TYPE *Cg);

void gemm_blis_B3A2C0( char, char, char, char, char, size_t, size_t, size_t, 
		       C_TYPE, AB_TYPE *, size_t, AB_TYPE *, size_t, 
		       C_TYPE, C_TYPE *, size_t,  AB_PACK_TYPE *, AB_PACK_TYPE *, 
		       size_t, size_t, size_t, int, int, int, int, C_TYPE *,
		       UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec);


void gemm_blis_A3B2C0( char, char, char, char, char, size_t, size_t, size_t, 
		       C_TYPE, AB_TYPE *, size_t, AB_TYPE *, size_t, 
		       C_TYPE, C_TYPE *, size_t, AB_PACK_TYPE *, AB_PACK_TYPE *, 
		       size_t, size_t, size_t, int, int, int, int, C_TYPE *,
		       UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec);

void dot_gemm( char orderA, char orderB, char orderC,
	       size_t m, size_t n, size_t k, 
               AB_TYPE *A, size_t ldA, AB_TYPE *B, size_t ldB, 
	       C_TYPE beta, C_TYPE *C, size_t ldC, AB_PACK_TYPE *Ac, AB_PACK_TYPE *Bc, 
	       size_t MC, size_t NC, size_t KC, int MR, int NR);


void pack_RB( char, char, int, int, AB_TYPE *, int, AB_PACK_TYPE *, int );
void pack_CB( char, char, int, int, AB_TYPE *, int, AB_PACK_TYPE *, int );

void pack_dot_A(char orderA, int mc, int kc, AB_TYPE *M, int ldM, AB_PACK_TYPE *Mc, int MR);
void pack_dot_B(char orderB, int kc, int nc, AB_TYPE *M, int ldM, AB_PACK_TYPE *Mc, int NR);

void vpack_dot_A(char orderA, int mc, int kc, AB_TYPE *M, int ldM, AB_PACK_TYPE *Mc, int MR);
void vpack_dot_B(char orderB, int kc, int nc, AB_TYPE *M, int ldM, AB_PACK_TYPE *Mc, int NR);

void prepack_dot_B( char orderB, size_t n, size_t k, AB_TYPE *B, size_t ldB, 
		    AB_PACK_TYPE *Bc, size_t NC, size_t KC, int NR);

void prepack_dot_A( char orderA, size_t m, size_t k, AB_TYPE *A, size_t ldA, AB_PACK_TYPE *Ac,
	           size_t MC, size_t KC, int MR);

void prepack_saxpy_A( char orderA, size_t m, size_t k, AB_TYPE *A, size_t ldA, AB_PACK_TYPE *Ac,
	           size_t MC, size_t KC, int MR);

