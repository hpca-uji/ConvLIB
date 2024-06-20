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
#define Mcol(a1,a2)  M[ (a2)*(ldM)+(a1) ]

#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Mrow(a1,a2)  M[ (a1)*(ldM)+(a2) ]


void gemm_blis_B3A2C0( char, char, char, char, char, size_t, size_t, size_t, 
		       C_TYPE, AB_TYPE *, size_t, AB_TYPE *, size_t, 
		       C_TYPE, C_TYPE *, size_t,  AB_TYPE *, AB_TYPE *, 
		       size_t, size_t, size_t, int, int, int, int, C_TYPE *,
		       UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec);


void gemm_blis_A3B2C0( char, char, char, char, char, size_t, size_t, size_t, 
		       C_TYPE, AB_TYPE *, size_t, AB_TYPE *, size_t, 
		       C_TYPE, C_TYPE *, size_t, AB_TYPE *, AB_TYPE *, 
		       size_t, size_t, size_t, int, int, int, int, C_TYPE *,
		       UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec);



void pack_RB( char, char, int, int, AB_TYPE *, int, AB_TYPE *, int );
void pack_CB( char, char, int, int, AB_TYPE *, int, AB_TYPE *, int );

void pack_RB_v( char, char, int, int, AB_TYPE *, int, AB_TYPE *, int );
void pack_CB_v( char, char, int, int, AB_TYPE *, int, AB_TYPE *, int );

