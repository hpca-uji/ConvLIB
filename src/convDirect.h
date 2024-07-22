#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef OMP_ENABLE
  #include <omp.h>
#endif

#ifdef ARMV8
  #include <arm_neon.h>
#endif

#include "formats.h"
#include "arrays.h"
#include "ukernels.h"

#include "asm_generator/ukernels/gemm_ukernel_headers.h"

#define min(a,b)     ( (a) > (b) ? (b) : (a) )


void convDirect_original( int, int, int,
                          int, int,
                          int, int,
                          int, int,
                          int, int,
                          AB_TYPE *, int, int, int,
                          AB_TYPE *, int, int, int,
                          C_TYPE *, int, int, int,
			  int);


void transform_filter_block_blis( int, int, 
				  int, int, 
				  AB_TYPE *, int, int, int,
				  AB_PACK_TYPE *, int, int, int, int,
				  int, int, int);

void convDirect_block_blis( int, int, int, 
		            int, int, 
		            int, int, 
			    int, int,
			    AB_TYPE *, int, int, int, 
			    AB_PACK_TYPE *, int, int, int, int,
			    C_TYPE *, int, int, int, 
			    AB_PACK_TYPE *, C_TYPE *,
			    int, int, int, int, int, int, int, 
			    UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec);


