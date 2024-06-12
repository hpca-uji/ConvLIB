#include "ukernels.h"

void generic_microkernel(int mr, int nr, int MR, int NR, AB_TYPE *A, AB_TYPE *B, 
		                C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
				C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge) {

  #ifdef FP32
    if (mr == MR && nr == NR)
      uk(kc, &alpha, A, B, &beta, C, ldC * sizeof(float));
    else
      uk_edge(mr, nr, MR, NR, kc, &alpha, A, B, &beta, aux, C, ldC);
  #elif INT8_INT32
    if (mr == MR && nr == NR)
      uk(kc, A, B, C, beta, ldC);
    else {
      uk(kc, A, B, aux, 0, MR);
      for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cptr[j*ldC + i] = (beta) * C[j*ldC + i] + aux[j * MR + i];
     }
  #endif

}

