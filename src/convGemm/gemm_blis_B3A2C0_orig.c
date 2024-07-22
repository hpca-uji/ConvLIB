/**
 * This file is part of convGemm
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <stdio.h>

#include "convgemm_blis.h"

/*
 * Computes the GEMM C := beta * C + alpha * A * B  following the BLIS approach
*/
void convgemm_blis_B3A2C0(char orderA, char orderB, char orderC,
                           char transA, char transB,
                           int m, int n, int k,
                           C_TYPE alpha, const AB_TYPE *A, int ldA,
                           const AB_TYPE *B, int ldB,
                           C_TYPE beta, C_TYPE *C, int ldC,
                           AB_PACK_TYPE *Ac, pack_func pack_RB,
                           AB_PACK_TYPE *Bc, pack_func pack_CB,
                           const conv_p *conv_params,
			   int MC, int NC, int KC, int MR, int NR, int TH, C_TYPE *Ctmp,
			   UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec) {

    // Quick return if possible
    C_TYPE zero = 0, one = 1;

    if ((m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)))
        return;

    UK_TYPE uk;
    UK_EDGE_TYPE uk_edge;

    fselector(MR, NR, CONVGEMM, UNKNOWN, uk_vec, uk_edge_vec, &uk, &uk_edge);

    //#ifdef FP32
      //uk_asm_selector_fp32(MR, NR, uk_vec, &uk);
      //uk_asm_edge_selector_fp32(MR, NR, uk_edge_vec, &uk_edge);
    //#elif FP16
      //uk_intrinsic_selector_fp16(MR, NR, uk_vec, &uk);
      //uk_edge = uk;
    //#elif INT8_INT32_U8
      //uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, &uk);
      //uk_edge = uk;
    //#elif INT8_INT32_S8
      //uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, &uk);
      //uk_edge = uk;
    //#endif

    int th_id = 0, i, j;

    #include "quick_gemm.h"

    for (int jc = 0; jc < n; jc += NC) {
        int nc = min(n - jc, NC);

        for (int pc = 0; pc < k; pc += KC) {
            int kc = min(k - pc, KC);
            bool last = (pc + KC) >= k;

            pack_CB(orderB, transB, kc, nc, B, ldB, Bc, NR, conv_params, pc, jc);

            float betaI = (pc == 0) ? beta : (float) 1.0;

            for (int ic = 0; ic < m; ic += MC) {
                int mc = min(m - ic, MC);

                pack_RB(orderA, transA, mc, kc, A, ldA, Ac, MR, conv_params, ic, pc);

		#ifdef OMP_ENABLE
                #pragma omp parallel for private(th_id)// collapse(2)
		#endif
                for (int jr = 0; jr < nc; jr += NR) {
		  #ifdef OMP_ENABLE
		  th_id = omp_get_thread_num();
 		  #else
		  th_id = 0;
		  #endif
		  C_TYPE *Ctmp_th = &Ctmp[th_id * MR * NR];

                  for (int ir = 0; ir < mc; ir += MR) {

                    int mr = min(mc - ir, MR);
                    int nr = min(nc - jr, NR);
                    C_TYPE *Cptr = (orderC == 'C') ? &Ccol(ic + ir, jc + jr) : &Crow(ic + ir, jc + jr);

                    generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc],
                                        Cptr, kc, ldC, alpha, betaI, Ctmp_th, uk, uk_edge);


                    }
                }

            }
        }
    }
}
