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

#include <stdbool.h>

#include "convgemm_blis.h"
#include "im2row_nhwc.h"

/*
 * BLIS pack for M-->Mc using implicit im2row
*/
void pack_CB_nhwc(char orderM, char transM, int mc, int nc, const AB_TYPE *restrict M, int ldM, AB_PACK_TYPE *restrict Mc,
                  int RR, const conv_p *conv_params, int start_row, int start_col) {
    if (((transM == 'N') && (orderM == 'C')) || ((transM == 'T') && (orderM == 'R'))) {
        // initial kernel positions
        int start_ky = start_row % conv_params->kwidth;
        int start_kx = (start_row / conv_params->kwidth) % conv_params->kheight;
        int start_c = (start_row / conv_params->kwidth) / conv_params->kheight;
	#ifdef OMP_ENABLE
        #pragma omp parallel for
        #endif
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int ky = start_ky;
            int kx = start_kx;
            int c = start_c;
            // initial pixel positions
            int start_y = (start_col + j) % conv_params->owidth;
            int start_x = ((start_col + j) / conv_params->owidth) % conv_params->oheight;
            int start_b = ((start_col + j) / conv_params->owidth) / conv_params->oheight;
            for (int i = 0; i < mc; i++) {
                int y = start_y;
                int x = start_x;
                int b = start_b;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(i,j+jj);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->height + ix) * conv_params->width + iy) * conv_params->channels +
                                  c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next pixel position
                    y++;
                    if (y >= conv_params->owidth) {
                        y = 0;
                        x++;
                        if (x >= conv_params->oheight) {
                            x = 0;
                            b++;
                        }
                    }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next kernel position
                ky++;
                if (ky >= conv_params->kwidth) {
                    ky = 0;
                    kx++;
                    if (kx >= conv_params->kheight) {
                        kx = 0;
                        c++;
                    }
                }
            }
        }
    } else {
        int start_y = (start_row) % conv_params->owidth;
        int start_x = ((start_row) / conv_params->owidth) % conv_params->oheight;
        int start_b = ((start_row) / conv_params->owidth) / conv_params->oheight;
	#ifdef OMP_ENABLE
	#pragma omp parallel for
	#endif
        for (int j = 0; j < nc; j += RR) {
            int k = j * mc;
            int nr = min(nc - j, RR);
            int y = start_y;
            int x = start_x;
            int b = start_b;
            int start_ky = (start_col + j) % conv_params->kwidth;
            int start_kx = ((start_col + j) / conv_params->kwidth) % conv_params->kheight;
            int start_c = ((start_col + j) / conv_params->kwidth) / conv_params->kheight;
            for (int i = 0; i < mc; i++) {
                int ky = start_ky;
                int kx = start_kx;
                int c = start_c;
                int jj = 0;
                for (; jj < nr; jj++) {
                    // Mc[k] = Mcol(j+jj,i);
                    int ix = conv_params->vstride * x + conv_params->vdilation * kx - conv_params->vpadding;
                    int iy = conv_params->hstride * y + conv_params->hdilation * ky - conv_params->hpadding;
                    if (0 <= ix && ix < conv_params->height && 0 <= iy && iy < conv_params->width) {
                        Mc[k] = M[((b * conv_params->height + ix) * conv_params->width + iy) * conv_params->channels +
                                  c];
                    } else Mc[k] = 0.0;
                    k++;
                    // next kernel position
                    ky++;
                    if (ky >= conv_params->kwidth) {
                        ky = 0;
                        kx++;
                        if (kx >= conv_params->kheight) {
                            kx = 0;
                            c++;
                        }
                    }
                }
                for (; jj < RR; jj++) {
                    Mc[k] = 0.0;
                    k++;
                }
                // k += (RR - nr);
                // next pixel position
                y++;
                if (y >= conv_params->owidth) {
                    y = 0;
                    x++;
                    if (x >= conv_params->oheight) {
                        x = 0;
                        b++;
                    }
                }
            }
        }
    }
}
