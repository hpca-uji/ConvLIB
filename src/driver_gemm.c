/* 
   Direct convolution 

   -----

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free Software
   Foundation, either version 3 of the License, or (at your option) any later
   version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
   You should have received a copy of the GNU General Public License along with
   this program. If not, see <http://www.gnu.org/licenses/>.

   -----

   author    = "Enrique S. Quintana-Orti" contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include <sys/time.h>

#include "ukernels.h"
#include "formats.h"
#include "arrays.h"
#include "sutils.h"
#include "convDirect.h"
#include "colors.h"
#include "inutils.h"

#include "im2row.h"
#include "im2col.h"

#include "modelLevel/model_level.h"
#include "gemm/gemm_blis.h"

#include "asm_generator/ukernels/gemm_ukernel_headers.h"

#undef min

#include "convGemm/convgemm_blis.h"
#include "convGemm/im2row_nhwc.h"

#ifdef ENABLE_BLIS
  #include "blis.h"
#endif

#ifdef ENABLE_OPENBLAS
  #include "cblas.h"
#endif

#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )



int main(int argc, char *argv[]) {
  char* variant;

  AB_TYPE *A, *B;
  AB_PACK_TYPE *Ac, *Bc;

  C_TYPE *C, *Cg, *Ctmp, alpha, beta;
  
  size_t mc_blis, nc_blis, kc_blis;
  int    mc, nc, kc;

  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  double best_error, best_flops, best_time;
  int best_mr, best_nr;
  int best_mc, best_nc, best_kc;

  int    m, t,
         nmin,  nmax,  nstep,
         kmin,  kmax,  kstep,
         cmin,  cmax,  cstep,
         hmin,  hmax,  hstep,
         wmin,  wmax,  wstep,
         rmin,  rmax,  rstep,
         smin,  smax,  sstep,
         prmax, psmax, ret,
         tformat, tformatmin, tformatmax,
         n, k, c,
         h, w,
         r, s,
         pr, ps,
         in, ir, is, ic, ik, ih, iw,
         ldD1,  ldD2,  ldD3,
         ldF1,  ldF2,  ldF3,
         ldFB1, ldFB2, ldFB3, ldFB4,
         ldY1,  ldY2,  ldY3,
         visual, nreps, ho, wo, homax, womax;
  
  int ib, i, i2, ii, Ci_Cib, Co_Cob, Co_Nr, Co_Mr;
  char *filename;
  FILE *fd;
  int cnn_test_num, cnn_i;
  int CIB, COB, WOB;
  size_t test_n = 0;
  int mm, nn, kk;
  int ldA, ldB, ldC;

  int MR, NR, TH;
  char *in_gemm;
  int mr_limit, nr_limit, mr_iter, nr_iter, mr_init, nr_init;

  int vpadding;
  int hpadding;
  int vstride; 
  int hstride; 
  int vdilation;
  int hdilation;

  int tile_H, tile_W;
  unsigned char wino_on, model_on;

  int params[15];

  testConfig_t* testConf=new_CNN_Test_Config(argv);
  load_model_level_params(argv[13], params);
  
  UK_TYPE uk;
  UK_EDGE_TYPE uk_edge;

  int gemm;
  int format;

  tmin    = testConf->tmin;
  tformat = testConf->format;
  TH      = testConf->TH;

  in_gemm = testConf->gemm; 
 
  mc_blis = testConf->MC;
  nc_blis = testConf->NC;
  kc_blis = testConf->KC;
  
  COB = testConf->MC;
  WOB = testConf->NC;
  CIB = testConf->KC;

  model_on = 1;
  if (mc_blis != -1 && nc_blis != -1 && kc_blis != -1) {
    model_on = 0;
  } else {
    if (mc_blis < 32 || nc_blis < 32 || kc_blis < 32) {
      printf("ERROR: MC, NC and KC must have a minimum value of 32\n");
      exit(-1);
    }
  }
  
  if      (strcmp("B3A2C0",   in_gemm)==0) gemm = B3A2C0;
  else if (strcmp("A3B2C0",   in_gemm)==0) gemm = A3B2C0;
  else if (strcmp("BLIS",     in_gemm)==0) gemm = BLIS;
  else if (strcmp("OPENBLAS", in_gemm)==0) gemm = OPENBLAS;
  else if (strcmp("SDOT_GEMM",in_gemm)==0) gemm = SDOT_GEMM;
  else                                     gemm = UNKNOWN;
  
  #if defined(NQ_FP32) || defined(FQ_FP32)
    UK_TYPE      *uk_vec      = new_uk_asm_selector_fp32();
    UK_EDGE_TYPE *uk_edge_vec = new_uk_asm_edge_selector_fp32();
  #elif defined(NQ_INT32) || defined(FQ_INT32)
    UK_TYPE      *uk_vec      = new_uk_intrinsic_selector_int32();
    UK_EDGE_TYPE *uk_edge_vec = NULL;
  #elif FP16
    UK_TYPE      *uk_vec      = new_uk_intrinsic_selector_fp16();
    UK_EDGE_TYPE *uk_edge_vec = NULL;
  #elif Q_INT8_INT32
    UK_TYPE      *uk_vec      = new_uk_intrinsic_selector_int8_int32();
    //UK_TYPE      *uk_vec      = new_uk_intrinsic_selector_int8_int32_u8();
    UK_EDGE_TYPE *uk_edge_vec = NULL;
  #else
    printf("ERROR: Type unsupported\n");
    exit(-1);
  #endif

  #if defined(NQ_FP32) || defined(FP16)
    #ifndef A78AE
      if (gemm == SDOT_GEMM) {
        printf(" ERROR: SDOT_GEMM only supported for INT8 - INT32 data type on A78AE Processor.\n\n"); exit(-1);
      }
    #endif
  #endif

  #if defined(NQ_INT8)
    errorthd = 0.5;
  #elif defined(NQ_FP16)
    errorthd = 1.0e-3;
  #elif defined(NQ_FP32) || defined(FQ_FP32)
    errorthd = 1.0e-5;
  #else
    errorthd = 1.0e-14;
  #endif

  fprintf(testConf->fd_csv, "Format;l;MC;NC;KC;M;N;K;n;k;c;ho;wo;kh;kw;Time;GFLOPS;Error;MR;NR\n");    

  printf(" +====================================================================================================================================+\n");
  printf(" |%s                                                    DRIVER FOR GEMM EVALUATION                                                      %s|\n",
  COLOR_BOLDYELLOW, COLOR_RESET);

  for (format=0; format < 2; format++) {
    if (format == 0) { //NHWC
      printf(" +====================================================================================================================================+\n");
      printf(" |%s                                                              NHWC                                                                  %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    } else {
      printf("\n");
      printf(" +====================================================================================================================================+\n");
      printf(" |%s                                                              NCHW                                                                  %s|\n", COLOR_BOLDYELLOW, COLOR_RESET);
    }
    printf(" +=========+=====================+=======================+======================================+==============================+======+\n");
    printf(" | %sMR   NR | MC      NC     KC   |   M      N      K     |  n     k     c   ho    wo   (kh,kw)  |  GFLOPS     Time     Error   | Test%s |\n",
    COLOR_RESET, COLOR_RESET);
    printf(" +=========+=====================+=======================+======================================+==============================+======+\n");
    for (cnn_i = 0; cnn_i < testConf->cnn_num; cnn_i++) {
    
      n  = testConf->cnn[cnn_i].nmin;
      k  = testConf->cnn[cnn_i].kmin;
      c  = testConf->cnn[cnn_i].cmin;
      ho = testConf->cnn[cnn_i].homin;
      wo = testConf->cnn[cnn_i].womin;
      h  = testConf->cnn[cnn_i].hmin;
      w  = testConf->cnn[cnn_i].wmin;
      r  = testConf->cnn[cnn_i].rmin;
      s  = testConf->cnn[cnn_i].smin;
      
      vpadding  = 0;
      hpadding  = 0;
      vstride   = 1;
      hstride   = 1;
      vdilation = 1;
      hdilation = 1;
  
      if (r == 3) {h += 2; w += 2;}
  
      //Matrix Order: Column-major
  
      if (format == 0) { //NHWC
        mm = k;
        nn = ho * wo * n;
        kk = r * s * c;
  
        alpha = 1.0;
        beta  = 0.0;
  
        ldA = mm;
        ldB = kk;
        ldC = mm;
      } else { //NCHW
        mm = ho * wo * n;
        nn = k;
        kk = r * s * c;
  
        alpha = 1.0;
        beta  = 0.0;
  
        ldA = mm;
        ldB = kk;
        ldC = mm;
      }
      //-------------------------------------------------
      //Iterate over MR-NR
      //-------------------------------------------------
      mr_limit = testConf->MR;
      nr_limit = testConf->NR;
      mr_init  = testConf->MR;
      nr_init  = testConf->NR;
  
      if (testConf->bestof=='T') {
        #if defined(NQ_FP32) || defined(FQ_FP32)
          mr_limit = 20;
          nr_limit = 20;
          mr_init  = 4;
          nr_init  = 4;
        #elif FP16
          mr_limit = 40;
          nr_limit = 40;
          mr_init  = 8;
          nr_init  = 8;
        #else
          mr_limit = 24;
          nr_limit = 24;
          mr_init  = 8;
          nr_init  = 4;
        #endif
        if (gemm == SDOT_GEMM) {
          mr_limit = 4;
          nr_limit = 16;
          mr_init  = 4;
          nr_init  = 16;
        }
      }
  
      best_error=0.0; best_flops=0.0; best_time = 0.0;
  
      for (mr_iter=mr_init; mr_iter < mr_limit + 1; mr_iter+=mr_init) {
        for (nr_iter=nr_init; nr_iter < nr_limit + 1; nr_iter+=nr_init) {
  
          MR = mr_iter;
          NR = nr_iter;
  
	  fselector(MR, NR, LOWERING, gemm, uk_vec, uk_edge_vec, &uk, &uk_edge);
    
	  if (uk == NULL) { continue; }
	  
          if (model_on) {
	    if (gemm==A3B2C0)
              get_optim_mc_nc_kc(sizeof(C_TYPE), nn, mm, kk, NR, MR, &mc, &nc, &kc, params);
	    else
              get_optim_mc_nc_kc(sizeof(C_TYPE), mm, nn, kk, MR, NR, &mc, &nc, &kc, params);
	    mc_blis = (size_t)mc;
	    nc_blis = (size_t)nc;
	    kc_blis = (size_t)kc;
	  }
  
          Ac = (AB_PACK_TYPE *)aligned_alloc(32, TH * 2 * (mm + MR) * (kk + 16) * sizeof(AB_PACK_TYPE));
          Bc = (AB_PACK_TYPE *)aligned_alloc(32, TH * 2 * (nn + NR) * (kk + 16) * sizeof(AB_PACK_TYPE));
  
          A  = (AB_TYPE *) malloc ( mm * kk * sizeof(AB_TYPE));
          B  = (AB_TYPE *) malloc ( nn * kk * sizeof(AB_TYPE));   
          C  = (C_TYPE  *) malloc ( mm * nn * sizeof(C_TYPE));
	  Cg = (C_TYPE  *) malloc ( mm * nn * sizeof(C_TYPE));
  
	  Ctmp = (C_TYPE *) malloc ( TH * MR * NR * sizeof(C_TYPE));
 
	  #ifdef NQ_FP32 
	    generate_matrix_fp32(COLUMN_MAJOR, mm, kk, A, ldA );
	    generate_matrix_fp32(COLUMN_MAJOR, kk, nn, B, ldB );
          #elif NQ_INT32
	    generate_matrix_int32(COLUMN_MAJOR, mm, kk, A, ldA );
	    generate_matrix_int32(COLUMN_MAJOR, kk, nn, B, ldB );
	  #elif defined(FQ_FP32) || defined(FQ_INT32) || defined(Q_INT8_INT32)
	    generate_matrix_int8(COLUMN_MAJOR, mm, kk, A, ldA );
	    generate_matrix_int8(COLUMN_MAJOR, kk, nn, B, ldB );
          #else
	    printf("ERROR: Data type model unsuported\n");
	    exit(-1);
          #endif

          // Set result to zeros
	  for (int i=0; i < mm * nn; i++) { C[i]=0; Cg[i]=0; }
	  
	  //if(gemm == SDOT_GEMM)
	  //prepack_dot_A( 'C', mm, kk, F, lda, Ac_blis, mc_blis, kc_blis, MR);
	  //else if(gemm == B3A2C0)
	  //prepack_saxpy_A( 'C', mm, kk, F, lda, Ac_blis, mc_blis, kc_blis, MR);
  
          time  = 0.0; 
          t1    = dclock();
          nreps = 0;
          while ( time <= tmin ) {
    
            if (gemm == BLIS) {
	      #if defined(ENABLE_BLIS) && defined(NQ_FP32)
	        sgemm_( "N", "N", &mm, &nn, &kk, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC );
	      #else
	        printf("ERROR: BLIS unsupported for this data type.\n"); exit(-1);
              #endif
	    } else if (gemm == OPENBLAS) {
              #if defined(ENABLE_OPENBLAS) &&  defined(NQ_FP32)
	        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
	                    mm, nn, kk, alpha, A, ldA, B, ldB, beta, C, ldC);
	      #else
	        printf("ERROR: OPENBLAS unsupported for this data type.\n"); exit(-1);
              #endif
	    } else if (gemm == B3A2C0) {
              gemm_blis_B3A2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
                                alpha, A, ldA, B, ldB, beta, C, ldC,
                                Ac, Bc, mc_blis, nc_blis, kc_blis, 
			        MR, NR, TH, testConf->LOOP, Ctmp, uk_vec, uk_edge_vec);
	    } else if (gemm == A3B2C0) {
              gemm_blis_A3B2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
                                alpha, A, ldA, B, ldB, beta, C, ldC,
                                Ac, Bc, mc_blis, nc_blis, kc_blis, 
			        MR, NR, TH, testConf->LOOP, Ctmp, uk_vec, uk_edge_vec);
	    } else if (gemm == SDOT_GEMM) {
	      ldC = ho * wo * n;
              dot_gemm( 'C', 'C', 'R', mm, nn, kk, A, ldA, B, ldB, beta, C, ldC,
	                 Ac, Bc, mc_blis, nc_blis, kc_blis, MR, NR);
	      } else {
	        printf("ERROR: Algorithm unsupported.\n"); exit(-1);
	      }
  
	    nreps++;
            t2 = dclock();
            time = ( t2 > t1 ? t2 - t1 : 0.0 );
      
          }
          time = time/nreps;
  
          if ( nreps == 0 ) continue; 
	      
          if ( testConf->test=='T' )
            error = gemm_validation(COLUMN_MAJOR, NO_TRANSPOSE, NO_TRANSPOSE, A, B, mm, nn, kk, alpha, beta, ldA, ldB, ldC, C, Cg);
          else
            error = -1.0;
  
	  if (gemm==SDOT_GEMM)  { 
            //C row-major, convert to col-major
            //C_TYPE *Y_tmp = (C_TYPE *) malloc (sizeof(C_TYPE) * n * ho * wo * k);
	    //convert_row2col(Y, Y_tmp, k, ho * wo * n);
	    //free(Y);
	    //Y = Y_tmp;
	    printf("Not implemented\n"); exit(-1);
	  }
  
              
          flops = 2.0 * mm * nn * kk;
  
          GFLOPS  = flops / (1.0e+9 * time );
  
          if ((gemm == BLIS) || (gemm == OPENBLAS))
            printf(" | -    -  |   -         -        -    | %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e |", n, k, c, ho, wo, r, s, COLOR_BOLDCYAN, GFLOPS, COLOR_RESET, time, error);
	  else
            printf(" | %-3d  %-2d | %-6zu %-6zu %-6zu|  %-6d %-6d %-6d | %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e |", MR, NR, mc_blis, nc_blis, kc_blis,  mm, nn, kk, n, k, c, ho, wo, r, s, COLOR_MAGENTA, GFLOPS, COLOR_RESET, time, error);
	  
  
	  if (GFLOPS > best_flops) {
	    best_error = error;
	    best_flops = GFLOPS; 
	    best_time  = time;
	    best_mr    = MR;
	    best_nr    = NR;
	    best_mc    = mc_blis;
	    best_nc    = nc_blis;
	    best_kc    = kc_blis;
	  }
	  
          if ( testConf->test=='T')
            if ( error < errorthd)
              printf("  %sOK%s  |", COLOR_GREEN, COLOR_RESET);
            else
              printf(" %sERR%s  |", COLOR_RED, COLOR_RESET);
          else
            printf("  %s-%s   |", COLOR_BOLDYELLOW, COLOR_RESET);
      
          printf("\n");
      
	  free(Ctmp);
  
          free(Ac); 
          free(Bc);
     
          free(A);
          free(B);
          free(C);
          free(Cg);
      
        }
      }
  
      if (testConf->bestof=='T') {
        printf(" +---------+---------------------+-----------------------+--------------------------------------+------------------------------+------+\n");
        printf(" | %s%-3d  %-2d | %-6d %-6d %-6d|   -       -     -     |   -    -     -     -      -     -    | %s%-9.2e%s      -       -     |      |", COLOR_RESET, best_mr, best_nr, best_mc, best_nc, best_kc,  COLOR_BOLDMAGENTA, best_flops, COLOR_RESET);
        printf("\n");
        printf(" +---------+---------------------+-----------------------+--------------------------------------+------------------------------+------+\n");
  
        fprintf(testConf->fd_csv,"%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e;%d;%d\n", format == 0 ? "NHWC" : "NCHW", testConf->cnn[cnn_i].layer, best_mc, best_nc, best_kc, mm, nn, kk, n, k, c, ho, wo, r, s, best_time, best_flops, best_error, best_mr, best_nr);
      } else
        fprintf(testConf->fd_csv,"%s;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e;%d;%d\n", format == 0 ? "NHWC" : "NCHW", testConf->cnn[cnn_i].layer, mc, nc, kc, mm, nn, kk, n, k, c, ho, wo, r, s, time, GFLOPS, error, MR, NR);
  
    }
    if (testConf->bestof != 'T') 
      printf(" +=========+=====================+=======================+======================================+==============================+======+\n");
  }
  fclose(testConf->fd_csv);
  free_CNN_Test_Config(testConf);
      
  printf("\n");
  
  return 0;
  
}



