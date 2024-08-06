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

  AB_TYPE *D, *DEXTr,*F, *DT, *DEXT;

  AB_PACK_TYPE *Ac, *FB, *Ac_blis, *Bc_blis;

  C_TYPE *Y, *Yg, *Ctmp, alphap, betap;
  
  size_t mc_blis, nc_blis, kc_blis;

  double t1, t2, time, tmin, error, nrm, tmp, errorthd, flops, GFLOPS;
  double best_error, best_flops, best_time;
  int best_mr, best_nr;
  int best_CIB, best_COB, best_WOB;

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
  int lda, ldb, ldc;
  int MR, NR, TH;
  char *in_algorithm, *in_gemm;
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

  int algorithm, gemm;

  tmin    = testConf->tmin;
  tformat = testConf->format;
  TH      = testConf->TH;

  in_algorithm = testConf->algorithm; 
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

  if      (strcmp("CONVDIRECT", in_algorithm)==0) algorithm = CONVDIRECT;
  else if (strcmp("LOWERING",   in_algorithm)==0) algorithm = LOWERING;
  else if (strcmp("CONVGEMM",   in_algorithm)==0) algorithm = CONVGEMM;
  else                                            algorithm = UNKNOWN;

  if      (strcmp("B3A2C0",   in_gemm)==0) gemm = B3A2C0;
  else if (strcmp("A3B2C0",   in_gemm)==0) gemm = A3B2C0;
  else if (strcmp("BLIS",     in_gemm)==0) gemm = BLIS;
  else if (strcmp("OPENBLAS", in_gemm)==0) gemm = OPENBLAS;
  else if (strcmp("SDOT_GEMM",in_gemm)==0) gemm = SDOT_GEMM;
  else                                     gemm = UNKNOWN;

  #if defined(NQ_FP32) || defined(FQ_FP32)
    UK_TYPE      *uk_vec      = new_uk_intrinsic_selector_fp32();
    UK_EDGE_TYPE *uk_edge_vec = NULL;
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
      if ((algorithm == LOWERING) && (gemm == SDOT_GEMM)) {
        printf(" ERROR: SDOT_GEMM only supported for INT8 - INT32 data type on A78AE Processor.\n\n"); exit(-1);
      }
    #endif
  #endif

  #if defined(INT8)
    errorthd = 0.5;
  #elif defined(FP16)
    errorthd = 1.0e-3;
  #elif defined(NQ_FP32)
    errorthd = 1.0e-5;
  #else
    errorthd = 1.0e-14;
  #endif

  fprintf(testConf->fd_csv, "l;WOB;COB;CIB;n;k;c;ho;wo;kh;kw;Time;GFLOPS;Error;MR;NR\n");    

  printf(" +==================================================================================================================+\n");
  printf(" |%s                                        DRIVER FOR NHWC CONVOLUTION EVALUATION                                    %s|\n",
  COLOR_BOLDYELLOW, COLOR_RESET);
  printf(" +=========+===========================+======================================+==============================+======+\n");
  printf(" | %sMR   NR | WOB(MC)  COB(NC)  CIB(KC) |   n     k     c   ho    wo   (kh,kw) |  GFLOPS     Time     Error   | Test%s |\n",
  COLOR_RESET, COLOR_RESET);
  printf(" +=========+===========================+======================================+==============================+======+\n");
    
        
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

    int m_gemm = k; 
    int n_gemm = n * ho * wo;
    int k_gemm = c * r * s;

    //-------------------------------------------------
    //Iterate over MR-NR
    //-------------------------------------------------
    mr_limit = testConf->MR;
    nr_limit = testConf->NR;
    mr_init  = testConf->MR;
    nr_init  = testConf->NR;

    if (testConf->bestof=='T') {
      #if NQ_FP32 || defined(FQ_FP32)
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
      if ((algorithm == LOWERING) &&  (gemm == SDOT_GEMM)) {
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

        if (algorithm == CONVDIRECT) fselector(NR, MR, algorithm, gemm, uk_vec, uk_edge_vec, &uk, &uk_edge);
	else                         fselector(MR, NR, algorithm, gemm, uk_vec, uk_edge_vec, &uk, &uk_edge);
  
	if (uk == NULL) continue;
	
        if ((algorithm==LOWERING) || (algorithm==CONVGEMM)) {
          if (model_on) {
	    if (gemm==A3B2C0)
              get_optim_mc_nc_kc(sizeof(C_TYPE), n_gemm, m_gemm, k_gemm, NR, MR, &COB, &WOB, &CIB, params);
	    else
              get_optim_mc_nc_kc(sizeof(C_TYPE), m_gemm, n_gemm, k_gemm, MR, NR, &WOB, &COB, &CIB, params);
            mc_blis = WOB; nc_blis = COB; kc_blis = CIB;
	  }
          Ac_blis = (AB_PACK_TYPE *)aligned_alloc(32, TH * 10 * m_gemm * k_gemm * sizeof(AB_PACK_TYPE));
          Bc_blis = (AB_PACK_TYPE *)aligned_alloc(32, TH * 10 * n_gemm * k_gemm * sizeof(AB_PACK_TYPE));
        } else {
          
	  if (model_on) {
	    //m=Wo; n=Co; k=Ci
            get_optim_mc_nc_kc(sizeof(C_TYPE), k, wo, c, NR, MR, &COB, &WOB, &CIB, params);
	    //TODO: Poor performance. Why?? Reverse micro-kernels??
            //get_optim_mc_nc_kc(sizeof(DTYPE), wo, k, c, MR, NR, &WOB, &COB, &CIB, params);
	  } else {
            if (WOB != -1) WOB = WOB / MR * MR;
            if (COB != -1) COB = COB / NR * NR;
	  }

          if (WOB != wo && WOB % MR != 0) {
            printf("ERROR: WOB must be multiple of MR. Now WOB=%d and MR=%d\n", WOB, MR);
            exit(-1);
          } else if (COB != k && COB % NR != 0) {
            printf("ERROR: COB must be multiple of NR. Now COB=%d and NR=%d\n", COB, NR);
            exit(-1);
          }

          Ac = (AB_PACK_TYPE *) aligned_alloc( 32, ((int) TH*WOB*MR*CIB*sizeof(AB_PACK_TYPE)));
          FB = (AB_PACK_TYPE *) malloc( ceil(((float) k)/NR)*NR*c*r*s*sizeof(AB_PACK_TYPE));

        }
    
        if (algorithm==LOWERING) {
          DEXT  = (AB_TYPE *) malloc( h*w*n*r*s*c*sizeof(AB_TYPE));
          //Input by rows. Only for sdot product!
          DEXTr = (AB_TYPE *) malloc( h*w*n*r*s*c*sizeof(AB_TYPE));
	}

        D    = (AB_TYPE *) malloc( n*c*h*w*sizeof(AB_TYPE));

        F = (AB_TYPE *) malloc( k*c*r*s*sizeof(AB_TYPE));   
        Y = (C_TYPE *) malloc( n*k*h*w*sizeof(C_TYPE));
          
        Ctmp = (C_TYPE *)malloc(TH * MR  * NR *sizeof(C_TYPE));
    
        Yg = (C_TYPE *) malloc( n*k*h*w*sizeof(C_TYPE) );   
          
        Ci_Cib = (int)ceil(((float) c)/CIB);
        Co_Cob = (int)ceil(((float) k)/COB);
        Co_Nr  = (int)ceil(((float) k)/NR);
        Co_Mr  = (int)ceil(((float) k)/MR);
    
        ldD3 = c;
        ldD2 = w * ldD3;
        ldD1 = h * ldD2;

        ldF3 = k;
        ldF2 = s*ldF3;
        ldF1 = r*ldF2;

        ldY3 = k;
        ldY2 = wo*ldY3;
        ldY1 = ho*ldY2;
          
        ldFB4 = NR;
        ldFB3 = c*ldFB4;
        ldFB2 = Co_Nr*ldFB3;
        ldFB1 = s*ldFB2;
         
        generate_tensor4D( n, h, w, c, D, ldD1, ldD2, ldD3 );
        generate_tensor4D( c, r, s, k, F, ldF1, ldF2, ldF3 );


        // Set result to zeros
	for (int i=0; i < n * k * ho * wo; i++) { Y[i]=0; Yg[i]=0; }

         if ( testConf->debug=='T' ) {
	   #ifdef NQ_FP32
             print_tensor4D_fp32( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
             print_tensor4D_fp32( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
	   #elif FQ_FP32 || FQ_INT32
             print_tensor4D_int8( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
             print_tensor4D_int8( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
	   #elif NQ_INT32
             print_tensor4D_int32( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
             print_tensor4D_int32( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
	   #elif FP16
             print_tensor4D_fp16( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
             print_tensor4D_fp16( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
           #else
             print_tensor4D_int8( "D", n, h, w, c, D, ldD1, ldD2, ldD3 );
             print_tensor4D_int8( "F", c, r, s, k, F, ldF1, ldF2, ldF3 );
           #endif
         }
   

	//Preparing lowering parameters.
	mm = k;
	nn = ho * wo * n;
	kk = r * s * c;
	alphap = 1.0;
	betap  = 0.0;
	lda = k;
	ldb = r * s * c;
	ldc = k;

	//Prepacking weights. Convdirect and Lowering + SDOT
        if (algorithm == CONVDIRECT)
          transform_filter_block_blis(c, k, r, s, F,  ldF1,  ldF2,  ldF3, FB, 
			              ldFB1, ldFB2, ldFB3, ldFB4, tformat, MR, NR);

	//else if (algorithm == LOWERING) 
	  //if(gemm == SDOT_GEMM)
	    //prepack_dot_A( 'C', mm, kk, F, lda, Ac_blis, mc_blis, kc_blis, MR);
	  //else if(gemm == B3A2C0)
	    //prepack_saxpy_A( 'C', mm, kk, F, lda, Ac_blis, mc_blis, kc_blis, MR);

        time  = 0.0; 
        t1    = dclock();
        nreps = 0;
        while ( time <= tmin ) {
          if (algorithm == LOWERING) {
            im2row(DEXT, c * r * s, D, n, h, w, c, ho, wo, r,
	           s, 0, 0, 1, 1, 1, 1, TH);
  
            if (gemm == BLIS) {
	      #if defined(ENABLE_BLIS) && defined(NQ_FP32)
	        sgemm_( "N", "N", &mm, &nn, &kk, &alphap, F, &lda, DEXT, &ldb, &betap, Y, &ldc );
	      #else
		printf("ERROR: BLIS unsupported for this data type.\n"); exit(-1);
              #endif
	    } else if (gemm == OPENBLAS) {
              #if defined(ENABLE_OPENBLAS) &&  defined(NQ_FP32)
	        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
		            mm, nn, kk, alphap, F, lda, DEXT, ldb, betap, Y, ldc);
	      #else
		printf("ERROR: OPENBLAS unsupported for this data type.\n"); exit(-1);
              #endif
	    } else if (gemm == B3A2C0) {
              gemm_blis_B3A2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
                                alphap, F, lda, DEXT, ldb, betap, Y, ldc,
                                Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, 
				MR, NR, TH, testConf->LOOP, Ctmp, uk_vec, uk_edge_vec);
	    } else if (gemm == A3B2C0) {
              gemm_blis_A3B2C0( 'C', 'C', 'C', 'N', 'N', mm, nn, kk, 
                                alphap, F, lda, DEXT, ldb, betap, Y, ldc,
                                Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, 
				MR, NR, TH, testConf->LOOP, Ctmp, uk_vec, uk_edge_vec);
	    } else if (gemm == SDOT_GEMM) {
	      #ifdef A78AE
	        ldc = ho * wo * n;
                dot_gemm( 'C', 'C', 'R', mm, nn, kk, F, lda, DEXT, ldb, betap, Y, ldc,
	                 Ac_blis, Bc_blis, mc_blis, nc_blis, kc_blis, MR, NR);
              #else
		printf("SDOT GEMM Only supported for A78AE arch.\n");
		exit(-1);
              #endif
	    } else {
	      printf("ERROR: Algorithm unsupported.\n"); exit(-1);
	    }

          } else if (algorithm == CONVGEMM) {
            conv_p conv_params = { n, h, w, c, k, r, s,
	    vstride, hstride, vpadding, hpadding,
	    vdilation, hdilation, ho, wo,
	    NULL, NULL, NULL, NULL, NULL, false };
    
            convgemm_blis_B3A2C0('C', 'C', 'C', 'N', 'N',
			           k, ho * wo * n, r * s * c,
			           1.0, F, k,
			           D, r * s * c,
			           0.0, Y, k,
			           Ac_blis, pack_RB_convgemm,
			           Bc_blis, pack_CB_nhwc,
			           &conv_params, mc_blis, 
				   nc_blis, kc_blis, MR, NR, TH, 
				   Ctmp, uk_vec, uk_edge_vec);

          } else if (algorithm == CONVDIRECT) {
            convDirect_block_blis(n, k, c, h, w, ho, wo, r, s, 
	    		          D,  ldD1, ldD2,  ldD3, 
	    		          FB, ldFB1, ldFB2, ldFB3, ldFB4,
	    		          Y,  ldY1, ldY2,  ldY3,
                                  Ac, Ctmp, tformat, CIB, COB, WOB, 
				  MR, NR, TH, uk_vec, uk_edge_vec);
          } else {
	    printf("ERROR: Algorithm unsuported\n"); exit(-1);
	  }
    
          nreps++;
          t2 = dclock();
          time = ( t2 > t1 ? t2 - t1 : 0.0 );
    
        }
        time = time/nreps;
        if ( nreps == 0 ) continue; 
	    
        // Test result
        if ( testConf->test=='T' ) {
          convDirect_original(n, k, c, h, w, ho, wo, 
			      r, s, vpadding, hpadding,
		              D,  ldD1, ldD2, ldD3, 
		              F,  ldF1, ldF2, ldF3, 
		              Yg, ldY1, ldY2, ldY3,
		              tformat);

	  if ((algorithm==LOWERING) &&  (gemm==SDOT_GEMM))  { //C row-major, convert to col-major
            C_TYPE *Y_tmp = (C_TYPE *) malloc (sizeof(C_TYPE) * n * ho * wo * k);
	    convert_row2col(Y, Y_tmp, k, ho * wo * n);
	    free(Y);
	    Y = Y_tmp;
	  }

          error = 0.0;
          nrm   = 0.0;
          for ( in=0; in<n; in++ )
          for ( ih=0; ih<ho; ih++ )
          for ( iw=0; iw<wo; iw++ )
          for ( ik=0; ik<k; ik++ ) {
            tmp = (double) Ygrow_NHWC(in,ik,ih,iw);
            nrm += tmp*tmp;
            tmp = (double) dabs(Yrow_NHWC(in,ik,ih,iw)-Ygrow_NHWC(in,ik,ih,iw));
            //printf("[%d, %d, %d, %d], Y=%.4f - Yg=%.4f\n", in, ik, ih, iw, Yrow_NHWC(in,ik,ih,iw), Ygrow_NHWC(in,ik,ih,iw));
	    error += tmp*tmp;
          }
          if ( nrm!=0.0 )
            error = sqrt(error) / sqrt(nrm);
          else
            error = sqrt(error);
        } else
          error = -1.0;
            
        flops = 2.0 * n * k * c * ho * wo * r * s;
        GFLOPS  = flops / (1.0e+9 * time );
	    
        if ( testConf->debug=='T' ) {
	  #if defined(NQ_FP32) || defined(FQ_FP32)
            print_tensor4D_fp32( "Ytest", n, h, w, k, Y, ldY1, ldY2, ldY3 );
            print_tensor4D_fp32( "Ycorrect", n, h, w, k, Yg, ldY1, ldY2, ldY3 );
	  #elif defined(NQ_INT32) || defined(FQ_INT32)
            print_tensor4D_int32( "Ytest", n, h, w, k, Y, ldY1, ldY2, ldY3 );
            print_tensor4D_int32( "Ycorrect", n, h, w, k, Yg, ldY1, ldY2, ldY3 );
          #endif
        }
	
          if ((algorithm == LOWERING)  && (gemm == BLIS || gemm == OPENBLAS))
            printf(" | -    -  |   -         -        -    | %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e |", n, k, c, ho, wo, r, s, COLOR_BOLDMAGENTA, GFLOPS, COLOR_RESET, time, error);
	  else
            printf(" | %-3d  %-2d | %-8d %-8d %-8d| %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e |", MR, NR, WOB, COB, CIB,  n, k, c, ho, wo, r, s, COLOR_BOLDMAGENTA, GFLOPS, COLOR_RESET, time, error);
	

	if (GFLOPS > best_flops) {
	  best_error = error;
	  best_flops = GFLOPS; 
	  best_time  = time;
	  best_mr    = MR;
	  best_nr    = NR;
	  best_COB   = COB;
	  best_CIB   = CIB;
	  best_WOB   = WOB;
	}
	
        if ( testConf->test=='T')
          if ( error < errorthd)
            printf("  %sOK%s  |", COLOR_GREEN, COLOR_RESET);
          else
            printf(" %sERR%s  |", COLOR_RED, COLOR_RESET);
        else
          printf("  %s-%s   |", COLOR_BOLDYELLOW, COLOR_RESET);
    
        printf("\n");
    
        if ((algorithm == LOWERING) || (algorithm == CONVGEMM)) {
          free(Ac_blis); 
          free(Bc_blis);
          if (algorithm == LOWERING){
            free(DEXT);
            free(DEXTr);
	  }
        } else {
          free(Ac); 
          free(FB);
        }
   
        free(Y);
        free(D);
        free(F);
        free(Yg);
    
      }
    }

    if (testConf->bestof=='T') {
      printf(" +---------+---------------------------+--------------------------------------+------------------------------+------+\n");
      printf(" | %s%-3d  %-2d | %-8d %-8d %-8d| %3d %5d %5d %5d %5d   (%1d,%1d)  | %s%-10.2e%s %-8.1e %8.1e %s|", COLOR_BOLDWHITE, best_mr, best_nr, best_WOB, best_COB, best_CIB,  n, k, c, ho, wo, r, s, COLOR_BOLDWHITE, best_flops, COLOR_RESET, best_time, best_error, COLOR_RESET);
      if ( testConf->test=='T' )
        if ( best_error < errorthd)
          printf("  %sOK%s  |", COLOR_GREEN, COLOR_RESET);
        else
          printf(" %sERR%s  |", COLOR_RED, COLOR_RESET);
       else
          printf("  %s-%s   |", COLOR_BOLDYELLOW, COLOR_RESET);
      printf("\n");
      printf(" +---------+---------------------------+--------------------------------------+------------------------------+------+\n");

      fprintf(testConf->fd_csv,"%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e;%d;%d\n",testConf->cnn[cnn_i].layer, best_WOB, best_COB, best_CIB, n, k, c, ho, wo, r, s, best_time, best_flops, best_error, best_mr, best_nr);
    } else
        fprintf(testConf->fd_csv,"%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%.2e;%.2f;%.2e;%d;%d\n",testConf->cnn[cnn_i].layer, WOB, COB, CIB, n, k, c, ho, wo, r, s, time, GFLOPS, error, MR, NR);
  }

  fclose(testConf->fd_csv);
  free_CNN_Test_Config(testConf);
    
  printf(" +=========+===========================+======================================+==============================+======+\n");
  printf("\n");

  return 0;

}

