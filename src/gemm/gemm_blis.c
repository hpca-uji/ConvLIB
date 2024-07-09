/* 
   GEMM FLAVOURS

   -----

   GEMM FLAVOURS is a family of algorithms for matrix multiplication based
   on the BLIS approach for this operation: https://github.com/flame/blis

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

   author    = "Enrique S. Quintana-Orti"
   contact   = "quintana@disca.upv.es"
   copyright = "Copyright 2021, Universitat Politecnica de Valencia"
   license   = "GPLv3"
   status    = "Production"
   version   = "1.1"
*/

#include "gemm_blis.h"


void gemm_base_Cresident( char orderC, int m, int n, int k, C_TYPE alpha, 
		          AB_TYPE *A, int ldA, AB_TYPE *B, int ldB, 
			  C_TYPE beta, C_TYPE *C, int ldC );

void gemm_blis_B3A2C0( char orderA, char orderB, char orderC,
                       char transA, char transB, size_t m, size_t n, size_t k, 
                       C_TYPE alpha, AB_TYPE *A, size_t ldA, AB_TYPE *B, 
		       size_t ldB, C_TYPE beta, C_TYPE *C, size_t ldC, 
		       AB_TYPE *Ac, AB_TYPE *Bc, size_t MC, size_t NC, size_t KC, 
		       int MR, int NR, int TH, int loop, C_TYPE *Ctmp, 
		       UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec) {

  int ic, jc, pc, mc, nc, kc, ir, jr, mr, nr, j, i, th, th_id; 
  C_TYPE  zero = 0.0, one = 1.0, beta_edge = 0.0, betaI, *Ctmp_th, *Cptr; 
  AB_TYPE *Aptr, *Bptr, *Acptr;

  UK_TYPE uk;
  UK_EDGE_TYPE uk_edge;

  fselector(MR, NR, uk_vec, uk_edge_vec, &uk, &uk_edge);

  //#ifdef FP32
    //uk_asm_selector_fp32(MR, NR, uk_vec, &uk);
    //uk_asm_edge_selector_fp32(MR, NR, uk_edge_vec, &uk_edge);
  //#elif FP16
    //uk_intrinsic_selector_fp16(MR, NR, uk_vec, &uk);
    //uk_edge = uk;
  //#elif INT8_INT32
    //uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, &uk);
    //uk_edge = uk;
  //#endif

  #if defined(CHECK)
  #include "check_params.h"
  #endif

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  if (TH == 1) {
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc,jc);
        
	pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);
        
	if ( pc==0 ) betaI = beta;
        else betaI = one;
        
	for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 
          Aptr = &Acol(ic, pc);
          pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);
          
          for (jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR); 
            for (ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);
	      
	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, Ctmp, uk, uk_edge);

            }
          }

        }
      }
    }
  
  } else {

    #ifdef OMP_ENABLE
    #pragma omp parallel num_threads(TH) private(jc, nc, pc, kc, Bptr, ic, mc, Aptr, Acptr, Cptr, Ctmp_th, jr, nr, ir, mr, th_id)
    {
      th_id = omp_get_thread_num();
    #else
      printf("ERROR: Parallel option configured but not compiled\n"); 
      exit(-1);
    #endif

    if(loop == 3) {
    //== LOOP 3 PARALLELIZATION ==//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC);
      int its_nc = (int) ceil((double)nc/NR/omp_get_num_threads());
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Bptr = &Bcol(pc,jc+its_nc*NR*omp_get_thread_num());

        pack_CB( orderB, transB, kc, min(its_nc*NR, nc - its_nc*NR*omp_get_thread_num()), Bptr, ldB, Bc+kc*its_nc*NR*omp_get_thread_num(), NR);

        if ( pc==0 )  betaI = beta;
        else betaI = one;

        int its_m = (int) ceil((double)m/MC/omp_get_num_threads());
        Acptr = Ac + ((MC + MR )* (KC + 1)) * omp_get_thread_num();

        #ifdef OMP_ENABLE
          #pragma omp barrier
        #endif

        for ( ic=omp_get_thread_num()*(its_m * MC); ic<min(m, (omp_get_thread_num()+1) * (its_m * MC)); ic+=MC ) {
          mc = min(m-ic, MC);
          Aptr = &Acol(ic, pc);

          pack_RB( orderA, transA, mc, kc, Aptr, ldA, Acptr, MR);

          for ( jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR);
            for ( ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR);
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Acptr[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);

	    }
          }
        }
        #ifdef OMP_ENABLE
          #pragma omp barrier
        #endif
      }
    }
    
    } else if(loop == 4) {

    //== LOOP 4 PARALLELIZATION ==//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      int its_nc = (int) ceil((double)nc/NR/TH);
      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
        pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                 Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);
          
        if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 

          int its_mc = (int) ceil((double)mc/MR/TH);
          Aptr = &Acol(ic + its_mc * MR * th_id,pc);

          pack_RB( orderA, transA, min(its_mc * MR, mc - its_mc * MR * th_id), kc, 
	           Aptr, ldA, Ac + kc * its_mc * MR * th_id, MR);

          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
          for ( jr=th_id*(its_nc*NR); jr<min(nc,(th_id+1)*(its_nc*NR)); jr+=NR ) {
            nr = min(nc-jr, NR); 

            for ( ir=0; ir<mc; ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);

            }
          }
          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
        }
      }
    }
    
    } else if(loop == 5) {
     
    //== LOOP 5 PARALLELIZATION ==//
    for ( jc=0; jc<n; jc+=NC ) {
      nc = min(n-jc, NC); 
      int its_nc = (int) ceil((double)nc/NR/TH);

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC); 
        Bptr = &Bcol(pc, jc + its_nc * NR * th_id);

        pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                 Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);
          
	if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( ic=0; ic<m; ic+=MC ) {
          mc = min(m-ic, MC); 

	  int its_mc = (int) ceil((double)mc/MR/TH);
          Aptr = &Acol(ic + its_mc * MR * th_id,pc);

          pack_RB( orderA, transA, min(its_mc * MR, mc - its_mc * MR * th_id), kc, 
		   Aptr, ldA, Ac + kc * its_mc * MR * th_id, MR);

          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
          for ( jr=0; jr<nc; jr+=NR ) {
            nr = min(nc-jr, NR); 
            for ( ir=th_id*(its_mc*MR); ir<min(mc,(th_id+1)*(its_mc*MR)); ir+=MR ) {
              mr = min(mc-ir, MR); 
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);

            }
          }
          #ifdef OMP_ENABLE
	  #pragma omp barrier
          #endif
        }
      }
    }
    }

    #ifdef OMP_ENABLE
    }
    #endif
  }
}


void gemm_blis_A3B2C0( char orderA, char orderB, char orderC, 
		       char transA, char transB, size_t m, size_t n, size_t k, 
		       C_TYPE alpha, AB_TYPE *A, size_t ldA, AB_TYPE *B, size_t ldB, 
		       C_TYPE beta, C_TYPE *C, size_t ldC, AB_TYPE *Ac, AB_TYPE *Bc, 
                       size_t MC, size_t NC, size_t KC, int MR, int NR, int TH, 
		       int loop, C_TYPE *Ctmp, UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec) {

  int ic, jc, pc, mc, nc, kc, ir, jr, mr, nr, j, i, th, th_id; 
  C_TYPE  zero = 0.0, one = 1.0, beta_edge = 0.0, betaI, *Ctmp_th, *Cptr; 
  AB_TYPE *Aptr, *Bptr, *Bcptr;

  UK_TYPE uk;
  UK_EDGE_TYPE uk_edge;

  fselector(MR, NR, uk_vec, uk_edge_vec, &uk, &uk_edge);

  //#ifdef FP32
    //uk_asm_selector_fp32(MR, NR, uk_vec, &uk);
    //uk_asm_edge_selector_fp32(MR, NR, uk_edge_vec, &uk_edge);
  //#elif FP16
    //uk_intrinsic_selector_fp16(MR, NR, uk_vec, &uk);
    //uk_edge = uk;
  //#elif INT8_INT32
    //uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, &uk);
    //uk_edge = uk;
  //#endif


  #if defined(CHECK)
  #include "check_params.h"
  #endif

  // Quick return if possible
  if ( (m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)) )
    return;

  #include "quick_gemm.h"

  
  if (TH == 1) {
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic,pc);

        pack_RB( orderA, transA, mc, kc, Aptr, ldA, Ac, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

        for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          Bptr = &Bcol(pc,jc);

          pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bc, NR);

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);
            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);
	      
	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, Ctmp, uk, uk_edge);

	    }
          }
        }
      }
    }
  } else {
    #ifdef OMP_ENABLE
    #pragma omp parallel num_threads(TH) private(jc, nc, pc, kc, Bptr, Bcptr, ic, mc, Aptr, Cptr, Ctmp_th, jr, nr, ir, mr, th_id)
    {
      th_id = omp_get_thread_num();
    #else
      printf("ERROR: Parallel option configured but not compiled\n"); 
      exit(-1);
    #endif
    
    if(loop == 3) {
    //== LOOP 3 PARALLELIZATION ==//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

        int its_n = (int) ceil((double)n/NC/omp_get_num_threads());
        Bcptr = Bc + ((NC + NR )* (KC + 1)) * th_id;

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
        for ( jc=omp_get_thread_num()*(its_n * NC); jc<min(n, (th_id+1) * (its_n * NC)); jc+=NC ) {
          nc = min(n-jc, NC);
          Bptr = &Bcol(pc,jc);

          pack_CB( orderB, transB, kc, nc, Bptr, ldB, Bcptr, NR);

          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bcptr[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);

            }
          }
        }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
      }
    }
    
    } else if(loop == 4) {
    //== LOOP 4 PARALLELIZATION ==//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

	for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          int its_nc = (int) ceil((double)nc/NR/TH);

          Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
          pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                   Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
          for ( ir=th_id*(its_mc*MR); ir<min(mc,(th_id+1)*(its_mc*MR)); ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=0; jr<nc; jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);

            }
          }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
	}
      }
    }

    } else if(loop == 5) {
    //== LOOP 5 PARALLELIZATION ==//
    for ( ic=0; ic<m; ic+=MC ) {
      mc = min(m-ic, MC);
      int its_mc = (int) ceil((double)mc/MR/omp_get_num_threads());

      for ( pc=0; pc<k; pc+=KC ) {
        kc = min(k-pc, KC);
        Aptr = &Acol(ic+its_mc*MR*th_id, pc);
      
        pack_RB( orderA, transA, min(its_mc*MR, mc - its_mc*MR*th_id), 
		 kc, Aptr, ldA, Ac + kc*its_mc*MR*th_id, MR);

        if ( pc==0 ) betaI = beta;
        else betaI = one;

	for ( jc=0; jc<n; jc+=NC ) {
          nc = min(n-jc, NC);
          int its_nc = (int) ceil((double)nc/NR/TH);

          Bptr = &Bcol(pc, jc + its_nc * NR * th_id);
          pack_CB( orderB, transB, kc, min(its_nc * NR, nc - its_nc * NR * th_id), 
                   Bptr, ldB, Bc + kc * its_nc * NR * th_id, NR);

        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
          for ( ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR);

            for ( jr=th_id*(its_nc*NR); jr<min(nc,(th_id+1)*(its_nc*NR)); jr+=NR ) {
              nr = min(nc-jr, NR);
              Cptr = &Ccol(ic+ir,jc+jr);

	      generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc], &Bc[jr*kc], 
                                  Cptr, kc, ldC, alpha, betaI, &Ctmp[th_id * MR * NR], uk, uk_edge);
            }
          }
        #ifdef OMP_ENABLE
        #pragma omp barrier
        #endif
	}
      }
    }
    
    }

    #ifdef OMP_ENABLE
    }
    #endif
  }

}


void pack_RB( char orderM, char transM, int mc, int nc, 
	      AB_TYPE *M, int ldM, AB_TYPE *Mc, int RR ){
  //BLIS pack for M-->Mc
  int    i, j, ii, k, rr;
  for ( i=0; i<mc; i+=RR ) { 
    k = i*nc;
    rr = min( mc-i, RR );
      for ( j=0; j<nc; j++ ) {
        for ( ii=0; ii<rr; ii++ ) {
	  Mc[k] = Mcol(i+ii, j);
          k++;
        } 
        k += (RR-rr);
      }
    }
}

void pack_CB( char orderM, char transM, int mc, int nc, 
              AB_TYPE *M, int ldM, AB_TYPE *Mc, int RR ) {
  //BLIS pack for M-->Mc
  int    i, j, jj, k, nr;
  for ( j=0; j<nc; j+=RR ) { 
    k = j*mc;
    nr = min( nc-j, RR );
    for ( i=0; i<mc; i++ ) {
      for ( jj=0; jj<nr; jj++ ) {
        Mc[k] = Mcol(i,j+jj);
        k++;
      }
      k += (RR-nr);
    }
  }

}


void gemm_base_Cresident( char orderC, int m, int n, int k, C_TYPE alpha, 
		          AB_TYPE *A, int ldA, AB_TYPE *B, int ldB, 
			  C_TYPE beta, C_TYPE *C, int ldC ){
  int    i, j, p;
  C_TYPE  zero = 0.0, tmp;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ) {
      tmp = 0.0; 
      for ( p=0; p<k; p++ ) 
        tmp += Acol(i,p) * Brow(p,j);

      if ( beta==zero ) {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp;
        else
          Crow(i,j) = alpha*tmp;
      }
      else {
        if ( orderC=='C' )
          Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
        else
          Crow(i,j) = alpha*tmp + beta*Crow(i,j);
      }
    }
}



//=======================================================================================
// DOT PRODCUTS GEMM BASED 
//=======================================================================================

void dot_gemm( char orderA, char orderB, char orderC,
	       size_t m, size_t n, size_t k, 
               AB_TYPE *A, size_t ldA, AB_TYPE *B, size_t ldB, 
	       C_TYPE beta, C_TYPE *C, size_t ldC, AB_TYPE *Ac, AB_TYPE *Bc, 
	       size_t MC, size_t NC, size_t KC, int MR, int NR) {


  int ic, jc, pc, mc, nc, kc, ir, jr, mr, nr, j, i, th, th_id, kc_pack; 
  C_TYPE  zero = 0, one = 1, beta_edge = 0, betaI, *Ctmp_th, *Cptr, alpha = 1; 
  AB_TYPE *Aptr, *Bptr, *Acptr;

  if ((m==0) || (n==0) || (k==0))
    return;

  for ( jc=0; jc<n; jc+=NC ) {
    nc = min(n-jc, NC); 
    for ( pc=0; pc<k; pc+=KC) {
      kc = min(k-pc, KC); 
      kc_pack = (int)ceil((double)kc / 16.0) * 16;

      if (orderB == 'C') Bptr = &Bcol(pc,jc);
      else               Bptr = &Brow(pc,jc);

      pack_dot_B(orderB, kc, nc, Bptr, ldB, Bc, NR);
        
      if ( pc==0 ) betaI = beta;
      else betaI = one;
        
      for ( ic=0; ic<m; ic+=MC ) {
        mc = min(m-ic, MC); 

	if (orderA == 'C') Aptr = &Acol(ic, pc);
	else               Aptr = &Arow(ic, pc);
        	
        pack_dot_A(orderA, mc, kc, Aptr, ldA, Ac, MR);
          
        for (jr=0; jr<nc; jr+=NR ) {
          nr = min(nc-jr, NR); 
          for (ir=0; ir<mc; ir+=MR ) {
            mr = min(mc-ir, MR); 

	    if ( orderC=='C' )  Cptr = &Ccol(ic+ir,jc+jr);
	    else                Cptr = &Crow(ic+ir,jc+jr);

	    generic_microkernel(mr, nr, MR, NR, &Ac[ir*kc_pack], &Bc[jr*kc_pack], 
                                Cptr, kc, ldC, alpha, betaI, NULL, NULL, NULL);

          }
        }

      }
    }
  }
  

}


//WARNING: Packings for int8_t quantization with support for sdot vector intrinsic instrucction
void pack_dot_A(char orderA, int mc, int kc, int8_t *M, int ldM, int8_t *Mc, int MR) {
  int p = 0;
  
  //16 values int8_t for a vectorial register
  int KR = 16; 
  int kr_lim, i, j, ii, jj;

  if (orderA == 'C') {
    for (i = 0; i < mc; i += MR) {
      for (j = 0; j < kc; j += KR) {
        for (ii = 0; ii < MR; ii++) {
          kr_lim = min(KR, kc - j);
	  for (jj = 0; jj < kr_lim; jj++) {
            Mc[p] = Mcol(i + ii, j + jj); 
            p++;
	    //printf("MC[%3d]=%d\n", p-1, Mc[p-1]);
          }
	  for (; jj < KR; jj++) { 
            Mc[p] = 0; p++; 
	    //printf("MC-padding[%3d]=%d\n", p-1, 0);
	  }
	}
      }
    }
  } else { printf("Not implemented\n"); exit(-1); }
  //for (int ti = 0; i < mc * (kc + KR); i++) { printf("%d, ", Mc[i]); } printf("\n");
}

//WARNING: Packings for int8_t quantization with support for sdot vector intrinsic instrucction
void pack_dot_B(char orderB, int kc, int nc, int8_t *M, int ldM, int8_t *Mc, int NR) {
  int p = 0;
  
  //16 values int8_t for a vectorial register
  int KR = 16; 
  int kr_lim, i, j, jj, jjj, ii;

  int kc_4_padding = 4 - kc % 4;
  int kc_padding   = kc % 16;
  //printf("Packing B:\n");
  if (orderB == 'C') {
    for (i = 0; i < nc; i += NR) {
      for (j = 0; j < kc; j += KR) {
        
	kr_lim = min(KR, kc - j);
        
	for (jj =0 ; jj < kr_lim - 3; jj += 4) {
          for (ii =0 ; ii < NR; ii++) {
            Mc[p] = Mcol(j + jj + 0, i + ii); p++;
            Mc[p] = Mcol(j + jj + 1, i + ii); p++;
            Mc[p] = Mcol(j + jj + 2, i + ii); p++;
            Mc[p] = Mcol(j + jj + 3, i + ii); p++;
	    //printf("Mc[%3d]=%3d, %3d, %3d, %3d\n",p-1, Mc[p-4], Mc[p-3], Mc[p-2], Mc[p-1]);
          }
	}

	if (jj != KR) {
          for (ii =0 ; ii < NR; ii++) {
            for (jjj=jj; jjj < kr_lim; jjj++) { //kc left
              Mc[p] = Mcol(j + jjj, i + ii); p++;
	      //printf("Mc-left[%3d]=%3d\n",p-1, Mc[p-1]);
	    }
	    for (jjj = 0; jjj < kc_4_padding; jjj++) { //kc padding (full 4 values)
	      Mc[p] = 0; p++; 
	      //printf("Mc-padding[%3d]=%3d\n",p-1, 0); 
	    }
	  }
          for (jjj=0; jjj < NR * (KR - (kr_lim + kc_4_padding)); jjj++) {Mc[p] = 0; p++;}
        }

      }
    }
  } else { printf("Not implemented\n"); exit(-1); }

  //printf("-------------------------------------------------------------------\n");
  //for (int i = 0; i < nc * (kc + KR); i++) { printf("%d, ", Mc[i]); if (i % 4 == 0) printf("\n");} printf("\n");
}


//=======================================================================================
//=======================================================================================
