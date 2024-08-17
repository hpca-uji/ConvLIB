#include "ukernels.h"

#define Ccol(a1,a2)  Cr[ (a2)*(ldC)+(a1) ]
#define Crow(a1,a2)  Cr[ (a1)*(ldC)+(a2) ]
#define Crref(i,j)   Cr[j*Clda+i]



//Micro-kernels selector
void fselector(int MR, int NR, int algorithm, int gemm, UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec, UK_TYPE *uk, UK_EDGE_TYPE *uk_edge) {

  #if defined(NQ_FP32) || defined(FQ_FP32)
    uk_intrinsic_selector_fp32(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #elif defined(NQ_INT32) || defined(FQ_INT32)
    uk_intrinsic_selector_int32(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #elif FP16
    uk_intrinsic_selector_fp16(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #elif Q_INT8_INT32
      if ((algorithm == LOWERING) && (gemm == SDOT_GEMM))
        #ifdef A78AE
        *uk = uk_int8_4x16_sdot;
        #else
        *uk = uk_intrinsic_quantize_int8_2x8;
        #endif
      else
        uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #endif

}



#ifdef A78AE

//============================================================================================
// MICRO-KERNELS FOR SAXPY
//============================================================================================

void ukernel_intrinsic_16x8_A78_fp16(int kc, float16_t *Ar, float16_t *Br, float16_t *Cr, float16_t beta, int Clda){
  int pr, bA = 0, bB = 0;
  
  float16x4_t  B0, B1, B2, B3;

  float16x8_t  A0, A1;
  float16x8_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  
	       C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17;
	       C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17;

  if (beta == 0) {
    C00 = vmovq_n_f16(0);
    C01 = vmovq_n_f16(0);
    C02 = vmovq_n_f16(0);
    C03 = vmovq_n_f16(0);
    C04 = vmovq_n_f16(0);
    C05 = vmovq_n_f16(0);
    C06 = vmovq_n_f16(0);
    C07 = vmovq_n_f16(0);
    C10 = vmovq_n_f16(0);
    C11 = vmovq_n_f16(0);
    C12 = vmovq_n_f16(0);
    C13 = vmovq_n_f16(0);
    C14 = vmovq_n_f16(0);
    C15 = vmovq_n_f16(0);
    C16 = vmovq_n_f16(0);
    C17 = vmovq_n_f16(0);
  } else {
    C00=vld1q_f16(&Crref(0, 0));
    C01=vld1q_f16(&Crref(0, 1));
    C02=vld1q_f16(&Crref(0, 2));
    C03=vld1q_f16(&Crref(0, 3));
    C04=vld1q_f16(&Crref(0, 4));
    C05=vld1q_f16(&Crref(0, 5));
    C06=vld1q_f16(&Crref(0, 6));
    C07=vld1q_f16(&Crref(0, 7));
    C10=vld1q_f16(&Crref(8, 0));
    C11=vld1q_f16(&Crref(8, 1));
    C12=vld1q_f16(&Crref(8, 2));
    C13=vld1q_f16(&Crref(8, 3));
    C14=vld1q_f16(&Crref(8, 4));
    C15=vld1q_f16(&Crref(8, 5));
    C16=vld1q_f16(&Crref(8, 6));
    C17=vld1q_f16(&Crref(8, 7));
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    A0 = vld1q_f16(&Ar[bA + 0]);
    A1 = vld1q_f16(&Ar[bA + 8]);

    B0 = vld1_f16(&Br[bB + 0]);
    B1 = vld1_f16(&Br[bB + 4]);
    
    C00 = vfmaq_lane_f16(C00, A0, B0, 0); 
    C10 = vfmaq_lane_f16(C10, A1, B0, 0); 

    C01 = vfmaq_lane_f16(C01, A0, B0, 1); 
    C11 = vfmaq_lane_f16(C11, A1, B0, 1); 

    C02 = vfmaq_lane_f16(C02, A0, B0, 2); 
    C12 = vfmaq_lane_f16(C12, A1, B0, 2); 

    C03 = vfmaq_lane_f16(C03, A0, B0, 3); 
    C13 = vfmaq_lane_f16(C13, A1, B0, 3); 

    C04 = vfmaq_lane_f16(C04, A0, B1, 0); 
    C14 = vfmaq_lane_f16(C14, A1, B1, 0); 

    C05 = vfmaq_lane_f16(C05, A0, B1, 1); 
    C15 = vfmaq_lane_f16(C15, A1, B1, 1); 

    C06 = vfmaq_lane_f16(C06, A0, B1, 2); 
    C16 = vfmaq_lane_f16(C16, A1, B1, 2); 

    C07 = vfmaq_lane_f16(C07, A0, B1, 3); 
    C17 = vfmaq_lane_f16(C17, A1, B1, 3); 

    bA+=16;
    bB+=8;
  }

  vst1q_f16(&Crref(0,0), C00); 
  vst1q_f16(&Crref(0,1), C01); 
  vst1q_f16(&Crref(0,2), C02); 
  vst1q_f16(&Crref(0,3), C03); 
  vst1q_f16(&Crref(0,4), C04); 
  vst1q_f16(&Crref(0,5), C05); 
  vst1q_f16(&Crref(0,6), C06); 
  vst1q_f16(&Crref(0,7), C07); 

  vst1q_f16(&Crref(8,0), C10); 
  vst1q_f16(&Crref(8,1), C11); 
  vst1q_f16(&Crref(8,2), C12); 
  vst1q_f16(&Crref(8,3), C13); 
  vst1q_f16(&Crref(8,4), C14); 
  vst1q_f16(&Crref(8,5), C15); 
  vst1q_f16(&Crref(8,6), C16); 
  vst1q_f16(&Crref(8,7), C17); 
}


//Generic micro-kernel for Lowering+GEMM based on DOT Products.
void uk_int8_4x16_sdot(int mr, int nr, uint32_t kc, int8_t *A, int8_t *B, int32_t *C, uint32_t kc, int32_t beta, uint32_t ldC) {

  //WARNING: C stored by rows!
  #if defined(Q_INT8_INT32)
    const int MR = 4;
    const int NR = 16;

    if (mr == MR && nr == NR) {
      uk_intrinsic_quantize_int8_4x16_sdot(mr, nr, kc, A, B, C, beta, ldC); 
    } else {
      C_TYPE aux[4 * 16];
      if (mr <= 4 && nr <= 4)
        uk_intrinsic_quantize_int8_4x4_sdot(mr, nr, kc, A, B, aux, 0, NR); //NR Because C is stored by rows
      else
        uk_intrinsic_quantize_int8_4x16_sdot(mr, nr, kc, A, B, aux, 0, NR); //NR Because C is stored by rows

      for (int i = 0; i < mr; i++)
      for (int j = 0; j < nr; j++)
        C[i*ldC + j] = (beta) * C[i*ldC + j] + aux[i * NR + j];
    }
  #else
    printf("Dot product only supported for INT8 - INT32 data types.\n");
    exit(-1);
  #endif

}

void uk_intrinsic_quantize_int8_4x16_sdot (int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {

  int KR = 16;

  int8x16_t A0, A1, A2, A3, B0, B1, B2, B3;

  int32x4_t C00, C01, C02, C03,
	    C10, C11, C12, C13,
	    C20, C21, C22, C23,
	    C30, C31, C32, C33;

  if (beta == 0) {
    C00 = vdupq_n_s32(0);
    C01 = vdupq_n_s32(0);
    C02 = vdupq_n_s32(0);
    C03 = vdupq_n_s32(0);
    C10 = vdupq_n_s32(0);
    C11 = vdupq_n_s32(0);
    C12 = vdupq_n_s32(0);
    C13 = vdupq_n_s32(0);
    C20 = vdupq_n_s32(0);
    C21 = vdupq_n_s32(0);
    C22 = vdupq_n_s32(0);
    C23 = vdupq_n_s32(0);
    C30 = vdupq_n_s32(0);
    C31 = vdupq_n_s32(0);
    C32 = vdupq_n_s32(0);
    C33 = vdupq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Crow(0, 0));  
    C01 = vld1q_s32(&Crow(0, 4));  
    C02 = vld1q_s32(&Crow(0, 8));  
    C03 = vld1q_s32(&Crow(0, 12));  

    C10 = vld1q_s32(&Crow(1, 0));  
    C11 = vld1q_s32(&Crow(1, 4));  
    C12 = vld1q_s32(&Crow(1, 8));  
    C13 = vld1q_s32(&Crow(1, 12));  
    
    C20 = vld1q_s32(&Crow(2, 0));  
    C21 = vld1q_s32(&Crow(2, 4));  
    C22 = vld1q_s32(&Crow(2, 8));  
    C23 = vld1q_s32(&Crow(2, 12));  
    
    C30 = vld1q_s32(&Crow(3, 0));  
    C31 = vld1q_s32(&Crow(3, 4));  
    C32 = vld1q_s32(&Crow(3, 8));  
    C33 = vld1q_s32(&Crow(3, 12));  
  }

  //Loop kc  (+=16)
  for (int i = 0; i < kc; i += KR) {
    //Load A: (4 x 16)
    A0 = vld1q_s8(&Ar[0]);
    A1 = vld1q_s8(&Ar[16]);
    A2 = vld1q_s8(&Ar[32]);
    A3 = vld1q_s8(&Ar[48]);

    //Dot Product
    //----------------------------------
    //Rep +1
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[0]);
    B1 = vld1q_s8(&Br[16]);
    B2 = vld1q_s8(&Br[32]);
    B3 = vld1q_s8(&Br[48]);

    C00 = vdotq_laneq_s32(C00, B0, A0, 0);
    C10 = vdotq_laneq_s32(C10, B0, A1, 0);
    C20 = vdotq_laneq_s32(C20, B0, A2, 0);
    C30 = vdotq_laneq_s32(C30, B0, A3, 0);
  
    C01 = vdotq_laneq_s32(C01, B1, A0, 0);
    C11 = vdotq_laneq_s32(C11, B1, A1, 0);
    C21 = vdotq_laneq_s32(C21, B1, A2, 0);
    C31 = vdotq_laneq_s32(C31, B1, A3, 0);

    C02 = vdotq_laneq_s32(C02, B2, A0, 0);
    C12 = vdotq_laneq_s32(C12, B2, A1, 0);
    C22 = vdotq_laneq_s32(C22, B2, A2, 0);
    C32 = vdotq_laneq_s32(C32, B2, A3, 0);
  
    C03 = vdotq_laneq_s32(C03, B3, A0, 0);
    C13 = vdotq_laneq_s32(C13, B3, A1, 0);
    C23 = vdotq_laneq_s32(C23, B3, A2, 0);
    C33 = vdotq_laneq_s32(C33, B3, A3, 0);
 
    //----------------------------------
    //Rep +2
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[64]);
    B1 = vld1q_s8(&Br[80]);
    B2 = vld1q_s8(&Br[96]);
    B3 = vld1q_s8(&Br[112]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 1);
    C10 = vdotq_laneq_s32(C10, B0, A1, 1);
    C20 = vdotq_laneq_s32(C20, B0, A2, 1);
    C30 = vdotq_laneq_s32(C30, B0, A3, 1);
  
    C01 = vdotq_laneq_s32(C01, B1, A0, 1);
    C11 = vdotq_laneq_s32(C11, B1, A1, 1);
    C21 = vdotq_laneq_s32(C21, B1, A2, 1);
    C31 = vdotq_laneq_s32(C31, B1, A3, 1);
  
    C02 = vdotq_laneq_s32(C02, B2, A0, 1);
    C12 = vdotq_laneq_s32(C12, B2, A1, 1);
    C22 = vdotq_laneq_s32(C22, B2, A2, 1);
    C32 = vdotq_laneq_s32(C32, B2, A3, 1);
    
    C03 = vdotq_laneq_s32(C03, B3, A0, 1);
    C13 = vdotq_laneq_s32(C13, B3, A1, 1);
    C23 = vdotq_laneq_s32(C23, B3, A2, 1);
    C33 = vdotq_laneq_s32(C33, B3, A3, 1);
    
    //----------------------------------
    //Rep +3
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[128]);
    B1 = vld1q_s8(&Br[144]);
    B2 = vld1q_s8(&Br[160]);
    B3 = vld1q_s8(&Br[176]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 2);
    C10 = vdotq_laneq_s32(C10, B0, A1, 2);
    C20 = vdotq_laneq_s32(C20, B0, A2, 2);
    C30 = vdotq_laneq_s32(C30, B0, A3, 2);
    
    C01 = vdotq_laneq_s32(C01, B1, A0, 2);
    C11 = vdotq_laneq_s32(C11, B1, A1, 2);
    C21 = vdotq_laneq_s32(C21, B1, A2, 2);
    C31 = vdotq_laneq_s32(C31, B1, A3, 2);
  
    C02 = vdotq_laneq_s32(C02, B2, A0, 2);
    C12 = vdotq_laneq_s32(C12, B2, A1, 2);
    C22 = vdotq_laneq_s32(C22, B2, A2, 2);
    C32 = vdotq_laneq_s32(C32, B2, A3, 2);
    
    C03 = vdotq_laneq_s32(C03, B3, A0, 2);
    C13 = vdotq_laneq_s32(C13, B3, A1, 2);
    C23 = vdotq_laneq_s32(C23, B3, A2, 2);
    C33 = vdotq_laneq_s32(C33, B3, A3, 2);
    
    //----------------------------------
    //Rep +4
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[192]);
    B1 = vld1q_s8(&Br[208]);
    B2 = vld1q_s8(&Br[224]);
    B3 = vld1q_s8(&Br[240]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 3);
    C10 = vdotq_laneq_s32(C10, B0, A1, 3);
    C20 = vdotq_laneq_s32(C20, B0, A2, 3);
    C30 = vdotq_laneq_s32(C30, B0, A3, 3);
    
    C01 = vdotq_laneq_s32(C01, B1, A0, 3);
    C11 = vdotq_laneq_s32(C11, B1, A1, 3);
    C21 = vdotq_laneq_s32(C21, B1, A2, 3);
    C31 = vdotq_laneq_s32(C31, B1, A3, 3);
  
    C02 = vdotq_laneq_s32(C02, B2, A0, 3);
    C12 = vdotq_laneq_s32(C12, B2, A1, 3);
    C22 = vdotq_laneq_s32(C22, B2, A2, 3);
    C32 = vdotq_laneq_s32(C32, B2, A3, 3);
    
    C03 = vdotq_laneq_s32(C03, B3, A0, 3);
    C13 = vdotq_laneq_s32(C13, B3, A1, 3);
    C23 = vdotq_laneq_s32(C23, B3, A2, 3);
    C33 = vdotq_laneq_s32(C33, B3, A3, 3);

    Ar = Ar + 64;
    Br = Br + 256;

  }

  vst1q_s32(&Crow(0, 0),  C00);  
  vst1q_s32(&Crow(0, 4),  C01);  
  vst1q_s32(&Crow(0, 8),  C02);  
  vst1q_s32(&Crow(0, 12), C03);  

  vst1q_s32(&Crow(1, 0),  C10);  
  vst1q_s32(&Crow(1, 4),  C11);  
  vst1q_s32(&Crow(1, 8),  C12);  
  vst1q_s32(&Crow(1, 12), C13);  
    
  vst1q_s32(&Crow(2, 0),  C20);  
  vst1q_s32(&Crow(2, 4),  C21);  
  vst1q_s32(&Crow(2, 8),  C22);  
  vst1q_s32(&Crow(2, 12), C23);  
    
  vst1q_s32(&Crow(3, 0),  C30);  
  vst1q_s32(&Crow(3, 4),  C31);  
  vst1q_s32(&Crow(3, 8),  C32);  
  vst1q_s32(&Crow(3, 12), C33);  

}

void uk_intrinsic_quantize_int8_4x4_sdot (int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {

  int KR = 16;

  int8x16_t A0, A1, A2, A3, B0, B1, B2, B3;

  int32x4_t C00, C10, C20, C30;


  if (beta == 0) {
    C00 = vdupq_n_s32(0);
    C10 = vdupq_n_s32(0);
    C20 = vdupq_n_s32(0);
    C30 = vdupq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Crow(0, 0));  
    C10 = vld1q_s32(&Crow(1, 0));  
    C20 = vld1q_s32(&Crow(2, 0));  
    C30 = vld1q_s32(&Crow(3, 0));  
  }

  //Loop kc  (+=16)
  for (int i = 0; i < kc; i += KR) {
    //Load A: (4 x 16)
    A0 = vld1q_s8(&Ar[0]);
    A1 = vld1q_s8(&Ar[16]);
    A2 = vld1q_s8(&Ar[32]);
    A3 = vld1q_s8(&Ar[48]);

    //Dot Product
    //----------------------------------
    //Rep +1
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[0]);
    B1 = vld1q_s8(&Br[16]);
    B2 = vld1q_s8(&Br[32]);
    B3 = vld1q_s8(&Br[48]);

    C00 = vdotq_laneq_s32(C00, B0, A0, 0);
    C10 = vdotq_laneq_s32(C10, B0, A1, 0);
    C20 = vdotq_laneq_s32(C20, B0, A2, 0);
    C30 = vdotq_laneq_s32(C30, B0, A3, 0);
  
 
    //----------------------------------
    //Rep +2
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[64]);
    B1 = vld1q_s8(&Br[80]);
    B2 = vld1q_s8(&Br[96]);
    B3 = vld1q_s8(&Br[112]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 1);
    C10 = vdotq_laneq_s32(C10, B0, A1, 1);
    C20 = vdotq_laneq_s32(C20, B0, A2, 1);
    C30 = vdotq_laneq_s32(C30, B0, A3, 1);
  
    
    //----------------------------------
    //Rep +3
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[128]);
    B1 = vld1q_s8(&Br[144]);
    B2 = vld1q_s8(&Br[160]);
    B3 = vld1q_s8(&Br[176]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 2);
    C10 = vdotq_laneq_s32(C10, B0, A1, 2);
    C20 = vdotq_laneq_s32(C20, B0, A2, 2);
    C30 = vdotq_laneq_s32(C30, B0, A3, 2);
    
    
    //----------------------------------
    //Rep +4
    //----------------------------------
    //Load B: (16 x 4) 
    B0 = vld1q_s8(&Br[192]);
    B1 = vld1q_s8(&Br[208]);
    B2 = vld1q_s8(&Br[224]);
    B3 = vld1q_s8(&Br[240]);
    C00 = vdotq_laneq_s32(C00, B0, A0, 3);
    C10 = vdotq_laneq_s32(C10, B0, A1, 3);
    C20 = vdotq_laneq_s32(C20, B0, A2, 3);
    C30 = vdotq_laneq_s32(C30, B0, A3, 3);

    Ar = Ar + 64;
    Br = Br + 256;

  }

  vst1q_s32(&Crow(0, 0),  C00);  
  vst1q_s32(&Crow(1, 0),  C10);  
  vst1q_s32(&Crow(2, 0),  C20);  
  vst1q_s32(&Crow(3, 0),  C30);  

}

#endif

#ifdef ARMV8
void uk_intrinsic_quantize_int8_2x8(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int8x8_t    A00,  A01, A10,  A11;
  int8x8_t    B00,  B01, B10,  B11;

  int16x8_t  _C00, _C01, _C10, _C11;
  
  int32x4_t  C00, C01, C02, C03, C04, C05, C06, C07, 
             C10, C11, C12, C13, C14, C15, C16, C17;

  const int MR = 2;
  const int NR = 8;


  if ( kc==0 ) return;


  C00 = vmovq_n_s32(0);
  C01 = vmovq_n_s32(0);
  C02 = vmovq_n_s32(0);
  C03 = vmovq_n_s32(0);
  C04 = vmovq_n_s32(0);
  C05 = vmovq_n_s32(0);
  C06 = vmovq_n_s32(0);
  C07 = vmovq_n_s32(0);
  
  C10 = vmovq_n_s32(0);
  C11 = vmovq_n_s32(0);
  C12 = vmovq_n_s32(0);
  C13 = vmovq_n_s32(0);
  C14 = vmovq_n_s32(0);
  C15 = vmovq_n_s32(0);
  C16 = vmovq_n_s32(0);
  C17 = vmovq_n_s32(0);

  _C00 = vmovq_n_s16(0);
  _C01 = vmovq_n_s16(0);
  _C10 = vmovq_n_s16(0);
  _C11 = vmovq_n_s16(0);


  baseA = 0;
  baseB = 0;

  if (nr <= 4) {
    if (nr <= 2) {
      for ( k=0; k<kc; k+=16 ) {
        //LOAD A
        //ARow-0
        A00 = vld1_s8(&Ar[baseA + 0]); //kc=8
        A01 = vld1_s8(&Ar[baseA + 8]); //kc=16
        //ARow-1
        A10 = vld1_s8(&Ar[baseA + 16]); //kc=8
        A11 = vld1_s8(&Ar[baseA + 24]); //kc=16
    
        //---------------------------------------
        //1st-Block (2 columns from B) 2x16
        //---------------------------------------
        //LOAD B
        //Bcolumn-0
        B00 = vld1_s8(&Br[baseB + 0]); //kc=8
        B10 = vld1_s8(&Br[baseB + 8]); //kc=16
        //BColumn-1
        B01 = vld1_s8(&Br[baseB + 16]); //kc=8
        B11 = vld1_s8(&Br[baseB + 24]); //kc=16
    
        //Multiply kc=8
        _C00 = vmull_s8(A00, B00);
        _C10 = vmull_s8(A10, B00);
        _C01 = vmull_s8(A00, B01);
        _C11 = vmull_s8(A10, B01);
        
        //Multiply kc=16
        _C00 = vmlal_s8(_C00, A01, B10);
        _C10 = vmlal_s8(_C10, A11, B10);
        _C01 = vmlal_s8(_C01, A01, B11);
        _C11 = vmlal_s8(_C11, A11, B11);
  
        //Reduction pairwise. From int8x8 to int32x4
        C00 = vpadalq_s16(C00, _C00);
        C01 = vpadalq_s16(C01, _C01);
        C10 = vpadalq_s16(C10, _C10);
        C11 = vpadalq_s16(C11, _C11);
    
        //---------------------------------------
    
        baseA = baseA + 32;
        baseB = baseB + 128;
      }
    } else {
      for ( k=0; k<kc; k+=16 ) {
        //LOAD A
        //ARow-0
        A00 = vld1_s8(&Ar[baseA + 0]); //kc=8
        A01 = vld1_s8(&Ar[baseA + 8]); //kc=16
        //ARow-1
        A10 = vld1_s8(&Ar[baseA + 16]); //kc=8
        A11 = vld1_s8(&Ar[baseA + 24]); //kc=16
    
        //---------------------------------------
        //1st-Block (2 columns from B) 2x16
        //---------------------------------------
        //LOAD B
        //Bcolumn-0
        B00 = vld1_s8(&Br[baseB + 0]); //kc=8
        B10 = vld1_s8(&Br[baseB + 8]); //kc=16
        //BColumn-1
        B01 = vld1_s8(&Br[baseB + 16]); //kc=8
        B11 = vld1_s8(&Br[baseB + 24]); //kc=16
    
        //Multiply kc=8
        _C00 = vmull_s8(A00, B00);
        _C10 = vmull_s8(A10, B00);
        _C01 = vmull_s8(A00, B01);
        _C11 = vmull_s8(A10, B01);
        
        //Multiply kc=16
        _C00 = vmlal_s8(_C00, A01, B10);
        _C10 = vmlal_s8(_C10, A11, B10);
        _C01 = vmlal_s8(_C01, A01, B11);
        _C11 = vmlal_s8(_C11, A11, B11);
    
        //Reduction pairwise. From int8x8 to int32x4
        C00 = vpadalq_s16(C00, _C00);
        C01 = vpadalq_s16(C01, _C01);
        C10 = vpadalq_s16(C10, _C10);
        C11 = vpadalq_s16(C11, _C11);
    
        //---------------------------------------
    
        //---------------------------------------
        //2nd-Block (2 columns from B) 2x16. Total a tile of 4x16
        //---------------------------------------
        //LOAD B
        //Bcolumn-0
        B00 = vld1_s8(&Br[baseB + 32]); //kc=8
        B10 = vld1_s8(&Br[baseB + 40]); //kc=16
        //BColumn-1
        B01 = vld1_s8(&Br[baseB + 48]); //kc=8
        B11 = vld1_s8(&Br[baseB + 56]); //kc=16
    
        //Multiply kc=8
        _C00 = vmull_s8(A00, B00);
        _C10 = vmull_s8(A10, B00);
        _C01 = vmull_s8(A00, B01);
        _C11 = vmull_s8(A10, B01);
        
        //Multiply kc=16
        _C00 = vmlal_s8(_C00, A01, B10);
        _C10 = vmlal_s8(_C10, A11, B10);
        _C01 = vmlal_s8(_C01, A01, B11);
        _C11 = vmlal_s8(_C11, A11, B11);
    
        //Reduction pairwise. From int8x8 to int32x4
        C02 = vpadalq_s16(C02, _C00);
        C03 = vpadalq_s16(C03, _C01);
        C12 = vpadalq_s16(C12, _C10);
        C13 = vpadalq_s16(C13, _C11);
        //---------------------------------------
    
        baseA = baseA + 32;
        baseB = baseB + 128;
    
      }
    }
  
    //Now, we need to make the last reduction.
    //With this reduction, we can concatenate the reductions by rows
    //Remember, C is stored by row-major
    C00 = vpaddq_s32(C00, C01);
    C02 = vpaddq_s32(C02, C03);
    C10 = vpaddq_s32(C10, C11);
    C12 = vpaddq_s32(C12, C13);

    //Last reduction 
    C00 = vpaddq_s32(C00, C02);
    C10 = vpaddq_s32(C10, C12);
    
    if (mr == MR && nr == 4) {
      if (beta != zero) {
        C01 = vld1q_s32(&Crow(0, 0));
        C11 = vld1q_s32(&Crow(1, 0));
  
        C00 = vaddq_s32(C00, C01);
        C10 = vaddq_s32(C10, C11);
      }
      vst1q_s32(&Crow(0, 0),  C00);
      vst1q_s32(&Crow(1, 0),  C10);
    } else {
      int32_t Ctmp[MR * NR];
      
      vst1q_s32(&Ctmp[0],  C00);
      vst1q_s32(&Ctmp[8],  C10);
      
      for (int i = 0; i < mr; i++)
       for (int j = 0; j < nr; j++)
          Cr[i*ldC + j] = (beta) * Cr[i*ldC + j] + Ctmp[i * NR + j];
    }

  } else { //nr == NR 
    for ( k=0; k<kc; k+=16 ) {
      //LOAD A
      //ARow-0
      A00 = vld1_s8(&Ar[baseA + 0]); //kc=8
      A01 = vld1_s8(&Ar[baseA + 8]); //kc=16
      //ARow-1
      A10 = vld1_s8(&Ar[baseA + 16]); //kc=8
      A11 = vld1_s8(&Ar[baseA + 24]); //kc=16
  
      //---------------------------------------
      //1st-Block (2 columns from B) 2x16
      //---------------------------------------
      //LOAD B
      //Bcolumn-0
      B00 = vld1_s8(&Br[baseB + 0]); //kc=8
      B10 = vld1_s8(&Br[baseB + 8]); //kc=16
      //BColumn-1
      B01 = vld1_s8(&Br[baseB + 16]); //kc=8
      B11 = vld1_s8(&Br[baseB + 24]); //kc=16
  
      //Multiply kc=8
      _C00 = vmull_s8(A00, B00);
      _C10 = vmull_s8(A10, B00);
      _C01 = vmull_s8(A00, B01);
      _C11 = vmull_s8(A10, B01);
      
      //Multiply kc=16
      _C00 = vmlal_s8(_C00, A01, B10);
      _C10 = vmlal_s8(_C10, A11, B10);
      _C01 = vmlal_s8(_C01, A01, B11);
      _C11 = vmlal_s8(_C11, A11, B11);
  
      //Reduction pairwise. From int8x8 to int32x4
      C00 = vpadalq_s16(C00, _C00);
      C01 = vpadalq_s16(C01, _C01);
      C10 = vpadalq_s16(C10, _C10);
      C11 = vpadalq_s16(C11, _C11);
  
      //---------------------------------------
  
      //---------------------------------------
      //2nd-Block (2 columns from B) 2x16. Total a tile of 4x16
      //---------------------------------------
      //LOAD B
      //Bcolumn-0
      B00 = vld1_s8(&Br[baseB + 32]); //kc=8
      B10 = vld1_s8(&Br[baseB + 40]); //kc=16
      //BColumn-1
      B01 = vld1_s8(&Br[baseB + 48]); //kc=8
      B11 = vld1_s8(&Br[baseB + 56]); //kc=16
  
      //Multiply kc=8
      _C00 = vmull_s8(A00, B00);
      _C10 = vmull_s8(A10, B00);
      _C01 = vmull_s8(A00, B01);
      _C11 = vmull_s8(A10, B01);
      
      //Multiply kc=16
      _C00 = vmlal_s8(_C00, A01, B10);
      _C10 = vmlal_s8(_C10, A11, B10);
      _C01 = vmlal_s8(_C01, A01, B11);
      _C11 = vmlal_s8(_C11, A11, B11);
  
      //Reduction pairwise. From int8x8 to int32x4
      C02 = vpadalq_s16(C02, _C00);
      C03 = vpadalq_s16(C03, _C01);
      C12 = vpadalq_s16(C12, _C10);
      C13 = vpadalq_s16(C13, _C11);
      //---------------------------------------
  
      //---------------------------------------
      //3th-Block (2 columns from B) 2x16. Total a tile of 6x16
      //---------------------------------------
      //LOAD B
      //Bcolumn-0
      B00 = vld1_s8(&Br[baseB + 64]); //kc=8
      B10 = vld1_s8(&Br[baseB + 72]); //kc=16
      //BColumn-1
      B01 = vld1_s8(&Br[baseB + 80]); //kc=8
      B11 = vld1_s8(&Br[baseB + 88]); //kc=16
  
      //Multiply kc=8
      _C00 = vmull_s8(A00, B00);
      _C10 = vmull_s8(A10, B00);
      _C01 = vmull_s8(A00, B01);
      _C11 = vmull_s8(A10, B01);
      
      //Multiply kc=16
      _C00 = vmlal_s8(_C00, A01, B10);
      _C10 = vmlal_s8(_C10, A11, B10);
      _C01 = vmlal_s8(_C01, A01, B11);
      _C11 = vmlal_s8(_C11, A11, B11);
  
      //Reduction pairwise. From int8x8 to int32x4
      C04 = vpadalq_s16(C04, _C00);
      C05 = vpadalq_s16(C05, _C01);
      C14 = vpadalq_s16(C14, _C10);
      C15 = vpadalq_s16(C15, _C11);
      //---------------------------------------
  
      //---------------------------------------
      //4th-Block (2 columns from B) 2x16. Total a tile of 8x16. Last Block!!
      //---------------------------------------
      //LOAD B
      //Bcolumn-0
      B00 = vld1_s8(&Br[baseB + 96]); //kc=8
      B10 = vld1_s8(&Br[baseB + 104]); //kc=16
      //BColumn-1
      B01 = vld1_s8(&Br[baseB + 112]); //kc=8
      B11 = vld1_s8(&Br[baseB + 120]); //kc=16
  
      //Multiply kc=8
      _C00 = vmull_s8(A00, B00);
      _C10 = vmull_s8(A10, B00);
      _C01 = vmull_s8(A00, B01);
      _C11 = vmull_s8(A10, B01);
      
      //Multiply kc=16
      _C00 = vmlal_s8(_C00, A01, B10);
      _C10 = vmlal_s8(_C10, A11, B10);
      _C01 = vmlal_s8(_C01, A01, B11);
      _C11 = vmlal_s8(_C11, A11, B11);
  
      //Reduction pairwise. From int8x8 to int32x4
      C06 = vpadalq_s16(C06, _C00);
      C07 = vpadalq_s16(C07, _C01);
      C16 = vpadalq_s16(C16, _C10);
      C17 = vpadalq_s16(C17, _C11);
      //---------------------------------------
      
      baseA = baseA + 32;
      baseB = baseB + 128;
  
    }
  
    //Now, we need to make the last reduction.
    //With this reduction, we can concatenate the reductions by rows
    //Remember, C is stored by row-major
    C00 = vpaddq_s32(C00, C01);
    C02 = vpaddq_s32(C02, C03);
    C04 = vpaddq_s32(C04, C05);
    C06 = vpaddq_s32(C06, C07);
    C10 = vpaddq_s32(C10, C11);
    C12 = vpaddq_s32(C12, C13);
    C14 = vpaddq_s32(C14, C15);
    C16 = vpaddq_s32(C16, C17);
  
    //Last reduction 
    C00 = vpaddq_s32(C00, C02);
    C04 = vpaddq_s32(C04, C06);
    C10 = vpaddq_s32(C10, C12);
    C14 = vpaddq_s32(C14, C16);

    if (mr == MR && nr == NR) {
      if (beta != zero) {
        C01 = vld1q_s32(&Crow(0, 0));
        C02 = vld1q_s32(&Crow(0, 4));
        C11 = vld1q_s32(&Crow(1, 0));
        C12 = vld1q_s32(&Crow(1, 4));
  
        C00 = vaddq_s32(C00, C01);
        C04 = vaddq_s32(C04, C02);
        C10 = vaddq_s32(C10, C11);
        C14 = vaddq_s32(C14, C12);
      }
      vst1q_s32(&Crow(0, 0),  C00);
      vst1q_s32(&Crow(0, 4),  C04);
      vst1q_s32(&Crow(1, 0),  C10);
      vst1q_s32(&Crow(1, 4),  C14);
    } else {
      int32_t Ctmp[MR * NR];
      
      vst1q_s32(&Ctmp[0],  C00);
      vst1q_s32(&Ctmp[4],  C04);
      vst1q_s32(&Ctmp[8],  C10);
      vst1q_s32(&Ctmp[12], C14);
      
      for (int i = 0; i < mr; i++)
       for (int j = 0; j < nr; j++)
          Cr[i*ldC + j] = (beta) * Cr[i*ldC + j] + Ctmp[i * NR + j];
    }

  }

}

#endif

