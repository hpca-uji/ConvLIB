#include "ukernels.h"

#define Ccol(a1,a2)  Cr[ (a2)*(ldC)+(a1) ]



void fselector(int MR, int NR, UK_TYPE *uk_vec, UK_EDGE_TYPE *uk_edge_vec, UK_TYPE *uk, UK_EDGE_TYPE *uk_edge) {

  #ifdef FP32
    uk_asm_selector_fp32(MR, NR, uk_vec, uk);
    uk_asm_edge_selector_fp32(MR, NR, uk_edge_vec, uk_edge);
  #elif FP16
    uk_intrinsic_selector_fp16(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #elif INT8_INT32_U8
    uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #elif INT8_INT32_S8
    uk_intrinsic_selector_int8_int32(MR, NR, uk_vec, uk);
    *uk_edge = *uk;
  #endif

}

void generic_microkernel(int mr, int nr, int MR, int NR, AB_TYPE *A, AB_TYPE *B, 
		                C_TYPE *C, uint32_t kc, uint32_t ldC, C_TYPE alpha, C_TYPE beta, 
				C_TYPE *aux, UK_TYPE uk, UK_EDGE_TYPE uk_edge) {

  #ifdef FP32
    if (mr == MR && nr == NR)
      uk(kc, &alpha, A, B, &beta, C, ldC * sizeof(float));
    else
      uk_edge(mr, nr, MR, NR, kc, &alpha, A, B, &beta, aux, C, ldC);
  #else
    if ((mr == MR) && (nr == NR))
      uk(kc, A, B, C, beta, ldC);
    else {
      uk(kc, A, B, aux, 0, MR);
      for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        C[j*ldC + i] = (beta) * C[j*ldC + i] + aux[j * MR + i];
     }
  #endif

}




//============================================================================================================
//============================================================================================================


void ukernel_intrinsic_vmlal_16x8_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {
  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=16 x NR=4

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, C02, C03, C04, C05, C06, C07, //
	    C10, C11, C12, C13, C14, C15, C16, C17, //
	    C20, C21, C22, C23, C24, C25, C26, C27, //
	    C30, C31, C32, C33, C34, C35, C36, C37; //+32

  int16x8_t  A0, A1;
  int16x4_t  A0low, A1low;
  int16x8_t  B0;

  if ( kc==0 ) return;

  if (beta==zero) {
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

    C20 = vmovq_n_s32(0); 
    C21 = vmovq_n_s32(0); 
    C22 = vmovq_n_s32(0); 
    C23 = vmovq_n_s32(0);
    C24 = vmovq_n_s32(0); 
    C25 = vmovq_n_s32(0); 
    C26 = vmovq_n_s32(0); 
    C27 = vmovq_n_s32(0);

    C30 = vmovq_n_s32(0); 
    C31 = vmovq_n_s32(0); 
    C32 = vmovq_n_s32(0); 
    C33 = vmovq_n_s32(0);
    C34 = vmovq_n_s32(0); 
    C35 = vmovq_n_s32(0); 
    C36 = vmovq_n_s32(0); 
    C37 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));  
    C01 = vld1q_s32(&Ccol(0,1));  
    C02 = vld1q_s32(&Ccol(0,2));  
    C03 = vld1q_s32(&Ccol(0,3));
    C04 = vld1q_s32(&Ccol(0,4));  
    C05 = vld1q_s32(&Ccol(0,5));  
    C06 = vld1q_s32(&Ccol(0,6));  
    C07 = vld1q_s32(&Ccol(0,7));

    C10 = vld1q_s32(&Ccol(4,0));  
    C11 = vld1q_s32(&Ccol(4,1));  
    C12 = vld1q_s32(&Ccol(4,2));  
    C13 = vld1q_s32(&Ccol(4,3));
    C14 = vld1q_s32(&Ccol(4,4));  
    C15 = vld1q_s32(&Ccol(4,5));  
    C16 = vld1q_s32(&Ccol(4,6));  
    C17 = vld1q_s32(&Ccol(4,7));

    C20 = vld1q_s32(&Ccol(8,0));  
    C21 = vld1q_s32(&Ccol(8,1));  
    C22 = vld1q_s32(&Ccol(8,2));  
    C23 = vld1q_s32(&Ccol(8,3));
    C24 = vld1q_s32(&Ccol(8,4));  
    C25 = vld1q_s32(&Ccol(8,5));  
    C26 = vld1q_s32(&Ccol(8,6));  
    C27 = vld1q_s32(&Ccol(8,7));

    C30 = vld1q_s32(&Ccol(12,0)); 
    C31 = vld1q_s32(&Ccol(12,1)); 
    C32 = vld1q_s32(&Ccol(12,2)); 
    C33 = vld1q_s32(&Ccol(12,3));
    C34 = vld1q_s32(&Ccol(12,4));  
    C35 = vld1q_s32(&Ccol(12,5));  
    C36 = vld1q_s32(&Ccol(12,6));  
    C37 = vld1q_s32(&Ccol(12,7));
  }

  baseA = 0;
  baseB = 0;

  for ( k=0; k<kc; k++ ) {
    A0 = vmovl_s8(vld1_s8(&Ar[baseA + 0])); //Load 8 elements from A #int8x8 ¿16?
    A1 = vmovl_s8(vld1_s8(&Ar[baseA + 8])); //Load 8 elements from A #int8x8

    B0 = vmovl_s8(vld1_s8(&Br[baseB])); //Load 8 elements from A #int8x8

    A0low = vget_low_s16(A0);
    A1low = vget_low_s16(A1);

    C00 = vmlal_laneq_s16     (C00, A0low, B0, 0); 
    C10 = vmlal_high_laneq_s16(C10, A0,    B0, 0);
    C20 = vmlal_laneq_s16     (C20, A1low, B0, 0); 
    C30 = vmlal_high_laneq_s16(C30, A1,    B0, 0);

    C01 = vmlal_laneq_s16     (C01, A0low, B0, 1); 
    C11 = vmlal_high_laneq_s16(C11, A0,    B0, 1);
    C21 = vmlal_laneq_s16     (C21, A1low, B0, 1); 
    C31 = vmlal_high_laneq_s16(C31, A1,    B0, 1);

    C02 = vmlal_laneq_s16     (C02, A0low, B0, 2); 
    C12 = vmlal_high_laneq_s16(C12, A0,    B0, 2);
    C22 = vmlal_laneq_s16     (C22, A1low, B0, 2); 
    C32 = vmlal_high_laneq_s16(C32, A1,    B0, 2);

    C03 = vmlal_laneq_s16     (C03, A0low, B0, 3); 
    C13 = vmlal_high_laneq_s16(C13, A0,    B0, 3);
    C23 = vmlal_laneq_s16     (C23, A1low, B0, 3); 
    C33 = vmlal_high_laneq_s16(C33, A1,    B0, 3);
    
    C04 = vmlal_laneq_s16     (C04, A0low, B0, 4); 
    C14 = vmlal_high_laneq_s16(C14, A0,    B0, 4);
    C24 = vmlal_laneq_s16     (C24, A1low, B0, 4); 
    C34 = vmlal_high_laneq_s16(C34, A1,    B0, 4);
    
    C05 = vmlal_laneq_s16     (C05, A0low, B0, 5); 
    C15 = vmlal_high_laneq_s16(C15, A0,    B0, 5);
    C25 = vmlal_laneq_s16     (C25, A1low, B0, 5); 
    C35 = vmlal_high_laneq_s16(C35, A1,    B0, 5);
    
    C06 = vmlal_laneq_s16     (C06, A0low, B0, 6); 
    C16 = vmlal_high_laneq_s16(C16, A0,    B0, 6);
    C26 = vmlal_laneq_s16     (C26, A1low, B0, 6); 
    C36 = vmlal_high_laneq_s16(C36, A1,    B0, 6);
    
    C07 = vmlal_laneq_s16     (C07, A0low, B0, 7); 
    C17 = vmlal_high_laneq_s16(C17, A0,    B0, 7);
    C27 = vmlal_laneq_s16     (C27, A1low, B0, 7); 
    C37 = vmlal_high_laneq_s16(C37, A1,    B0, 7);

    //vr = vmlal_laneq_s16();      // int32x4_t = int32x4_t + int16x4_t * int16x8_t;
    //vr = vmlal_high_laneq_s16(); // int32x4_t = int32x4_t + int16x8_t * int16x8_t

    baseA = baseA + 16; 
    baseB = baseB + 8;

  }
  
  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);
  vst1q_s32(&Ccol(0,2), C02);
  vst1q_s32(&Ccol(0,3), C03);
  vst1q_s32(&Ccol(0,4), C04);
  vst1q_s32(&Ccol(0,5), C05);
  vst1q_s32(&Ccol(0,6), C06);
  vst1q_s32(&Ccol(0,7), C07);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);
  vst1q_s32(&Ccol(4,2), C12);
  vst1q_s32(&Ccol(4,3), C13);
  vst1q_s32(&Ccol(4,4), C14);
  vst1q_s32(&Ccol(4,5), C15);
  vst1q_s32(&Ccol(4,6), C16);
  vst1q_s32(&Ccol(4,7), C17);
    
  vst1q_s32(&Ccol(8,0), C20);
  vst1q_s32(&Ccol(8,1), C21);
  vst1q_s32(&Ccol(8,2), C22);
  vst1q_s32(&Ccol(8,3), C23);
  vst1q_s32(&Ccol(8,4), C24);
  vst1q_s32(&Ccol(8,5), C25);
  vst1q_s32(&Ccol(8,6), C26);
  vst1q_s32(&Ccol(8,7), C27);
  
  vst1q_s32(&Ccol(12,0), C30);
  vst1q_s32(&Ccol(12,1), C31);
  vst1q_s32(&Ccol(12,2), C32);
  vst1q_s32(&Ccol(12,3), C33);
  vst1q_s32(&Ccol(12,4), C34);
  vst1q_s32(&Ccol(12,5), C35);
  vst1q_s32(&Ccol(12,6), C36);
  vst1q_s32(&Ccol(12,7), C37);


 }

void ukernel_intrinsic_vmlal_16x4_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {
  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=16 x NR=4

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, C02, C03, 
	    C10, C11, C12, C13, 
	    C20, C21, C22, C23, 
	    C30, C31, C32, C33;

  int16x8_t  A0, A1;
  int16x4_t  A0low, A1low;
  int16x4_t  B0;

  if ( kc==0 ) return;

  if (beta==zero) {
    C00 = vmovq_n_s32(0); 
    C01 = vmovq_n_s32(0); 
    C02 = vmovq_n_s32(0); 
    C03 = vmovq_n_s32(0);

    C10 = vmovq_n_s32(0); 
    C11 = vmovq_n_s32(0); 
    C12 = vmovq_n_s32(0); 
    C13 = vmovq_n_s32(0);

    C20 = vmovq_n_s32(0); 
    C21 = vmovq_n_s32(0); 
    C22 = vmovq_n_s32(0); 
    C23 = vmovq_n_s32(0);

    C30 = vmovq_n_s32(0); 
    C31 = vmovq_n_s32(0); 
    C32 = vmovq_n_s32(0); 
    C33 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));  
    C01 = vld1q_s32(&Ccol(0,1));  
    C02 = vld1q_s32(&Ccol(0,2));  
    C03 = vld1q_s32(&Ccol(0,3));

    C10 = vld1q_s32(&Ccol(4,0));  
    C11 = vld1q_s32(&Ccol(4,1));  
    C12 = vld1q_s32(&Ccol(4,2));  
    C13 = vld1q_s32(&Ccol(4,3));

    C20 = vld1q_s32(&Ccol(8,0));  
    C21 = vld1q_s32(&Ccol(8,1));  
    C22 = vld1q_s32(&Ccol(8,2));  
    C23 = vld1q_s32(&Ccol(8,3));

    C30 = vld1q_s32(&Ccol(12,0)); 
    C31 = vld1q_s32(&Ccol(12,1)); 
    C32 = vld1q_s32(&Ccol(12,2)); 
    C33 = vld1q_s32(&Ccol(12,3));
  }

  baseA = 0;
  baseB = 0;

  for ( k=0; k<kc; k++ ) {
    A0 = vmovl_s8(vld1_s8(&Ar[baseA + 0])); //Load 8 elements from A #int8x8 ¿16?
    A1 = vmovl_s8(vld1_s8(&Ar[baseA + 8])); //Load 8 elements from A #int8x8

    B0 = vget_low_s16(vmovl_s8(vld1_s8(&Br[baseB]))); //Load 8 elements from A #int8x8

    A0low = vget_low_s16(A0);
    A1low = vget_low_s16(A1);

    C00 = vmlal_lane_s16     (C00, A0low, B0, 0); 
    C10 = vmlal_high_lane_s16(C10, A0,    B0, 0);
    C20 = vmlal_lane_s16     (C20, A1low, B0, 0); 
    C30 = vmlal_high_lane_s16(C30, A1,    B0, 0);

    C01 = vmlal_lane_s16     (C01, A0low, B0, 1); 
    C11 = vmlal_high_lane_s16(C11, A0,    B0, 1);
    C21 = vmlal_lane_s16     (C21, A1low, B0, 1); 
    C31 = vmlal_high_lane_s16(C31, A1,    B0, 1);

    C02 = vmlal_lane_s16     (C02, A0low, B0, 2); 
    C12 = vmlal_high_lane_s16(C12, A0,    B0, 2);
    C22 = vmlal_lane_s16     (C22, A1low, B0, 2); 
    C32 = vmlal_high_lane_s16(C32, A1,    B0, 2);

    C03 = vmlal_lane_s16     (C03, A0low, B0, 3); 
    C13 = vmlal_high_lane_s16(C13, A0,    B0, 3);
    C23 = vmlal_lane_s16     (C23, A1low, B0, 3); 
    C33 = vmlal_high_lane_s16(C33, A1,    B0, 3);
    
    baseA = baseA + 16; 
    baseB = baseB + 4;

  }
  
  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);
  vst1q_s32(&Ccol(0,2), C02);
  vst1q_s32(&Ccol(0,3), C03);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);
  vst1q_s32(&Ccol(4,2), C12);
  vst1q_s32(&Ccol(4,3), C13);
    
  vst1q_s32(&Ccol(8,0), C20);
  vst1q_s32(&Ccol(8,1), C21);
  vst1q_s32(&Ccol(8,2), C22);
  vst1q_s32(&Ccol(8,3), C23);
  
  vst1q_s32(&Ccol(12,0), C30);
  vst1q_s32(&Ccol(12,1), C31);
  vst1q_s32(&Ccol(12,2), C32);
  vst1q_s32(&Ccol(12,3), C33);

}

void ukernel_intrinsic_vmull_8x2_ux2_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {
  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=8 x NR=2

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, 
            C10, C11;

  int8x16_t _A;
  int8x8_t  A0, A1;

  int8x8_t  B00, B01,
            B10, B11;

  int16x8_t VPa, VPb;

  if ( kc==0 ) return;

  if (beta==zero) {
    C00 = vmovq_n_s32(0); C01 = vmovq_n_s32(0);
    C10 = vmovq_n_s32(0); C11 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));
    C01 = vld1q_s32(&Ccol(0,1));

    C10 = vld1q_s32(&Ccol(4,0));
    C11 = vld1q_s32(&Ccol(4,1));
  }

  baseA = 0;
  baseB = 0;

  // Iterate from 0 to kc, in steps of 2
  // This loop can be unrolled in a larger factor to reduce the cost of vdup
  for ( k=0; k<kc-1; k+=2 ) {
    _A = vld1q_s8(&Ar[baseA]);

    A0 = vget_low_s8(_A);
    A1 = vget_high_s8(_A);
    baseA = baseA + 16; 

    B00 = vdup_n_s8(Br[baseB+0]); 
    B01 = vdup_n_s8(Br[baseB+1]); 
    B10 = vdup_n_s8(Br[baseB+2]); 
    B11 = vdup_n_s8(Br[baseB+3]); 
    baseB = baseB + 4;

    VPa = vmull_s8(A0, B00);
    VPb = vmlal_s8(VPa, A1, B10);
    
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16(VPb)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VPb)));
    
    VPa = vmull_s8(A0, B01);
    VPb = vmlal_s8(VPa, A1, B11);

    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16(VPb)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VPb)));
  }
  

  // Last iteration if kc is an odd number
  if ((kc%2) != 0) {
    A0  = vld1_s8(&Ar[baseA]); 
    
    B00 = vdup_n_s8(Br[baseB+0]); 
    B01 = vdup_n_s8(Br[baseB+1]); 

    VPb = vmull_s8(A0, B00);
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16(VPb)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VPb)));

    VPb = vmull_s8(A0, B01);
    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16(VPb)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VPb)));
  }


  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);

}


void ukernel_intrinsic_vmull_16x4_ux2_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {
  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=16 x NR=4

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, C02, C03,
	    C10, C11, C12, C13,
	    C20, C21, C22, C23,
	    C30, C31, C32, C33; //+16

  int8x16_t _A, _An;

  int8x8_t  A0, A1, A0n, A1n;
  int8x8_t  _B, B0, B1, B2, B3, B0n, B1n, B2n, B3n;

  int16x8_t VM0, VM1, VM2, VM3;

  if ( kc==0 ) return;

  if (beta==zero) {
    C00 = vmovq_n_s32(0); 
    C01 = vmovq_n_s32(0); 
    C02 = vmovq_n_s32(0); 
    C03 = vmovq_n_s32(0);

    C10 = vmovq_n_s32(0); 
    C11 = vmovq_n_s32(0); 
    C12 = vmovq_n_s32(0); 
    C13 = vmovq_n_s32(0);

    C20 = vmovq_n_s32(0); 
    C21 = vmovq_n_s32(0); 
    C22 = vmovq_n_s32(0); 
    C23 = vmovq_n_s32(0);

    C30 = vmovq_n_s32(0); 
    C31 = vmovq_n_s32(0); 
    C32 = vmovq_n_s32(0); 
    C33 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));  
    C01 = vld1q_s32(&Ccol(0,1));  
    C02 = vld1q_s32(&Ccol(0,2));  
    C03 = vld1q_s32(&Ccol(0,3));

    C10 = vld1q_s32(&Ccol(4,0));  
    C11 = vld1q_s32(&Ccol(4,1));  
    C12 = vld1q_s32(&Ccol(4,2));  
    C13 = vld1q_s32(&Ccol(4,3));

    C20 = vld1q_s32(&Ccol(8,0));  
    C21 = vld1q_s32(&Ccol(8,1));  
    C22 = vld1q_s32(&Ccol(8,2));  
    C23 = vld1q_s32(&Ccol(8,3));

    C30 = vld1q_s32(&Ccol(12,0)); 
    C31 = vld1q_s32(&Ccol(12,1)); 
    C32 = vld1q_s32(&Ccol(12,2)); 
    C33 = vld1q_s32(&Ccol(12,3));
  }

  baseA = 0;
  baseB = 0;

  // Iterate from 0 to kc, in steps of 2
  // This loop can be unrolled in a larger factor to reduce the cost of vdup
  for ( k=0; k<kc-1; k+=2 ) {
    //Load A
    _A = vld1q_s8(&Ar[baseA]);
    A0 = vget_low_s8 (_A);
    A1 = vget_high_s8(_A);

    //Load A: Next Iteration
    _An = vld1q_s8(&Ar[baseA+16]);
    A0n = vget_low_s8 (_An);
    A1n = vget_high_s8(_An);

    //Load B
    _B = vld1_s8(&Br[baseB]);
    B0 = vdup_lane_s8(_B, 0);
    B1 = vdup_lane_s8(_B, 1);
    B2 = vdup_lane_s8(_B, 2);
    B3 = vdup_lane_s8(_B, 3);
    //B: Next Iteration
    B0n = vdup_lane_s8(_B, 4);
    B1n = vdup_lane_s8(_B, 5);
    B2n = vdup_lane_s8(_B, 6);
    B3n = vdup_lane_s8(_B, 7);

    //Update C: 8x4     
    VM0 = vmull_s8(A0, B0);
    VM0 = vmlal_s8(VM0, A0n, B0n);
    VM1 = vmull_s8(A0, B1);
    VM1 = vmlal_s8(VM1, A0n, B1n);
    VM2 = vmull_s8(A0, B2);
    VM2 = vmlal_s8(VM2, A0n, B2n);
    VM3 = vmull_s8(A0, B3);
    VM3 = vmlal_s8(VM3, A0n, B3n);
    
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16 (VM0)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VM0)));
    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16 (VM1)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VM1)));
    C02 = vaddq_s32(C02, vmovl_s16(vget_low_s16 (VM2)));
    C12 = vaddq_s32(C12, vmovl_s16(vget_high_s16(VM2)));
    C03 = vaddq_s32(C03, vmovl_s16(vget_low_s16 (VM3)));
    C13 = vaddq_s32(C13, vmovl_s16(vget_high_s16(VM3)));

    //Update-next C: 8x4 (Total 16x4)
    VM0 = vmull_s8(A1, B0);
    VM0 = vmlal_s8(VM0, A1n, B0n);
    VM1 = vmull_s8(A1, B1);
    VM1 = vmlal_s8(VM1, A1n, B1n);
    VM2 = vmull_s8(A1, B2);
    VM2 = vmlal_s8(VM2, A1n, B2n);
    VM3 = vmull_s8(A1, B3);
    VM3 = vmlal_s8(VM3, A1n, B3n);
    
    C20 = vaddq_s32(C20, vmovl_s16(vget_low_s16 (VM0)));
    C30 = vaddq_s32(C30, vmovl_s16(vget_high_s16(VM0)));
    C21 = vaddq_s32(C21, vmovl_s16(vget_low_s16 (VM1)));
    C31 = vaddq_s32(C31, vmovl_s16(vget_high_s16(VM1)));
    C22 = vaddq_s32(C22, vmovl_s16(vget_low_s16 (VM2)));
    C32 = vaddq_s32(C32, vmovl_s16(vget_high_s16(VM2)));
    C23 = vaddq_s32(C23, vmovl_s16(vget_low_s16 (VM3)));
    C33 = vaddq_s32(C33, vmovl_s16(vget_high_s16(VM3)));
    
    baseA = baseA + 32; 
    baseB = baseB + 8;
  }
  

  // Last iteration if kc is an odd number
  if ((kc%2) != 0) {
    //Load A
    _A  = vld1q_s8(&Ar[baseA]);
    A0 = vget_low_s8 (_A);
    A1 = vget_high_s8(_A);

    //Load B
    B0 = vdup_n_s8(Br[baseB+0]); 
    B1 = vdup_n_s8(Br[baseB+1]); 
    B2 = vdup_n_s8(Br[baseB+2]); 
    B3 = vdup_n_s8(Br[baseB+3]); 
    
    //Update C: 8x4     
    VM0 = vmull_s8(A0, B0);
    VM1 = vmull_s8(A0, B1);
    VM2 = vmull_s8(A0, B2);
    VM3 = vmull_s8(A0, B3);
    
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16 (VM0)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VM0)));
    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16 (VM1)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VM1)));
    C02 = vaddq_s32(C02, vmovl_s16(vget_low_s16 (VM2)));
    C12 = vaddq_s32(C12, vmovl_s16(vget_high_s16(VM2)));
    C03 = vaddq_s32(C03, vmovl_s16(vget_low_s16 (VM3)));
    C13 = vaddq_s32(C13, vmovl_s16(vget_high_s16(VM3)));

    //Update-next C: 8x4 (Total 16x4)
    VM0 = vmull_s8(A1, B0);
    VM1 = vmull_s8(A1, B1);
    VM2 = vmull_s8(A1, B2);
    VM3 = vmull_s8(A1, B3);
    
    C20 = vaddq_s32(C20, vmovl_s16(vget_low_s16 (VM0)));
    C30 = vaddq_s32(C30, vmovl_s16(vget_high_s16(VM0)));
    C21 = vaddq_s32(C21, vmovl_s16(vget_low_s16 (VM1)));
    C31 = vaddq_s32(C31, vmovl_s16(vget_high_s16(VM1)));
    C22 = vaddq_s32(C22, vmovl_s16(vget_low_s16 (VM2)));
    C32 = vaddq_s32(C32, vmovl_s16(vget_high_s16(VM2)));
    C23 = vaddq_s32(C23, vmovl_s16(vget_low_s16 (VM3)));
    C33 = vaddq_s32(C33, vmovl_s16(vget_high_s16(VM3)));
  }


  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);
  vst1q_s32(&Ccol(0,2), C02);
  vst1q_s32(&Ccol(0,3), C03);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);
  vst1q_s32(&Ccol(4,2), C12);
  vst1q_s32(&Ccol(4,3), C13);
    
  vst1q_s32(&Ccol(8,0), C20);
  vst1q_s32(&Ccol(8,1), C21);
  vst1q_s32(&Ccol(8,2), C22);
  vst1q_s32(&Ccol(8,3), C23);
  
  vst1q_s32(&Ccol(12,0), C30);
  vst1q_s32(&Ccol(12,1), C31);
  vst1q_s32(&Ccol(12,2), C32);
  vst1q_s32(&Ccol(12,3), C33);


}

void ukernel_intrinsic_vmull_16x8_ux2_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {
  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=16 x NR=4

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, C02, C03, C04, C05, C06, C07,
	    C10, C11, C12, C13, C14, C15, C16, C17,
	    C20, C21, C22, C23, C24, C25, C26, C27,
	    C30, C31, C32, C33, C34, C35, C36, C37;

  int8x16_t _A, _An;
  int8x8_t  A0, A1, A0n, A1n, B, Bn;

  int16x8_t VM;

  if ( kc==0 ) return;

  if (beta==zero) {
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

    C20 = vmovq_n_s32(0); 
    C21 = vmovq_n_s32(0); 
    C22 = vmovq_n_s32(0); 
    C23 = vmovq_n_s32(0);
    C24 = vmovq_n_s32(0); 
    C25 = vmovq_n_s32(0);
    C26 = vmovq_n_s32(0); 
    C27 = vmovq_n_s32(0);

    C30 = vmovq_n_s32(0); 
    C31 = vmovq_n_s32(0); 
    C32 = vmovq_n_s32(0); 
    C33 = vmovq_n_s32(0);
    C34 = vmovq_n_s32(0); 
    C35 = vmovq_n_s32(0);
    C36 = vmovq_n_s32(0); 
    C37 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));  
    C01 = vld1q_s32(&Ccol(0,1));  
    C02 = vld1q_s32(&Ccol(0,2));  
    C03 = vld1q_s32(&Ccol(0,3));
    C04 = vld1q_s32(&Ccol(0,4));  
    C05 = vld1q_s32(&Ccol(0,5));
    C06 = vld1q_s32(&Ccol(0,6));  
    C07 = vld1q_s32(&Ccol(0,7));

    C10 = vld1q_s32(&Ccol(4,0));  
    C11 = vld1q_s32(&Ccol(4,1));  
    C12 = vld1q_s32(&Ccol(4,2));  
    C13 = vld1q_s32(&Ccol(4,3));
    C14 = vld1q_s32(&Ccol(4,4));  
    C15 = vld1q_s32(&Ccol(4,5));
    C16 = vld1q_s32(&Ccol(4,6));  
    C17 = vld1q_s32(&Ccol(4,7));

    C20 = vld1q_s32(&Ccol(8,0));  
    C21 = vld1q_s32(&Ccol(8,1));  
    C22 = vld1q_s32(&Ccol(8,2));  
    C23 = vld1q_s32(&Ccol(8,3));
    C24 = vld1q_s32(&Ccol(8,4));  
    C25 = vld1q_s32(&Ccol(8,5));
    C26 = vld1q_s32(&Ccol(8,6));  
    C27 = vld1q_s32(&Ccol(8,7));

    C30 = vld1q_s32(&Ccol(12,0)); 
    C31 = vld1q_s32(&Ccol(12,1)); 
    C32 = vld1q_s32(&Ccol(12,2)); 
    C33 = vld1q_s32(&Ccol(12,3));
    C34 = vld1q_s32(&Ccol(12,4));  
    C35 = vld1q_s32(&Ccol(12,5));
    C36 = vld1q_s32(&Ccol(12,6));  
    C37 = vld1q_s32(&Ccol(12,7));
  }

  baseA = 0;
  baseB = 0;

  // Iterate from 0 to kc, in steps of 2
  // his loop can be unrolled in a larger factor to reduce the cost of vdup
  for ( k=0; k<kc-1; k+=2 ) {
    _A = vld1q_s8(&Ar[baseA]);
    A0 = vget_low_s8 (_A);
    A1 = vget_high_s8(_A);

    _An = vld1q_s8(&Ar[baseA+16]);
    A0n = vget_low_s8 (_An);
    A1n = vget_high_s8(_An);

    B  = vdup_n_s8(Br[baseB+0]);
    Bn = vdup_n_s8(Br[baseB+8]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16 (VM)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C20 = vaddq_s32(C20, vmovl_s16(vget_low_s16 (VM)));
    C30 = vaddq_s32(C30, vmovl_s16(vget_high_s16(VM)));


    B  = vdup_n_s8(Br[baseB+1]);
    Bn = vdup_n_s8(Br[baseB+9]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16 (VM)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C21 = vaddq_s32(C21, vmovl_s16(vget_low_s16 (VM)));
    C31 = vaddq_s32(C31, vmovl_s16(vget_high_s16(VM)));
   

    B  = vdup_n_s8(Br[baseB+2]);
    Bn = vdup_n_s8(Br[baseB+10]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C02 = vaddq_s32(C02, vmovl_s16(vget_low_s16 (VM)));
    C12 = vaddq_s32(C12, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C22 = vaddq_s32(C22, vmovl_s16(vget_low_s16 (VM)));
    C32 = vaddq_s32(C32, vmovl_s16(vget_high_s16(VM)));
   

    B  = vdup_n_s8(Br[baseB+3]);
    Bn = vdup_n_s8(Br[baseB+11]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C03 = vaddq_s32(C03, vmovl_s16(vget_low_s16 (VM)));
    C13 = vaddq_s32(C13, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C23 = vaddq_s32(C23, vmovl_s16(vget_low_s16 (VM)));
    C33 = vaddq_s32(C33, vmovl_s16(vget_high_s16(VM)));


    B  = vdup_n_s8(Br[baseB+4]);
    Bn = vdup_n_s8(Br[baseB+12]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C04 = vaddq_s32(C04, vmovl_s16(vget_low_s16 (VM)));
    C14 = vaddq_s32(C14, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C24 = vaddq_s32(C24, vmovl_s16(vget_low_s16 (VM)));
    C34 = vaddq_s32(C34, vmovl_s16(vget_high_s16(VM)));
    
    
    B  = vdup_n_s8(Br[baseB+5]);
    Bn = vdup_n_s8(Br[baseB+13]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C05 = vaddq_s32(C05, vmovl_s16(vget_low_s16 (VM)));
    C15 = vaddq_s32(C15, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C25 = vaddq_s32(C25, vmovl_s16(vget_low_s16 (VM)));
    C35 = vaddq_s32(C35, vmovl_s16(vget_high_s16(VM)));
    
    B  = vdup_n_s8(Br[baseB+6]);
    Bn = vdup_n_s8(Br[baseB+14]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C06 = vaddq_s32(C06, vmovl_s16(vget_low_s16 (VM)));
    C16 = vaddq_s32(C16, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C26 = vaddq_s32(C26, vmovl_s16(vget_low_s16 (VM)));
    C36 = vaddq_s32(C36, vmovl_s16(vget_high_s16(VM)));

    B  = vdup_n_s8(Br[baseB+7]);
    Bn = vdup_n_s8(Br[baseB+15]);
    VM = vmull_s8(A0, B);
    VM = vmlal_s8(VM, A0n, Bn);
    C07 = vaddq_s32(C07, vmovl_s16(vget_low_s16 (VM)));
    C17 = vaddq_s32(C17, vmovl_s16(vget_high_s16(VM)));
    VM = vmull_s8(A1, B);
    VM = vmlal_s8(VM, A1n, Bn);
    C27 = vaddq_s32(C27, vmovl_s16(vget_low_s16 (VM)));
    C37 = vaddq_s32(C37, vmovl_s16(vget_high_s16(VM)));
    
    baseA = baseA + 32; 
    baseB = baseB + 16;
  }
  

  // Last iteration if kc is an odd number
  if ((kc%2) != 0) {
    //Load A
    _A  = vld1q_s8(&Ar[baseA]);
    A0 = vget_low_s8 (_A);
    A1 = vget_high_s8(_A);

    //Load B
    B  = vdup_n_s8(Br[baseB+0]); 
    VM  = vmull_s8(A0, B);
    C00 = vaddq_s32(C00, vmovl_s16(vget_low_s16 (VM)));
    C10 = vaddq_s32(C10, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C20 = vaddq_s32(C20, vmovl_s16(vget_low_s16 (VM)));
    C30 = vaddq_s32(C30, vmovl_s16(vget_high_s16(VM)));

    B  = vdup_n_s8(Br[baseB+1]); 
    VM  = vmull_s8(A0, B);
    C01 = vaddq_s32(C01, vmovl_s16(vget_low_s16 (VM)));
    C11 = vaddq_s32(C11, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C21 = vaddq_s32(C21, vmovl_s16(vget_low_s16 (VM)));
    C31 = vaddq_s32(C31, vmovl_s16(vget_high_s16(VM)));

    B  = vdup_n_s8(Br[baseB+2]); 
    VM  = vmull_s8(A0, B);
    C02 = vaddq_s32(C02, vmovl_s16(vget_low_s16 (VM)));
    C12 = vaddq_s32(C12, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C22 = vaddq_s32(C22, vmovl_s16(vget_low_s16 (VM)));
    C32 = vaddq_s32(C32, vmovl_s16(vget_high_s16(VM)));

    B  = vdup_n_s8(Br[baseB+3]); 
    VM  = vmull_s8(A0, B);
    C03 = vaddq_s32(C03, vmovl_s16(vget_low_s16 (VM)));
    C13 = vaddq_s32(C13, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C23 = vaddq_s32(C23, vmovl_s16(vget_low_s16 (VM)));
    C33 = vaddq_s32(C33, vmovl_s16(vget_high_s16(VM)));
    
    B  = vdup_n_s8(Br[baseB+4]); 
    VM  = vmull_s8(A0, B);
    C04 = vaddq_s32(C04, vmovl_s16(vget_low_s16 (VM)));
    C14 = vaddq_s32(C14, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C24 = vaddq_s32(C24, vmovl_s16(vget_low_s16 (VM)));
    C34 = vaddq_s32(C34, vmovl_s16(vget_high_s16(VM)));
    
    B  = vdup_n_s8(Br[baseB+5]); 
    VM  = vmull_s8(A0, B);
    C05 = vaddq_s32(C05, vmovl_s16(vget_low_s16 (VM)));
    C15 = vaddq_s32(C15, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C25 = vaddq_s32(C25, vmovl_s16(vget_low_s16 (VM)));
    C35 = vaddq_s32(C35, vmovl_s16(vget_high_s16(VM)));
    
    B  = vdup_n_s8(Br[baseB+6]); 
    VM  = vmull_s8(A0, B);
    C06 = vaddq_s32(C06, vmovl_s16(vget_low_s16 (VM)));
    C16 = vaddq_s32(C16, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C26 = vaddq_s32(C26, vmovl_s16(vget_low_s16 (VM)));
    C36 = vaddq_s32(C36, vmovl_s16(vget_high_s16(VM)));
    
    B  = vdup_n_s8(Br[baseB+7]); 
    VM  = vmull_s8(A0, B);
    C07 = vaddq_s32(C07, vmovl_s16(vget_low_s16 (VM)));
    C17 = vaddq_s32(C17, vmovl_s16(vget_high_s16(VM)));
    VM  = vmull_s8(A1, B);
    C27 = vaddq_s32(C27, vmovl_s16(vget_low_s16 (VM)));
    C37 = vaddq_s32(C37, vmovl_s16(vget_high_s16(VM)));
  }


  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);
  vst1q_s32(&Ccol(0,2), C02);
  vst1q_s32(&Ccol(0,3), C03);
  vst1q_s32(&Ccol(0,4), C04);
  vst1q_s32(&Ccol(0,5), C05);
  vst1q_s32(&Ccol(0,6), C06);
  vst1q_s32(&Ccol(0,7), C07);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);
  vst1q_s32(&Ccol(4,2), C12);
  vst1q_s32(&Ccol(4,3), C13);
  vst1q_s32(&Ccol(4,4), C14);
  vst1q_s32(&Ccol(4,5), C15);
  vst1q_s32(&Ccol(4,6), C16);
  vst1q_s32(&Ccol(4,7), C17);
    
    
  vst1q_s32(&Ccol(8,0), C20);
  vst1q_s32(&Ccol(8,1), C21);
  vst1q_s32(&Ccol(8,2), C22);
  vst1q_s32(&Ccol(8,3), C23);
  vst1q_s32(&Ccol(8,4), C24);
  vst1q_s32(&Ccol(8,5), C25);
  vst1q_s32(&Ccol(8,6), C26);
  vst1q_s32(&Ccol(8,7), C27);
    
  
  vst1q_s32(&Ccol(12,0), C30);
  vst1q_s32(&Ccol(12,1), C31);
  vst1q_s32(&Ccol(12,2), C32);
  vst1q_s32(&Ccol(12,3), C33);
  vst1q_s32(&Ccol(12,4), C34);
  vst1q_s32(&Ccol(12,5), C35);
  vst1q_s32(&Ccol(12,6), C36);
  vst1q_s32(&Ccol(12,7), C37);
    


}

void ukernel_intrinsic_qu8_24x4_unrollx2_int8_int32(int kc, int8_t  *Ar, int8_t *Br, int32_t *Cr, int32_t beta, int ldC) {

  //BLIS GEMM microkernel, computes the product Cr := Cr + Ar * Br
  //Update micro-tile of C: MR=16 x NR=4

  int i, j, k, baseA, baseB, Amr, Bnr;
  int zero = 0, one = 1, *Aptr, *Bptr;

  int32x4_t C00, C01, C02, C03,
	    C10, C11, C12, C13,
	    C20, C21, C22, C23,
	    C30, C31, C32, C33,
	    C40, C41, C42, C43,
	    C50, C51, C52, C53; //+24

  int16x8_t  A0, A1, A2;
  int16x4_t  A0low, A1low, A2low, _B0;
  int16x8_t  B0;

  if ( kc==0 ) return;

  if (beta==zero) {
    C00 = vmovq_n_s32(0); 
    C01 = vmovq_n_s32(0); 
    C02 = vmovq_n_s32(0); 
    C03 = vmovq_n_s32(0);

    C10 = vmovq_n_s32(0); 
    C11 = vmovq_n_s32(0); 
    C12 = vmovq_n_s32(0); 
    C13 = vmovq_n_s32(0);

    C20 = vmovq_n_s32(0); 
    C21 = vmovq_n_s32(0); 
    C22 = vmovq_n_s32(0); 
    C23 = vmovq_n_s32(0);

    C30 = vmovq_n_s32(0); 
    C31 = vmovq_n_s32(0); 
    C32 = vmovq_n_s32(0); 
    C33 = vmovq_n_s32(0);

    C40 = vmovq_n_s32(0); 
    C41 = vmovq_n_s32(0); 
    C42 = vmovq_n_s32(0); 
    C43 = vmovq_n_s32(0);

    C50 = vmovq_n_s32(0); 
    C51 = vmovq_n_s32(0); 
    C52 = vmovq_n_s32(0); 
    C53 = vmovq_n_s32(0);
  } else {
    C00 = vld1q_s32(&Ccol(0,0));  
    C01 = vld1q_s32(&Ccol(0,1));  
    C02 = vld1q_s32(&Ccol(0,2));  
    C03 = vld1q_s32(&Ccol(0,3));

    C10 = vld1q_s32(&Ccol(4,0));  
    C11 = vld1q_s32(&Ccol(4,1));  
    C12 = vld1q_s32(&Ccol(4,2));  
    C13 = vld1q_s32(&Ccol(4,3));

    C20 = vld1q_s32(&Ccol(8,0));  
    C21 = vld1q_s32(&Ccol(8,1));  
    C22 = vld1q_s32(&Ccol(8,2));  
    C23 = vld1q_s32(&Ccol(8,3));

    C30 = vld1q_s32(&Ccol(12,0)); 
    C31 = vld1q_s32(&Ccol(12,1)); 
    C32 = vld1q_s32(&Ccol(12,2)); 
    C33 = vld1q_s32(&Ccol(12,3));

    C40 = vld1q_s32(&Ccol(16,0)); 
    C41 = vld1q_s32(&Ccol(16,1)); 
    C42 = vld1q_s32(&Ccol(16,2)); 
    C43 = vld1q_s32(&Ccol(16,3));

    C50 = vld1q_s32(&Ccol(20,0)); 
    C51 = vld1q_s32(&Ccol(20,1)); 
    C52 = vld1q_s32(&Ccol(20,2)); 
    C53 = vld1q_s32(&Ccol(20,3));
  }

  baseA = 0;
  baseB = 0;

  for ( k=0; k<kc-1; k+=2 ) {
    A0    = vmovl_s8(vld1_s8(&Ar[baseA + 0 ])); //Load 8 elements from A #int8x8 ¿16?
    A1    = vmovl_s8(vld1_s8(&Ar[baseA + 8 ])); //Load 8 elements from A #int8x8
    A2    = vmovl_s8(vld1_s8(&Ar[baseA + 16])); //Load 8 elements from A #int8x8 ¿16?
    A0low = vget_low_s16(A0);
    A1low = vget_low_s16(A1);
    A2low = vget_low_s16(A2);
    
    B0    = vmovl_s8(vld1_s8(&Br[baseB])); //Load 8 elements from A #int8x8

    //x1
    C00 = vmlal_laneq_s16     (C00, A0low, B0, 0); 
    C10 = vmlal_high_laneq_s16(C10, A0,    B0, 0);
    C20 = vmlal_laneq_s16     (C20, A1low, B0, 0); 
    C30 = vmlal_high_laneq_s16(C30, A1,    B0, 0);
    C40 = vmlal_laneq_s16     (C40, A2low, B0, 0); 
    C50 = vmlal_high_laneq_s16(C50, A2,    B0, 0);

    C01 = vmlal_laneq_s16     (C01, A0low, B0, 1); 
    C11 = vmlal_high_laneq_s16(C11, A0,    B0, 1);
    C21 = vmlal_laneq_s16     (C21, A1low, B0, 1); 
    C31 = vmlal_high_laneq_s16(C31, A1,    B0, 1);
    C41 = vmlal_laneq_s16     (C41, A2low, B0, 1); 
    C51 = vmlal_high_laneq_s16(C51, A2,    B0, 1);

    C02 = vmlal_laneq_s16     (C02, A0low, B0, 2); 
    C12 = vmlal_high_laneq_s16(C12, A0,    B0, 2);
    C22 = vmlal_laneq_s16     (C22, A1low, B0, 2); 
    C32 = vmlal_high_laneq_s16(C32, A1,    B0, 2);
    C42 = vmlal_laneq_s16     (C42, A2low, B0, 2); 
    C52 = vmlal_high_laneq_s16(C52, A2,    B0, 2);

    C03 = vmlal_laneq_s16     (C03, A0low, B0, 3); 
    C13 = vmlal_high_laneq_s16(C13, A0,    B0, 3);
    C23 = vmlal_laneq_s16     (C23, A1low, B0, 3); 
    C33 = vmlal_high_laneq_s16(C33, A1,    B0, 3);
    C43 = vmlal_laneq_s16     (C43, A2low, B0, 3); 
    C53 = vmlal_high_laneq_s16(C53, A2,    B0, 3);
    
    A0    = vmovl_s8(vld1_s8(&Ar[baseA + 24])); //Load 8 elements from A #int8x8 ¿16?
    A1    = vmovl_s8(vld1_s8(&Ar[baseA + 32])); //Load 8 elements from A #int8x8
    A2    = vmovl_s8(vld1_s8(&Ar[baseA + 40])); //Load 8 elements from A #int8x8 ¿16?
    A0low = vget_low_s16(A0);
    A1low = vget_low_s16(A1);
    A2low = vget_low_s16(A2);
    
    //x2
    C00 = vmlal_laneq_s16     (C00, A0low, B0, 4); 
    C10 = vmlal_high_laneq_s16(C10, A0,    B0, 4);
    C20 = vmlal_laneq_s16     (C20, A1low, B0, 4); 
    C30 = vmlal_high_laneq_s16(C30, A1,    B0, 4);
    C40 = vmlal_laneq_s16     (C40, A2low, B0, 4); 
    C50 = vmlal_high_laneq_s16(C50, A2,    B0, 4);

    C01 = vmlal_laneq_s16     (C01, A0low, B0, 5); 
    C11 = vmlal_high_laneq_s16(C11, A0,    B0, 5);
    C21 = vmlal_laneq_s16     (C21, A1low, B0, 5); 
    C31 = vmlal_high_laneq_s16(C31, A1,    B0, 5);
    C41 = vmlal_laneq_s16     (C41, A2low, B0, 5); 
    C51 = vmlal_high_laneq_s16(C51, A2,    B0, 5);

    C02 = vmlal_laneq_s16     (C02, A0low, B0, 6); 
    C12 = vmlal_high_laneq_s16(C12, A0,    B0, 6);
    C22 = vmlal_laneq_s16     (C22, A1low, B0, 6); 
    C32 = vmlal_high_laneq_s16(C32, A1,    B0, 6);
    C42 = vmlal_laneq_s16     (C42, A2low, B0, 6); 
    C52 = vmlal_high_laneq_s16(C52, A2,    B0, 6);

    C03 = vmlal_laneq_s16     (C03, A0low, B0, 7); 
    C13 = vmlal_high_laneq_s16(C13, A0,    B0, 7);
    C23 = vmlal_laneq_s16     (C23, A1low, B0, 7); 
    C33 = vmlal_high_laneq_s16(C33, A1,    B0, 7);
    C43 = vmlal_laneq_s16     (C43, A2low, B0, 7); 
    C53 = vmlal_high_laneq_s16(C53, A2,    B0, 7);

    baseA = baseA + 48; 
    baseB = baseB + 8;
  }
  

  // Last iteration if kc is an odd number
  if ((kc%2) != 0) {
    A0 = vmovl_s8(vld1_s8(&Ar[baseA + 0 ])); //Load 8 elements from A #int8x8 ¿16?
    A1 = vmovl_s8(vld1_s8(&Ar[baseA + 8 ])); //Load 8 elements from A #int8x8
    A2 = vmovl_s8(vld1_s8(&Ar[baseA + 16])); //Load 8 elements from A #int8x8

    _B0 = vget_low_s16(vmovl_s8(vld1_s8(&Br[baseB]))); //Load 8 elements from A #int8x8

    A0low = vget_low_s16(A0);
    A1low = vget_low_s16(A1);
    A2low = vget_low_s16(A2);

    C00 = vmlal_lane_s16      (C00, A0low, _B0, 0); 
    C10 = vmlal_high_lane_s16 (C10, A0,    _B0, 0);
    C20 = vmlal_lane_s16      (C20, A1low, _B0, 0); 
    C30 = vmlal_high_lane_s16 (C30, A1,    _B0, 0);
    C40 = vmlal_lane_s16     (C40, A2low, _B0, 0); 
    C50 = vmlal_high_lane_s16(C50, A2,    _B0, 0);

    C01 = vmlal_lane_s16      (C01, A0low, _B0, 1); 
    C11 = vmlal_high_lane_s16 (C11, A0,    _B0, 1);
    C21 = vmlal_lane_s16      (C21, A1low, _B0, 1); 
    C31 = vmlal_high_lane_s16 (C31, A1,    _B0, 1);
    C41 = vmlal_lane_s16     (C41, A2low, _B0, 1); 
    C51 = vmlal_high_lane_s16(C51, A2,    _B0, 1);

    C02 = vmlal_lane_s16      (C02, A0low, _B0, 2); 
    C12 = vmlal_high_lane_s16 (C12, A0,    _B0, 2);
    C22 = vmlal_lane_s16      (C22, A1low, _B0, 2); 
    C32 = vmlal_high_lane_s16 (C32, A1,    _B0, 2);
    C42 = vmlal_lane_s16     (C42, A2low, _B0, 2); 
    C52 = vmlal_high_lane_s16(C52, A2,    _B0, 2);

    C03 = vmlal_lane_s16      (C03, A0low, _B0, 3); 
    C13 = vmlal_high_lane_s16 (C13, A0,    _B0, 3);
    C23 = vmlal_lane_s16      (C23, A1low, _B0, 3); 
    C33 = vmlal_high_lane_s16 (C33, A1,    _B0, 3);
    C43 = vmlal_lane_s16     (C43, A2low, _B0, 3); 
    C53 = vmlal_high_lane_s16(C53, A2,    _B0, 3);
  }


  vst1q_s32(&Ccol(0,0), C00);
  vst1q_s32(&Ccol(0,1), C01);
  vst1q_s32(&Ccol(0,2), C02);
  vst1q_s32(&Ccol(0,3), C03);

  vst1q_s32(&Ccol(4,0), C10);
  vst1q_s32(&Ccol(4,1), C11);
  vst1q_s32(&Ccol(4,2), C12);
  vst1q_s32(&Ccol(4,3), C13);
    
  vst1q_s32(&Ccol(8,0), C20);
  vst1q_s32(&Ccol(8,1), C21);
  vst1q_s32(&Ccol(8,2), C22);
  vst1q_s32(&Ccol(8,3), C23);
  
  vst1q_s32(&Ccol(12,0), C30);
  vst1q_s32(&Ccol(12,1), C31);
  vst1q_s32(&Ccol(12,2), C32);
  vst1q_s32(&Ccol(12,3), C33);

  vst1q_s32(&Ccol(16,0), C40);
  vst1q_s32(&Ccol(16,1), C41);
  vst1q_s32(&Ccol(16,2), C42);
  vst1q_s32(&Ccol(16,3), C43);
  
  vst1q_s32(&Ccol(20,0), C50);
  vst1q_s32(&Ccol(20,1), C51);
  vst1q_s32(&Ccol(20,2), C52);
  vst1q_s32(&Ccol(20,3), C53);

}
