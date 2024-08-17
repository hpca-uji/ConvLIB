#include "uKernels_intrinsic_int8_int32.h"

#define Crref(i,j) Cr[j*ldC+i]
#define vstoreC_int8_int32(mem, vreg)                    vst1q_s32(mem, vreg)
#define vinit_int8_int32(vreg, value)                    vreg  = vmovq_n_s32(value)
#define vloadC_int8_int32(vreg, mem)                     vreg  = vld1q_s32(mem)
#define vload_int8_int32(vreg, mem)                      vreg  = vld1q_s8(mem)
#define vload_s_int8_int32(vreg, mem)                     vreg  = vld1_s8(mem)
#define vgetlow_int8_int32(vreg1, vreg2)                 vreg1 = vget_low_s8(vreg2)
#define vgethigh_int8_int32(vreg1, vreg2)                vreg1 = vget_high_s8(vreg2)
#define vdup_int8_int32(vreg, mem)                       vreg  = vdup_n_s8(mem)
#define vmull_int8_int32(vreg1, vreg2, vreg3)            vreg1 = vmull_s8(vreg2, vreg3)
#define vmlal_int8_int32(vreg1, vreg2, vreg3)            vreg1 = vmlal_s8(vreg1, vreg2, vreg3)
#define vaddq_low_int8_int32(vreg1, vreg2)               vreg1 = vaddq_s32(vreg1, vmovl_s16(vget_low_s16(vreg2)))
#define vaddq_high_int8_int32(vreg1, vreg2)              vreg1 = vaddq_s32(vreg1, vmovl_s16(vget_high_s16(vreg2)))
void ukernel_intrinsic_8x4_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=4;
  int32_t Ctmp[8*4];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0;
  int32x4_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vload_s_int8_int32(An0, &Ar[bA + 8]);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    bA+=16;
    bB+=8;
  }

  if ((kc%2) != 0) {
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x8_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=8;
  int32_t Ctmp[8*8];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vload_s_int8_int32(An0, &Ar[bA + 8]);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    bA+=16;
    bB+=16;
  }

  if ((kc%2) != 0) {
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x12_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=12;
  int32_t Ctmp[8*12];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vload_s_int8_int32(An0, &Ar[bA + 8]);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    bA+=16;
    bB+=24;
  }

  if ((kc%2) != 0) {
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x16_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=16;
  int32_t Ctmp[8*16];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C012,  C013,  C014,  C015,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111,  C112,  C113,  C114,  C115;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C012, 0);
    vinit_int8_int32(C013, 0);
    vinit_int8_int32(C014, 0);
    vinit_int8_int32(C015, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
    vinit_int8_int32(C112, 0);
    vinit_int8_int32(C113, 0);
    vinit_int8_int32(C114, 0);
    vinit_int8_int32(C115, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C012, &Crref(0, 12));
    vloadC_int8_int32(C013, &Crref(0, 13));
    vloadC_int8_int32(C014, &Crref(0, 14));
    vloadC_int8_int32(C015, &Crref(0, 15));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
    vloadC_int8_int32(C112, &Crref(4, 12));
    vloadC_int8_int32(C113, &Crref(4, 13));
    vloadC_int8_int32(C114, &Crref(4, 14));
    vloadC_int8_int32(C115, &Crref(4, 15));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vload_s_int8_int32(An0, &Ar[bA + 8]);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+24]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+25]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+26]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+27]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    vdup_int8_int32(B, Br[bB+12]);
    vdup_int8_int32(Bn, Br[bB+28]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);

    vdup_int8_int32(B, Br[bB+13]);
    vdup_int8_int32(Bn, Br[bB+29]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);

    vdup_int8_int32(B, Br[bB+14]);
    vdup_int8_int32(Bn, Br[bB+30]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);

    vdup_int8_int32(B, Br[bB+15]);
    vdup_int8_int32(Bn, Br[bB+31]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);

    bA+=16;
    bB+=32;
  }

  if ((kc%2) != 0) {
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
    vdup_int8_int32(B, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);
    vdup_int8_int32(B, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);
    vdup_int8_int32(B, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);
    vdup_int8_int32(B, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(0,12), C012); 
  vstoreC_int8_int32(&Crref(0,13), C013); 
  vstoreC_int8_int32(&Crref(0,14), C014); 
  vstoreC_int8_int32(&Crref(0,15), C015); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  vstoreC_int8_int32(&Crref(4,12), C112); 
  vstoreC_int8_int32(&Crref(4,13), C113); 
  vstoreC_int8_int32(&Crref(4,14), C114); 
  vstoreC_int8_int32(&Crref(4,15), C115); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x20_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=20;
  int32_t Ctmp[8*20];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C012,  C013,  C014,  C015,  C016,  C017,  C018,  C019,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111,  C112,  C113,  C114,  C115,  C116,  C117,  C118,  C119;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C012, 0);
    vinit_int8_int32(C013, 0);
    vinit_int8_int32(C014, 0);
    vinit_int8_int32(C015, 0);
    vinit_int8_int32(C016, 0);
    vinit_int8_int32(C017, 0);
    vinit_int8_int32(C018, 0);
    vinit_int8_int32(C019, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
    vinit_int8_int32(C112, 0);
    vinit_int8_int32(C113, 0);
    vinit_int8_int32(C114, 0);
    vinit_int8_int32(C115, 0);
    vinit_int8_int32(C116, 0);
    vinit_int8_int32(C117, 0);
    vinit_int8_int32(C118, 0);
    vinit_int8_int32(C119, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C012, &Crref(0, 12));
    vloadC_int8_int32(C013, &Crref(0, 13));
    vloadC_int8_int32(C014, &Crref(0, 14));
    vloadC_int8_int32(C015, &Crref(0, 15));
    vloadC_int8_int32(C016, &Crref(0, 16));
    vloadC_int8_int32(C017, &Crref(0, 17));
    vloadC_int8_int32(C018, &Crref(0, 18));
    vloadC_int8_int32(C019, &Crref(0, 19));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
    vloadC_int8_int32(C112, &Crref(4, 12));
    vloadC_int8_int32(C113, &Crref(4, 13));
    vloadC_int8_int32(C114, &Crref(4, 14));
    vloadC_int8_int32(C115, &Crref(4, 15));
    vloadC_int8_int32(C116, &Crref(4, 16));
    vloadC_int8_int32(C117, &Crref(4, 17));
    vloadC_int8_int32(C118, &Crref(4, 18));
    vloadC_int8_int32(C119, &Crref(4, 19));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vload_s_int8_int32(An0, &Ar[bA + 8]);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+24]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+25]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+26]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+27]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+28]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+29]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+30]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+31]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    vdup_int8_int32(B, Br[bB+12]);
    vdup_int8_int32(Bn, Br[bB+32]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);

    vdup_int8_int32(B, Br[bB+13]);
    vdup_int8_int32(Bn, Br[bB+33]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);

    vdup_int8_int32(B, Br[bB+14]);
    vdup_int8_int32(Bn, Br[bB+34]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);

    vdup_int8_int32(B, Br[bB+15]);
    vdup_int8_int32(Bn, Br[bB+35]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);

    vdup_int8_int32(B, Br[bB+16]);
    vdup_int8_int32(Bn, Br[bB+36]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C016, VM);
    vaddq_high_int8_int32(C116, VM);

    vdup_int8_int32(B, Br[bB+17]);
    vdup_int8_int32(Bn, Br[bB+37]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C017, VM);
    vaddq_high_int8_int32(C117, VM);

    vdup_int8_int32(B, Br[bB+18]);
    vdup_int8_int32(Bn, Br[bB+38]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C018, VM);
    vaddq_high_int8_int32(C118, VM);

    vdup_int8_int32(B, Br[bB+19]);
    vdup_int8_int32(Bn, Br[bB+39]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C019, VM);
    vaddq_high_int8_int32(C119, VM);

    bA+=16;
    bB+=40;
  }

  if ((kc%2) != 0) {
    vload_s_int8_int32(A0, &Ar[bA + 0]);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
    vdup_int8_int32(B, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);
    vdup_int8_int32(B, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);
    vdup_int8_int32(B, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);
    vdup_int8_int32(B, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);
    vdup_int8_int32(B, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C016, VM);
    vaddq_high_int8_int32(C116, VM);
    vdup_int8_int32(B, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C017, VM);
    vaddq_high_int8_int32(C117, VM);
    vdup_int8_int32(B, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C018, VM);
    vaddq_high_int8_int32(C118, VM);
    vdup_int8_int32(B, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C019, VM);
    vaddq_high_int8_int32(C119, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(0,12), C012); 
  vstoreC_int8_int32(&Crref(0,13), C013); 
  vstoreC_int8_int32(&Crref(0,14), C014); 
  vstoreC_int8_int32(&Crref(0,15), C015); 
  vstoreC_int8_int32(&Crref(0,16), C016); 
  vstoreC_int8_int32(&Crref(0,17), C017); 
  vstoreC_int8_int32(&Crref(0,18), C018); 
  vstoreC_int8_int32(&Crref(0,19), C019); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  vstoreC_int8_int32(&Crref(4,12), C112); 
  vstoreC_int8_int32(&Crref(4,13), C113); 
  vstoreC_int8_int32(&Crref(4,14), C114); 
  vstoreC_int8_int32(&Crref(4,15), C115); 
  vstoreC_int8_int32(&Crref(4,16), C116); 
  vstoreC_int8_int32(&Crref(4,17), C117); 
  vstoreC_int8_int32(&Crref(4,18), C118); 
  vstoreC_int8_int32(&Crref(4,19), C119); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x4_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=4;
  int32_t Ctmp[16*4];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0,  A1, An1;
  int32x4_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13,  C20,  C21,  C22,  C23,  C30,  C31,  C32,  C33;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C20, 0);
    vinit_int8_int32(C21, 0);
    vinit_int8_int32(C22, 0);
    vinit_int8_int32(C23, 0);
    vinit_int8_int32(C30, 0);
    vinit_int8_int32(C31, 0);
    vinit_int8_int32(C32, 0);
    vinit_int8_int32(C33, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C20, &Crref(8, 0));
    vloadC_int8_int32(C21, &Crref(8, 1));
    vloadC_int8_int32(C22, &Crref(8, 2));
    vloadC_int8_int32(C23, &Crref(8, 3));
    vloadC_int8_int32(C30, &Crref(12, 0));
    vloadC_int8_int32(C31, &Crref(12, 1));
    vloadC_int8_int32(C32, &Crref(12, 2));
    vloadC_int8_int32(C33, &Crref(12, 3));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);

    vload_int8_int32(_A, &Ar[bA + 16]);
    vgetlow_int8_int32(An0, _A);
    vgethigh_int8_int32(An1, _A);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);

    bA+=32;
    bB+=8;
  }

  if ((kc%2) != 0) {
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(8,0), C20); 
  vstoreC_int8_int32(&Crref(8,1), C21); 
  vstoreC_int8_int32(&Crref(8,2), C22); 
  vstoreC_int8_int32(&Crref(8,3), C23); 
  vstoreC_int8_int32(&Crref(12,0), C30); 
  vstoreC_int8_int32(&Crref(12,1), C31); 
  vstoreC_int8_int32(&Crref(12,2), C32); 
  vstoreC_int8_int32(&Crref(12,3), C33); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x8_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=8;
  int32_t Ctmp[16*8];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0,  A1, An1;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C20, 0);
    vinit_int8_int32(C21, 0);
    vinit_int8_int32(C22, 0);
    vinit_int8_int32(C23, 0);
    vinit_int8_int32(C24, 0);
    vinit_int8_int32(C25, 0);
    vinit_int8_int32(C26, 0);
    vinit_int8_int32(C27, 0);
    vinit_int8_int32(C30, 0);
    vinit_int8_int32(C31, 0);
    vinit_int8_int32(C32, 0);
    vinit_int8_int32(C33, 0);
    vinit_int8_int32(C34, 0);
    vinit_int8_int32(C35, 0);
    vinit_int8_int32(C36, 0);
    vinit_int8_int32(C37, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C20, &Crref(8, 0));
    vloadC_int8_int32(C21, &Crref(8, 1));
    vloadC_int8_int32(C22, &Crref(8, 2));
    vloadC_int8_int32(C23, &Crref(8, 3));
    vloadC_int8_int32(C24, &Crref(8, 4));
    vloadC_int8_int32(C25, &Crref(8, 5));
    vloadC_int8_int32(C26, &Crref(8, 6));
    vloadC_int8_int32(C27, &Crref(8, 7));
    vloadC_int8_int32(C30, &Crref(12, 0));
    vloadC_int8_int32(C31, &Crref(12, 1));
    vloadC_int8_int32(C32, &Crref(12, 2));
    vloadC_int8_int32(C33, &Crref(12, 3));
    vloadC_int8_int32(C34, &Crref(12, 4));
    vloadC_int8_int32(C35, &Crref(12, 5));
    vloadC_int8_int32(C36, &Crref(12, 6));
    vloadC_int8_int32(C37, &Crref(12, 7));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);

    vload_int8_int32(_A, &Ar[bA + 16]);
    vgetlow_int8_int32(An0, _A);
    vgethigh_int8_int32(An1, _A);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);

    bA+=32;
    bB+=16;
  }

  if ((kc%2) != 0) {
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(8,0), C20); 
  vstoreC_int8_int32(&Crref(8,1), C21); 
  vstoreC_int8_int32(&Crref(8,2), C22); 
  vstoreC_int8_int32(&Crref(8,3), C23); 
  vstoreC_int8_int32(&Crref(8,4), C24); 
  vstoreC_int8_int32(&Crref(8,5), C25); 
  vstoreC_int8_int32(&Crref(8,6), C26); 
  vstoreC_int8_int32(&Crref(8,7), C27); 
  vstoreC_int8_int32(&Crref(12,0), C30); 
  vstoreC_int8_int32(&Crref(12,1), C31); 
  vstoreC_int8_int32(&Crref(12,2), C32); 
  vstoreC_int8_int32(&Crref(12,3), C33); 
  vstoreC_int8_int32(&Crref(12,4), C34); 
  vstoreC_int8_int32(&Crref(12,5), C35); 
  vstoreC_int8_int32(&Crref(12,6), C36); 
  vstoreC_int8_int32(&Crref(12,7), C37); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x12_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=12;
  int32_t Ctmp[16*12];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0,  A1, An1;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C28,  C29,  C210,  C211,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37,  C38,  C39,  C310,  C311;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
    vinit_int8_int32(C20, 0);
    vinit_int8_int32(C21, 0);
    vinit_int8_int32(C22, 0);
    vinit_int8_int32(C23, 0);
    vinit_int8_int32(C24, 0);
    vinit_int8_int32(C25, 0);
    vinit_int8_int32(C26, 0);
    vinit_int8_int32(C27, 0);
    vinit_int8_int32(C28, 0);
    vinit_int8_int32(C29, 0);
    vinit_int8_int32(C210, 0);
    vinit_int8_int32(C211, 0);
    vinit_int8_int32(C30, 0);
    vinit_int8_int32(C31, 0);
    vinit_int8_int32(C32, 0);
    vinit_int8_int32(C33, 0);
    vinit_int8_int32(C34, 0);
    vinit_int8_int32(C35, 0);
    vinit_int8_int32(C36, 0);
    vinit_int8_int32(C37, 0);
    vinit_int8_int32(C38, 0);
    vinit_int8_int32(C39, 0);
    vinit_int8_int32(C310, 0);
    vinit_int8_int32(C311, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
    vloadC_int8_int32(C20, &Crref(8, 0));
    vloadC_int8_int32(C21, &Crref(8, 1));
    vloadC_int8_int32(C22, &Crref(8, 2));
    vloadC_int8_int32(C23, &Crref(8, 3));
    vloadC_int8_int32(C24, &Crref(8, 4));
    vloadC_int8_int32(C25, &Crref(8, 5));
    vloadC_int8_int32(C26, &Crref(8, 6));
    vloadC_int8_int32(C27, &Crref(8, 7));
    vloadC_int8_int32(C28, &Crref(8, 8));
    vloadC_int8_int32(C29, &Crref(8, 9));
    vloadC_int8_int32(C210, &Crref(8, 10));
    vloadC_int8_int32(C211, &Crref(8, 11));
    vloadC_int8_int32(C30, &Crref(12, 0));
    vloadC_int8_int32(C31, &Crref(12, 1));
    vloadC_int8_int32(C32, &Crref(12, 2));
    vloadC_int8_int32(C33, &Crref(12, 3));
    vloadC_int8_int32(C34, &Crref(12, 4));
    vloadC_int8_int32(C35, &Crref(12, 5));
    vloadC_int8_int32(C36, &Crref(12, 6));
    vloadC_int8_int32(C37, &Crref(12, 7));
    vloadC_int8_int32(C38, &Crref(12, 8));
    vloadC_int8_int32(C39, &Crref(12, 9));
    vloadC_int8_int32(C310, &Crref(12, 10));
    vloadC_int8_int32(C311, &Crref(12, 11));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);

    vload_int8_int32(_A, &Ar[bA + 16]);
    vgetlow_int8_int32(An0, _A);
    vgethigh_int8_int32(An1, _A);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);

    bA+=32;
    bB+=24;
  }

  if ((kc%2) != 0) {
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  vstoreC_int8_int32(&Crref(8,0), C20); 
  vstoreC_int8_int32(&Crref(8,1), C21); 
  vstoreC_int8_int32(&Crref(8,2), C22); 
  vstoreC_int8_int32(&Crref(8,3), C23); 
  vstoreC_int8_int32(&Crref(8,4), C24); 
  vstoreC_int8_int32(&Crref(8,5), C25); 
  vstoreC_int8_int32(&Crref(8,6), C26); 
  vstoreC_int8_int32(&Crref(8,7), C27); 
  vstoreC_int8_int32(&Crref(8,8), C28); 
  vstoreC_int8_int32(&Crref(8,9), C29); 
  vstoreC_int8_int32(&Crref(8,10), C210); 
  vstoreC_int8_int32(&Crref(8,11), C211); 
  vstoreC_int8_int32(&Crref(12,0), C30); 
  vstoreC_int8_int32(&Crref(12,1), C31); 
  vstoreC_int8_int32(&Crref(12,2), C32); 
  vstoreC_int8_int32(&Crref(12,3), C33); 
  vstoreC_int8_int32(&Crref(12,4), C34); 
  vstoreC_int8_int32(&Crref(12,5), C35); 
  vstoreC_int8_int32(&Crref(12,6), C36); 
  vstoreC_int8_int32(&Crref(12,7), C37); 
  vstoreC_int8_int32(&Crref(12,8), C38); 
  vstoreC_int8_int32(&Crref(12,9), C39); 
  vstoreC_int8_int32(&Crref(12,10), C310); 
  vstoreC_int8_int32(&Crref(12,11), C311); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x16_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=16;
  int32_t Ctmp[16*16];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0,  A1, An1;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C012,  C013,  C014,  C015,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111,  C112,  C113,  C114,  C115,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C28,  C29,  C210,  C211,  C212,  C213,  C214,  C215,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37,  C38,  C39,  C310,  C311,  C312,  C313,  C314,  C315;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C012, 0);
    vinit_int8_int32(C013, 0);
    vinit_int8_int32(C014, 0);
    vinit_int8_int32(C015, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
    vinit_int8_int32(C112, 0);
    vinit_int8_int32(C113, 0);
    vinit_int8_int32(C114, 0);
    vinit_int8_int32(C115, 0);
    vinit_int8_int32(C20, 0);
    vinit_int8_int32(C21, 0);
    vinit_int8_int32(C22, 0);
    vinit_int8_int32(C23, 0);
    vinit_int8_int32(C24, 0);
    vinit_int8_int32(C25, 0);
    vinit_int8_int32(C26, 0);
    vinit_int8_int32(C27, 0);
    vinit_int8_int32(C28, 0);
    vinit_int8_int32(C29, 0);
    vinit_int8_int32(C210, 0);
    vinit_int8_int32(C211, 0);
    vinit_int8_int32(C212, 0);
    vinit_int8_int32(C213, 0);
    vinit_int8_int32(C214, 0);
    vinit_int8_int32(C215, 0);
    vinit_int8_int32(C30, 0);
    vinit_int8_int32(C31, 0);
    vinit_int8_int32(C32, 0);
    vinit_int8_int32(C33, 0);
    vinit_int8_int32(C34, 0);
    vinit_int8_int32(C35, 0);
    vinit_int8_int32(C36, 0);
    vinit_int8_int32(C37, 0);
    vinit_int8_int32(C38, 0);
    vinit_int8_int32(C39, 0);
    vinit_int8_int32(C310, 0);
    vinit_int8_int32(C311, 0);
    vinit_int8_int32(C312, 0);
    vinit_int8_int32(C313, 0);
    vinit_int8_int32(C314, 0);
    vinit_int8_int32(C315, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C012, &Crref(0, 12));
    vloadC_int8_int32(C013, &Crref(0, 13));
    vloadC_int8_int32(C014, &Crref(0, 14));
    vloadC_int8_int32(C015, &Crref(0, 15));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
    vloadC_int8_int32(C112, &Crref(4, 12));
    vloadC_int8_int32(C113, &Crref(4, 13));
    vloadC_int8_int32(C114, &Crref(4, 14));
    vloadC_int8_int32(C115, &Crref(4, 15));
    vloadC_int8_int32(C20, &Crref(8, 0));
    vloadC_int8_int32(C21, &Crref(8, 1));
    vloadC_int8_int32(C22, &Crref(8, 2));
    vloadC_int8_int32(C23, &Crref(8, 3));
    vloadC_int8_int32(C24, &Crref(8, 4));
    vloadC_int8_int32(C25, &Crref(8, 5));
    vloadC_int8_int32(C26, &Crref(8, 6));
    vloadC_int8_int32(C27, &Crref(8, 7));
    vloadC_int8_int32(C28, &Crref(8, 8));
    vloadC_int8_int32(C29, &Crref(8, 9));
    vloadC_int8_int32(C210, &Crref(8, 10));
    vloadC_int8_int32(C211, &Crref(8, 11));
    vloadC_int8_int32(C212, &Crref(8, 12));
    vloadC_int8_int32(C213, &Crref(8, 13));
    vloadC_int8_int32(C214, &Crref(8, 14));
    vloadC_int8_int32(C215, &Crref(8, 15));
    vloadC_int8_int32(C30, &Crref(12, 0));
    vloadC_int8_int32(C31, &Crref(12, 1));
    vloadC_int8_int32(C32, &Crref(12, 2));
    vloadC_int8_int32(C33, &Crref(12, 3));
    vloadC_int8_int32(C34, &Crref(12, 4));
    vloadC_int8_int32(C35, &Crref(12, 5));
    vloadC_int8_int32(C36, &Crref(12, 6));
    vloadC_int8_int32(C37, &Crref(12, 7));
    vloadC_int8_int32(C38, &Crref(12, 8));
    vloadC_int8_int32(C39, &Crref(12, 9));
    vloadC_int8_int32(C310, &Crref(12, 10));
    vloadC_int8_int32(C311, &Crref(12, 11));
    vloadC_int8_int32(C312, &Crref(12, 12));
    vloadC_int8_int32(C313, &Crref(12, 13));
    vloadC_int8_int32(C314, &Crref(12, 14));
    vloadC_int8_int32(C315, &Crref(12, 15));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);

    vload_int8_int32(_A, &Ar[bA + 16]);
    vgetlow_int8_int32(An0, _A);
    vgethigh_int8_int32(An1, _A);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+24]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+25]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+26]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+27]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);

    vdup_int8_int32(B, Br[bB+12]);
    vdup_int8_int32(Bn, Br[bB+28]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C212, VM);
    vaddq_high_int8_int32(C312, VM);

    vdup_int8_int32(B, Br[bB+13]);
    vdup_int8_int32(Bn, Br[bB+29]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C213, VM);
    vaddq_high_int8_int32(C313, VM);

    vdup_int8_int32(B, Br[bB+14]);
    vdup_int8_int32(Bn, Br[bB+30]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C214, VM);
    vaddq_high_int8_int32(C314, VM);

    vdup_int8_int32(B, Br[bB+15]);
    vdup_int8_int32(Bn, Br[bB+31]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C215, VM);
    vaddq_high_int8_int32(C315, VM);

    bA+=32;
    bB+=32;
  }

  if ((kc%2) != 0) {
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);
    vdup_int8_int32(B, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C212, VM);
    vaddq_high_int8_int32(C312, VM);
    vdup_int8_int32(B, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C213, VM);
    vaddq_high_int8_int32(C313, VM);
    vdup_int8_int32(B, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C214, VM);
    vaddq_high_int8_int32(C314, VM);
    vdup_int8_int32(B, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C215, VM);
    vaddq_high_int8_int32(C315, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(0,12), C012); 
  vstoreC_int8_int32(&Crref(0,13), C013); 
  vstoreC_int8_int32(&Crref(0,14), C014); 
  vstoreC_int8_int32(&Crref(0,15), C015); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  vstoreC_int8_int32(&Crref(4,12), C112); 
  vstoreC_int8_int32(&Crref(4,13), C113); 
  vstoreC_int8_int32(&Crref(4,14), C114); 
  vstoreC_int8_int32(&Crref(4,15), C115); 
  vstoreC_int8_int32(&Crref(8,0), C20); 
  vstoreC_int8_int32(&Crref(8,1), C21); 
  vstoreC_int8_int32(&Crref(8,2), C22); 
  vstoreC_int8_int32(&Crref(8,3), C23); 
  vstoreC_int8_int32(&Crref(8,4), C24); 
  vstoreC_int8_int32(&Crref(8,5), C25); 
  vstoreC_int8_int32(&Crref(8,6), C26); 
  vstoreC_int8_int32(&Crref(8,7), C27); 
  vstoreC_int8_int32(&Crref(8,8), C28); 
  vstoreC_int8_int32(&Crref(8,9), C29); 
  vstoreC_int8_int32(&Crref(8,10), C210); 
  vstoreC_int8_int32(&Crref(8,11), C211); 
  vstoreC_int8_int32(&Crref(8,12), C212); 
  vstoreC_int8_int32(&Crref(8,13), C213); 
  vstoreC_int8_int32(&Crref(8,14), C214); 
  vstoreC_int8_int32(&Crref(8,15), C215); 
  vstoreC_int8_int32(&Crref(12,0), C30); 
  vstoreC_int8_int32(&Crref(12,1), C31); 
  vstoreC_int8_int32(&Crref(12,2), C32); 
  vstoreC_int8_int32(&Crref(12,3), C33); 
  vstoreC_int8_int32(&Crref(12,4), C34); 
  vstoreC_int8_int32(&Crref(12,5), C35); 
  vstoreC_int8_int32(&Crref(12,6), C36); 
  vstoreC_int8_int32(&Crref(12,7), C37); 
  vstoreC_int8_int32(&Crref(12,8), C38); 
  vstoreC_int8_int32(&Crref(12,9), C39); 
  vstoreC_int8_int32(&Crref(12,10), C310); 
  vstoreC_int8_int32(&Crref(12,11), C311); 
  vstoreC_int8_int32(&Crref(12,12), C312); 
  vstoreC_int8_int32(&Crref(12,13), C313); 
  vstoreC_int8_int32(&Crref(12,14), C314); 
  vstoreC_int8_int32(&Crref(12,15), C315); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x20_int8_int32(int mr, int nr, int kc, int8_t  *Ar, int8_t *Br, int32_t *Cor, int32_t betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=20;
  int32_t Ctmp[16*20];
  int32_t beta;
  int32_t *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int8x16_t _A, _An;
  int16x8_t VM;
   int8x8_t B, Bn;
  int8x8_t  A0, An0,  A1, An1;
  int32x4_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C08,  C09,  C010,  C011,  C012,  C013,  C014,  C015,  C016,  C017,  C018,  C019,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C18,  C19,  C110,  C111,  C112,  C113,  C114,  C115,  C116,  C117,  C118,  C119,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C28,  C29,  C210,  C211,  C212,  C213,  C214,  C215,  C216,  C217,  C218,  C219,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37,  C38,  C39,  C310,  C311,  C312,  C313,  C314,  C315,  C316,  C317,  C318,  C319;

  if (beta == 0) {
    vinit_int8_int32(C00, 0);
    vinit_int8_int32(C01, 0);
    vinit_int8_int32(C02, 0);
    vinit_int8_int32(C03, 0);
    vinit_int8_int32(C04, 0);
    vinit_int8_int32(C05, 0);
    vinit_int8_int32(C06, 0);
    vinit_int8_int32(C07, 0);
    vinit_int8_int32(C08, 0);
    vinit_int8_int32(C09, 0);
    vinit_int8_int32(C010, 0);
    vinit_int8_int32(C011, 0);
    vinit_int8_int32(C012, 0);
    vinit_int8_int32(C013, 0);
    vinit_int8_int32(C014, 0);
    vinit_int8_int32(C015, 0);
    vinit_int8_int32(C016, 0);
    vinit_int8_int32(C017, 0);
    vinit_int8_int32(C018, 0);
    vinit_int8_int32(C019, 0);
    vinit_int8_int32(C10, 0);
    vinit_int8_int32(C11, 0);
    vinit_int8_int32(C12, 0);
    vinit_int8_int32(C13, 0);
    vinit_int8_int32(C14, 0);
    vinit_int8_int32(C15, 0);
    vinit_int8_int32(C16, 0);
    vinit_int8_int32(C17, 0);
    vinit_int8_int32(C18, 0);
    vinit_int8_int32(C19, 0);
    vinit_int8_int32(C110, 0);
    vinit_int8_int32(C111, 0);
    vinit_int8_int32(C112, 0);
    vinit_int8_int32(C113, 0);
    vinit_int8_int32(C114, 0);
    vinit_int8_int32(C115, 0);
    vinit_int8_int32(C116, 0);
    vinit_int8_int32(C117, 0);
    vinit_int8_int32(C118, 0);
    vinit_int8_int32(C119, 0);
    vinit_int8_int32(C20, 0);
    vinit_int8_int32(C21, 0);
    vinit_int8_int32(C22, 0);
    vinit_int8_int32(C23, 0);
    vinit_int8_int32(C24, 0);
    vinit_int8_int32(C25, 0);
    vinit_int8_int32(C26, 0);
    vinit_int8_int32(C27, 0);
    vinit_int8_int32(C28, 0);
    vinit_int8_int32(C29, 0);
    vinit_int8_int32(C210, 0);
    vinit_int8_int32(C211, 0);
    vinit_int8_int32(C212, 0);
    vinit_int8_int32(C213, 0);
    vinit_int8_int32(C214, 0);
    vinit_int8_int32(C215, 0);
    vinit_int8_int32(C216, 0);
    vinit_int8_int32(C217, 0);
    vinit_int8_int32(C218, 0);
    vinit_int8_int32(C219, 0);
    vinit_int8_int32(C30, 0);
    vinit_int8_int32(C31, 0);
    vinit_int8_int32(C32, 0);
    vinit_int8_int32(C33, 0);
    vinit_int8_int32(C34, 0);
    vinit_int8_int32(C35, 0);
    vinit_int8_int32(C36, 0);
    vinit_int8_int32(C37, 0);
    vinit_int8_int32(C38, 0);
    vinit_int8_int32(C39, 0);
    vinit_int8_int32(C310, 0);
    vinit_int8_int32(C311, 0);
    vinit_int8_int32(C312, 0);
    vinit_int8_int32(C313, 0);
    vinit_int8_int32(C314, 0);
    vinit_int8_int32(C315, 0);
    vinit_int8_int32(C316, 0);
    vinit_int8_int32(C317, 0);
    vinit_int8_int32(C318, 0);
    vinit_int8_int32(C319, 0);
  } else {
    vloadC_int8_int32(C00, &Crref(0, 0));
    vloadC_int8_int32(C01, &Crref(0, 1));
    vloadC_int8_int32(C02, &Crref(0, 2));
    vloadC_int8_int32(C03, &Crref(0, 3));
    vloadC_int8_int32(C04, &Crref(0, 4));
    vloadC_int8_int32(C05, &Crref(0, 5));
    vloadC_int8_int32(C06, &Crref(0, 6));
    vloadC_int8_int32(C07, &Crref(0, 7));
    vloadC_int8_int32(C08, &Crref(0, 8));
    vloadC_int8_int32(C09, &Crref(0, 9));
    vloadC_int8_int32(C010, &Crref(0, 10));
    vloadC_int8_int32(C011, &Crref(0, 11));
    vloadC_int8_int32(C012, &Crref(0, 12));
    vloadC_int8_int32(C013, &Crref(0, 13));
    vloadC_int8_int32(C014, &Crref(0, 14));
    vloadC_int8_int32(C015, &Crref(0, 15));
    vloadC_int8_int32(C016, &Crref(0, 16));
    vloadC_int8_int32(C017, &Crref(0, 17));
    vloadC_int8_int32(C018, &Crref(0, 18));
    vloadC_int8_int32(C019, &Crref(0, 19));
    vloadC_int8_int32(C10, &Crref(4, 0));
    vloadC_int8_int32(C11, &Crref(4, 1));
    vloadC_int8_int32(C12, &Crref(4, 2));
    vloadC_int8_int32(C13, &Crref(4, 3));
    vloadC_int8_int32(C14, &Crref(4, 4));
    vloadC_int8_int32(C15, &Crref(4, 5));
    vloadC_int8_int32(C16, &Crref(4, 6));
    vloadC_int8_int32(C17, &Crref(4, 7));
    vloadC_int8_int32(C18, &Crref(4, 8));
    vloadC_int8_int32(C19, &Crref(4, 9));
    vloadC_int8_int32(C110, &Crref(4, 10));
    vloadC_int8_int32(C111, &Crref(4, 11));
    vloadC_int8_int32(C112, &Crref(4, 12));
    vloadC_int8_int32(C113, &Crref(4, 13));
    vloadC_int8_int32(C114, &Crref(4, 14));
    vloadC_int8_int32(C115, &Crref(4, 15));
    vloadC_int8_int32(C116, &Crref(4, 16));
    vloadC_int8_int32(C117, &Crref(4, 17));
    vloadC_int8_int32(C118, &Crref(4, 18));
    vloadC_int8_int32(C119, &Crref(4, 19));
    vloadC_int8_int32(C20, &Crref(8, 0));
    vloadC_int8_int32(C21, &Crref(8, 1));
    vloadC_int8_int32(C22, &Crref(8, 2));
    vloadC_int8_int32(C23, &Crref(8, 3));
    vloadC_int8_int32(C24, &Crref(8, 4));
    vloadC_int8_int32(C25, &Crref(8, 5));
    vloadC_int8_int32(C26, &Crref(8, 6));
    vloadC_int8_int32(C27, &Crref(8, 7));
    vloadC_int8_int32(C28, &Crref(8, 8));
    vloadC_int8_int32(C29, &Crref(8, 9));
    vloadC_int8_int32(C210, &Crref(8, 10));
    vloadC_int8_int32(C211, &Crref(8, 11));
    vloadC_int8_int32(C212, &Crref(8, 12));
    vloadC_int8_int32(C213, &Crref(8, 13));
    vloadC_int8_int32(C214, &Crref(8, 14));
    vloadC_int8_int32(C215, &Crref(8, 15));
    vloadC_int8_int32(C216, &Crref(8, 16));
    vloadC_int8_int32(C217, &Crref(8, 17));
    vloadC_int8_int32(C218, &Crref(8, 18));
    vloadC_int8_int32(C219, &Crref(8, 19));
    vloadC_int8_int32(C30, &Crref(12, 0));
    vloadC_int8_int32(C31, &Crref(12, 1));
    vloadC_int8_int32(C32, &Crref(12, 2));
    vloadC_int8_int32(C33, &Crref(12, 3));
    vloadC_int8_int32(C34, &Crref(12, 4));
    vloadC_int8_int32(C35, &Crref(12, 5));
    vloadC_int8_int32(C36, &Crref(12, 6));
    vloadC_int8_int32(C37, &Crref(12, 7));
    vloadC_int8_int32(C38, &Crref(12, 8));
    vloadC_int8_int32(C39, &Crref(12, 9));
    vloadC_int8_int32(C310, &Crref(12, 10));
    vloadC_int8_int32(C311, &Crref(12, 11));
    vloadC_int8_int32(C312, &Crref(12, 12));
    vloadC_int8_int32(C313, &Crref(12, 13));
    vloadC_int8_int32(C314, &Crref(12, 14));
    vloadC_int8_int32(C315, &Crref(12, 15));
    vloadC_int8_int32(C316, &Crref(12, 16));
    vloadC_int8_int32(C317, &Crref(12, 17));
    vloadC_int8_int32(C318, &Crref(12, 18));
    vloadC_int8_int32(C319, &Crref(12, 19));
  }

  for (pr=0; pr<kc-1; pr+=2) { // Loop L6
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);

    vload_int8_int32(_A, &Ar[bA + 16]);
    vgetlow_int8_int32(An0, _A);
    vgethigh_int8_int32(An1, _A);

    vdup_int8_int32(B, Br[bB+0]);
    vdup_int8_int32(Bn, Br[bB+20]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);

    vdup_int8_int32(B, Br[bB+1]);
    vdup_int8_int32(Bn, Br[bB+21]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);

    vdup_int8_int32(B, Br[bB+2]);
    vdup_int8_int32(Bn, Br[bB+22]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);

    vdup_int8_int32(B, Br[bB+3]);
    vdup_int8_int32(Bn, Br[bB+23]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);

    vdup_int8_int32(B, Br[bB+4]);
    vdup_int8_int32(Bn, Br[bB+24]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);

    vdup_int8_int32(B, Br[bB+5]);
    vdup_int8_int32(Bn, Br[bB+25]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);

    vdup_int8_int32(B, Br[bB+6]);
    vdup_int8_int32(Bn, Br[bB+26]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);

    vdup_int8_int32(B, Br[bB+7]);
    vdup_int8_int32(Bn, Br[bB+27]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);

    vdup_int8_int32(B, Br[bB+8]);
    vdup_int8_int32(Bn, Br[bB+28]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);

    vdup_int8_int32(B, Br[bB+9]);
    vdup_int8_int32(Bn, Br[bB+29]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);

    vdup_int8_int32(B, Br[bB+10]);
    vdup_int8_int32(Bn, Br[bB+30]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);

    vdup_int8_int32(B, Br[bB+11]);
    vdup_int8_int32(Bn, Br[bB+31]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);

    vdup_int8_int32(B, Br[bB+12]);
    vdup_int8_int32(Bn, Br[bB+32]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C212, VM);
    vaddq_high_int8_int32(C312, VM);

    vdup_int8_int32(B, Br[bB+13]);
    vdup_int8_int32(Bn, Br[bB+33]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C213, VM);
    vaddq_high_int8_int32(C313, VM);

    vdup_int8_int32(B, Br[bB+14]);
    vdup_int8_int32(Bn, Br[bB+34]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C214, VM);
    vaddq_high_int8_int32(C314, VM);

    vdup_int8_int32(B, Br[bB+15]);
    vdup_int8_int32(Bn, Br[bB+35]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C215, VM);
    vaddq_high_int8_int32(C315, VM);

    vdup_int8_int32(B, Br[bB+16]);
    vdup_int8_int32(Bn, Br[bB+36]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C016, VM);
    vaddq_high_int8_int32(C116, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C216, VM);
    vaddq_high_int8_int32(C316, VM);

    vdup_int8_int32(B, Br[bB+17]);
    vdup_int8_int32(Bn, Br[bB+37]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C017, VM);
    vaddq_high_int8_int32(C117, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C217, VM);
    vaddq_high_int8_int32(C317, VM);

    vdup_int8_int32(B, Br[bB+18]);
    vdup_int8_int32(Bn, Br[bB+38]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C018, VM);
    vaddq_high_int8_int32(C118, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C218, VM);
    vaddq_high_int8_int32(C318, VM);

    vdup_int8_int32(B, Br[bB+19]);
    vdup_int8_int32(Bn, Br[bB+39]);
    vmull_int8_int32(VM, A0, B);
    vmlal_int8_int32(VM, An0, Bn);
    vaddq_low_int8_int32(C019, VM);
    vaddq_high_int8_int32(C119, VM);

    vmull_int8_int32(VM, A1, B);
    vmlal_int8_int32(VM, An1, Bn);
    vaddq_low_int8_int32(C219, VM);
    vaddq_high_int8_int32(C319, VM);

    bA+=32;
    bB+=40;
  }

  if ((kc%2) != 0) {
    vload_int8_int32(_A, &Ar[bA + 0]);
    vgetlow_int8_int32(A0, _A);
    vgethigh_int8_int32(A1, _A);
    vdup_int8_int32(B, Br[bB+0]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C00, VM);
    vaddq_high_int8_int32(C10, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C20, VM);
    vaddq_high_int8_int32(C30, VM);
    vdup_int8_int32(B, Br[bB+1]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C01, VM);
    vaddq_high_int8_int32(C11, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C21, VM);
    vaddq_high_int8_int32(C31, VM);
    vdup_int8_int32(B, Br[bB+2]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C02, VM);
    vaddq_high_int8_int32(C12, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C22, VM);
    vaddq_high_int8_int32(C32, VM);
    vdup_int8_int32(B, Br[bB+3]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C03, VM);
    vaddq_high_int8_int32(C13, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C23, VM);
    vaddq_high_int8_int32(C33, VM);
    vdup_int8_int32(B, Br[bB+4]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C04, VM);
    vaddq_high_int8_int32(C14, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C24, VM);
    vaddq_high_int8_int32(C34, VM);
    vdup_int8_int32(B, Br[bB+5]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C05, VM);
    vaddq_high_int8_int32(C15, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C25, VM);
    vaddq_high_int8_int32(C35, VM);
    vdup_int8_int32(B, Br[bB+6]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C06, VM);
    vaddq_high_int8_int32(C16, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C26, VM);
    vaddq_high_int8_int32(C36, VM);
    vdup_int8_int32(B, Br[bB+7]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C07, VM);
    vaddq_high_int8_int32(C17, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C27, VM);
    vaddq_high_int8_int32(C37, VM);
    vdup_int8_int32(B, Br[bB+8]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C08, VM);
    vaddq_high_int8_int32(C18, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C28, VM);
    vaddq_high_int8_int32(C38, VM);
    vdup_int8_int32(B, Br[bB+9]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C09, VM);
    vaddq_high_int8_int32(C19, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C29, VM);
    vaddq_high_int8_int32(C39, VM);
    vdup_int8_int32(B, Br[bB+10]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C010, VM);
    vaddq_high_int8_int32(C110, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C210, VM);
    vaddq_high_int8_int32(C310, VM);
    vdup_int8_int32(B, Br[bB+11]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C011, VM);
    vaddq_high_int8_int32(C111, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C211, VM);
    vaddq_high_int8_int32(C311, VM);
    vdup_int8_int32(B, Br[bB+12]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C012, VM);
    vaddq_high_int8_int32(C112, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C212, VM);
    vaddq_high_int8_int32(C312, VM);
    vdup_int8_int32(B, Br[bB+13]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C013, VM);
    vaddq_high_int8_int32(C113, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C213, VM);
    vaddq_high_int8_int32(C313, VM);
    vdup_int8_int32(B, Br[bB+14]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C014, VM);
    vaddq_high_int8_int32(C114, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C214, VM);
    vaddq_high_int8_int32(C314, VM);
    vdup_int8_int32(B, Br[bB+15]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C015, VM);
    vaddq_high_int8_int32(C115, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C215, VM);
    vaddq_high_int8_int32(C315, VM);
    vdup_int8_int32(B, Br[bB+16]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C016, VM);
    vaddq_high_int8_int32(C116, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C216, VM);
    vaddq_high_int8_int32(C316, VM);
    vdup_int8_int32(B, Br[bB+17]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C017, VM);
    vaddq_high_int8_int32(C117, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C217, VM);
    vaddq_high_int8_int32(C317, VM);
    vdup_int8_int32(B, Br[bB+18]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C018, VM);
    vaddq_high_int8_int32(C118, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C218, VM);
    vaddq_high_int8_int32(C318, VM);
    vdup_int8_int32(B, Br[bB+19]);
    vmull_int8_int32(VM, A0, B);
    vaddq_low_int8_int32(C019, VM);
    vaddq_high_int8_int32(C119, VM);
    vmull_int8_int32(VM, A1, B);
    vaddq_low_int8_int32(C219, VM);
    vaddq_high_int8_int32(C319, VM);
  }
  vstoreC_int8_int32(&Crref(0,0), C00); 
  vstoreC_int8_int32(&Crref(0,1), C01); 
  vstoreC_int8_int32(&Crref(0,2), C02); 
  vstoreC_int8_int32(&Crref(0,3), C03); 
  vstoreC_int8_int32(&Crref(0,4), C04); 
  vstoreC_int8_int32(&Crref(0,5), C05); 
  vstoreC_int8_int32(&Crref(0,6), C06); 
  vstoreC_int8_int32(&Crref(0,7), C07); 
  vstoreC_int8_int32(&Crref(0,8), C08); 
  vstoreC_int8_int32(&Crref(0,9), C09); 
  vstoreC_int8_int32(&Crref(0,10), C010); 
  vstoreC_int8_int32(&Crref(0,11), C011); 
  vstoreC_int8_int32(&Crref(0,12), C012); 
  vstoreC_int8_int32(&Crref(0,13), C013); 
  vstoreC_int8_int32(&Crref(0,14), C014); 
  vstoreC_int8_int32(&Crref(0,15), C015); 
  vstoreC_int8_int32(&Crref(0,16), C016); 
  vstoreC_int8_int32(&Crref(0,17), C017); 
  vstoreC_int8_int32(&Crref(0,18), C018); 
  vstoreC_int8_int32(&Crref(0,19), C019); 
  vstoreC_int8_int32(&Crref(4,0), C10); 
  vstoreC_int8_int32(&Crref(4,1), C11); 
  vstoreC_int8_int32(&Crref(4,2), C12); 
  vstoreC_int8_int32(&Crref(4,3), C13); 
  vstoreC_int8_int32(&Crref(4,4), C14); 
  vstoreC_int8_int32(&Crref(4,5), C15); 
  vstoreC_int8_int32(&Crref(4,6), C16); 
  vstoreC_int8_int32(&Crref(4,7), C17); 
  vstoreC_int8_int32(&Crref(4,8), C18); 
  vstoreC_int8_int32(&Crref(4,9), C19); 
  vstoreC_int8_int32(&Crref(4,10), C110); 
  vstoreC_int8_int32(&Crref(4,11), C111); 
  vstoreC_int8_int32(&Crref(4,12), C112); 
  vstoreC_int8_int32(&Crref(4,13), C113); 
  vstoreC_int8_int32(&Crref(4,14), C114); 
  vstoreC_int8_int32(&Crref(4,15), C115); 
  vstoreC_int8_int32(&Crref(4,16), C116); 
  vstoreC_int8_int32(&Crref(4,17), C117); 
  vstoreC_int8_int32(&Crref(4,18), C118); 
  vstoreC_int8_int32(&Crref(4,19), C119); 
  vstoreC_int8_int32(&Crref(8,0), C20); 
  vstoreC_int8_int32(&Crref(8,1), C21); 
  vstoreC_int8_int32(&Crref(8,2), C22); 
  vstoreC_int8_int32(&Crref(8,3), C23); 
  vstoreC_int8_int32(&Crref(8,4), C24); 
  vstoreC_int8_int32(&Crref(8,5), C25); 
  vstoreC_int8_int32(&Crref(8,6), C26); 
  vstoreC_int8_int32(&Crref(8,7), C27); 
  vstoreC_int8_int32(&Crref(8,8), C28); 
  vstoreC_int8_int32(&Crref(8,9), C29); 
  vstoreC_int8_int32(&Crref(8,10), C210); 
  vstoreC_int8_int32(&Crref(8,11), C211); 
  vstoreC_int8_int32(&Crref(8,12), C212); 
  vstoreC_int8_int32(&Crref(8,13), C213); 
  vstoreC_int8_int32(&Crref(8,14), C214); 
  vstoreC_int8_int32(&Crref(8,15), C215); 
  vstoreC_int8_int32(&Crref(8,16), C216); 
  vstoreC_int8_int32(&Crref(8,17), C217); 
  vstoreC_int8_int32(&Crref(8,18), C218); 
  vstoreC_int8_int32(&Crref(8,19), C219); 
  vstoreC_int8_int32(&Crref(12,0), C30); 
  vstoreC_int8_int32(&Crref(12,1), C31); 
  vstoreC_int8_int32(&Crref(12,2), C32); 
  vstoreC_int8_int32(&Crref(12,3), C33); 
  vstoreC_int8_int32(&Crref(12,4), C34); 
  vstoreC_int8_int32(&Crref(12,5), C35); 
  vstoreC_int8_int32(&Crref(12,6), C36); 
  vstoreC_int8_int32(&Crref(12,7), C37); 
  vstoreC_int8_int32(&Crref(12,8), C38); 
  vstoreC_int8_int32(&Crref(12,9), C39); 
  vstoreC_int8_int32(&Crref(12,10), C310); 
  vstoreC_int8_int32(&Crref(12,11), C311); 
  vstoreC_int8_int32(&Crref(12,12), C312); 
  vstoreC_int8_int32(&Crref(12,13), C313); 
  vstoreC_int8_int32(&Crref(12,14), C314); 
  vstoreC_int8_int32(&Crref(12,15), C315); 
  vstoreC_int8_int32(&Crref(12,16), C316); 
  vstoreC_int8_int32(&Crref(12,17), C317); 
  vstoreC_int8_int32(&Crref(12,18), C318); 
  vstoreC_int8_int32(&Crref(12,19), C319); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}

uk_intrinsic_int8_int32 *new_uk_intrinsic_selector_int8_int32() { 
  uk_intrinsic_int8_int32 *uk_vec = (uk_intrinsic_int8_int32 *)malloc(sizeof(uk_intrinsic_int8_int32) * 21 * 21);
  uk_vec[4*21 + 8] = ukernel_intrinsic_8x4_int8_int32;
  uk_vec[8*21 + 8] = ukernel_intrinsic_8x8_int8_int32;
  uk_vec[12*21 + 8] = ukernel_intrinsic_8x12_int8_int32;
  uk_vec[16*21 + 8] = ukernel_intrinsic_8x16_int8_int32;
  uk_vec[20*21 + 8] = ukernel_intrinsic_8x20_int8_int32;
  uk_vec[4*21 + 16] = ukernel_intrinsic_16x4_int8_int32;
  uk_vec[8*21 + 16] = ukernel_intrinsic_16x8_int8_int32;
  uk_vec[12*21 + 16] = ukernel_intrinsic_16x12_int8_int32;
  uk_vec[16*21 + 16] = ukernel_intrinsic_16x16_int8_int32;
  uk_vec[20*21 + 16] = ukernel_intrinsic_16x20_int8_int32;
  return uk_vec;
}

void uk_intrinsic_selector_int8_int32(int mr, int nr, uk_intrinsic_int8_int32 *uk_vec, uk_intrinsic_int8_int32 *ukr) {
  (*ukr) = uk_vec[nr*21 + mr];
}

uk_config_int8_int32_t *new_uk_intrinsic_config_int8_int32() {
  uk_config_int8_int32_t *uk_config = (uk_config_int8_int32_t *)malloc(sizeof(uk_config_int8_int32_t));
  uk_config->uk_num = 0;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 12;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 16;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 20;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 12;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 16;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 20;
  uk_config->uk_num++;
  return uk_config;
}

