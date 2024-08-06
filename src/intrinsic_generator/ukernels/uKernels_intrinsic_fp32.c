#include "uKernels_intrinsic_fp32.h"
#define Crref(i,j)  Cr[j*ldC+i]
#define min(a,b) a >= b ? a : b
#define vinit(vreg, value, vlength)        vreg = __riscv_vfmv_v_f_f32m1(value, vlength)
#define vloadC(vreg, mem,  vlength)        vreg = __riscv_vle32_v_f32m1 (mem, vlength)
#define vstoreC(mem, vreg, vlength)               __riscv_vse32_v_f32m1 (mem, vreg, vlength)
#define vload(vreg, vtmp, mem, vlength)    vreg = __riscv_vle32_v_f32m1 (mem, vlength); 
#define vdup(vreg, mem, vlength)           vreg = __riscv_vfmv_v_f_f32m1(mem, vlength)
#define vupdate(vreg1, vreg2, value, vlength) vreg1=__riscv_vfmacc_vf_f32m1(vreg1, value, vreg2, vlength)

void ukernel_intrinsic_8x2_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=2;
  float Ctmp[8*2];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  B0,  B1;
  vfloat32m1_t  C00,  C01;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    bA+=8;
    bB+=2;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x4_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=4;
  float Ctmp[8*4];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  B0,  B1,  B2,  B3;
  vfloat32m1_t  C00,  C01,  C02,  C03;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    bA+=8;
    bB+=4;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x6_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=6;
  float Ctmp[8*6];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  B0,  B1,  B2,  B3,  B4,  B5;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    bA+=8;
    bB+=6;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_8x8_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=8;
  const int NR=8;
  float Ctmp[8*8];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C06, 0, vlength);
    vinit(C07, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C06, &Crref(0, 6), vlength);
    vloadC(C07, &Crref(0, 7), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C06, A0, Br[bB + 6], vlength); 
    vupdate(C07, A0, Br[bB + 7], vlength); 
    bA+=8;
    bB+=8;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(0,6), C06, vlength); 
  vstoreC(&Crref(0,7), C07, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x2_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=2;
  float Ctmp[16*2];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  B0,  B1;
  vfloat32m1_t  C00,  C01,  C10,  C11;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    bA+=16;
    bB+=2;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x4_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=4;
  float Ctmp[16*4];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  B0,  B1,  B2,  B3;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    bA+=16;
    bB+=4;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x6_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=6;
  float Ctmp[16*6];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  B0,  B1,  B2,  B3,  B4,  B5;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C10,  C11,  C12,  C13,  C14,  C15;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    bA+=16;
    bB+=6;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_16x8_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=16;
  const int NR=8;
  float Ctmp[16*8];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C06, 0, vlength);
    vinit(C07, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C16, 0, vlength);
    vinit(C17, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C06, &Crref(0, 6), vlength);
    vloadC(C07, &Crref(0, 7), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C16, &Crref(8, 6), vlength);
    vloadC(C17, &Crref(8, 7), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C06, A0, Br[bB + 6], vlength); 
    vupdate(C07, A0, Br[bB + 7], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C16, A1, Br[bB + 6], vlength); 
    vupdate(C17, A1, Br[bB + 7], vlength); 
    bA+=16;
    bB+=8;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(0,6), C06, vlength); 
  vstoreC(&Crref(0,7), C07, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(8,6), C16, vlength); 
  vstoreC(&Crref(8,7), C17, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_24x2_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=24;
  const int NR=2;
  float Ctmp[24*2];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  B0,  B1;
  vfloat32m1_t  C00,  C01,  C10,  C11,  C20,  C21;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    bA+=24;
    bB+=2;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_24x4_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=24;
  const int NR=4;
  float Ctmp[24*4];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  B0,  B1,  B2,  B3;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13,  C20,  C21,  C22,  C23;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    bA+=24;
    bB+=4;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_24x6_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=24;
  const int NR=6;
  float Ctmp[24*6];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  B0,  B1,  B2,  B3,  B4,  B5;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C10,  C11,  C12,  C13,  C14,  C15,  C20,  C21,  C22,  C23,  C24,  C25;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    bA+=24;
    bB+=6;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_24x8_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=24;
  const int NR=8;
  float Ctmp[24*8];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C06, 0, vlength);
    vinit(C07, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C16, 0, vlength);
    vinit(C17, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
    vinit(C26, 0, vlength);
    vinit(C27, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C06, &Crref(0, 6), vlength);
    vloadC(C07, &Crref(0, 7), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C16, &Crref(8, 6), vlength);
    vloadC(C17, &Crref(8, 7), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
    vloadC(C26, &Crref(16, 6), vlength);
    vloadC(C27, &Crref(16, 7), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C06, A0, Br[bB + 6], vlength); 
    vupdate(C07, A0, Br[bB + 7], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C16, A1, Br[bB + 6], vlength); 
    vupdate(C17, A1, Br[bB + 7], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    vupdate(C26, A2, Br[bB + 6], vlength); 
    vupdate(C27, A2, Br[bB + 7], vlength); 
    bA+=24;
    bB+=8;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(0,6), C06, vlength); 
  vstoreC(&Crref(0,7), C07, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(8,6), C16, vlength); 
  vstoreC(&Crref(8,7), C17, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  vstoreC(&Crref(16,6), C26, vlength); 
  vstoreC(&Crref(16,7), C27, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_32x2_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=32;
  const int NR=2;
  float Ctmp[32*2];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  B0,  B1;
  vfloat32m1_t  C00,  C01,  C10,  C11,  C20,  C21,  C30,  C31;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    bA+=32;
    bB+=2;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_32x4_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=32;
  const int NR=4;
  float Ctmp[32*4];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  B0,  B1,  B2,  B3;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13,  C20,  C21,  C22,  C23,  C30,  C31,  C32,  C33;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    bA+=32;
    bB+=4;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_32x6_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=32;
  const int NR=6;
  float Ctmp[32*6];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  B0,  B1,  B2,  B3,  B4,  B5;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C10,  C11,  C12,  C13,  C14,  C15,  C20,  C21,  C22,  C23,  C24,  C25,  C30,  C31,  C32,  C33,  C34,  C35;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
    vinit(C34, 0, vlength);
    vinit(C35, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
    vloadC(C34, &Crref(24, 4), vlength);
    vloadC(C35, &Crref(24, 5), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    vupdate(C34, A3, Br[bB + 4], vlength); 
    vupdate(C35, A3, Br[bB + 5], vlength); 
    bA+=32;
    bB+=6;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  vstoreC(&Crref(24,4), C34, vlength); 
  vstoreC(&Crref(24,5), C35, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_32x8_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=32;
  const int NR=8;
  float Ctmp[32*8];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C06, 0, vlength);
    vinit(C07, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C16, 0, vlength);
    vinit(C17, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
    vinit(C26, 0, vlength);
    vinit(C27, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
    vinit(C34, 0, vlength);
    vinit(C35, 0, vlength);
    vinit(C36, 0, vlength);
    vinit(C37, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C06, &Crref(0, 6), vlength);
    vloadC(C07, &Crref(0, 7), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C16, &Crref(8, 6), vlength);
    vloadC(C17, &Crref(8, 7), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
    vloadC(C26, &Crref(16, 6), vlength);
    vloadC(C27, &Crref(16, 7), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
    vloadC(C34, &Crref(24, 4), vlength);
    vloadC(C35, &Crref(24, 5), vlength);
    vloadC(C36, &Crref(24, 6), vlength);
    vloadC(C37, &Crref(24, 7), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C06, A0, Br[bB + 6], vlength); 
    vupdate(C07, A0, Br[bB + 7], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C16, A1, Br[bB + 6], vlength); 
    vupdate(C17, A1, Br[bB + 7], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    vupdate(C26, A2, Br[bB + 6], vlength); 
    vupdate(C27, A2, Br[bB + 7], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    vupdate(C34, A3, Br[bB + 4], vlength); 
    vupdate(C35, A3, Br[bB + 5], vlength); 
    vupdate(C36, A3, Br[bB + 6], vlength); 
    vupdate(C37, A3, Br[bB + 7], vlength); 
    bA+=32;
    bB+=8;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(0,6), C06, vlength); 
  vstoreC(&Crref(0,7), C07, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(8,6), C16, vlength); 
  vstoreC(&Crref(8,7), C17, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  vstoreC(&Crref(16,6), C26, vlength); 
  vstoreC(&Crref(16,7), C27, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  vstoreC(&Crref(24,4), C34, vlength); 
  vstoreC(&Crref(24,5), C35, vlength); 
  vstoreC(&Crref(24,6), C36, vlength); 
  vstoreC(&Crref(24,7), C37, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_40x2_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=40;
  const int NR=2;
  float Ctmp[40*2];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  A4,  B0,  B1;
  vfloat32m1_t  C00,  C01,  C10,  C11,  C20,  C21,  C30,  C31,  C40,  C41;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C40, 0, vlength);
    vinit(C41, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C40, &Crref(32, 0), vlength);
    vloadC(C41, &Crref(32, 1), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vload(A4, vtmp, &Ar[bA + 32], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C40, A4, Br[bB + 0], vlength); 
    vupdate(C41, A4, Br[bB + 1], vlength); 
    bA+=40;
    bB+=2;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(32,0), C40, vlength); 
  vstoreC(&Crref(32,1), C41, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_40x4_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=40;
  const int NR=4;
  float Ctmp[40*4];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  A4,  B0,  B1,  B2,  B3;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C10,  C11,  C12,  C13,  C20,  C21,  C22,  C23,  C30,  C31,  C32,  C33,  C40,  C41,  C42,  C43;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
    vinit(C40, 0, vlength);
    vinit(C41, 0, vlength);
    vinit(C42, 0, vlength);
    vinit(C43, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
    vloadC(C40, &Crref(32, 0), vlength);
    vloadC(C41, &Crref(32, 1), vlength);
    vloadC(C42, &Crref(32, 2), vlength);
    vloadC(C43, &Crref(32, 3), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vload(A4, vtmp, &Ar[bA + 32], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    vupdate(C40, A4, Br[bB + 0], vlength); 
    vupdate(C41, A4, Br[bB + 1], vlength); 
    vupdate(C42, A4, Br[bB + 2], vlength); 
    vupdate(C43, A4, Br[bB + 3], vlength); 
    bA+=40;
    bB+=4;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  vstoreC(&Crref(32,0), C40, vlength); 
  vstoreC(&Crref(32,1), C41, vlength); 
  vstoreC(&Crref(32,2), C42, vlength); 
  vstoreC(&Crref(32,3), C43, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_40x6_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=40;
  const int NR=6;
  float Ctmp[40*6];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  A4,  B0,  B1,  B2,  B3,  B4,  B5;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C10,  C11,  C12,  C13,  C14,  C15,  C20,  C21,  C22,  C23,  C24,  C25,  C30,  C31,  C32,  C33,  C34,  C35,  C40,  C41,  C42,  C43,  C44,  C45;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
    vinit(C34, 0, vlength);
    vinit(C35, 0, vlength);
    vinit(C40, 0, vlength);
    vinit(C41, 0, vlength);
    vinit(C42, 0, vlength);
    vinit(C43, 0, vlength);
    vinit(C44, 0, vlength);
    vinit(C45, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
    vloadC(C34, &Crref(24, 4), vlength);
    vloadC(C35, &Crref(24, 5), vlength);
    vloadC(C40, &Crref(32, 0), vlength);
    vloadC(C41, &Crref(32, 1), vlength);
    vloadC(C42, &Crref(32, 2), vlength);
    vloadC(C43, &Crref(32, 3), vlength);
    vloadC(C44, &Crref(32, 4), vlength);
    vloadC(C45, &Crref(32, 5), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vload(A4, vtmp, &Ar[bA + 32], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    vupdate(C34, A3, Br[bB + 4], vlength); 
    vupdate(C35, A3, Br[bB + 5], vlength); 
    vupdate(C40, A4, Br[bB + 0], vlength); 
    vupdate(C41, A4, Br[bB + 1], vlength); 
    vupdate(C42, A4, Br[bB + 2], vlength); 
    vupdate(C43, A4, Br[bB + 3], vlength); 
    vupdate(C44, A4, Br[bB + 4], vlength); 
    vupdate(C45, A4, Br[bB + 5], vlength); 
    bA+=40;
    bB+=6;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  vstoreC(&Crref(24,4), C34, vlength); 
  vstoreC(&Crref(24,5), C35, vlength); 
  vstoreC(&Crref(32,0), C40, vlength); 
  vstoreC(&Crref(32,1), C41, vlength); 
  vstoreC(&Crref(32,2), C42, vlength); 
  vstoreC(&Crref(32,3), C43, vlength); 
  vstoreC(&Crref(32,4), C44, vlength); 
  vstoreC(&Crref(32,5), C45, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}
void ukernel_intrinsic_40x8_fp32(int mr, int nr, int kc, float  *Ar, float *Br, float *Cor, float betaI, int Clda){
  int pr, bA = 0, bB = 0;
  int ldC;
  const int MR=40;
  const int NR=8;
  float Ctmp[40*8];
  float beta;
  float *Cr;
  if (mr == MR && nr == NR) {Cr=Cor; ldC = Clda; beta=betaI;} else {Cr=Ctmp; ldC = MR; beta=0;}
  int vlength = 8;
  vfloat32m1_t vtmp;
  vfloat32m1_t  A0,  A1,  A2,  A3,  A4,  B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7;
  vfloat32m1_t  C00,  C01,  C02,  C03,  C04,  C05,  C06,  C07,  C10,  C11,  C12,  C13,  C14,  C15,  C16,  C17,  C20,  C21,  C22,  C23,  C24,  C25,  C26,  C27,  C30,  C31,  C32,  C33,  C34,  C35,  C36,  C37,  C40,  C41,  C42,  C43,  C44,  C45,  C46,  C47;

  if (beta == 0) {
    vinit(C00, 0, vlength);
    vinit(C01, 0, vlength);
    vinit(C02, 0, vlength);
    vinit(C03, 0, vlength);
    vinit(C04, 0, vlength);
    vinit(C05, 0, vlength);
    vinit(C06, 0, vlength);
    vinit(C07, 0, vlength);
    vinit(C10, 0, vlength);
    vinit(C11, 0, vlength);
    vinit(C12, 0, vlength);
    vinit(C13, 0, vlength);
    vinit(C14, 0, vlength);
    vinit(C15, 0, vlength);
    vinit(C16, 0, vlength);
    vinit(C17, 0, vlength);
    vinit(C20, 0, vlength);
    vinit(C21, 0, vlength);
    vinit(C22, 0, vlength);
    vinit(C23, 0, vlength);
    vinit(C24, 0, vlength);
    vinit(C25, 0, vlength);
    vinit(C26, 0, vlength);
    vinit(C27, 0, vlength);
    vinit(C30, 0, vlength);
    vinit(C31, 0, vlength);
    vinit(C32, 0, vlength);
    vinit(C33, 0, vlength);
    vinit(C34, 0, vlength);
    vinit(C35, 0, vlength);
    vinit(C36, 0, vlength);
    vinit(C37, 0, vlength);
    vinit(C40, 0, vlength);
    vinit(C41, 0, vlength);
    vinit(C42, 0, vlength);
    vinit(C43, 0, vlength);
    vinit(C44, 0, vlength);
    vinit(C45, 0, vlength);
    vinit(C46, 0, vlength);
    vinit(C47, 0, vlength);
  } else {
    vloadC(C00, &Crref(0, 0), vlength);
    vloadC(C01, &Crref(0, 1), vlength);
    vloadC(C02, &Crref(0, 2), vlength);
    vloadC(C03, &Crref(0, 3), vlength);
    vloadC(C04, &Crref(0, 4), vlength);
    vloadC(C05, &Crref(0, 5), vlength);
    vloadC(C06, &Crref(0, 6), vlength);
    vloadC(C07, &Crref(0, 7), vlength);
    vloadC(C10, &Crref(8, 0), vlength);
    vloadC(C11, &Crref(8, 1), vlength);
    vloadC(C12, &Crref(8, 2), vlength);
    vloadC(C13, &Crref(8, 3), vlength);
    vloadC(C14, &Crref(8, 4), vlength);
    vloadC(C15, &Crref(8, 5), vlength);
    vloadC(C16, &Crref(8, 6), vlength);
    vloadC(C17, &Crref(8, 7), vlength);
    vloadC(C20, &Crref(16, 0), vlength);
    vloadC(C21, &Crref(16, 1), vlength);
    vloadC(C22, &Crref(16, 2), vlength);
    vloadC(C23, &Crref(16, 3), vlength);
    vloadC(C24, &Crref(16, 4), vlength);
    vloadC(C25, &Crref(16, 5), vlength);
    vloadC(C26, &Crref(16, 6), vlength);
    vloadC(C27, &Crref(16, 7), vlength);
    vloadC(C30, &Crref(24, 0), vlength);
    vloadC(C31, &Crref(24, 1), vlength);
    vloadC(C32, &Crref(24, 2), vlength);
    vloadC(C33, &Crref(24, 3), vlength);
    vloadC(C34, &Crref(24, 4), vlength);
    vloadC(C35, &Crref(24, 5), vlength);
    vloadC(C36, &Crref(24, 6), vlength);
    vloadC(C37, &Crref(24, 7), vlength);
    vloadC(C40, &Crref(32, 0), vlength);
    vloadC(C41, &Crref(32, 1), vlength);
    vloadC(C42, &Crref(32, 2), vlength);
    vloadC(C43, &Crref(32, 3), vlength);
    vloadC(C44, &Crref(32, 4), vlength);
    vloadC(C45, &Crref(32, 5), vlength);
    vloadC(C46, &Crref(32, 6), vlength);
    vloadC(C47, &Crref(32, 7), vlength);
  }

  for (pr=0; pr<kc; pr++) { // Loop L6
    vload(A0, vtmp, &Ar[bA + 0], vlength);
    vload(A1, vtmp, &Ar[bA + 8], vlength);
    vload(A2, vtmp, &Ar[bA + 16], vlength);
    vload(A3, vtmp, &Ar[bA + 24], vlength);
    vload(A4, vtmp, &Ar[bA + 32], vlength);
    vupdate(C00, A0, Br[bB + 0], vlength); 
    vupdate(C01, A0, Br[bB + 1], vlength); 
    vupdate(C02, A0, Br[bB + 2], vlength); 
    vupdate(C03, A0, Br[bB + 3], vlength); 
    vupdate(C04, A0, Br[bB + 4], vlength); 
    vupdate(C05, A0, Br[bB + 5], vlength); 
    vupdate(C06, A0, Br[bB + 6], vlength); 
    vupdate(C07, A0, Br[bB + 7], vlength); 
    vupdate(C10, A1, Br[bB + 0], vlength); 
    vupdate(C11, A1, Br[bB + 1], vlength); 
    vupdate(C12, A1, Br[bB + 2], vlength); 
    vupdate(C13, A1, Br[bB + 3], vlength); 
    vupdate(C14, A1, Br[bB + 4], vlength); 
    vupdate(C15, A1, Br[bB + 5], vlength); 
    vupdate(C16, A1, Br[bB + 6], vlength); 
    vupdate(C17, A1, Br[bB + 7], vlength); 
    vupdate(C20, A2, Br[bB + 0], vlength); 
    vupdate(C21, A2, Br[bB + 1], vlength); 
    vupdate(C22, A2, Br[bB + 2], vlength); 
    vupdate(C23, A2, Br[bB + 3], vlength); 
    vupdate(C24, A2, Br[bB + 4], vlength); 
    vupdate(C25, A2, Br[bB + 5], vlength); 
    vupdate(C26, A2, Br[bB + 6], vlength); 
    vupdate(C27, A2, Br[bB + 7], vlength); 
    vupdate(C30, A3, Br[bB + 0], vlength); 
    vupdate(C31, A3, Br[bB + 1], vlength); 
    vupdate(C32, A3, Br[bB + 2], vlength); 
    vupdate(C33, A3, Br[bB + 3], vlength); 
    vupdate(C34, A3, Br[bB + 4], vlength); 
    vupdate(C35, A3, Br[bB + 5], vlength); 
    vupdate(C36, A3, Br[bB + 6], vlength); 
    vupdate(C37, A3, Br[bB + 7], vlength); 
    vupdate(C40, A4, Br[bB + 0], vlength); 
    vupdate(C41, A4, Br[bB + 1], vlength); 
    vupdate(C42, A4, Br[bB + 2], vlength); 
    vupdate(C43, A4, Br[bB + 3], vlength); 
    vupdate(C44, A4, Br[bB + 4], vlength); 
    vupdate(C45, A4, Br[bB + 5], vlength); 
    vupdate(C46, A4, Br[bB + 6], vlength); 
    vupdate(C47, A4, Br[bB + 7], vlength); 
    bA+=40;
    bB+=8;
  }

  vstoreC(&Crref(0,0), C00, vlength); 
  vstoreC(&Crref(0,1), C01, vlength); 
  vstoreC(&Crref(0,2), C02, vlength); 
  vstoreC(&Crref(0,3), C03, vlength); 
  vstoreC(&Crref(0,4), C04, vlength); 
  vstoreC(&Crref(0,5), C05, vlength); 
  vstoreC(&Crref(0,6), C06, vlength); 
  vstoreC(&Crref(0,7), C07, vlength); 
  vstoreC(&Crref(8,0), C10, vlength); 
  vstoreC(&Crref(8,1), C11, vlength); 
  vstoreC(&Crref(8,2), C12, vlength); 
  vstoreC(&Crref(8,3), C13, vlength); 
  vstoreC(&Crref(8,4), C14, vlength); 
  vstoreC(&Crref(8,5), C15, vlength); 
  vstoreC(&Crref(8,6), C16, vlength); 
  vstoreC(&Crref(8,7), C17, vlength); 
  vstoreC(&Crref(16,0), C20, vlength); 
  vstoreC(&Crref(16,1), C21, vlength); 
  vstoreC(&Crref(16,2), C22, vlength); 
  vstoreC(&Crref(16,3), C23, vlength); 
  vstoreC(&Crref(16,4), C24, vlength); 
  vstoreC(&Crref(16,5), C25, vlength); 
  vstoreC(&Crref(16,6), C26, vlength); 
  vstoreC(&Crref(16,7), C27, vlength); 
  vstoreC(&Crref(24,0), C30, vlength); 
  vstoreC(&Crref(24,1), C31, vlength); 
  vstoreC(&Crref(24,2), C32, vlength); 
  vstoreC(&Crref(24,3), C33, vlength); 
  vstoreC(&Crref(24,4), C34, vlength); 
  vstoreC(&Crref(24,5), C35, vlength); 
  vstoreC(&Crref(24,6), C36, vlength); 
  vstoreC(&Crref(24,7), C37, vlength); 
  vstoreC(&Crref(32,0), C40, vlength); 
  vstoreC(&Crref(32,1), C41, vlength); 
  vstoreC(&Crref(32,2), C42, vlength); 
  vstoreC(&Crref(32,3), C43, vlength); 
  vstoreC(&Crref(32,4), C44, vlength); 
  vstoreC(&Crref(32,5), C45, vlength); 
  vstoreC(&Crref(32,6), C46, vlength); 
  vstoreC(&Crref(32,7), C47, vlength); 
  if (mr != MR || nr != NR)
    for (int j = 0; j < nr; j++)
      for (int i = 0; i < mr; i++)
        Cor[j*Clda + i] = (betaI) * Cor[j*Clda + i] + Ctmp[j * MR + i];
}

uk_intrinsic_fp32 *new_uk_intrinsic_selector_fp32() { 
  uk_intrinsic_fp32 *uk_vec = (uk_intrinsic_fp32 *)malloc(sizeof(uk_intrinsic_fp32) * 41 * 9);
  uk_vec[2*41 + 8] = ukernel_intrinsic_8x2_fp32;
  uk_vec[4*41 + 8] = ukernel_intrinsic_8x4_fp32;
  uk_vec[6*41 + 8] = ukernel_intrinsic_8x6_fp32;
  uk_vec[8*41 + 8] = ukernel_intrinsic_8x8_fp32;
  uk_vec[2*41 + 16] = ukernel_intrinsic_16x2_fp32;
  uk_vec[4*41 + 16] = ukernel_intrinsic_16x4_fp32;
  uk_vec[6*41 + 16] = ukernel_intrinsic_16x6_fp32;
  uk_vec[8*41 + 16] = ukernel_intrinsic_16x8_fp32;
  uk_vec[2*41 + 24] = ukernel_intrinsic_24x2_fp32;
  uk_vec[4*41 + 24] = ukernel_intrinsic_24x4_fp32;
  uk_vec[6*41 + 24] = ukernel_intrinsic_24x6_fp32;
  uk_vec[8*41 + 24] = ukernel_intrinsic_24x8_fp32;
  uk_vec[2*41 + 32] = ukernel_intrinsic_32x2_fp32;
  uk_vec[4*41 + 32] = ukernel_intrinsic_32x4_fp32;
  uk_vec[6*41 + 32] = ukernel_intrinsic_32x6_fp32;
  uk_vec[8*41 + 32] = ukernel_intrinsic_32x8_fp32;
  uk_vec[2*41 + 40] = ukernel_intrinsic_40x2_fp32;
  uk_vec[4*41 + 40] = ukernel_intrinsic_40x4_fp32;
  uk_vec[6*41 + 40] = ukernel_intrinsic_40x6_fp32;
  uk_vec[8*41 + 40] = ukernel_intrinsic_40x8_fp32;
  return uk_vec;
}

void uk_intrinsic_selector_fp32(int mr, int nr, uk_intrinsic_fp32 *uk_vec, uk_intrinsic_fp32 *ukr) {
  (*ukr) = uk_vec[nr*41 + mr];
}

uk_config_fp32_t *new_uk_intrinsic_config_fp32() {
  uk_config_fp32_t *uk_config = (uk_config_fp32_t *)malloc(sizeof(uk_config_fp32_t));
  uk_config->uk_num = 0;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 2;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 6;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 8;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 2;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 6;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 16;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 24;
  uk_config->nr_pool[uk_config->uk_num] = 2;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 24;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 24;
  uk_config->nr_pool[uk_config->uk_num] = 6;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 24;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 32;
  uk_config->nr_pool[uk_config->uk_num] = 2;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 32;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 32;
  uk_config->nr_pool[uk_config->uk_num] = 6;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 32;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 40;
  uk_config->nr_pool[uk_config->uk_num] = 2;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 40;
  uk_config->nr_pool[uk_config->uk_num] = 4;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 40;
  uk_config->nr_pool[uk_config->uk_num] = 6;
  uk_config->uk_num++;
  uk_config->mr_pool[uk_config->uk_num] = 40;
  uk_config->nr_pool[uk_config->uk_num] = 8;
  uk_config->uk_num++;
  return uk_config;
}

