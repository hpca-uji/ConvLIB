#!/bin/bash

source src/bash/arch.sh

source SIMD_generator.config

CONFIG_PATH=SIMD-arch

#-Set pipelining option for ukernel generator
_PIPELINING=""
if [ "$PIPELINING" = "T" ]; then
  _PIPELINING="--pipelining"
  UNROLL=0
fi

_DUP=""
if [ "$DUP" = "T" ]; then
  _DUP="--dup"
fi

#-Set reorder loads option for ukernel generator
_REORDER=""
if [ "$REORDER" = "T" ]; then
  _REORDER="--reorder"
fi

#-Set reorder loads option for ukernel generator
_UNROLL=""
if [ $UNROLL -ne 0 ]; then
  _UNROLL="--unroll ${UNROLL}"
fi

#-Set reorder loads option for ukernel generator
_OP=""
if [ "$OP" != "NORMAL" ]; then
  _OP="--op_b ${OP,,}"
fi

if [ ! -f $CONFIG_PATH/$ARCH_CONFIG ]; then
    echo "ERROR: The Architecture configure doesn't exist. Please, enter a valid filename."
    exit -1
fi

VLEN=$(tail -1 $CONFIG_PATH/$ARCH_CONFIG | cut -f1)
MAXVECTOR=$(tail -1 $CONFIG_PATH/$ARCH_CONFIG | cut -f2)



#Generating ASM micro-kernels: FP32
rm src/asm_generator/ukernels/*

if [ "$ARCH" = "$RISCV" ]; then
  ./src/asm_generator/ukernels_generator.py --arch riscv --data FP32 --vlen ${VLEN} \
	                                    --maxvec ${MAXVECTOR} ${_OP} ${_REORDER} ${_PIPELINING} ${_UNROLL}
else
    ./src/asm_generator/ukernels_generator.py --arch armv8 --data FP32 --vlen ${VLEN} \
	                                      --maxvec ${MAXVECTOR} ${_PIPELINING} ${_UNROLL}

    #Generating Intrinsic micro-kernels: INT8-INT32 | INT8-INT16 | fp16 | FP32 
    rm src/intrinsic_generator/ukernels/*
    cd src/intrinsic_generator/ 
    ./instrinsics_generator.py --arch armv8 ${_DUP}
    cd ../..
fi



