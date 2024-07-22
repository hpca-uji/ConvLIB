#----------------------------------------------------------
# Processor Architecture. Processors Avaibles:
#   ARM:
#    [*] A57
#    [*] A78AE
#    [*] Carmel
#   RISC-V:
#    [*] C906
#    [*] C910
#----------------------------------------------------------
# Architectures with ARMv8 support:
#    > A57    (Jetson Nano)
#    > A78AE  (Jetson Orin)
#    > Carmel (Jetson Xavier)
# Architectures with RISC-V v0.7 support:
#    > C906
#    > C910
#----------------------------------------------------------
PROCESSOR = A78AE
#----------------------------------------------------------

#----------------------------------------------------------
#Compilation Options. Possible values [T | F]
#  [*] T: True and F: False
#----------------------------------------------------------
OPENBLAS_ENABLE = F
BLIS_ENABLE     = F
#----------------------------------------------------------


#----------------------------------------------------------
#Convolution Data Type. 
#----------------------------------------------------------
# Options availables for Direct Convolution:
#  [*] NQ_FP32        #Only float data arithmetic
#  [*] NQ_FP16        #Only float16 data arithmetic
#  [*] Q_INT8_INT32   #Quantization int8 to int32
#----------------------------------------------------------
# Options availables for GEMM:
#  [*] NQ_FP32        #Input:FP32  + Pack:FP32  + Micro:FP32
#  [*] NQ_INT32       #Input:INT32 + Pack:INT32 + Micro:INT32
#  [*] FQ_FP32        #Input:INT8  + Pack:FP32  + Micro:FP32
#  [*] FQ_INT32       #Input:INT8  + Pack:FP32  + Micro:FP32
#  [*] Q_INT8_INT32   #Input:INT8  + Pack:INT8  + Micro:INT8+INT32
#  [] NQ_FP16        #Input:FP16  + Pack:FP16  + Micro:FP16
#  [] FQ_FP16        #Input:INT8  + Pack:FP32  + Micro:FP32
#----------------------------------------------------------
DTYPE = Q_INT8_INT32
#----------------------------------------------------------


#----------------------------------------------------------
#Path Libraries. Only necessary if it is enabled
#----------------------------------------------------------
OPENBLAS_HOME = ./src/OpenBLAS/install/
BLIS_HOME     = ./src/blis/install/
#----------------------------------------------------------

