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
#Convolution Data Type. Options Availables:
#  [*] FP32
#  [*] FP16
#  [*] INT8_INT32_U8
#  [*] INT8_INT32_S8
#----------------------------------------------------------
DTYPE = FP16
#----------------------------------------------------------


#----------------------------------------------------------
#Path Libraries. Only necessary if it is enabled
#----------------------------------------------------------
OPENBLAS_HOME = ./src/OpenBLAS/install/
BLIS_HOME     = ./src/blis/install/
#----------------------------------------------------------

