
include Makefile.inc

#------------------------------------------
#| COMPILERS                              |
#------------------------------------------

FLAGS=-O3 -DCHECK

ifeq ($(DTYPE), FP32)
    FLAGS += -DFP32
else ifeq ($(DTYPE), FP16)
    FLAGS += -DFP16
else ifeq ($(DTYPE), INT8_INT32_U8)
    FLAGS += -DINT8_INT32_U8
else ifeq ($(DTYPE), INT8_INT32_S8)
    FLAGS += -DINT8_INT32_S8
endif

ifneq ($(MAKECMDGOALS),clean)
    ifeq ($(arch), riscv)
        CC       = riscv64-unknown-linux-gnu-gcc
        CLINKER  = riscv64-unknown-linux-gnu-gcc
        #OPTFLAGS   +=  -O3 -fopenmp -march=rv64imafdcv0p7_zfh_xtheadc -mabi=lp64d -mtune=c910
        OPTFLAGS = -O0 -g3 -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DFP32 -mtune=c906 -static -DRISCV 
    else ifeq ($(arch), armv8)
        CC       = gcc
        CLINKER  = gcc
        ifeq ($(FP16_ENABLE), T)
          ifeq ($(SDOT_ENABLE), T)
            OPTFLAGS = -march=armv8.2-a+fp16+dotprod -DARMV8 $(FLAGS) -DSDOT
          else
            OPTFLAGS = -march=armv8.2-a+fp16 -DARMV8 $(FLAGS) 
	  endif
	else
            OPTFLAGS = -march=armv8-a -DARMV8 $(FLAGS)
        endif
    else
        $(error Architecture unsuported. Please use arch=[riscv|armv8])
    endif
endif


#------------------------------------------
OPTFLAGS   += -fopenmp -DOMP_ENABLE
OBJDIR      = build
BIN         = convolution_driver.x
#------------------------------------------
LIBS        = -lm
LIBS_LINKER = $(LIBS)
INCLUDE     = 
#------------------------------------------


ifeq ($(BLIS_ENABLE), T)
	INCLUDE     += -I$(BLIS_HOME)/include/blis/ 
	LIBS_LINKER += $(BLIS_HOME)/lib/libblis.a 
	OPTFLAGS    += -DENABLE_BLIS
endif

ifeq ($(OPENBLAS_ENABLE), T)
	INCLUDE     += -I$(OPENBLAS_HOME)/include/
	LIBS_LINKER += $(OPENBLAS_HOME)/lib/libopenblas.a 
	OPTFLAGS    += -DENABLE_OPENBLAS
endif
#------------------------------------------

OBJ_FILES = $(OBJDIR)/model_level.o $(OBJDIR)/selector_ukernel.o $(OBJDIR)/gemm_ukernel.o $(OBJDIR)/ukernels.o

SRC_ASM_FILES = $(wildcard ./src/asm_generator/ukernels/*.S)
OBJ_ASM_FILES = $(patsubst ./src/asm_generator/ukernels/%.S, $(OBJDIR)/%.o, $(SRC_ASM_FILES))
OBJ_FILES += $(OBJ_ASM_FILES)

ifeq ($(arch), armv8)
    ifeq ($(FP16_ENABLE), T)
        OBJ_FILES += $(OBJDIR)/uKernels_intrinsic_fp16.o
    endif
    OBJ_FILES += $(OBJDIR)/uKernels_intrinsic_fp32.o
    #OBJ_FILES += $(OBJDIR)/uKernels_intrinsic_int8_int16.o 
    OBJ_FILES += $(OBJDIR)/uKernels_intrinsic_int8_int32_s8.o
    OBJ_FILES += $(OBJDIR)/uKernels_intrinsic_int8_int32_u8.o
endif

SRC_CONV_FILES = $(wildcard ./src/*.c)
OBJ_CONV_FILES = $(patsubst ./src/%.c, $(OBJDIR)/%.o, $(SRC_CONV_FILES))

SRC_GEMM_FILES = $(wildcard ./src/gemm/*.c)
OBJ_GEMM_FILES = $(patsubst ./src/gemm/%.c, $(OBJDIR)/%.o, $(SRC_GEMM_FILES))

SRC_CONVGEMM_FILES = $(wildcard ./src/convGemm/*.c)
OBJ_CONVGEMM_FILES = $(patsubst ./src/convGemm/%.c, $(OBJDIR)/%.o, $(SRC_CONVGEMM_FILES))

OBJ_FILES  += $(OBJ_CONV_FILES) $(OBJ_GEMM_FILES) $(OBJ_CONVGEMM_FILES) 




all: $(OBJDIR)/$(BIN)

$(OBJDIR)/$(BIN): $(OBJ_FILES)
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS_LINKER)

$(OBJDIR)/%.o: ./src/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/gemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/intrinsic_generator/ukernels/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/convGemm/%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/%.o: ./src/asm_generator/ukernels/%.S
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/model_level.o: ./src/modelLevel/model_level.c 
	mkdir -p $(OBJDIR)
	mkdir -p output
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/gemm_ukernel.o: ./src/asm_generator/ukernels/gemm_ukernel.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/selector_ukernel.o: ./src/asm_generator/ukernels/selector_ukernel.c
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

clean:
	rm $(OBJDIR)/* 

