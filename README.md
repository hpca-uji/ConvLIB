# ConvLIB
ConvLib is a library of multi-threaded routines for convolution operators on multicore processors with ARM (NEON) and RISC-V architecture. Two major features of ConvLIB, which makes it different from other packages is that the routines in the library self-adapt to the cache hierarchy and hardware-specific micro-kernels are generated automatically.

## Requisites
- OpenMP if a parallel execution is required.

## Supported Hardware
- ARM A57, A78AE, CARMEL
- RISC-V Xuantie C906, C910
- Any other processor with ARM NEON or RISC-V RVV 1.0 architecture.

## How to install
1. Modify the `Makefile.inc` file for configuring the installation.
2. Configure the micro-kernel generation process in `SIMD_generator.config` file.
3. Run the `build.sh` script seleccting the architecture type as follows:
   ``` sh
   $ ./build.sh armv8
   ```

## How to use 
1. Configure the convolution features in the `convolution.config` file.
2. Run the `convolution.sh` script as follows:
   ``` sh
   $ ./convolution.sh cnn/MODEL output/OUT
   ```
Where `MODEL` is the desired CNN model and `OUT` is the name of the output file.

## Adding new CNN model
Adding a new CNN model is as easy as adding a new file to the `cnn` folder following the format of already existing ones. 

## Adding new hardware
1. Add a new file in the `cache-arch` folder following the `cache-TEMPLATE` file.
2. Add a new file in the `SIMD-arch` folder following the `SIMD-TEMPLATE` file.

## How to cite
Pending
