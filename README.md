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
Parallel GEMM-based Convolutions for Deep Learning on Multicore ARM and RISC-V Architectures.
Héctor Martínez, Sandra Catalán, Adrián Castelló, and Enrique S. Quintana-Ortí. 
Parallel GEMM-Based Convolutions for Deep Learning on Multicore Arm and Risc-V Architectures. 
J. of Systems Architecture, 2023. Under review.
Available at SSRN: https://ssrn.com/abstract=4676008 or http://dx.doi.org/10.2139/ssrn.4676008

## Related work
Micro-kernels for portable and efficient matrix multiplication in deep learning. 
Guillermo Alajeos, Adrián Castelló, Héctor Martínez, Pedro Alonso-Jordá, Francisco D. Igual, Enrique S. Quintana-Ortí.
J. Supercomputing 79, 8124–8147 (2023). 
https://doi.org/10.1007/s11227-022-05003-3

AlgorithmXXX: Automatic Generators for a Family of Matrix Multiplication Routines with Apache TVM. 
Guillermo Alajeos, Adrián Castelló, Francisco D. Igual, Héctor Martínez, Enrique S. Quintana-Ortí.
ACM Transactions on Mathematical Software, 2024. To appear.
arXiv:2310.20347v1, 2023. https://arxiv.org/pdf/2310.20347.pdf

## Funding
This work was supported by the research project PID2020-113656RB-C22 of MCIN/AEI/10.13039/501100011033. 
The project has also received funding from the European High-Performance
Computing Joint Undertaking (JU) under grant agreement No 955558 (eFlows4HPC project). The
JU receives support from the European Union’s Horizon 2020 research and innovation programme,
and Spain, Germany, France, Italy, Poland, Switzerland, and Norway.
