#ifndef PARAMS_HPP
#define PARAMS_HPP

constexpr unsigned int BLAS_m = 256;
constexpr unsigned int BLAS_k = 256;
constexpr unsigned int BLAS_n = 256;

constexpr unsigned int BLAS_lda = BLAS_k;           
constexpr unsigned int BLAS_ldb = BLAS_n;
constexpr unsigned int BLAS_ldc = BLAS_n;

constexpr unsigned int BLAS_memWidth = 16;
constexpr unsigned int BLAS_gemmMBlocks = 4;
constexpr unsigned int BLAS_gemmKBlocks = 4;
constexpr unsigned int BLAS_gemmNBlocks = 4;

#endif
