#include "uut_top.hpp"

/**
 * @brief UUT 顶层函数
 * m_K矩阵 A 的列数，以元素为单位
 */
void uut_top(
    MemIntType* l_aAddr,        // 矩阵 A 在外部存储器中的基地址
    MemIntType* l_bAddr,
    MemIntType* l_cAddr,
    unsigned int l_aColBlocks,  // 矩阵 A 的列块数 
    unsigned int l_aRowBlocks,
    unsigned int l_bColBlocks, 
    unsigned int l_aLd,         // 矩阵 A 的 leading dimension（主维度）以内存字为单位的数量
    unsigned int l_bLd,
    unsigned int l_cLd
) {
    #pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_aAddr
    #pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_bAddr
    #pragma HLS INTERFACE m_axi bundle = gmem0 depth = BLAS_m * (BLAS_k / BLAS_memWidth) port = l_cAddr

    GemmTypeBaseline l_gemmKernel;
    const unsigned int l_transpBlocks = l_aColBlocks * l_aRowBlocks * l_bColBlocks * BLAS_gemmMBlocks;
    l_gemmKernel.GemmBlocks(l_aAddr, l_bAddr, l_cAddr, l_cAddr, l_aColBlocks, l_aRowBlocks, l_bColBlocks, l_aLd, l_bLd, l_cLd, l_cLd, l_transpBlocks, 1);

}