#ifndef XF_BLAS_GEMM_KERNEL_HPP
#define XF_BLAS_GEMM_KERNEL_HPP

#include "types.hpp"
// #include "transpose.hpp"
// #include "matrixBuffer.hpp"
#include <hls_stream.h>
#include "params.hpp"

namespace xf{

namespace blas {

/**
 * @brief GEMM kernel
 * @tparam t_DataType Data type for matrix A, B, C
 * @tparam t_MemWidth number of elements in one memory word
 * @tparam t_aColMemWords 
 * @tparam t_aRowMemWords 
 * @tparam t_bColMemWords 
 */
template <typename t_DataType,    // matrix A, B entry data type
          unsigned int t_MemWidth, // number of matrix elements in one memory word
          unsigned int t_aColMemWords = 1, 
          unsigned int t_aRowMemWords = 1, 
          unsigned int t_bColMemWords = 1  
          >
class GemmKernel {
   public:
    static const unsigned int t_aMH = t_MemWidth * t_aRowMemWords;  //m维度
    static const unsigned int t_bKD = t_MemWidth * t_aColMemWords;  //k维度

    typedef WideType<t_DataType, t_MemWidth> MemWideType;
    typedef typename MemWideType::t_TypeInt MemIntType;
    typedef hls::stream<MemIntType> MemStream;

    typedef hls::stream<typename TaggedWideType<t_DataType, t_MemWidth>::t_TypeInt> EdgeStream;

    typedef t_DataType MacBitType;
    typedef MemWideType WideMacBitType;
    typedef MemStream WideMacBitStream;


   public:
    void GemmReadAB(
        MemIntType* l_aAddr,
        MemIntType* l_bAddr,
        unsigned int l_aColBlocks,
        unsigned int l_aRowBlocks,
        unsigned int l_bColBlocks,
        unsigned int l_aWordLd,
        unsigned int l_bWordLd,
        MemStream& p_As,
        MemStream& p_Bs
    ) {
        loop_m_block:
        for(int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
            #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
            loop_n_block:
            for(int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
                #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmNBlocks max=BLAS_gemmNBlocks avg=BLAS_gemmNBlocks
                loop_k_block:
                for(int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
                    #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmKBlocks max=BLAS_gemmKBlocks avg=BLAS_gemmKBlocks
                    // l_bufferB
                    loop_B_k:
                    for (int i = 0; i < t_bKD; ++i){
                        #pragma HLS LOOP_TRIPCOUNT min=BLAS_memWidth*BLAS_gemmKBlocks max=BLAS_memWidth*BLAS_gemmKBlocks avg=BLAS_memWidth*BLAS_gemmKBlocks
                        #pragma HLS PIPELINE II=t_bColMemWords
                        loop_B_n:
                        for (int j = 0; j < t_bColMemWords; ++j){
                            #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmNBlocks max=BLAS_gemmNBlocks avg=BLAS_gemmNBlocks
                            unsigned int l_bSrcOffset = 
                                i * l_bWordLd + l_bWordLd * t_bKD * l_aColBlock + l_bColBlock * t_bColMemWords + j; //地址计算可能过于复杂
                            MemIntType l_bVal = l_bAddr[l_bSrcOffset];
                            p_Bs.write(l_bVal);
                        }
                    }
                    // l_bufferA
                    loop_A_m:
                    for (int i = 0; i < t_aMH; i++){
                        #pragma HLS LOOP_TRIPCOUNT min=BLAS_memWidth*BLAS_gemmMBlocks max=BLAS_memWidth*BLAS_gemmMBlocks avg=BLAS_memWidth*BLAS_gemmMBlocks
                        #pragma HLS PIPELINE II = t_aColMemWords
                        loop_A_k:
                        for (int j = 0; j < t_aColMemWords; j++) {
                            #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
                            unsigned int l_aSrcOffset =
                                l_aWordLd * t_aMH * l_aRowBlock + l_aColBlock * t_aColMemWords + i * l_aWordLd + j;
                            MemIntType l_word = l_aAddr[l_aSrcOffset];
                            p_As.write(l_word);
                        }
                    }
                }
            }
        }
    }

    void GemmCBuffer(
        WideMacBitStream& p_Cs,
        unsigned int p_aColBlocks,
        unsigned int p_cBlocks,
        WideMacBitStream& p_Cout
    ){
        WideMacBitType l_bufferC[t_aMH * t_bColMemWords];
        loop_buffer_C_init:
        for (int i = 0; i < t_aMH * t_bColMemWords; i++){
            #pragma HLS LOOP_TRIPCOUNT min=64*BLAS_gemmNBlocks max=64*BLAS_gemmNBlocks avg=64*BLAS_gemmNBlocks
            #pragma HLS PIPELINE
            for (int j = 0; j < t_MemWidth; j++){
                l_bufferC[i][j] = 0;
            }
        }

        for (int l_block = 0; l_block < p_cBlocks; ++l_block) {
            #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks*BLAS_gemmNBlocks max=BLAS_gemmMBlocks*BLAS_gemmNBlocks avg=BLAS_gemmMBlocks*BLAS_gemmNBlocks
            for (int m = 0; m < p_aColBlocks; ++m) {
                #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
                for (int i = 0; i < t_aRowMemWords; ++i) {
                    #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
                    for (int j = 0; j < t_bColMemWords; ++j) {
                        #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
                        for (int l = 0; l < t_MemWidth; ++l) {
                            #pragma HLS DEPENDENCE variable = l_bufferC array inter RAW false
                            #pragma HLS PIPELINE
                            unsigned int l_arrIdx = (l + i * t_MemWidth) * t_bColMemWords + j;
                            WideMacBitType l_val = p_Cs.read();
                            for (int k = 0; k < t_MemWidth; ++k) {
                                #pragma HLS LOOP_TRIPCOUNT min=BLAS_memWidth max=BLAS_memWidth avg=BLAS_memWidth
                                l_bufferC[l_arrIdx][k] += l_val[k];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < t_bColMemWords * t_aRowMemWords * t_MemWidth; ++i) {
                #pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
                #pragma HLS PIPELINE
                WideMacBitType l_val = l_bufferC[i];
                p_Cout.write(l_val);
                for (int k = 0; k < t_MemWidth; k++) l_bufferC[i][k] = 0;
            }
        }
    }

    void GemmBlockStream(
        MemStream& p_As,
        MemStream& p_Bs,
        MemStream& p_Cs,
        unsigned int p_aColBlocks,
        unsigned int p_aRowBlocks,
        unsigned int p_bColBlocks,
        unsigned int p_transpBlocks,
        int32_t p_postScale
    ){
        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;
        unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;

        #pragma HLS DATAFLOW

        MemStream p_Bs1, p_AoutS, p_CBufferS;
        EdgeStream p_AEdgeS0, p_BEdgeS0;
        WideMacBitStream p_CEdgeS, p_COutS;

        #pragma HLS STREAM variable = p_CEdgeS depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
        #pragma HLS RESOURCE variable = p_CEdgeS core = fifo_uram

        // 转置A
        // Transpose<t_DataType, t_aColMemWords, t_MemWidth> l_transp(p_transpBlocks, t_bColMemWords);
        // l_transp.process(p_As, p_AoutS);

        // B缓冲
        // MatrixBuffer<typename MemWideType::t_TypeInt, t_MemWidth * t_aColMemWords, t_bColMemWords, true, false>()
        //     .process(p_Bs, p_Bs1, l_abBlocks, t_aRowMemWords);

        // 矩阵乘内核

        // C缓冲
        GemmCBuffer(p_CEdgeS, p_aColBlocks, l_cBlocks, p_Cs);


    }


    /**
     * @brief Write the result matrix C from stream to memory
     * @param l_cAddr the base address of matrix C in external memory
     * @param p_Cs the stream of matrix C blocks
     * @param l_aRowBlocks number of row blocks of matrix A
     * @param l_bColBlocks number of column blocks of matrix B
     * @param l_cWordLd leading dimension of matrix C in memory, in unit of memory words
     */
    void GemmWriteMemStream(
        MemIntType* l_cAddr,
        MemStream& p_Cs,
        unsigned int l_aRowBlocks,
        unsigned int l_bColBlocks,
        unsigned int l_cWordLd
    ){
        unsigned int l_rowOffset = 0;
        unsigned int l_colOffset = 0;

        loop_m_block:
        for (int rowBlock = 0; rowBlock < l_aRowBlocks; ++rowBlock) {
            #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmMBlocks max=BLAS_gemmMBlocks avg=BLAS_gemmMBlocks
            loop_n_block:
            for (int colBlock = 0; colBlock < l_bColBlocks; ++colBlock) {
                #pragma HLS LOOP_TRIPCOUTN min=BLAS_gemmNBlocks max=BLAS_gemmNBlocks avg=BLAS_gemmNBlocks
                loop_m:
                for (int i = 0; i < t_aRowMemWords * t_MemWidth; i++) {
                    #pragma HLS LOOP_TRIPCOUNT min=BLAS_m/BLAS_gemmMBlocks max=BLAS_m/BLAS_gemmMBlocks avg=BLAS_m/BLAS_gemmMBlocks
                    #pragma HLS PIPELINE II = t_bColMemWords
                    loop_n:
                    for (int j = 0; j < t_bColMemWords; j++) {
                        #pragma HLS LOOP_TRIPCOUNT min=BLAS_gemmNBlocks max=BLAS_gemmNBlocks avg=BLAS_gemmNBlocks
                        unsigned int l_dstOffset = i * l_cWordLd + l_cWordLd * t_MemWidth * t_aRowMemWords * rowBlock +
                                                   colBlock * t_bColMemWords;
                        MemIntType l_word = p_Cs.read();
                        l_cAddr[l_dstOffset+j] = l_word;
                    }
                }
            }
        }
    }

    void GemmBlocks(
        MemIntType* p_aAddr,
        MemIntType* p_bAddr,
        MemIntType* p_cAddr,
        unsigned int p_aColBlocks,
        unsigned int p_aRowBlocks,
        unsigned int p_bColBlocks,
        unsigned int p_aLd,
        unsigned int p_bLd,
        unsigned int p_cLd,
        unsigned int p_transpBlocks,
        int32_t p_postScale
    ) {
        #pragma HLS DATAFLOW

        MemStream l_As, l_Bs;
        MemStream l_Cs;

        #pragma HLS STREAM variable = l_Cs depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
        #pragma HLS RESOURCE variable = l_Cs core = fifo_uram

        #pragma HLS STREAM variable = l_As depth = t_aColMemWords * t_MemWidth * t_aRowMemWords
        #pragma HLS RESOURCE variable = l_As core = fifo_uram

        unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;

        GemmReadAB(p_aAddr, p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, l_As, l_Bs);
        GemmBlockStream(l_As, l_Bs, l_Cs, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_transpBlocks, p_postScale);
        GemmWriteMemStream(p_cAddr, l_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
    }

};
} // namespace blas

} // namespace xf
#endif