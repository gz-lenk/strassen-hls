#ifndef XF_BLAS_GEMM_KERNEL_HPP
#define XF_BLAS_GEMM_KERNEL_HPP

/**
 * @brief GEMM kernel
 * @tparam t_DataType Data type for matrix A, B, C
 * @tparam t_MemWidth number of elements in one memory word
 * @tparam t_aColMemWords number of memory words in one row of the matrix A buffer
 * @tparam t_aRowMemWords number of memory words in one column of the matrix A buffer
 * @tparam t_bColMemWords number of memory words in one row of the matrix
 */
template <typename t_DataType,    // matrix A, B entry data type
          unsigned int t_MemWidth, // number of matrix elements in one memory word
          unsigned int t_aColMemWords = 1, // number of memory words in one row of the matrix A buffer
          unsigned int t_aRowMemWords = 1, // number of memory words in one column of the matrix A buffer
          unsigned int t_bColMemWords = 1  // number of memory words in one row of the matrix B buffer
          >
class GemmKernel {
   public:
    static const unsigned int t_aMH = t_MemWidth * t_aRowMemWords;
    static const unsigned int t_aMW = t_MemWidth * t_aColMemWords;


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
        for(int l_aRowBlock = 0; l_aRowBlock < l_aRowBlocks; ++l_aRowBlock) {
            for(int l_bColBlock = 0; l_bColBlock < l_bColBlocks; ++l_bColBlock) {
                for(int l_aColBlock = 0; l_aColBlock < l_aColBlocks; ++l_aColBlock) {
                    // l_bufferB
                    for (int i = 0; i < t_bKD; ++i){
                        #pragma HLS PIPELINE II=t_bColMemWords
                        for (int j = 0; j < t_bColMemWords; ++j){
                            unsigned int l_bAddrIdx = (l_bColBlock * t_bColMemWords + j) + (i + l_aColBlock * t_bKD) * l_bWordLd;
                            MemIntType l_bVal = l_bAddr[l_bAddrIdx];
                            p_Bs.write(l_bVal);
                        }
                    }
                    // l_bufferA
                    for (int i = 0; i < t_aMH; i++){  // Number of matrix elements in one col of matrix A buffer
                        #pragma HLS PIPELINE II = t_aColMemWords
                        for (int j = 0; j < t_aColMemWords; j++) {
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

    void GemmCBuffer(){}

    void GemmBlockStream(
        MemStream& p_As,
        MemStream& p_Bs,
        MemStream& p_Cs,
        unsigned int l_aColBlocks,
        unsigned int l_aRowBlocks,
        unsigned int l_bColBlocks,
        unsigned int p_transpBlocks
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

        // B缓冲

        // 矩阵乘内核

        // C缓冲


    }

    void GemmWriteMemStream(){}

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

        GemmReadABX(p_aAddr, p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, l_As,l_Bs);
        GemmBlockStream(l_As, l_Bs, l_Cs, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_transpBlocks, p_postScale);
        GemmWriteMemStream(p_cAddr, l_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);
    }

};
#endif