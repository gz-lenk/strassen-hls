#ifndef XF_BLAS_STRASSENSSQUAREDKERNEL_HPP
#define XF_BLAS_STRASSENSSQUAREDKERNEL_HPP

namespace xf {

namespace blas {

template <typename t_FloatType,             // 矩阵ABC中元素的数据类型
          unsigned int t_MemWidth,          // 每个内存字中的矩阵元素数量
          unsigned int t_aColMemWords = 1,  // 矩阵A缓冲区每行的内存字数量
          unsigned int t_aRowMemWords = 1,  // 矩阵A缓冲区每列的内存字数量
          unsigned int t_bColMemWords = 1   // 矩阵B缓冲区每行的内存字数量
          >
class StrassensSquaredKernel {
   public:
    static const unsigned int t_aMH = t_MemWidth * t_aRowMemWords; // 矩阵 A 缓冲区中一列的矩阵元素数量
    static const unsigned int t_bKD = t_MemWidth * t_aColMemWords; // 矩阵 A 缓冲区中一行的矩阵元素数量 / 矩阵 B 缓冲区中一列的矩阵元素数量

    typedef WideType<t_FloatType, t_MemWidth> MemWideType;
    typedef typename MemWideType::t_TypeInt MemIntType;
    typedef hls::stream<MemIntType> MemStream;

    typedef hls::stream<typename TaggedWideType<t_FloatType, t_MemWidth>::t_TypeInt> EdgeStream;

    typedef t_FloatType MacBitType;
    typedef MemWideType WideMacBitType;
    typedef MemStream WideMacBitStream;

    typedef GemmArgs GemmArgsType;
    static const unsigned int t_StrassensFactor = 4;
    SubMatrixOps<t_FloatType, t_MemWidth, t_aRowMemWords, t_aColMemWords, t_StrassensFactor> t_subMatOps;

   public:
    /**
     * @brief 从A和B矩阵的外部存储器地址读取数据块，并将其写入流中
     * @param p_aAddr 矩阵 A 在外部存储器中的基地址
     * @param p_bAddr 矩阵 B 在外部存储器中的基地址
     * @param p_aColBlocks 矩阵 A 的列块数
     * @param p_aRowBlocks 矩阵 A 的行块数
     * @param p_bColBlocks 矩阵 B 的列块数
     * @param p_aLd 矩阵 A 的 leading dimension（主维度）以内存字为单位的数量
     * @param p_bLd 矩阵 B 的 leading dimension（主维度）以内存字为单位的数量
     * @param l_lhs 输入流
     * @param l_rhs 输入流
     */
    void GemmReadAB(
        MemIntType* p_aAddr,        // 矩阵 A 在外部存储器中的基地址
        MemIntType* p_bAddr,
        unsigned int p_aColBlocks,  // 矩阵 A 的列块数 
        unsigned int p_aRowBlocks,
        unsigned int p_bColBlocks, 
        unsigned int p_aLd,         // 矩阵 A 的 leading dimension（主维度）以内存字为单位的数量
        unsigned int p_bLd,
        MemStream& l_lhs,           // 输入流
        MemStream& l_rhs
    ){
        MemWideType buffer_a[t_StrassensFactor*t_StrassensFactor*t_aMH*t_aColMemWords];
        // #pragma HLS BIND_STORAGE variable = buffer_a type = RAM_2P impl = BRAM
        MemWideType buffer_b[t_StrassensFactor*t_StrassensFactor*t_aMH*t_aColMemWords];
        // #pragma HLS BIND_STORAGE variable = buffer_b type = RAM_2P impl = BRAM

        


    }



    void GemmBlocks(
        MemIntType* p_aAddr,        // 矩阵 A 在外部存储器中的基地址
        MemIntType* p_bAddr,
        MemIntType* p_cAddr,
        unsigned int p_aColBlocks,  // 矩阵 A 的列块数 
        unsigned int p_aRowBlocks,
        unsigned int p_bColBlocks, 
        unsigned int p_aLd,         // 矩阵 A 的 leading dimension（主维度）以内存字为单位的数量
        unsigned int p_bLd,
        unsigned int p_cLd
    ){
        const unsigned int l_cBlocks = p_aRowBlocks * p_bColBlocks;     // 结果矩阵 C 的总块数
        const unsigned int l_abBlocks = l_cBlocks * p_aColBlocks;       // A-B 矩阵对的总块数，每个 C 块对应一个 A-B 矩阵对
        const unsigned int num_blocks_to_multiply = 49 * l_abBlocks;    // 需要相乘的子块总数（在 Strassen's squared 算法中）

        #pragma HLS DATAFLOW

        MemStream l_Cs;             // 结果流
        #pragma HLS STREAM variable = l_Cs depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
        #pragma HLS bind_storage variable = l_Cs type = fifo impl = uram

        MemStream l_res;            // 中间结果流
        #pragma HLS STREAM variable = l_res depth = t_MemWidth * t_aRowMemWords * t_bColMemWords
        #pragma HLS bind_storage variable = l_res type = fifo impl = uram

        MemStream l_lhs, l_rhs;     // 输入流
        #pragma HLS STREAM variable = l_lhs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
        #pragma HLS bind_storage variable = l_lhs type = fifo impl = uram
        #pragma HLS STREAM variable = l_rhs depth = t_aColMemWords * t_MemWidth * t_aRowMemWords + 2
        #pragma HLS bind_storage variable = l_rhs type = fifo impl = uram

        GemmReadAB(p_aAddr, p_bAddr, p_aColBlocks, p_aRowBlocks, p_bColBlocks, p_aLd, p_bLd, l_lhs, l_rhs);

        GemmMicroKernel(l_lhs, l_rhs, l_res, num_blocks_to_multiply);

        StrassensOutBuffer(l_res, l_Cs, l_cBlocks, p_aColBlocks);

        StrassensWriteC(p_cAddr, l_Cs, p_aRowBlocks, p_bColBlocks, p_cLd);


    }

};

} // namespace blas

} // namespace xf

#endif

