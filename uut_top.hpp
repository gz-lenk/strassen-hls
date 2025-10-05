#ifndef UUT_TOP_HPP
#define UUT_TOP_HPP

#include <hls_stream.h>
#include "gemmKernel.hpp"
#include "params.hpp"

typedef xf::blas:: GemmKernel<BLAS_dataType,
			      BLAS_memWidth,
			      BLAS_gemmKBlocks,
			      BLAS_gemmMBlocks,
			      BLAS_gemmNBlocks> GemmTypeBaseline;

typedef WideType<BLAS_dataType, BLAS_memWidth> MemWideType;
typedef typename MemWideType::t_TypeInt MemIntType;
typedef hls::stream<MemIntType> MemStream;

void uut_top( MemIntType* l_aAddr, 
        MemIntType* l_bAddr, 
        MemIntType* l_cAddr, 
        unsigned int l_aColBlocks, 
        unsigned int l_aRowBlocks,
        unsigned int l_bColBlocks, 
        unsigned int l_aLd, 
        unsigned int l_bLd,
        unsigned int l_cLd
        );                 

#endif // UUT_TOP_HPP