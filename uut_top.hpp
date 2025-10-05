#ifndef UUT_TOP_HPP
#define UUT_TOP_HPP

#include <hls_stream.h>
#include "gemmKernel.hpp"

typedef GemmKernel<BLAS_dataType,
			      BLAS_dataType,
			      BLAS_memWidth,
			      BLAS_memWidth,
			      BLAS_gemmKBlocks,
			      BLAS_gemmMBlocks,
			      BLAS_gemmNBlocks> GemmTypeBaseline;

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