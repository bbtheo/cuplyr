#ifndef CUPLR_CUDA_UTILS_HPP
#define CUPLR_CUDA_UTILS_HPP

#include <Rcpp.h>
#include <cuda_runtime.h>

namespace cuplr {

inline void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        Rcpp::stop("CUDA error (%s): %s", context, cudaGetErrorString(err));
    }
}

} // namespace cuplr

#endif // CUPLR_CUDA_UTILS_HPP
