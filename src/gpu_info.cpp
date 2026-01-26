// src/gpu_info.cpp
#include <cuda_runtime.h>

#include <string>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
bool gpu_is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// [[Rcpp::export]]
List gpu_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return List::create(
            Named("available") = false,
            Named("device_count") = 0
        );
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    return List::create(
        Named("available") = true,
        Named("device_count") = device_count,
        Named("device_id") = device_id,
        Named("name") = std::string(prop.name),
        Named("compute_capability") = std::to_string(prop.major) + "." + std::to_string(prop.minor),
        Named("total_memory") = static_cast<double>(total_mem),
        Named("free_memory") = static_cast<double>(free_mem),
        Named("multiprocessors") = prop.multiProcessorCount
    );
}
