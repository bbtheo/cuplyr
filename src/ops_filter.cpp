// src/ops_filter.cpp
//
// Down to the two handlers filter_eval_mask() (R/filter.R) still needs for
// its CPU-eval fallback path: gpu_filter_bool() (uniform TRUE/FALSE) and
// gpu_filter_mask() (a mixed R logical vector applied as a boolean mask).
// Every predicate the expression IR understands now goes through
// gpu_filter_expr() (src/ops_expr.cpp) instead -- gpu_filter_scalar()/
// gpu_filter_col() were deleted in Phase 1 T8 once that cutover left them
// with no remaining R callers.
#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/null_mask.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

// [[Rcpp::export]]
SEXP gpu_filter_bool(SEXP xptr, bool keep_all) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (keep_all) {
        return xptr;
    }

    auto result = cudf::empty_like(view);
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_filter_mask(SEXP xptr, LogicalVector mask) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    size_type n = view.num_rows();
    if (mask.size() != n) {
        Rcpp::stop("Mask length (%d) must match table rows (%d)", mask.size(), n);
    }

    std::vector<int8_t> bool_data(n);
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    size_type null_count = 0;

    for (size_type i = 0; i < n; ++i) {
        if (LogicalVector::is_na(mask[i])) {
            bool_data[i] = 0;
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        } else {
            bool_data[i] = mask[i] ? 1 : 0;
        }
    }

    rmm::device_buffer data(n * sizeof(int8_t),
                           rmm::cuda_stream_view(),
                           rmm::mr::get_current_device_resource_ref());
    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), bool_data.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice),
                   "gpu_filter_mask memcpy");
    }

    rmm::device_buffer null_mask;
    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    auto mask_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::BOOL8},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );

    auto result = cudf::apply_boolean_mask(view, mask_col->view());
    return make_gpu_table_xptr(std::move(result));
}
