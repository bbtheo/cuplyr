// src/ops_select.cpp
#include "gpu_table.hpp"

#include <cudf/table/table.hpp>

#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

// [[Rcpp::export]]
SEXP gpu_select(SEXP xptr, IntegerVector col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Share the selected columns (F57): select allocates nothing.
    std::vector<SharedColumn> columns;
    for (int i = 0; i < col_indices.size(); ++i) {
        int idx = col_indices[i];
        if (idx < 0 || idx >= view.num_columns()) {
            Rcpp::stop("Column index out of bounds: %d", idx);
        }
        columns.push_back(get_shared_column(ptr, idx));
    }

    return make_gpu_table_xptr(std::move(columns));
}
