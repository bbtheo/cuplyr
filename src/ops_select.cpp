// src/ops_select.cpp
#include "gpu_table.hpp"

#include <cudf/table/table.hpp>

#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

// [[Rcpp::export]]
SEXP gpu_select(SEXP xptr, IntegerVector col_indices) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    std::vector<cudf::column_view> selected_views;
    for (int i = 0; i < col_indices.size(); ++i) {
        int idx = col_indices[i];
        if (idx < 0 || idx >= view.num_columns()) {
            Rcpp::stop("Column index out of bounds: %d", idx);
        }
        selected_views.push_back(view.column(idx));
    }

    cudf::table_view selected_view(selected_views);

    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < selected_view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(selected_view.column(i)));
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}
