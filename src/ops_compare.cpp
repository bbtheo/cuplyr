// src/ops_compare.cpp
#include "gpu_table.hpp"
#include "ops_common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

// [[Rcpp::export]]
SEXP gpu_compare_scalar(SEXP xptr, int col_idx, std::string op, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col = view.column(col_idx);

    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    auto bool_col = cudf::binary_operation(
        col, *scalar, get_compare_op(op),
        cudf::data_type{cudf::type_id::BOOL8}
    );

    auto int_col = cudf::cast(bool_col->view(), cudf::data_type{cudf::type_id::INT32});

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        result_columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    result_columns.push_back(std::move(int_col));

    auto result = std::make_unique<cudf::table>(std::move(result_columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_compare_cols(SEXP xptr, int col_idx, std::string op, int col_idx2) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns() ||
        col_idx2 < 0 || col_idx2 >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col1 = view.column(col_idx);
    cudf::column_view col2 = view.column(col_idx2);

    auto bool_col = cudf::binary_operation(
        col1, col2, get_compare_op(op),
        cudf::data_type{cudf::type_id::BOOL8}
    );

    auto int_col = cudf::cast(bool_col->view(), cudf::data_type{cudf::type_id::INT32});

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        result_columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    result_columns.push_back(std::move(int_col));

    auto result = std::make_unique<cudf::table>(std::move(result_columns));
    return make_gpu_table_xptr(std::move(result));
}
