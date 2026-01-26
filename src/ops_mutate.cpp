// src/ops_mutate.cpp
#include "gpu_table.hpp"
#include "ops_common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>

#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

// [[Rcpp::export]]
SEXP gpu_copy_column(SEXP xptr, int col_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    columns.push_back(std::make_unique<cudf::column>(view.column(col_idx)));

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_scalar(SEXP xptr, int col_idx, std::string op, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col = view.column(col_idx);

    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    auto new_col = cudf::binary_operation(
        col, *scalar, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    columns.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_scalar_replace(SEXP xptr, int col_idx, std::string op, double value, int replace_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns() ||
        replace_idx < 0 || replace_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col = view.column(col_idx);

    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    auto new_col = cudf::binary_operation(
        col, *scalar, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(view.num_columns());
    for (int i = 0; i < view.num_columns(); ++i) {
        if (i == replace_idx) {
            columns.push_back(std::move(new_col));
        } else {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_cols(SEXP xptr, int col_idx1, std::string op, int col_idx2) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx1 < 0 || col_idx1 >= view.num_columns() ||
        col_idx2 < 0 || col_idx2 >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col1 = view.column(col_idx1);
    cudf::column_view col2 = view.column(col_idx2);

    auto new_col = cudf::binary_operation(
        col1, col2, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    columns.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_cols_replace(SEXP xptr, int col_idx1, std::string op, int col_idx2, int replace_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx1 < 0 || col_idx1 >= view.num_columns() ||
        col_idx2 < 0 || col_idx2 >= view.num_columns() ||
        replace_idx < 0 || replace_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col1 = view.column(col_idx1);
    cudf::column_view col2 = view.column(col_idx2);

    auto new_col = cudf::binary_operation(
        col1, col2, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(view.num_columns());
    for (int i = 0; i < view.num_columns(); ++i) {
        if (i == replace_idx) {
            columns.push_back(std::move(new_col));
        } else {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_copy_column_replace(SEXP xptr, int source_idx, int replace_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (source_idx < 0 || source_idx >= view.num_columns() ||
        replace_idx < 0 || replace_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    auto new_col = std::make_unique<cudf::column>(view.column(source_idx));

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(view.num_columns());
    for (int i = 0; i < view.num_columns(); ++i) {
        if (i == replace_idx) {
            columns.push_back(std::move(new_col));
        } else {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}
