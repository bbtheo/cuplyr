// src/ops_filter_fused.cpp
// Fused filter operations - apply multiple predicates with single AND mask
#include "gpu_table.hpp"
#include "ops_common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <vector>
#include <string>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_filter_fused(SEXP xptr, Rcpp::IntegerVector col_indices,
                      Rcpp::CharacterVector ops, Rcpp::NumericVector values) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_preds = col_indices.size();
    if (n_preds == 0) {
        // No predicates - return copy
        auto result = std::make_unique<cudf::table>(view);
        return make_gpu_table_xptr(std::move(result));
    }

    if (n_preds != ops.size() || n_preds != values.size()) {
        Rcpp::stop("Mismatched predicate specification lengths");
    }

    // Create first mask
    int col_idx = col_indices[0];
    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds: %d", col_idx);
    }

    cudf::column_view col = view.column(col_idx);
    double value = values[0];
    std::string op = Rcpp::as<std::string>(ops[0]);

    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    auto mask = cudf::binary_operation(
        col, *scalar, get_compare_op(op),
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // AND with subsequent masks
    for (int i = 1; i < n_preds; ++i) {
        col_idx = col_indices[i];
        if (col_idx < 0 || col_idx >= view.num_columns()) {
            Rcpp::stop("Column index out of bounds: %d", col_idx);
        }

        col = view.column(col_idx);
        value = values[i];
        op = Rcpp::as<std::string>(ops[i]);

        auto scalar_i = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
        static_cast<cudf::numeric_scalar<double>*>(scalar_i.get())->set_value(value);

        auto mask_i = cudf::binary_operation(
            col, *scalar_i, get_compare_op(op),
            cudf::data_type{cudf::type_id::BOOL8}
        );

        // AND the masks together
        mask = cudf::binary_operation(
            mask->view(), mask_i->view(),
            cudf::binary_operator::BITWISE_AND,
            cudf::data_type{cudf::type_id::BOOL8}
        );
    }

    // Apply the combined mask
    auto result = cudf::apply_boolean_mask(view, mask->view());
    return make_gpu_table_xptr(std::move(result));
}
