// src/ops_join.cpp
// GPU join operations for cuplyr using cuDF

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include <cudf/join/join.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/device_uvector.hpp>
#include <set>
#include <Rcpp.h>

using namespace Rcpp;

namespace {

// Helper: extract key columns as table_view
cudf::table_view extract_keys(cudf::table_view const& tbl,
                              std::vector<cudf::size_type> const& key_indices) {
    std::vector<cudf::column_view> key_cols;
    key_cols.reserve(key_indices.size());
    for (auto idx : key_indices) {
        if (idx < 0 || idx >= tbl.num_columns()) {
            Rcpp::stop("Key column index out of bounds: %d (table has %d columns)",
                       idx, tbl.num_columns());
        }
        key_cols.push_back(tbl.column(idx));
    }
    return cudf::table_view{key_cols};
}

// Helper: create a column_view from device_uvector for gather operations
// This allows proper handling of JoinNoMatch indices (-1) with NULLIFY policy
cudf::column_view indices_to_column_view(rmm::device_uvector<cudf::size_type> const& indices) {
    return cudf::column_view(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(indices.size()),
        indices.data(),
        nullptr,  // no null mask
        0         // no nulls
    );
}

// Helper: build result table from gather maps
// Combines columns from both tables, dropping specified right-side columns
//
// NOTE: cudf::left_join/inner_join/full_join return:
//   std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
//             std::unique_ptr<rmm::device_uvector<size_type>>>
// The gather operation uses column_view to properly handle JoinNoMatch (-1)
// indices with out_of_bounds_policy::NULLIFY, ensuring unmatched rows get NULL.
std::unique_ptr<cudf::table> build_join_result(
    cudf::table_view const& left,
    cudf::table_view const& right,
    rmm::device_uvector<cudf::size_type> const& left_indices,
    rmm::device_uvector<cudf::size_type> const& right_indices,
    std::vector<cudf::size_type> const& right_drop_indices
) {
    // Convert device_uvector to column_view for gather
    // Using column_view ensures proper handling of negative indices (-1 = JoinNoMatch)
    // with out_of_bounds_policy::NULLIFY - negative indices are treated as out-of-bounds
    cudf::column_view left_map = indices_to_column_view(left_indices);
    cudf::column_view right_map = indices_to_column_view(right_indices);

    auto left_gathered = cudf::gather(
        left,
        left_map,
        cudf::out_of_bounds_policy::NULLIFY
    );
    auto right_gathered = cudf::gather(
        right,
        right_map,
        cudf::out_of_bounds_policy::NULLIFY
    );

    std::vector<std::unique_ptr<cudf::column>> result_cols;
    result_cols.reserve(left.num_columns() + right.num_columns() - right_drop_indices.size());

    auto left_cols = left_gathered->release();
    auto right_cols = right_gathered->release();

    // Add all left columns
    for (auto& col : left_cols) {
        result_cols.push_back(std::move(col));
    }

    // Build drop set for O(1) lookup
    std::set<cudf::size_type> drop_set(right_drop_indices.begin(), right_drop_indices.end());

    // Add non-dropped right columns
    for (cudf::size_type i = 0; i < static_cast<cudf::size_type>(right_cols.size()); i++) {
        if (drop_set.find(i) == drop_set.end()) {
            result_cols.push_back(std::move(right_cols[i]));
        }
    }

    return std::make_unique<cudf::table>(std::move(result_cols));
}

} // anonymous namespace

// [[Rcpp::export]]
SEXP gpu_left_join(SEXP xptr_left, SEXP xptr_right,
                   IntegerVector left_key_cols,
                   IntegerVector right_key_cols,
                   IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr_left(xptr_left);
    Rcpp::XPtr<GpuTablePtr> ptr_right(xptr_right);

    cudf::table_view left_view = get_table_view(ptr_left);
    cudf::table_view right_view = get_table_view(ptr_right);

    // Convert R indices to C++ vectors (already 0-based from R side)
    std::vector<cudf::size_type> left_keys(
        left_key_cols.begin(), left_key_cols.end()
    );
    std::vector<cudf::size_type> right_keys(
        right_key_cols.begin(), right_key_cols.end()
    );
    std::vector<cudf::size_type> right_drop(
        right_drop_cols.begin(), right_drop_cols.end()
    );

    // Extract key column views
    cudf::table_view left_key_view = extract_keys(left_view, left_keys);
    cudf::table_view right_key_view = extract_keys(right_view, right_keys);

    // Perform left join - returns pair of device_uvector<size_type>
    auto [left_map, right_map] = cudf::left_join(
        left_key_view, right_key_view,
        cudf::null_equality::EQUAL
    );

    // Build result table using gather maps
    auto result = build_join_result(
        left_view, right_view,
        *left_map, *right_map,
        right_drop
    );

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_inner_join(SEXP xptr_left, SEXP xptr_right,
                    IntegerVector left_key_cols,
                    IntegerVector right_key_cols,
                    IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr_left(xptr_left);
    Rcpp::XPtr<GpuTablePtr> ptr_right(xptr_right);

    cudf::table_view left_view = get_table_view(ptr_left);
    cudf::table_view right_view = get_table_view(ptr_right);

    std::vector<cudf::size_type> left_keys(left_key_cols.begin(), left_key_cols.end());
    std::vector<cudf::size_type> right_keys(right_key_cols.begin(), right_key_cols.end());
    std::vector<cudf::size_type> right_drop(right_drop_cols.begin(), right_drop_cols.end());

    cudf::table_view left_key_view = extract_keys(left_view, left_keys);
    cudf::table_view right_key_view = extract_keys(right_view, right_keys);

    auto [left_map, right_map] = cudf::inner_join(
        left_key_view, right_key_view,
        cudf::null_equality::EQUAL
    );

    auto result = build_join_result(
        left_view, right_view,
        *left_map, *right_map,
        right_drop
    );

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_full_join(SEXP xptr_left, SEXP xptr_right,
                   IntegerVector left_key_cols,
                   IntegerVector right_key_cols,
                   IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr_left(xptr_left);
    Rcpp::XPtr<GpuTablePtr> ptr_right(xptr_right);

    cudf::table_view left_view = get_table_view(ptr_left);
    cudf::table_view right_view = get_table_view(ptr_right);

    std::vector<cudf::size_type> left_keys(left_key_cols.begin(), left_key_cols.end());
    std::vector<cudf::size_type> right_keys(right_key_cols.begin(), right_key_cols.end());
    std::vector<cudf::size_type> right_drop(right_drop_cols.begin(), right_drop_cols.end());

    cudf::table_view left_key_view = extract_keys(left_view, left_keys);
    cudf::table_view right_key_view = extract_keys(right_view, right_keys);

    auto [left_map, right_map] = cudf::full_join(
        left_key_view, right_key_view,
        cudf::null_equality::EQUAL
    );

    auto result = build_join_result(
        left_view, right_view,
        *left_map, *right_map,
        right_drop
    );

    return make_gpu_table_xptr(std::move(result));
}
