// src/ops_join.cpp
// GPU join implementations using cuDF
#include <Rcpp.h>

#include "cuda_utils.hpp"
#include "gpu_table.hpp"

#include <cudf/join/join.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/sorting.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>

#include <set>
#include <vector>

namespace {

std::vector<cudf::size_type> to_index_vec(const Rcpp::IntegerVector& r_idx) {
    std::vector<cudf::size_type> out;
    out.reserve(r_idx.size());
    for (int i = 0; i < r_idx.size(); ++i) {
        out.push_back(static_cast<cudf::size_type>(r_idx[i]));
    }
    return out;
}

cudf::table_view select_table_view(const cudf::table_view& view,
                                   const std::vector<cudf::size_type>& indices) {
    std::vector<cudf::column_view> cols;
    cols.reserve(indices.size());
    for (auto idx : indices) {
        cols.push_back(view.column(idx));
    }
    return cudf::table_view(cols);
}

std::unique_ptr<cudf::table> build_join_result(
    const cudf::table_view& left_view,
    const cudf::table_view& right_view,
    const rmm::device_uvector<cudf::size_type>& left_map_in,
    const rmm::device_uvector<cudf::size_type>& right_map_in,
    const std::vector<cudf::size_type>& right_keep_cols) {

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    auto sanitize_map = [&](const rmm::device_uvector<cudf::size_type>& map,
                            cudf::size_type nrows) {
        std::vector<cudf::size_type> host(map.size());
        cuplyr::check_cuda(
            cudaMemcpy(host.data(), map.data(), map.size() * sizeof(cudf::size_type),
                       cudaMemcpyDeviceToHost),
            "copy join map to host");

        for (auto& v : host) {
            if (v == cudf::JoinNoMatch) v = nrows;
        }

        rmm::device_uvector<cudf::size_type> out(map.size(), stream);
        cuplyr::check_cuda(
            cudaMemcpy(out.data(), host.data(), host.size() * sizeof(cudf::size_type),
                       cudaMemcpyHostToDevice),
            "copy join map to device");

        return out;
    };

    auto left_map = sanitize_map(left_map_in, left_view.num_rows());
    auto right_map = sanitize_map(right_map_in, right_view.num_rows());

    cudf::device_span<cudf::size_type const> left_span(left_map.data(), left_map.size());
    cudf::device_span<cudf::size_type const> right_span(right_map.data(), right_map.size());

    cudf::column_view left_map_view(left_span);
    cudf::column_view right_map_view(right_span);

    // Stable sort by left_map then right_map to preserve left-row order
    cudf::table_view map_tbl({left_map_view, right_map_view});
    std::vector<cudf::order> order_cols = {cudf::order::ASCENDING, cudf::order::ASCENDING};
    auto order = cudf::stable_sorted_order(map_tbl, order_cols);
    auto sorted_maps = cudf::gather(map_tbl, order->view(),
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    stream, mr);
    auto sorted_view = sorted_maps->view();
    left_map_view = sorted_view.column(0);
    right_map_view = sorted_view.column(1);

    auto right_subview = select_table_view(right_view, right_keep_cols);

    auto left_gathered = cudf::gather(
        left_view, left_map_view,
        cudf::out_of_bounds_policy::NULLIFY,
        stream, mr
    );

    auto right_gathered = cudf::gather(
        right_subview, right_map_view,
        cudf::out_of_bounds_policy::NULLIFY,
        stream, mr
    );

    auto left_cols = left_gathered->release();
    auto right_cols = right_gathered->release();

    std::vector<std::unique_ptr<cudf::column>> result_cols;
    result_cols.reserve(left_cols.size() + right_cols.size());

    for (auto& col : left_cols) {
        result_cols.push_back(std::move(col));
    }
    for (auto& col : right_cols) {
        result_cols.push_back(std::move(col));
    }

    return std::make_unique<cudf::table>(std::move(result_cols));
}

std::vector<cudf::size_type> compute_right_keep_cols(
    const cudf::table_view& right_view,
    const std::vector<cudf::size_type>& right_drop_cols) {
    std::set<cudf::size_type> drop_set(right_drop_cols.begin(), right_drop_cols.end());
    std::vector<cudf::size_type> keep;
    keep.reserve(right_view.num_columns() - drop_set.size());
    for (cudf::size_type i = 0; i < right_view.num_columns(); ++i) {
        if (drop_set.count(i) == 0) {
            keep.push_back(i);
        }
    }
    return keep;
}

} // namespace

// [[Rcpp::export]]
SEXP gpu_left_join(SEXP xptr_left,
                   SEXP xptr_right,
                   Rcpp::IntegerVector left_key_cols,
                   Rcpp::IntegerVector right_key_cols,
                   Rcpp::IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    auto left_keys = to_index_vec(left_key_cols);
    auto right_keys = to_index_vec(right_key_cols);
    auto right_drop = to_index_vec(right_drop_cols);

    auto left_key_view = select_table_view(left_view, left_keys);
    auto right_key_view = select_table_view(right_view, right_keys);

    auto [left_map, right_map] = cudf::left_join(
        left_key_view, right_key_view, cudf::null_equality::EQUAL);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);
    auto result = build_join_result(left_view, right_view, *left_map, *right_map, right_keep);

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_inner_join(SEXP xptr_left,
                    SEXP xptr_right,
                    Rcpp::IntegerVector left_key_cols,
                    Rcpp::IntegerVector right_key_cols,
                    Rcpp::IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    auto left_keys = to_index_vec(left_key_cols);
    auto right_keys = to_index_vec(right_key_cols);
    auto right_drop = to_index_vec(right_drop_cols);

    auto left_key_view = select_table_view(left_view, left_keys);
    auto right_key_view = select_table_view(right_view, right_keys);

    auto [left_map, right_map] = cudf::inner_join(
        left_key_view, right_key_view, cudf::null_equality::EQUAL);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);
    auto result = build_join_result(left_view, right_view, *left_map, *right_map, right_keep);

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_full_join(SEXP xptr_left,
                   SEXP xptr_right,
                   Rcpp::IntegerVector left_key_cols,
                   Rcpp::IntegerVector right_key_cols,
                   Rcpp::IntegerVector right_drop_cols) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    auto left_keys = to_index_vec(left_key_cols);
    auto right_keys = to_index_vec(right_key_cols);
    auto right_drop = to_index_vec(right_drop_cols);

    auto left_key_view = select_table_view(left_view, left_keys);
    auto right_key_view = select_table_view(right_view, right_keys);

    auto [left_map, right_map] = cudf::full_join(
        left_key_view, right_key_view, cudf::null_equality::EQUAL);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);
    auto result = build_join_result(left_view, right_view, *left_map, *right_map, right_keep);

    return make_gpu_table_xptr(std::move(result));
}
