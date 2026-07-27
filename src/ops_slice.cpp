// src/ops_slice.cpp
//
// GPU-native slice() family (Phase 3, task 2): ungrouped slice_head()/
// slice_tail()/slice() (integer indices)/slice_min()/slice_max(). Grouped
// input and slice_sample() stay on the CPU fallback (R/fallback.R) -- see
// R/slice.R's module doc for the full scope decision. Semantics below were
// verified empirically against dplyr 1.2.1 (and cross-checked against
// dplyr's own `get_slice_size()`/`slice_rank_idx()` source) before writing
// any of this.
//
// - gpu_slice_head()/gpu_slice_tail(): `cudf::slice()` gives a zero-copy
//   view into [0, size) or [nrow-size, nrow); wrapped in an owned
//   `cudf::table` (a `table_view` -> `table` copy, same trick already used
//   by `gpu_arrange()`'s empty-sort-spec early return).
// - gpu_slice_indices(): dplyr's `slice(df, ...)` index-vector semantics
//   (duplicates/reordering allowed for positive indices, 0/NA/out-of-range
//   silently dropped, negative indices mean "drop these", positive and
//   negative can't mix) are resolved here in C++, not in R, because `nrow`
//   is only known once the input table actually exists -- which, for a
//   lazy pipeline, is only true once upstream ops have actually executed
//   (see R/lower.R::lower_slice()). The already-resolved 0-based keep list
//   is uploaded as a device gather map and gathered with `DONT_CHECK`
//   (safe: every index in it is already validated in-range).
// - gpu_slice_rank(): implements slice_min()/slice_max(). dplyr's own
//   algorithm (`dplyr:::slice_rank_idx()`) is: rank `order_by` with
//   NA always sorting *last* regardless of direction (ties="min" when
//   with_ties=TRUE, i.e. SQL RANK(); ties="sequential"/first-occurrence-wins
//   when with_ties=FALSE), keep every row whose rank is <= the resolved
//   size, optionally drop NA rows outright (na_rm=TRUE), and return the
//   kept rows *in sorted order* (not original row order) -- confirmed
//   empirically: `slice_min(df, x, n=3)` on x=c(1,2,2,3,NA,NA,1) returns
//   rows in ascending-x order, not original-row order.
//
//   This is implemented as: stable-sort the whole table by `order_by`
//   (NA last, either direction -- see the null_order comment below),
//   compute a per-original-row `keep` mask via `cudf::rank()` (MIN method
//   for with_ties=TRUE == SQL RANK(), FIRST method for with_ties=FALSE ==
//   stable position, both exactly matching dplyr's `ties=` choice), gather
//   that mask into the same sorted order, and `apply_boolean_mask()` the
//   sorted table by the sorted mask. `apply_boolean_mask()` preserves the
//   relative order of TRUE rows, so the result is exactly "kept rows, in
//   sorted order" -- dplyr's contract, achieved with no CPU round-trip at
//   all (an earlier design considered collecting a single boundary value
//   to determine the with_ties cutoff on the CPU; `cudf::rank()` makes
//   that unnecessary since it's a direct public API for exactly this
//   ranking/tie-breaking problem).

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "ops_common.hpp"

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <climits>
#include <cmath>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

namespace {

void check_row_limit(const cudf::table_view& view, const char* who) {
    if (view.num_rows() > static_cast<cudf::size_type>(INT32_MAX)) {
        Rcpp::stop("Table has too many rows for %s (max ~2.1 billion)", who);
    }
}

} // namespace

// [[Rcpp::export]]
SEXP gpu_slice_head(SEXP xptr, double amount, bool is_prop) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice_head()");

    cudf::size_type nrow = view.num_rows();
    cudf::size_type size = compute_slice_size(amount, is_prop, nrow);

    std::vector<cudf::table_view> parts = cudf::slice(view, {0, size});
    auto result = std::make_unique<cudf::table>(parts[0]);
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_slice_tail(SEXP xptr, double amount, bool is_prop) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice_tail()");

    cudf::size_type nrow = view.num_rows();
    cudf::size_type size = compute_slice_size(amount, is_prop, nrow);

    std::vector<cudf::table_view> parts = cudf::slice(view, {nrow - size, nrow});
    auto result = std::make_unique<cudf::table>(parts[0]);
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_slice_indices(SEXP xptr, NumericVector raw_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice()");

    cudf::size_type nrow = view.num_rows();

    // Resolve dplyr::slice()'s index-vector contract (verified empirically
    // against dplyr 1.2.1): drop NA/0, error on mixing positive and negative,
    // negative means "drop these" (result is every remaining row, ascending,
    // duplicates collapse), positive means "keep these, in exactly the given
    // order/duplication", dropping any magnitude > nrow.
    bool has_pos = false, has_neg = false;
    std::vector<double> vals;
    vals.reserve(raw_indices.size());
    for (double v : raw_indices) {
        if (NumericVector::is_na(v) || v == 0) continue;
        if (std::floor(v) != v) {
            Rcpp::stop("slice() indices must be whole numbers, not %f", v);
        }
        if (v > 0) has_pos = true; else has_neg = true;
        vals.push_back(v);
    }
    if (has_pos && has_neg) {
        Rcpp::stop("slice() indices must be either all positive or all negative");
    }

    std::vector<cudf::size_type> keep;
    if (has_neg) {
        std::vector<bool> excluded(static_cast<size_t>(nrow), false);
        for (double v : vals) {
            cudf::size_type idx0 = static_cast<cudf::size_type>(-v) - 1;
            if (idx0 >= 0 && idx0 < nrow) excluded[static_cast<size_t>(idx0)] = true;
        }
        keep.reserve(static_cast<size_t>(nrow));
        for (cudf::size_type i = 0; i < nrow; ++i) {
            if (!excluded[static_cast<size_t>(i)]) keep.push_back(i);
        }
    } else {
        keep.reserve(vals.size());
        for (double v : vals) {
            cudf::size_type idx1 = static_cast<cudf::size_type>(v);
            if (idx1 >= 1 && idx1 <= nrow) keep.push_back(idx1 - 1);
        }
    }

    if (keep.empty()) {
        return make_gpu_table_xptr(cudf::empty_like(view));
    }

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    rmm::device_uvector<cudf::size_type> d_idx(keep.size(), stream);
    check_cuda(
        cudaMemcpy(d_idx.data(), keep.data(), keep.size() * sizeof(cudf::size_type),
                   cudaMemcpyHostToDevice),
        "slice() gather-map upload");

    cudf::column_view gather_map(
        cudf::device_span<cudf::size_type const>(d_idx.data(), d_idx.size()));

    auto result = cudf::gather(view, gather_map, cudf::out_of_bounds_policy::DONT_CHECK,
                                stream, mr);
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_slice_rank(SEXP xptr, int order_col_idx, bool descending, double amount,
                     bool is_prop, bool with_ties, bool na_rm) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice_min()/slice_max()");

    if (order_col_idx < 0 || order_col_idx >= view.num_columns()) {
        Rcpp::stop("order_by column index out of bounds: %d (table has %d columns)",
                   order_col_idx, view.num_columns());
    }

    cudf::column_view order_col = view.column(order_col_idx);
    cudf::size_type nrow = view.num_rows();
    cudf::size_type size = compute_slice_size(amount, is_prop, nrow);

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    cudf::order col_order = descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;

    // NA must sort *last* regardless of direction (dplyr:
    // `slice_rank_idx()`'s `na_value` is "largest" for ascending (min) and
    // "smallest" for descending (max) -- in both cases the row that ends up
    // last). cudf's null_order is a *value* convention (AFTER = "null
    // compares as larger than everything", BEFORE = "smaller than
    // everything"), so achieving "last" in both sort directions means
    // flipping the null_order together with the column order -- the mirror
    // image of what R/lower.R::lower_arrange() does for plain arrange()
    // (which wants dplyr's actual "NA first when descending" behavior and
    // so uses AFTER unconditionally).
    cudf::null_order null_prec = descending ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

    cudf::rank_method method = with_ties ? cudf::rank_method::MIN : cudf::rank_method::FIRST;
    std::unique_ptr<cudf::column> rank_col = cudf::rank(
        order_col, method, col_order, cudf::null_policy::INCLUDE, null_prec, false,
        stream, mr);

    cudf::numeric_scalar<int32_t> size_scalar(size, true);
    std::unique_ptr<cudf::column> keep_mask = cudf::binary_operation(
        rank_col->view(), size_scalar, cudf::binary_operator::LESS_EQUAL,
        cudf::data_type(cudf::type_id::BOOL8), stream, mr);

    if (na_rm && order_col.nullable()) {
        std::unique_ptr<cudf::column> valid_bools = cudf::mask_to_bools(
            order_col.null_mask(), order_col.offset(),
            order_col.offset() + order_col.size(), stream, mr);
        keep_mask = cudf::binary_operation(
            keep_mask->view(), valid_bools->view(), cudf::binary_operator::LOGICAL_AND,
            cudf::data_type(cudf::type_id::BOOL8), stream, mr);
    }

    std::unique_ptr<cudf::column> sort_indices = cudf::stable_sorted_order(
        cudf::table_view({order_col}), {col_order}, {null_prec}, stream, mr);

    std::unique_ptr<cudf::table> sorted_table = cudf::gather(
        view, sort_indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

    std::unique_ptr<cudf::table> sorted_mask_tbl = cudf::gather(
        cudf::table_view({keep_mask->view()}), sort_indices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

    std::unique_ptr<cudf::table> result = cudf::apply_boolean_mask(
        sorted_table->view(), sorted_mask_tbl->view().column(0), stream, mr);

    return make_gpu_table_xptr(std::move(result));
}
