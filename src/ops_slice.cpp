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
//   Three routes, picked in gpu_slice_rank():
//   - top-k (numeric/date keys, 1 <= size <= non-NA rows, and under "auto" a
//     size of at most kTopkFraction of the rows): find the size-th best key
//     with cudf::top_k_order(), then stable-sort only the rows no worse than
//     it (all ties included), so the cost follows the survivors.
//   - with_ties = FALSE otherwise: stable-sort the key (NA last, either
//     direction -- see the null_order comment below), keep the first `size`
//     positions, and gather; na_rm drops NA rows among them afterwards.
//   - with_ties = TRUE otherwise: stable-sort, build a keep mask from
//     `cudf::rank()` (MIN method == SQL RANK()), gather the mask into sorted
//     order and apply_boolean_mask(), which preserves the sorted order.
//   Every route returns the kept rows in sorted order, with no CPU round-trip.
//   The "cuplyr.topk_path" option ("auto", "topk", "sort") forces a route and
//   the result pointer's "topk" attribute records which one ran.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "ops_common.hpp"

#include <cudf/copying.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/filling.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <climits>
#include <cstdint>
#include <string>
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

// group_split(): gather the rows into group order (`idx0`, 0-based, all
// groups concatenated) and cut at the cumulative group `sizes` with one
// cudf::contiguous_split, which allocates one packed buffer per piece. A chunk
// owns that buffer alone (its columns are SharedColumns over it), so a kept
// chunk holds its own rows' bytes whether or not it is ever used, and the
// gathered table and the input are free to go. An empty `idx0` means "already
// in group order" (ungrouped input): the source is split as is. A single group
// is not split at all: its table shares the gathered (or input) columns, which
// are all its own rows. Returns one XPtr per size (an empty list for no sizes).
// [[Rcpp::export]]
Rcpp::List gpu_split_rows(SEXP xptr, IntegerVector idx0, IntegerVector sizes) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "group_split()");

    R_xlen_t ngroups = sizes.size();
    Rcpp::List out(ngroups);
    if (ngroups == 0) return out;

    cudf::size_type total = 0;
    for (R_xlen_t i = 0; i < ngroups; ++i) {
        if (sizes[i] < 0) Rcpp::stop("group_split(): negative group size");
        total += sizes[i];
    }

    std::unique_ptr<cudf::table> gathered;
    cudf::table_view source = view;
    if (idx0.size() > 0) {
        if (idx0.size() != total) Rcpp::stop("group_split(): row indices do not match group sizes");
        if (total > view.num_rows()) Rcpp::stop("group_split(): group sizes exceed the table's rows");
        for (R_xlen_t i = 0; i < idx0.size(); ++i) {
            if (idx0[i] == NA_INTEGER || idx0[i] < 0 || idx0[i] >= view.num_rows())
                Rcpp::stop("group_split(): row index out of range");
        }
        auto stream = cudf::get_default_stream();
        auto mr = rmm::mr::get_current_device_resource();
        rmm::device_uvector<cudf::size_type> d_idx(idx0.size(), stream);
        check_cuda(
            cudaMemcpy(d_idx.data(), &idx0[0], idx0.size() * sizeof(cudf::size_type),
                       cudaMemcpyHostToDevice),
            "group_split() gather-map upload");
        cudf::column_view gather_map(
            cudf::device_span<cudf::size_type const>(d_idx.data(), d_idx.size()));
        gathered = cudf::gather(view, gather_map, cudf::out_of_bounds_policy::DONT_CHECK,
                                stream, mr);
        source = gathered->view();
    } else if (total != view.num_rows()) {
        Rcpp::stop("group_split(): group sizes do not cover the table");
    }

    if (ngroups == 1) {
        if (gathered) {
            out[0] = make_gpu_table_xptr(std::move(gathered));
        } else {
            std::vector<SharedColumn> shared;
            for (int i = 0; i < view.num_columns(); ++i) {
                shared.push_back(get_shared_column(ptr, i));
            }
            out[0] = make_gpu_table_xptr(std::move(shared));
        }
        return out;
    }

    std::vector<cudf::size_type> splits;
    splits.reserve(static_cast<size_t>(ngroups) - 1);
    cudf::size_type edge = 0;
    for (R_xlen_t i = 0; i + 1 < ngroups; ++i) {
        edge += sizes[i];
        splits.push_back(edge);
    }

    std::vector<cudf::packed_table> pieces = cudf::contiguous_split(source, splits);
    gathered.reset();
    for (R_xlen_t i = 0; i < ngroups; ++i) {
        cudf::packed_table& piece = pieces[static_cast<size_t>(i)];
        // The piece's buffer is shared by its columns and by nothing else.
        std::shared_ptr<const void> owner(
            std::shared_ptr<rmm::device_buffer>(std::move(piece.data.gpu_data)));
        std::vector<SharedColumn> columns;
        columns.reserve(static_cast<size_t>(piece.table.num_columns()));
        for (cudf::size_type c = 0; c < piece.table.num_columns(); ++c) {
            columns.emplace_back(owner, piece.table.column(c));
        }
        out[i] = make_gpu_table_xptr(std::move(columns));
    }
    return out;
}

namespace {

// Largest fraction of the rows that "auto" still sends down the top-k route.
// Above it the survivors of the cut, which the route stable-sorts, approach the
// whole table and the plain sort is as fast (measured in the F98 brief).
constexpr double kTopkFraction = 0.25;

bool topk_key_type(cudf::type_id id) {
    switch (id) {
        case cudf::type_id::INT32:
        case cudf::type_id::INT64:
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64:
        case cudf::type_id::TIMESTAMP_DAYS:
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
            return true;
        default:
            return false;
    }
}

// A non-null float key as an INT64 whose signed order is the float order, so
// cudf::top_k_order() takes its radix-select fast path (it sorts any float or
// nullable key in full). Negative floats have the sign bit set; flipping their
// other 63 bits reverses their order. -0 lands below +0 here, which is why the
// cut value is compared in the original domain.
std::unique_ptr<cudf::column> sortable_int64(cudf::column_view key,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) {
    std::unique_ptr<cudf::column> wide;
    if (key.type().id() == cudf::type_id::FLOAT32) {
        wide = cudf::cast(key, cudf::data_type(cudf::type_id::FLOAT64), stream, mr);
        key = wide->view();
    }
    cudf::column_view bits = cudf::bit_cast(key, cudf::data_type(cudf::type_id::INT64));
    auto i64 = cudf::data_type(cudf::type_id::INT64);
    auto sign = cudf::binary_operation(bits, cudf::numeric_scalar<int64_t>(63, true),
                                       cudf::binary_operator::SHIFT_RIGHT, i64, stream, mr);
    auto flip = cudf::binary_operation(
        sign->view(), cudf::numeric_scalar<int64_t>(INT64_MAX, true),
        cudf::binary_operator::BITWISE_AND, i64, stream, mr);
    sign.reset();
    return cudf::binary_operation(bits, flip->view(), cudf::binary_operator::BITWISE_XOR, i64,
                                  stream, mr);
}

// Row positions (INT32, in result order) of the `size` best rows of `key` for
// slice_min()/slice_max(), `1 <= size <= valid rows`. The cut value t is the
// size-th best key; rows no worse than t (all ties included, in input order)
// are the only ones sorted. Equal to a stable sort of the whole key, first
// `size` rows (or every survivor with ties), at a cost that depends on the
// survivors, not the table.
std::unique_ptr<cudf::column> topk_positions(cudf::column_view key, cudf::size_type size,
                                             bool descending, bool with_ties,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) {
    const cudf::order col_order = descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    const bool is_float = key.type().id() == cudf::type_id::FLOAT64 ||
                          key.type().id() == cudf::type_id::FLOAT32;

    // top_k_order() wants no nulls (and NaN is null here)
    std::unique_ptr<cudf::table> valid_only;
    cudf::column_view dense = key;
    if (key.null_count() > 0) {
        valid_only = cudf::drop_nulls(cudf::table_view({key}), {0}, stream, mr);
        dense = valid_only->view().column(0);
    }

    std::unique_ptr<cudf::column> mapped;
    cudf::column_view top_key = dense;
    if (is_float) {
        mapped = sortable_int64(dense, stream, mr);
        top_key = mapped->view();
    }
    std::unique_ptr<cudf::column> best = cudf::top_k_order(top_key, size, col_order, stream, mr);
    mapped.reset();

    std::unique_ptr<cudf::table> best_keys = cudf::gather(
        cudf::table_view({dense}), best->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    best.reset();
    valid_only.reset();
    std::unique_ptr<cudf::scalar> cut = cudf::reduce(
        best_keys->view().column(0),
        descending ? *cudf::make_min_aggregation<cudf::reduce_aggregation>()
                   : *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
        key.type(), stream, mr);
    best_keys.reset();

    // A null or NaN key compares to null, which apply_boolean_mask() drops
    std::unique_ptr<cudf::column> within = cudf::binary_operation(
        key, *cut,
        descending ? cudf::binary_operator::GREATER_EQUAL : cudf::binary_operator::LESS_EQUAL,
        cudf::data_type(cudf::type_id::BOOL8), stream, mr);

    std::unique_ptr<cudf::column> seq = cudf::sequence(
        key.size(), cudf::numeric_scalar<int32_t>(0, true), cudf::numeric_scalar<int32_t>(1, true),
        stream, mr);
    std::unique_ptr<cudf::table> survivors = cudf::apply_boolean_mask(
        cudf::table_view({seq->view(), key}), within->view(), stream, mr);
    seq.reset();
    within.reset();

    cudf::column_view surv_key = survivors->view().column(1);
    std::unique_ptr<cudf::column> order = cudf::stable_sorted_order(
        cudf::table_view({surv_key}), {col_order}, {cudf::null_order::AFTER}, stream, mr);
    std::unique_ptr<cudf::table> sorted_pos = cudf::gather(
        cudf::table_view({survivors->view().column(0)}), order->view(),
        cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    std::unique_ptr<cudf::column> positions = std::move(sorted_pos->release().front());
    if (!with_ties && positions->size() > size) {
        positions = std::make_unique<cudf::column>(
            cudf::slice(positions->view(), {0, size}).front(), stream, mr);
    }
    return positions;
}

} // namespace

// `path` picks the route: "auto" (top-k when the cut is selective, see
// kTopkFraction), "topk" (whenever it applies) or "sort" (never). The result
// pointer's "topk" attribute says which route ran.
// [[Rcpp::export]]
SEXP gpu_slice_rank(SEXP xptr, int order_col_idx, bool descending, double amount,
                     bool is_prop, bool with_ties, bool na_rm,
                     std::string path = "auto") {
    using namespace cuplyr;

    if (path != "auto" && path != "topk" && path != "sort") {
        Rcpp::stop("path must be \"auto\", \"topk\" or \"sort\"");
    }

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice_min()/slice_max()");

    if (order_col_idx < 0 || order_col_idx >= view.num_columns()) {
        Rcpp::stop("order_by column index out of bounds: %d (table has %d columns)",
                   order_col_idx, view.num_columns());
    }

    cudf::column_view order_col = view.column(order_col_idx);
    // A floating NaN ranks as NA, tied with the real NAs and dropped by
    // na_rm (vec_rank() and vec_detect_complete() both treat NaN as
    // missing). cudf orders NaN above +Inf, which put a GPU-computed NaN
    // first for slice_max(). Only the sort/rank key is nulled; the gathered
    // rows keep their NaN. Same rule as gpu_window()'s rank frames.
    std::unique_ptr<cudf::column> nan_nulled;
    if (order_col.type().id() == cudf::type_id::FLOAT64 ||
        order_col.type().id() == cudf::type_id::FLOAT32) {
        auto mask = cudf::nans_to_nulls(order_col);
        if (mask.second > order_col.null_count()) {
            nan_nulled = std::make_unique<cudf::column>(order_col);
            nan_nulled->set_null_mask(std::move(*mask.first), mask.second);
            order_col = nan_nulled->view();
        }
    }
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
    // flipping the null_order together with the column order -- the same
    // technique src/ops_arrange.cpp's gpu_arrange() now uses too (Phase 6:
    // plain arrange(desc(x)) used to place NA FIRST, a real divergence from
    // dplyr that predated this comment; ops_arrange.cpp's null_order was
    // fixed to flip with direction the same way this function already did).
    cudf::null_order null_prec = descending ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

    // Top-k route: needs 1 <= size <= valid rows (a larger size puts NA rows
    // in the result, size 0 has nothing to select), so na_rm cannot matter.
    const cudf::size_type valid = nrow - order_col.null_count();
    if (path != "sort" && topk_key_type(order_col.type().id()) && size >= 1 &&
        size <= valid &&
        (path == "topk" || static_cast<double>(size) <= kTopkFraction * static_cast<double>(nrow))) {
        // Timestamps compare as their integer ticks (no timestamp scalar needed)
        cudf::column_view key = order_col;
        if (order_col.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
            key = cudf::bit_cast(order_col, cudf::data_type(cudf::type_id::INT32));
        } else if (order_col.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
            key = cudf::bit_cast(order_col, cudf::data_type(cudf::type_id::INT64));
        }
        std::unique_ptr<cudf::column> positions =
            topk_positions(key, size, descending, with_ties, stream, mr);
        auto out = make_gpu_table_xptr(cudf::gather(
            view, positions->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr));
        out.attr("topk") = true;
        return out;
    }

    // Without ties the answer is a stable sort and its first `size` rows,
    // arrange() |> head()'s cost; the rank below costs about twice that.
    if (!with_ties) {
        std::unique_ptr<cudf::column> sorted = cudf::stable_sorted_order(
            cudf::table_view({order_col}), {col_order}, {null_prec}, stream, mr);
        std::unique_ptr<cudf::column> first = std::make_unique<cudf::column>(
            cudf::slice(sorted->view(), {0, size}).front(), stream, mr);
        sorted.reset();
        if (na_rm && order_col.nullable() && size > 0) {
            // dplyr ranks first and then drops the NA rows among the first `size`
            std::unique_ptr<cudf::table> keys = cudf::gather(
                cudf::table_view({order_col}), first->view(),
                cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
            std::unique_ptr<cudf::column> present = cudf::is_valid(keys->view().column(0), stream, mr);
            std::unique_ptr<cudf::table> kept = cudf::apply_boolean_mask(
                cudf::table_view({first->view()}), present->view(), stream, mr);
            first = std::move(kept->release().front());
        }
        auto out = make_gpu_table_xptr(cudf::gather(
            view, first->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr));
        out.attr("topk") = false;
        return out;
    }

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

    auto out = make_gpu_table_xptr(std::move(result));
    out.attr("topk") = false;
    return out;
}
