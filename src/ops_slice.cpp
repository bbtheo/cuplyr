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
//
// - gpu_slice_rank_grouped() (F115): the same for group_by()/by= input and a
//   numeric or date key. It is the one grouped route; any other key type, and
//   "cuplyr.topk_path" = "sort", stay on R/slice.R's window desugar. It sets no
//   "topk" attribute: the route shows as a slice node of mode "rank_grouped" in
//   the plan. See the comment above it for the algorithm.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "ops_common.hpp"

#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/replace.hpp>
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
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>

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

// ---------------------------------------------------------------------------
// F115: grouped slice_min()/slice_max() on a numeric or date key.
//
// dplyr's rule per group: rank the key (NA last in both directions, NaN is NA,
// ties "min" or "sequential"), keep rank <= size, order the kept rows by rank.
// The route sorts row POSITIONS only, with stable single-column radix sorts
// (least significant first): the key, then "is the key missing" (so missing
// rows end up last, in row order, in either direction), then the group
// columns. Equal keys keep row order throughout, which is dplyr's tie order.
// The rows are then in (group, rank) order; the in-group position comes from a
// sorted-groupby scan, the rank of a tied row from the start of its run of
// equal keys, and one boolean mask picks the survivors. Only the survivors are
// gathered from the table. cudf::segmented_top_k_order() is not used: it is not
// stable, takes one k for every segment and cannot keep ties (see the F115
// brief). `by_first` (by=/.by=) orders the groups by the first row each has
// instead of by group value.
// ---------------------------------------------------------------------------

namespace {

using ColumnPtr = std::unique_ptr<cudf::column>;

// rows of `cols` picked by `perm` (DONT_CHECK: perm is a sort result)
std::unique_ptr<cudf::table> gather_rows(const cudf::table_view& cols, cudf::column_view perm,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr) {
    return cudf::gather(cols, perm, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

// The composed permutation: element i is inner[outer[i]]
ColumnPtr compose(cudf::column_view inner, cudf::column_view outer, rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr) {
    auto t = gather_rows(cudf::table_view({inner}), outer, stream, mr);
    return std::move(t->release().front());
}

// A key as an INT64 or INT32 column whose signed order is the key's order, with
// no nulls and -0 equal to 0 (sort passes need a dense, comparable column).
// Missing rows (null or NaN-nulled) take 0; they are put last by their own pass.
// New columns are owned by `keep`.
cudf::column_view dense_sort_key(cudf::column_view key, rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr, std::vector<ColumnPtr>& keep) {
    if (key.type().id() == cudf::type_id::FLOAT32) {
        keep.push_back(cudf::cast(key, cudf::data_type(cudf::type_id::FLOAT64), stream, mr));
        key = keep.back()->view();
    }
    if (key.type().id() == cudf::type_id::FLOAT64) {
        if (key.null_count() > 0) {
            keep.push_back(cudf::replace_nulls(
                key, *cudf::make_fixed_width_scalar<double>(0.0, stream, mr), stream, mr));
            key = keep.back()->view();
        }
        // -0 + 0 is +0: the two must tie, and as INT64 they would sort apart
        keep.push_back(cudf::binary_operation(
            key, cudf::numeric_scalar<double>(0.0, true, stream, mr), cudf::binary_operator::ADD,
            cudf::data_type(cudf::type_id::FLOAT64), stream, mr));
        keep.push_back(sortable_int64(keep.back()->view(), stream, mr));
        return keep.back()->view();
    }
    // integers and (bit-cast) timestamps
    if (key.null_count() > 0) {
        if (key.type().id() == cudf::type_id::INT32) {
            keep.push_back(cudf::replace_nulls(
                key, *cudf::make_fixed_width_scalar<int32_t>(0, stream, mr), stream, mr));
        } else {
            keep.push_back(cudf::replace_nulls(
                key, *cudf::make_fixed_width_scalar<int64_t>(0, stream, mr), stream, mr));
        }
        return keep.back()->view();
    }
    return key;
}

// 0, 1, ..., n-1 as INT32
ColumnPtr iota(cudf::size_type n, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
    return cudf::sequence(n, cudf::numeric_scalar<int32_t>(0, true, stream, mr),
                          cudf::numeric_scalar<int32_t>(1, true, stream, mr), stream, mr);
}

} // namespace

// [[Rcpp::export]]
SEXP gpu_slice_rank_grouped(SEXP xptr, IntegerVector group_idx, int order_col_idx,
                            bool descending, double amount, bool is_prop, bool with_ties,
                            bool na_rm, bool by_first) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view, "slice_min()/slice_max()");

    if (order_col_idx < 0 || order_col_idx >= view.num_columns()) {
        Rcpp::stop("order_by column index out of bounds: %d (table has %d columns)",
                   order_col_idx, view.num_columns());
    }
    if (group_idx.size() == 0) {
        Rcpp::stop("gpu_slice_rank_grouped: no group columns");
    }
    std::vector<cudf::column_view> group_views;
    for (int i = 0; i < group_idx.size(); ++i) {
        if (group_idx[i] < 0 || group_idx[i] >= view.num_columns()) {
            Rcpp::stop("group column index out of bounds: %d (table has %d columns)",
                       group_idx[i], view.num_columns());
        }
        group_views.push_back(view.column(group_idx[i]));
    }

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    const cudf::size_type m = view.num_rows();
    if (m == 0) {
        return make_gpu_table_xptr(cudf::empty_like(view));
    }

    // The key as the ranking sees it: a floating NaN is missing (as in
    // gpu_slice_rank()); the gathered rows keep their NaN. Timestamps compare
    // as their integer ticks.
    cudf::column_view key = view.column(order_col_idx);
    std::unique_ptr<cudf::column> nan_nulled = nan_nulled_copy(key);
    if (nan_nulled) key = nan_nulled->view();
    if (key.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
        key = cudf::bit_cast(key, cudf::data_type(cudf::type_id::INT32));
    } else if (key.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
        key = cudf::bit_cast(key, cudf::data_type(cudf::type_id::INT64));
    }
    if (!topk_key_type(key.type().id())) {
        Rcpp::stop("gpu_slice_rank_grouped: unsupported order_by type");
    }
    const bool has_missing = key.null_count() > 0;
    const cudf::order col_order = descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;

    // Pass 1: the key. `order` holds row positions, composed pass by pass.
    std::vector<ColumnPtr> keepalive;
    ColumnPtr order;
    {
        cudf::column_view sort_key = dense_sort_key(key, stream, mr, keepalive);
        order = cudf::stable_sorted_order(cudf::table_view({sort_key}), {col_order},
                                          {cudf::null_order::AFTER}, stream, mr);
    }
    keepalive.clear();

    // Pass 2: missing rows last, in row order (they all tied at 0 in pass 1)
    if (has_missing) {
        ColumnPtr missing = cudf::is_null(key, stream, mr);
        ColumnPtr by_order = std::move(gather_rows(cudf::table_view({missing->view()}), order->view(), stream, mr)->release().front());
        missing.reset();
        ColumnPtr o2 = cudf::stable_sorted_order(cudf::table_view({by_order->view()}),
                                                 {cudf::order::ASCENDING}, {cudf::null_order::AFTER},
                                                 stream, mr);
        by_order.reset();
        order = compose(order->view(), o2->view(), stream, mr);
    }

    // Pass 3: the groups (a single radix sort for one integer column, a
    // comparator sort of the group columns otherwise)
    {
        auto g = gather_rows(cudf::table_view(group_views), order->view(), stream, mr);
        std::vector<cudf::order> g_order(group_views.size(), cudf::order::ASCENDING);
        std::vector<cudf::null_order> g_null(group_views.size(), cudf::null_order::AFTER);
        ColumnPtr o3 = cudf::stable_sorted_order(g->view(), g_order, g_null, stream, mr);
        g.reset();
        order = compose(order->view(), o3->view(), stream, mr);
    }

    // The group columns and the key in (group, rank) order
    std::vector<cudf::column_view> gk_views = group_views;
    gk_views.push_back(key);
    std::unique_ptr<cudf::table> sorted = gather_rows(cudf::table_view(gk_views), order->view(), stream, mr);
    const cudf::size_type ng = static_cast<cudf::size_type>(group_views.size());
    std::vector<cudf::column_view> sg_views;
    for (cudf::size_type i = 0; i < ng; ++i) sg_views.push_back(sorted->view().column(i));
    cudf::table_view sorted_groups(sg_views);
    cudf::column_view sorted_key = sorted->view().column(ng);

    // Partition exactly as gpu_window() does: the rows are sorted by group, NaN
    // and NA group keys included (sorted::YES, never NOT_EQUAL on the values)
    std::vector<cudf::order> key_order(ng, cudf::order::ASCENDING);
    std::vector<cudf::null_order> key_null(ng, cudf::null_order::AFTER);
    cudf::groupby::groupby gb(sorted_groups, cudf::null_policy::INCLUDE, cudf::sorted::YES,
                              key_order, key_null);

    // 1-based position in the group
    ColumnPtr rn;
    {
        std::vector<cudf::groupby::scan_request> reqs(1);
        reqs[0].values = sorted_groups.column(0);
        reqs[0].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(cudf::null_policy::INCLUDE));
        auto res = gb.scan(reqs, stream, mr);
        rn = std::move(res.second[0].results[0]);
        if (rn->type().id() != cudf::type_id::INT32) {
            rn = cudf::cast(rn->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
        }
    }
    const auto i32 = cudf::data_type(cudf::type_id::INT32);
    const auto bool8 = cudf::data_type(cudf::type_id::BOOL8);
    const auto f64 = cudf::data_type(cudf::type_id::FLOAT64);

    // 0-based rank within the group
    ColumnPtr rank0 = cudf::binary_operation(rn->view(), cudf::numeric_scalar<int32_t>(1, true, stream, mr),
                                             cudf::binary_operator::SUB, i32, stream, mr);
    if (with_ties) {
        // the start of the run of equal keys (missing keys are all equal) the
        // row sits in, as a global position; its offset from the row, which is
        // <= 0, moves the rank back to the run start
        ColumnPtr seq = iota(m, stream, mr);
        ColumnPtr first_of_group = cudf::binary_operation(
            rn->view(), cudf::numeric_scalar<int32_t>(1, true, stream, mr),
            cudf::binary_operator::EQUAL, bool8, stream, mr);
        ColumnPtr flag;  // row starts a run: first of a group, or key differs from the row above
        {
            auto cur = cudf::slice(sorted_key, {1, m}).front();
            auto prev = cudf::slice(sorted_key, {0, m - 1}).front();
            ColumnPtr same = cudf::binary_operation(cur, prev, cudf::binary_operator::NULL_EQUALS, bool8, stream, mr);
            ColumnPtr differs = cudf::unary_operation(same->view(), cudf::unary_operator::NOT, stream, mr);
            same.reset();
            auto first_tail = cudf::slice(first_of_group->view(), {1, m}).front();
            ColumnPtr tail = cudf::binary_operation(differs->view(), first_tail,
                                                    cudf::binary_operator::LOGICAL_OR, bool8, stream, mr);
            differs.reset();
            ColumnPtr head = cudf::make_column_from_scalar(cudf::numeric_scalar<bool>(true, true, stream, mr), 1, stream, mr);
            flag = cudf::concatenate(std::vector<cudf::column_view>{head->view(), tail->view()}, stream, mr);
        }
        first_of_group.reset();
        ColumnPtr flag32 = cudf::cast(flag->view(), i32, stream, mr);
        flag.reset();
        ColumnPtr marked = cudf::binary_operation(seq->view(), flag32->view(), cudf::binary_operator::MUL, i32, stream, mr);
        flag32.reset();
        ColumnPtr run_start = cudf::scan(marked->view(), *cudf::make_max_aggregation<cudf::scan_aggregation>(),
                                         cudf::scan_type::INCLUSIVE, cudf::null_policy::INCLUDE, stream, mr);
        marked.reset();
        ColumnPtr back = cudf::binary_operation(run_start->view(), seq->view(), cudf::binary_operator::SUB, i32, stream, mr);
        rank0 = cudf::binary_operation(rank0->view(), back->view(), cudf::binary_operator::ADD, i32, stream, mr);
    }
    rn.reset();

    // Per group: its row count (for prop/negative sizes) and, for by=, the
    // first row it has. Both are per sorted group, then repeated to every row.
    ColumnPtr group_rows;   // rows of the row's group, INT32
    ColumnPtr group_first;  // first row position of the row's group, INT32
    const bool size_per_row = is_prop || amount < 0;
    if (size_per_row || by_first) {
        std::vector<cudf::groupby::aggregation_request> reqs(by_first ? 2 : 1);
        reqs[0].values = sorted_groups.column(0);
        reqs[0].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
        if (by_first) {
            reqs[1].values = order->view();
            reqs[1].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
        }
        auto res = gb.aggregate(reqs, stream, mr);
        cudf::column_view counts = res.second[0].results[0]->view();
        std::vector<cudf::column_view> per_group = {counts};
        if (by_first) per_group.push_back(res.second[1].results[0]->view());
        auto rep = cudf::repeat(cudf::table_view(per_group), counts, stream, mr);
        auto cols = rep->release();
        group_rows = std::move(cols[0]);
        if (by_first) group_first = std::move(cols[1]);
    }

    // keep = rank0 < size, i.e. rank <= size (sizes are whole numbers). The
    // size is dplyr's get_slice_size() in FLOAT64, so Inf works; it needs no
    // clamp, since a size below 0 keeps nothing and one past the group keeps all.
    ColumnPtr keep;
    if (!size_per_row) {
        // n >= 0: one size for every group
        const double whole = std::floor(amount);
        const int32_t size = whole >= static_cast<double>(INT32_MAX) ? INT32_MAX
                                                                     : static_cast<int32_t>(whole);
        keep = cudf::binary_operation(rank0->view(), cudf::numeric_scalar<int32_t>(size, true, stream, mr),
                                      cudf::binary_operator::LESS, bool8, stream, mr);
    } else {
        ColumnPtr gs = cudf::cast(group_rows->view(), f64, stream, mr);
        ColumnPtr size;
        if (is_prop) {
            ColumnPtr raw = cudf::binary_operation(gs->view(), cudf::numeric_scalar<double>(amount, true, stream, mr),
                                                   cudf::binary_operator::MUL, f64, stream, mr);
            if (amount >= 0) {
                size = cudf::unary_operation(raw->view(), cudf::unary_operator::FLOOR, stream, mr);
            } else {
                ColumnPtr total = cudf::binary_operation(gs->view(), raw->view(), cudf::binary_operator::ADD, f64, stream, mr);
                size = cudf::unary_operation(total->view(), cudf::unary_operator::CEIL, stream, mr);
            }
        } else {
            ColumnPtr total = cudf::binary_operation(gs->view(), cudf::numeric_scalar<double>(amount, true, stream, mr),
                                                     cudf::binary_operator::ADD, f64, stream, mr);
            size = cudf::unary_operation(total->view(), cudf::unary_operator::CEIL, stream, mr);
        }
        gs.reset();
        ColumnPtr r = cudf::cast(rank0->view(), f64, stream, mr);
        keep = cudf::binary_operation(r->view(), size->view(), cudf::binary_operator::LESS, bool8, stream, mr);
    }
    rank0.reset();

    if (na_rm && has_missing) {
        ColumnPtr present = cudf::is_valid(sorted_key, stream, mr);
        keep = cudf::binary_operation(keep->view(), present->view(), cudf::binary_operator::LOGICAL_AND,
                                      bool8, stream, mr);
    }

    // Survivors, still in (group, rank) order; by= puts the groups in order of
    // their first row (stable, so each group keeps its rank order)
    std::vector<cudf::column_view> carried = {order->view()};
    if (by_first) carried.push_back(group_first->view());
    std::unique_ptr<cudf::table> kept =
        cudf::apply_boolean_mask(cudf::table_view(carried), keep->view(), stream, mr);
    std::vector<ColumnPtr> kept_cols = kept->release();
    ColumnPtr positions = std::move(kept_cols[0]);
    if (by_first && positions->size() > 1) {
        ColumnPtr by_group = cudf::stable_sorted_order(
            cudf::table_view({kept_cols[1]->view()}), {cudf::order::ASCENDING},
            {cudf::null_order::AFTER}, stream, mr);
        positions = compose(positions->view(), by_group->view(), stream, mr);
    }
    return make_gpu_table_xptr(gather_rows(view, positions->view(), stream, mr));
}
