// src/ops_join.cpp
// GPU join implementations using cuDF
#include <Rcpp.h>

#include "cuda_utils.hpp"
#include "gpu_table.hpp"

#if __has_include(<cudf/join/join.hpp>)
#include <cudf/join/join.hpp>
#elif __has_include(<cudf/join.hpp>)
#include <cudf/join.hpp>
#else
#error "cuDF join headers not found (expected cudf/join/join.hpp or cudf/join.hpp)"
#endif
#include <cudf/join/filtered_join.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/unary.hpp>
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

// Wrap a raw join-map device_uvector<size_type> as a column_view (no copy).
cudf::column_view map_as_column_view(const rmm::device_uvector<cudf::size_type>& map) {
    cudf::device_span<cudf::size_type const> span(map.data(), map.size());
    return cudf::column_view(span);
}

// -----------------------------------------------------------------------
// Device-side join-map sanitization (Phase 7 J4 pipeline restructure).
//
// cuDF's join primitives (left_join/full_join, and this file's own
// right_join/semi/anti assembly below) report an unmatched row via a
// negative sentinel (cudf::JoinNoMatch == numeric_limits<size_type>::min(),
// per cudf/join/join.hpp). The OLD implementation round-tripped the entire
// map to the host, remapped negatives to `nrows` there, and copied it back
// -- two D2H/H2D transfers per join, on the map's full row count. This does
// the equivalent remap entirely on-device via one compute_column() (an AST
// boolean mask: `v < 0 || v >= nrows`) and one copy_if_else() (replace
// flagged entries with the literal `nrows`), with zero host round-trips.
//
// The `>= nrows` half of the check is a real (if latent) gap the old code
// never had: it only ever checked `v < 0`. cudf's own join primitives never
// emit an out-of-range *positive* sentinel today, but gpu_right_join()
// below deliberately constructs its assembled map using `nrows` itself as
// the sentinel for unmatched-right rows (see there for why) -- so any
// consumer of a join map, this function included, needs to treat both
// "negative" and "already exactly nrows" (or beyond) as the canonical
// out-of-bounds marker uniformly. After this function, every join map in
// the pipeline uses exactly one canonical sentinel value (`nrows`), which
// both (a) makes cudf::gather()'s out_of_bounds_policy::NULLIFY behave
// correctly and (b) sorts strictly after every real index [0, nrows) in the
// ascending stable sort build_join_result() performs -- the mechanism
// gpu_right_join() relies on to reproduce dplyr's row-order contract.
std::unique_ptr<cudf::column> sanitize_join_map(
    const cudf::column_view& map_col,
    cudf::size_type nrows,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {

    cudf::numeric_scalar<cudf::size_type> zero_scalar(0, true, stream);
    cudf::numeric_scalar<cudf::size_type> nrows_scalar(nrows, true, stream);

    cudf::ast::tree tree;
    auto const& col_ref  = tree.emplace<cudf::ast::column_reference>(0);
    auto const& lit_zero  = tree.emplace<cudf::ast::literal>(zero_scalar);
    auto const& lit_nrows = tree.emplace<cudf::ast::literal>(nrows_scalar);
    auto const& is_neg    = tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LESS, col_ref, lit_zero);
    auto const& is_oob    = tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::GREATER_EQUAL, col_ref, lit_nrows);
    auto const& invalid   = tree.emplace<cudf::ast::operation>(
        cudf::ast::ast_operator::LOGICAL_OR, is_neg, is_oob);

    cudf::table_view single_col_tbl({map_col});
    auto mask = cudf::compute_column(single_col_tbl, invalid, stream, mr);

    return cudf::copy_if_else(nrows_scalar, map_col, mask->view(), stream, mr);
}

// -----------------------------------------------------------------------
// join_map_stats(): four "first offender" diagnostics over an already-
// sanitized pair of join maps (see sanitize_join_map() above -- both maps
// use exactly one canonical sentinel value, `n_left`/`n_right`
// respectively, for "no match").
//
//   - left_multi_first / right_multi_first: the smallest left/right row
//     index that appears more than once among that map's MATCHED (i.e.
//     non-sentinel) entries -- the first row whose join key had more than
//     one match on the other side. -1 if no such row exists.
//   - left_unmatched_first / right_unmatched_first: the smallest left/right
//     row index in [0, n_left)/[0, n_right) that never appears among that
//     map's matched entries at all -- the first row with no match on the
//     other side. -1 if every row matched at least once.
//
// These are dormant in Phase 7 J4 (nothing consumes `diag` yet --
// check_join_cardinality()'s multiple=/unmatched=/relationship= semantics
// land in J5), but the computation itself is real: every reduction below
// only ever transfers a single scalar to the host (`scalar::value()`), so
// this stays cheap and never round-trips a full map to host memory --
// consistent with the "single conditional D2H" restructure goal (the
// removed host round-trip was sanitize_join_map()'s old job, not this
// function's; see the block comment above).
Rcpp::List join_map_stats(
    const cudf::column_view& left_map,
    const cudf::column_view& right_map,
    cudf::size_type n_left,
    cudf::size_type n_right,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {

    // Smallest value that appears more than once in `map_col`'s entries
    // that are NOT equal to `sentinel` (i.e. among real matches only).
    // -1 if there is no such duplicated value.
    auto first_multi = [&](const cudf::column_view& map_col,
                           cudf::size_type sentinel) -> int {
        cudf::numeric_scalar<cudf::size_type> sentinel_scalar(sentinel, true, stream);
        auto valid_mask = cudf::binary_operation(
            map_col, sentinel_scalar, cudf::binary_operator::NOT_EQUAL,
            cudf::data_type(cudf::type_id::BOOL8), stream, mr);

        cudf::table_view map_tbl({map_col});
        auto valid_tbl = cudf::apply_boolean_mask(map_tbl, valid_mask->view(), stream, mr);
        auto valid_col = valid_tbl->view().column(0);
        if (valid_col.size() == 0) return -1;

        cudf::groupby::groupby gb(cudf::table_view({valid_col}));
        std::vector<cudf::groupby::aggregation_request> requests(1);
        requests[0].values = valid_col;
        requests[0].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>());
        auto agg_result = gb.aggregate(requests, stream, mr);
        auto& keys_tbl = agg_result.first;
        auto& counts_col = agg_result.second[0].results[0];

        auto counts_i32 = cudf::cast(
            counts_col->view(), cudf::data_type(cudf::type_id::INT32), stream, mr);
        cudf::numeric_scalar<cudf::size_type> one_scalar(1, true, stream);
        auto dup_mask = cudf::binary_operation(
            counts_i32->view(), one_scalar, cudf::binary_operator::GREATER,
            cudf::data_type(cudf::type_id::BOOL8), stream, mr);

        cudf::table_view keys_view({keys_tbl->view().column(0)});
        auto dup_keys_tbl = cudf::apply_boolean_mask(keys_view, dup_mask->view(), stream, mr);
        auto dup_keys = dup_keys_tbl->view().column(0);
        if (dup_keys.size() == 0) return -1;

        auto min_scalar = cudf::reduce(
            dup_keys, *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type(cudf::type_id::INT32), stream, mr);
        if (!min_scalar->is_valid(stream)) return -1;
        return static_cast<cudf::numeric_scalar<cudf::size_type> const&>(*min_scalar)
            .value(stream);
    };

    // Smallest value in [0, n) that never appears among `map_col`'s
    // non-sentinel entries. -1 if every value in [0, n) is matched (or
    // n == 0).
    auto first_unmatched = [&](const cudf::column_view& map_col,
                               cudf::size_type sentinel,
                               cudf::size_type n) -> int {
        if (n == 0) return -1;

        cudf::numeric_scalar<cudf::size_type> sentinel_scalar(sentinel, true, stream);
        auto valid_mask = cudf::binary_operation(
            map_col, sentinel_scalar, cudf::binary_operator::NOT_EQUAL,
            cudf::data_type(cudf::type_id::BOOL8), stream, mr);
        cudf::table_view map_tbl({map_col});
        auto valid_tbl = cudf::apply_boolean_mask(map_tbl, valid_mask->view(), stream, mr);
        auto valid_col = valid_tbl->view().column(0);

        cudf::numeric_scalar<cudf::size_type> zero_scalar(0, true, stream);
        auto seq = cudf::sequence(n, zero_scalar, stream, mr);

        cudf::table_view build_tbl({valid_col});
        cudf::filtered_join joiner(build_tbl, cudf::null_equality::EQUAL,
                                   cudf::set_as_build_table::RIGHT, stream);
        cudf::table_view probe_tbl({seq->view()});
        auto unmatched_idx = joiner.anti_join(probe_tbl, stream, mr);
        if (unmatched_idx->size() == 0) return -1;

        cudf::column_view idx_view(cudf::device_span<cudf::size_type const>(
            unmatched_idx->data(), unmatched_idx->size()));
        auto min_scalar = cudf::reduce(
            idx_view, *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type(cudf::type_id::INT32), stream, mr);
        if (!min_scalar->is_valid(stream)) return -1;
        return static_cast<cudf::numeric_scalar<cudf::size_type> const&>(*min_scalar)
            .value(stream);
    };

    int left_multi_first = first_multi(left_map, n_left);
    int right_multi_first = first_multi(right_map, n_right);
    int left_unmatched_first = first_unmatched(left_map, n_left, n_left);
    int right_unmatched_first = first_unmatched(right_map, n_right, n_right);

    return Rcpp::List::create(
        Rcpp::Named("left_multi_first") = left_multi_first,
        Rcpp::Named("right_multi_first") = right_multi_first,
        Rcpp::Named("left_unmatched_first") = left_unmatched_first,
        Rcpp::Named("right_unmatched_first") = right_unmatched_first
    );
}

struct JoinBuildResult {
    std::unique_ptr<cudf::table> table;
    Rcpp::List diag;
};

// The six-step join-map pipeline (Phase 7 J4, scratchpad/phase7_joins_design.md
// section 1.3): device sanitize (incl. the >= nrows sentinel gap) -> stable
// sort (this IS dplyr's row-order contract) -> gather/coalesce, plus a
// join_map_stats() diagnostics hook alongside the sanitize step. Applied
// uniformly to left/inner/full/right -- semi/anti/cross never call this
// (they have no multiple=/unmatched=/relationship= in dplyr's own
// signatures, so no diagnostics are needed there).
JoinBuildResult build_join_result(
    const cudf::table_view& left_view,
    const cudf::table_view& right_view,
    const cudf::column_view& left_map_in,
    const cudf::column_view& right_map_in,
    const std::vector<cudf::size_type>& right_keep_cols,
    const std::vector<cudf::size_type>& left_key_cols = {},
    const std::vector<cudf::size_type>& right_key_cols = {}) {

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    // Step 1: device-side sanitize (replaces the old host round-trip;
    // fixes the latent >= nrows sentinel gap). Every join map downstream of
    // this point uses exactly one canonical sentinel value per side
    // (left_view.num_rows() / right_view.num_rows()).
    auto left_map_col = sanitize_join_map(left_map_in, left_view.num_rows(), stream, mr);
    auto right_map_col = sanitize_join_map(right_map_in, right_view.num_rows(), stream, mr);

    // Step 2 (dormant hook until J5): first-offender diagnostics. Computed
    // from the already-sanitized, still-on-device maps -- only ever
    // transfers individual scalars to host (see join_map_stats()'s own
    // comment), never the full map.
    Rcpp::List diag = join_map_stats(left_map_col->view(), right_map_col->view(),
                                     left_view.num_rows(), right_view.num_rows(),
                                     stream, mr);

    cudf::column_view left_map_view = left_map_col->view();
    cudf::column_view right_map_view = right_map_col->view();

    // Step 3: stable sort by (left_map, right_map) -- KEEP, this IS the
    // dplyr row-order contract (left/inner/full: left-row order, right
    // matches in right-index order; right: x-matched order then unmatched-y
    // tail, since gpu_right_join() below assembles its map so unmatched-y
    // rows carry the sentinel n_left as their left_map value, which sorts
    // strictly after every real left index).
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

    // Step 4/6: gather + key coalescing.
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

    // Join-key coalescing (scratchpad/unification_design.md Part C, P1):
    // right-only rows (full_join) or unmatched-right rows (right_join) have
    // no left source row, so left_map points out of bounds and
    // left_gathered's key columns come back NULL for them, even though the
    // right table's own key columns (which this function's caller has NOT
    // included in right_keep_cols, since keep = FALSE drops them) hold the
    // real values for those exact rows. Gather the right table's raw key
    // columns via the same right_map (before they were dropped) and
    // coalesce them into the left-gathered key columns so those rows get
    // the key filled in from the right side, matching dplyr's full_join()/
    // right_join() contract. Only full_join()/right_join() pass non-empty
    // left_key_cols/right_key_cols; left/inner joins are unaffected (and
    // for those join types every left-gathered key value is already
    // non-null, so this would be a no-op anyway).
    std::vector<std::unique_ptr<cudf::column>> coalesced_keys;
    if (!left_key_cols.empty()) {
        auto right_key_view = select_table_view(right_view, right_key_cols);
        auto right_key_gathered = cudf::gather(
            right_key_view, right_map_view,
            cudf::out_of_bounds_policy::NULLIFY,
            stream, mr
        );
        auto right_key_cols_vec = right_key_gathered->release();
        coalesced_keys.reserve(left_key_cols.size());
        for (size_t i = 0; i < left_key_cols.size(); ++i) {
            coalesced_keys.push_back(cudf::replace_nulls(
                left_gathered->view().column(left_key_cols[i]),
                right_key_cols_vec[i]->view(),
                stream, mr));
        }
    }

    auto left_cols = left_gathered->release();
    auto right_cols = right_gathered->release();

    for (size_t i = 0; i < left_key_cols.size(); ++i) {
        left_cols[left_key_cols[i]] = std::move(coalesced_keys[i]);
    }

    std::vector<std::unique_ptr<cudf::column>> result_cols;
    result_cols.reserve(left_cols.size() + right_cols.size());

    for (auto& col : left_cols) {
        result_cols.push_back(std::move(col));
    }
    for (auto& col : right_cols) {
        result_cols.push_back(std::move(col));
    }

    return JoinBuildResult{
        std::make_unique<cudf::table>(std::move(result_cols)),
        diag
    };
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

SEXP wrap_join_build_result(JoinBuildResult&& build) {
    return Rcpp::List::create(
        Rcpp::Named("ptr") = cuplyr::make_gpu_table_xptr(std::move(build.table)),
        Rcpp::Named("diag") = build.diag
    );
}

} // namespace

// [[Rcpp::export]]
SEXP gpu_left_join(SEXP xptr_left,
                   SEXP xptr_right,
                   Rcpp::IntegerVector left_key_cols,
                   Rcpp::IntegerVector right_key_cols,
                   Rcpp::IntegerVector right_drop_cols,
                   bool nulls_equal = true) {
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

    auto null_equality = nulls_equal ? cudf::null_equality::EQUAL
                                     : cudf::null_equality::UNEQUAL;
    auto [left_map, right_map] = cudf::left_join(
        left_key_view, right_key_view, null_equality);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);
    auto build = build_join_result(left_view, right_view,
                                   map_as_column_view(*left_map),
                                   map_as_column_view(*right_map),
                                   right_keep);

    return wrap_join_build_result(std::move(build));
}

// [[Rcpp::export]]
SEXP gpu_inner_join(SEXP xptr_left,
                    SEXP xptr_right,
                    Rcpp::IntegerVector left_key_cols,
                    Rcpp::IntegerVector right_key_cols,
                    Rcpp::IntegerVector right_drop_cols,
                    bool nulls_equal = true) {
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

    auto null_equality = nulls_equal ? cudf::null_equality::EQUAL
                                     : cudf::null_equality::UNEQUAL;
    auto [left_map, right_map] = cudf::inner_join(
        left_key_view, right_key_view, null_equality);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);
    auto build = build_join_result(left_view, right_view,
                                   map_as_column_view(*left_map),
                                   map_as_column_view(*right_map),
                                   right_keep);

    return wrap_join_build_result(std::move(build));
}

// [[Rcpp::export]]
SEXP gpu_full_join(SEXP xptr_left,
                   SEXP xptr_right,
                   Rcpp::IntegerVector left_key_cols,
                   Rcpp::IntegerVector right_key_cols,
                   Rcpp::IntegerVector right_drop_cols,
                   bool nulls_equal = true) {
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

    auto null_equality = nulls_equal ? cudf::null_equality::EQUAL
                                     : cudf::null_equality::UNEQUAL;
    auto [left_map, right_map] = cudf::full_join(
        left_key_view, right_key_view, null_equality);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);

    // Only coalesce when the right table's key columns were actually dropped
    // (i.e. `keep = FALSE`, dplyr's default). With `keep = TRUE` both tables'
    // raw key columns survive uncombined in the output, and the left copy
    // must stay exactly as gathered (NA for right-only rows) -- coalescing
    // it would silently corrupt the `keep = TRUE` contract.
    std::vector<cudf::size_type> coalesce_left_keys;
    std::vector<cudf::size_type> coalesce_right_keys;
    if (!right_drop.empty()) {
        coalesce_left_keys = left_keys;
        coalesce_right_keys = right_keys;
    }

    auto build = build_join_result(left_view, right_view,
                                   map_as_column_view(*left_map),
                                   map_as_column_view(*right_map),
                                   right_keep, coalesce_left_keys, coalesce_right_keys);

    return wrap_join_build_result(std::move(build));
}

// [[Rcpp::export]]
SEXP gpu_right_join(SEXP xptr_left,
                    SEXP xptr_right,
                    Rcpp::IntegerVector left_key_cols,
                    Rcpp::IntegerVector right_key_cols,
                    Rcpp::IntegerVector right_drop_cols,
                    bool nulls_equal = true) {
    using namespace cuplyr;

    // Native right_join (Phase 7 J4 -- see scratchpad/phase7_joins_design.md
    // section 2). The OLD implementation ran a swapped left_join(y, x) and
    // reordered columns after the fact, which put unmatched-y rows in
    // *y's own row order* (cudf::left_join's left-table-order contract,
    // applied to the swapped call where y played the "left" role). dplyr's
    // actual right_join() contract is different: x-matched rows in
    // x('s left_join-style) order FIRST, then unmatched-y rows appended
    // LAST -- i.e. the same shape as build_join_result()'s existing stable
    // sort already produces for left_join, just with the roles of "which
    // side always survives" swapped.
    //
    // This reproduces that contract directly: cudf::inner_join(x, y) gives
    // every MATCHED (x, y) index pair (unspecified order -- the stable sort
    // below fixes that). Every remaining y row (no x match at all) is found
    // via a filtered_join anti-join built from x's keys and probed with y's
    // keys, which returns Y-side (probe-side) indices per the same
    // build/probe contract gpu_semi_anti_join() above already established --
    // here the build/probe roles are just swapped (x is the build/filter
    // side) so the returned indices are into y instead of x. Those
    // unmatched-y rows are assigned the sentinel `n_left` (one past x's
    // last valid row index) as their LEFT map value: build_join_result()'s
    // ascending stable sort on (left_map, right_map) then places every real
    // left index [0, n_left) -- i.e. every matched row, in x order -- before
    // the sentinel-tagged unmatched-y rows, which is exactly dplyr's
    // x-matched-then-unmatched-y-tail contract. (Empirically verified: see
    // tests/testthat/test-dplyr-join.R's right_join order-pin tests.)
    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    auto left_keys = to_index_vec(left_key_cols);
    auto right_keys = to_index_vec(right_key_cols);
    auto right_drop = to_index_vec(right_drop_cols);

    auto left_key_view = select_table_view(left_view, left_keys);
    auto right_key_view = select_table_view(right_view, right_keys);

    auto null_equality = nulls_equal ? cudf::null_equality::EQUAL
                                     : cudf::null_equality::UNEQUAL;

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    auto [inner_left_map, inner_right_map] = cudf::inner_join(
        left_key_view, right_key_view, null_equality);

    // Build the hash table from x's keys, probe with y's keys, so the
    // returned indices are PROBE-side (y) indices with no match in the
    // build (x) table -- the mirror image of gpu_semi_anti_join()'s
    // build=right/probe=left arrangement above (there, build=y/probe=x
    // returns x-side unmatched/matched indices; here it's flipped).
    //
    // `reuse_tbl` is passed RIGHT here even though x (this call's build
    // table) is NOT this join's right-hand table -- verified empirically
    // (this cudf build's filtered_join.cu) that `set_as_build_table::LEFT`
    // is simply unimplemented ("Left table reuse is yet to be implemented.
    // Filtered join requires the right table to be the build table"), so
    // RIGHT is the only supported value regardless of which real table is
    // passed as `build`. It only selects the underlying hash structure
    // (`cuco::static_set`, a plain membership set -- exactly what an
    // anti-join's "is this key present at all" check needs); it does not
    // change which table is `build` vs which is the `probe` argument to
    // anti_join() below.
    cudf::filtered_join x_filter(left_key_view, null_equality,
                                cudf::set_as_build_table::RIGHT, stream);
    auto unmatched_y = x_filter.anti_join(right_key_view, stream, mr);

    cudf::size_type n_left = left_view.num_rows();

    cudf::column_view inner_left_view = map_as_column_view(*inner_left_map);
    cudf::column_view inner_right_view = map_as_column_view(*inner_right_map);
    cudf::column_view unmatched_y_view = map_as_column_view(*unmatched_y);

    // Sentinel column: n_left, repeated once per unmatched-y row. This is
    // deliberately NOT a negative JoinNoMatch-style sentinel -- it needs to
    // sort strictly after every real left index [0, n_left) in the
    // ascending stable sort build_join_result() performs, and `n_left`
    // itself already is (and is also the same canonical sentinel
    // sanitize_join_map() would have produced for a genuine negative
    // sentinel, so this assembled map is already in "sanitized" form for
    // the left side -- sanitize_join_map() is still applied uniformly
    // below, but is a no-op for these particular entries).
    auto sentinel_col = cudf::make_numeric_column(
        cudf::data_type(cudf::type_to_id<cudf::size_type>()),
        unmatched_y->size(),
        cudf::mask_state::UNALLOCATED, stream, mr);
    if (unmatched_y->size() > 0) {
        cudf::numeric_scalar<cudf::size_type> sentinel_scalar(n_left, true, stream);
        cudf::mutable_column_view sentinel_mview = sentinel_col->mutable_view();
        cudf::fill_in_place(sentinel_mview, 0, unmatched_y->size(), sentinel_scalar, stream);
    }

    auto full_left_map = cudf::concatenate(
        std::vector<cudf::column_view>{inner_left_view, sentinel_col->view()}, stream, mr);
    auto full_right_map = cudf::concatenate(
        std::vector<cudf::column_view>{inner_right_view, unmatched_y_view}, stream, mr);

    auto right_keep = compute_right_keep_cols(right_view, right_drop);

    // keep = FALSE key coalescing: right_join's single output key column
    // displays under x's name (dplyr's by = c(x_col = y_col) convention,
    // handled entirely R-side by build_join_output_info()) but its VALUES
    // must come from y, since right_join is driven by y (every y row
    // survives; the rows needing coalescing here are exactly the
    // unmatched-y ones, whose left_map is the sentinel above -- the same
    // shape of gap full_join()'s coalescing already handles for its
    // right-only rows). Reuse that exact machinery: pass the same
    // left_keys/right_keys full_join() does when keep = FALSE.
    std::vector<cudf::size_type> coalesce_left_keys;
    std::vector<cudf::size_type> coalesce_right_keys;
    if (!right_drop.empty()) {
        coalesce_left_keys = left_keys;
        coalesce_right_keys = right_keys;
    }

    auto build = build_join_result(left_view, right_view,
                                   full_left_map->view(), full_right_map->view(),
                                   right_keep, coalesce_left_keys, coalesce_right_keys);

    return wrap_join_build_result(std::move(build));
}

// [[Rcpp::export]]
SEXP gpu_semi_anti_join(SEXP xptr_left,
                        SEXP xptr_right,
                        Rcpp::IntegerVector left_key_cols,
                        Rcpp::IntegerVector right_key_cols,
                        bool is_anti,
                        bool nulls_equal = true) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    auto left_keys = to_index_vec(left_key_cols);
    auto right_keys = to_index_vec(right_key_cols);

    auto left_key_view = select_table_view(left_view, left_keys);
    auto right_key_view = select_table_view(right_view, right_keys);

    auto null_equality = nulls_equal ? cudf::null_equality::EQUAL
                                     : cudf::null_equality::UNEQUAL;

    auto stream = cudf::get_default_stream();
    auto mr = rmm::mr::get_current_device_resource();

    // Build the hash table from the right/filter table (scratchpad/
    // phase7_joins_design.md J2's verified fact); probe with the left table.
    // Returned indices are UNORDERED left-table row indices with no
    // duplication -- even when multiple right rows match the same left key,
    // a left row appears at most once.
    cudf::filtered_join joiner(right_key_view, null_equality,
                               cudf::set_as_build_table::RIGHT, stream);

    auto indices = is_anti ? joiner.anti_join(left_key_view, stream, mr)
                           : joiner.semi_join(left_key_view, stream, mr);

    // Sort the index vector ascending before gathering, to reproduce dplyr's
    // left-row-order contract (filtered_join's result order is unspecified).
    cudf::device_span<cudf::size_type const> idx_span(indices->data(), indices->size());
    cudf::column_view idx_col_view(idx_span);
    cudf::table_view idx_tbl({idx_col_view});
    auto sorted_idx_tbl = cudf::sort(idx_tbl, {cudf::order::ASCENDING},
                                     {}, stream, mr);
    auto sorted_idx_view = sorted_idx_tbl->view().column(0);

    auto result = cudf::gather(left_view, sorted_idx_view,
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream, mr);

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_cross_join(SEXP xptr_left, SEXP xptr_right) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> left_ptr(xptr_left);
    Rcpp::XPtr<GpuTablePtr> right_ptr(xptr_right);

    cudf::table_view left_view = get_table_view(left_ptr);
    cudf::table_view right_view = get_table_view(right_ptr);

    // 0-column guard: cudf::cross_join() throws cudf::logic_error if either
    // side has 0 columns (scratchpad/phase7_joins_design.md J3 spec, verified
    // against the cudf/join/join.hpp doc comment). The R side
    // (cross_join.tbl_gpu(), R/join.R) already stops with a clearer,
    // dplyr-flavored message before this is ever called; this is a
    // defensive backstop against any other caller reaching this entry point
    // directly with a malformed/empty table.
    if (left_view.num_columns() == 0 || right_view.num_columns() == 0) {
        Rcpp::stop("cross_join() requires both tables to have at least one column.");
    }

    // No keys, no drops: cudf::cross_join() returns the full cartesian
    // product with left's columns first (in their original order) followed
    // by right's columns (in their original order) -- exactly the column
    // order build_join_output_info() produces for an empty spec with
    // keep = TRUE (every common name suffixed, nothing dropped), so no
    // further reordering/selecting is needed here. 0-row inputs on either
    // side are handled natively by cudf (empty cartesian product, no error);
    // only the 0-column case above needs a guard.
    auto result = cudf::cross_join(left_view, right_view);

    return make_gpu_table_xptr(std::move(result));
}
