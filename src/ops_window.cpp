// src/ops_window.cpp
//
// GPU-native window functions.
//
// Phase 5 task W2 implemented the "shift" family (lag()/lead()) and the
// "scan" family (cumsum()/cummax()/cummin()/cumprod(), plus bare
// row_number()'s no-order COUNT_ALL form).
//
// Phase 5 task W4 (this file, extended -- see R/lower.R::
// validate_window_spec(), which now accepts every window kind) adds:
//   - the "rank" family: row_number(x)/min_rank()/dense_rank()/
//     percent_rank()/cume_dist(), via groupby SCAN + make_rank_aggregation()
//     (still one gb.scan() call, alongside the W2 scan family -- RANK only
//     works with scan algorithms). Section 2.2's constraint applies: a rank
//     spec's value column must equal the frame's sole order column (checked
//     both R-side, when the spec is built, and defensively here).
//   - the "agg" family: mean()/sum()/min()/max()/n()/sd()/var() (plain
//     per-group aggregates) and first()/last()/nth() (via
//     make_nth_element_aggregation(), honoring an order_by frame when one
//     is supplied) -- both computed via ONE gb.aggregate() call per frame,
//     then broadcast back out to every row of that group via
//     cudf::repeat(), see section 4's "aggregate + repeat + scatter" note
//     and empirical check E4 (recorded in the W4 commit message).
//
// This is the six-step algorithm from
// scratchpad/phase5_window_design.md section 2.2, implemented EXACTLY:
//
//   1. sort_keys = [group cols] ++ [order cols]; ASCENDING for every group
//      col, per-spec order for order cols; null_order::AFTER for every group
//      col, and (fixed post-W6 -- see the "nulls sort LAST regardless of
//      direction" comment at this step's call site below) null_order::AFTER
//      for an ASCENDING order col but BEFORE for a DESCENDING one, so nulls
//      land last within a group's frame regardless of sort direction,
//      matching real dplyr's own order()/arrange() convention.
//   2. need_sort = !(group_indices.empty() && order_idx.empty()).
//      perm = need_sort ? stable_sorted_order(sort_keys, ...) : nullptr.
//      STABLE is mandatory: cumsum()/lag()/row_number() must follow
//      ORIGINAL row order within a group.
//   3. work = need_sort ? gather(needed_cols, *perm) : needed_cols_view
//      (group + order + value cols only).
//   4. keys = group cols of `work`, or (when there are no real group
//      columns at all) a single constant INT32 zero column. groupby
//      constructed with null_policy::INCLUDE, sorted::YES -- this is
//      what makes cudf's internal key_sort_order() the identity permutation
//      (verified against cudf source, see section 0 of the design doc),
//      so step 5's results come back aligned with `work`'s row order.
//      NEVER construct the groupby with sorted::NO here.
//   5. compute (section 4 of the design doc) -> one result column per
//      spec, aligned with `work`'s row order. cum*'s sticky-NA fix (cudf's
//      own group-scan null semantics are NON-sticky, unlike R's
//      cumsum(c(1,2,NA,4)) == 1,3,NA,NA) is applied here, still in `work`
//      order, before any scatter.
//   6. if (perm) results = scatter(results, perm, results) -- perm[i] is
//      the ORIGINAL row index of the row at sorted position i, so
//      dest[perm[i]] = value_for_original_row_perm[i]; since perm is a
//      full permutation of [0,n), every destination row is written
//      exactly once. `target` (self-scattered here) is only a shape/type
//      template.
//
// A later task (Theo-requested, scratchpad/todo.md "Rolling-window
// benchmark target") adds a sixth family, "rolling":
// roll_mean()/roll_sum()/roll_min()/roll_max()/roll_sd() (NOT
// roll_median() -- cudf 25.12 has no rolling MEDIAN aggregation in this
// environment, see R/lower.R's window_spec_lowerable()), via
// cudf::grouped_rolling_window() (used uniformly whether or not the
// frame is actually grouped, same "one code path via a substituted
// zero-key column" idiom as the families above). See the file-level
// design comment above R/ir.R's ir_parse_roll_call() for the full
// empirical derivation of the min_periods/na.rm scheme this family uses --
// notably, na.rm=FALSE poisoning falls out of a FIXED w-row window's own
// min_periods=w requirement for free (no is-null-indicator masking pass
// needed, unlike the cum*/rank sticky-NA fix), while na.rm=TRUE needs an
// extra rolling COUNT_ALL pass to distinguish "leading incomplete window"
// (always NA) from "internal window with some nulls" (skip them).
//
// Empirical checks recorded in the W2 commit message: E1 (grouped cumsum
// on a shuffled 100k-row table matches dplyr exactly -- validates the
// whole perm/gather/scan/scatter round trip), E2 (grouped row_number()
// with duplicate + NA group keys), E5 (cudf group-scan NA IS non-sticky
// before the fix; the fix reproduces R's sticky semantics), E6 (lag()/
// lead() fill type + NA default). W4 adds E3 (min_rank()/dense_rank()/
// percent_rank()/cume_dist() with ties + NAs match dplyr exactly, incl.
// percent_rank()'s (rank-1)/(n_non_NA-1) denominator and all-NA groups)
// and E4 (gb.aggregate() result row i <-> sorted group i: the repeat()
// broadcast validated against dplyr's own grouped mean() on shuffled data
// with duplicate + NA keys) -- both recorded in the W4 commit message.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "expr_eval.hpp"  // build_scalar(), expr_type_from_str()

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <climits>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

namespace cuplyr {
// Defined in src/ops_groupby.cpp (external linkage, namespace cuplyr):
// reused here so mean()/sum()/min()/max()/n()/sd()/var() window
// aggregations pick EXACTLY summarise()'s own cudf aggregation choice
// (ddof=1 for std/variance, null_policy::INCLUDE for count, ...) --
// R/ast.R::make_aggregation()'s output-type table (reused verbatim by the
// "agg"-kind registry entries in R/ir.R) documents the type side of this
// same contract.
std::unique_ptr<cudf::groupby_aggregation> get_groupby_agg(const std::string& agg_type);
}  // namespace cuplyr

namespace {

// One C++-side resolved spec, mirroring R/lower.R::build_window_cpp_spec()'s
// plain-list shape exactly: list(fn, value_idx, out_type, out_pos, n,
// default_value, default_valid, na_rm).
struct WindowSpec {
    std::string fn;
    bool has_value;
    cudf::size_type value_idx;
    std::string out_type;
    int out_pos;
    bool has_n;
    cudf::size_type n;
    bool default_valid;
    SEXP default_value;
    bool na_rm;
};

WindowSpec parse_window_spec(Rcpp::List spec) {
    WindowSpec s;
    s.fn = Rcpp::as<std::string>(spec["fn"]);

    Rcpp::IntegerVector value_idx_v = spec["value_idx"];
    int value_idx_r = value_idx_v[0];
    s.has_value = !Rcpp::IntegerVector::is_na(value_idx_r);
    s.value_idx = s.has_value ? static_cast<cudf::size_type>(value_idx_r) : -1;

    s.out_type = Rcpp::as<std::string>(spec["out_type"]);
    s.out_pos = Rcpp::as<int>(spec["out_pos"]);

    Rcpp::IntegerVector n_v = spec["n"];
    int n_r = n_v[0];
    s.has_n = !Rcpp::IntegerVector::is_na(n_r);
    s.n = s.has_n ? static_cast<cudf::size_type>(n_r) : 0;

    s.default_valid = Rcpp::as<bool>(spec["default_valid"]);
    s.default_value = spec["default_value"];
    s.na_rm = Rcpp::as<bool>(spec["na_rm"]);
    return s;
}

bool is_cum_scan_fn(const std::string& fn) {
    return fn == "cumsum" || fn == "cummax" || fn == "cummin" || fn == "cumprod";
}

std::unique_ptr<cudf::groupby_scan_aggregation> make_cum_scan_agg(const std::string& fn) {
    if (fn == "cumsum")  return cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
    if (fn == "cummax")  return cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
    if (fn == "cummin")  return cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
    if (fn == "cumprod") return cudf::make_product_aggregation<cudf::groupby_scan_aggregation>();
    Rcpp::stop("gpu_window: internal: unknown scan fn '%s'", fn.c_str());
}

// --- Phase 5 W4: rank family (scan-based, via make_rank_aggregation()) ---

bool is_rank_fn(const std::string& fn) {
    return fn == "row_number" || fn == "min_rank" || fn == "dense_rank" ||
           fn == "percent_rank" || fn == "cume_dist";
}

// dplyr <-> cudf::rank_method/rank_percentage mapping, verified empirically
// against dplyr 1.2.1 (scratchpad/phase5_window_design.md section 4 + the
// W4 commit message's E3 check): row_number(x) is ties="sequential"
// (rank_method::FIRST); min_rank()/dense_rank() are MIN/DENSE; percent_rank()
// is `(min_rank(x)-1)/(n_non_NA-1)` == MIN method + ONE_NORMALIZED;
// cume_dist() is `rank(ties="max")/n_non_NA` == MAX method +
// ZERO_NORMALIZED. All five use null_policy::EXCLUDE (dplyr's own
// `incomplete = "na"` default: an NA input value gets an NA rank).
std::unique_ptr<cudf::groupby_scan_aggregation> make_rank_scan_agg(
        const std::string& fn, cudf::order col_order, cudf::null_order null_prec) {
    if (fn == "row_number") {
        return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::FIRST, col_order, cudf::null_policy::EXCLUDE, null_prec,
            cudf::rank_percentage::NONE);
    }
    if (fn == "min_rank") {
        return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::MIN, col_order, cudf::null_policy::EXCLUDE, null_prec,
            cudf::rank_percentage::NONE);
    }
    if (fn == "dense_rank") {
        return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::DENSE, col_order, cudf::null_policy::EXCLUDE, null_prec,
            cudf::rank_percentage::NONE);
    }
    if (fn == "percent_rank") {
        return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::MIN, col_order, cudf::null_policy::EXCLUDE, null_prec,
            cudf::rank_percentage::ONE_NORMALIZED);
    }
    if (fn == "cume_dist") {
        return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::MAX, col_order, cudf::null_policy::EXCLUDE, null_prec,
            cudf::rank_percentage::ZERO_NORMALIZED);
    }
    Rcpp::stop("gpu_window: internal: unknown rank fn '%s'", fn.c_str());
}

// --- Phase 5 W4: agg family (aggregate + cudf::repeat() broadcast) ---

bool is_agg_fn(const std::string& fn) {
    // Phase 6, task 6.2: median()/n_distinct() join this list, reusing
    // get_groupby_agg() (src/ops_groupby.cpp, genuinely shared source
    // between the summarise() and window-mutate() paths) with NO other
    // change needed here beyond this membership check + (for median())
    // the na.rm=FALSE propagation list a few lines below -- verified
    // empirically (test-dplyr-summarise.R's window-parity oracle tests)
    // against dplyr 1.2.1's own `group_by(g) |> mutate(m = median(x))` /
    // `mutate(nd = n_distinct(x))`. Like every other function in this list,
    // neither supports a na.rm=/na_rm= ARGUMENT in mutate() context (no
    // window "agg" function does) -- median() always gets the na.rm=FALSE
    // whole-group-null propagation below; n_distinct() always counts NA as
    // its own distinct value (get_groupby_agg()'s fixed
    // null_policy::INCLUDE overload for "n_distinct", see that function's
    // own docs). quantile()/first()/last()/nth()/any()/all() are NOT
    // added here: quantile() needs a probability parameter the WindowSpec/
    // registry shape doesn't carry, and any()/all() have no
    // groupby_aggregation instantiation in cudf at all (see
    // combine_any_all()'s docs in src/ops_groupby.cpp) -- both would need
    // real new plumbing, not just a membership check, so neither is
    // attempted this wave (first()/last()/nth() already have their OWN,
    // separate, pre-existing window support below, unrelated to this task).
    return fn == "mean" || fn == "sum" || fn == "min" || fn == "max" ||
           fn == "n" || fn == "sd" || fn == "var" ||
           fn == "median" || fn == "n_distinct";
}

bool is_nth_fn(const std::string& fn) {
    return fn == "first" || fn == "last" || fn == "nth";
}

// window's registry names ("sd"/"var") differ from the cudf-accepted
// names get_groupby_agg() (src/ops_groupby.cpp) expects ("std"/"variance")
// -- R/summarise.R's parse_aggregations() makes the exact same translation
// for summarise()'s own sd()/var(). sum/mean/min/max/n pass through
// unchanged.
std::string window_agg_cudf_name(const std::string& fn) {
    if (fn == "sd")  return "std";
    if (fn == "var") return "variance";
    return fn;
}

// --- Rolling windows (Theo-requested benchmark task, a sixth window
// family alongside shift/rank/scan/agg -- see the file-level design
// comment above ir_parse_roll_call() in R/ir.R for the full empirical
// derivation of the min_periods/na.rm scheme implemented below). ---

bool is_rolling_fn(const std::string& fn) {
    return fn == "roll_mean" || fn == "roll_sum" || fn == "roll_min" ||
           fn == "roll_max" || fn == "roll_sd";
    // "roll_median" is intentionally excluded: cudf has no rolling MEDIAN
    // aggregation in this environment (verified via `nm -DC libcudf.so`,
    // see R/ir.R's comment) -- window_spec_lowerable() (R/lower.R) already
    // keeps it from ever reaching gpu_window() at all, so a "roll_median"
    // spec here would be a cuplyr bug (caught defensively by
    // make_rolling_agg()'s Rcpp::stop() below, never by this predicate
    // returning FALSE for it -- callers dispatch on is_rolling_fn() to
    // decide bucket membership, not to validate lowerability).
}

std::unique_ptr<cudf::rolling_aggregation> make_rolling_agg(const std::string& fn) {
    if (fn == "roll_mean") return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
    if (fn == "roll_sum")  return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
    if (fn == "roll_min")  return cudf::make_min_aggregation<cudf::rolling_aggregation>();
    if (fn == "roll_max")  return cudf::make_max_aggregation<cudf::rolling_aggregation>();
    // ddof = 1 (sample standard deviation): matches R's own sd() and this
    // file's/ops_groupby.cpp's existing "sd"/"std" ddof choice.
    if (fn == "roll_sd")   return cudf::make_std_aggregation<cudf::rolling_aggregation>(1);
    Rcpp::stop("gpu_window: internal: unknown rolling fn '%s'", fn.c_str());
}

void check_row_limit(const cudf::table_view& view) {
    if (view.num_rows() > static_cast<cudf::size_type>(INT32_MAX)) {
        Rcpp::stop("Table has too many rows for window functions (max ~2.1 billion)");
    }
}

} // namespace

// [[Rcpp::export]]
SEXP gpu_window(SEXP xptr, Rcpp::List frames, Rcpp::IntegerVector group_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    check_row_limit(view);

    cudf::size_type nrow = view.num_rows();
    int n_input = view.num_columns();

    for (int i = 0; i < group_indices.size(); ++i) {
        if (group_indices[i] < 0 || group_indices[i] >= n_input) {
            Rcpp::stop("gpu_window: group column index out of bounds: %d", group_indices[i]);
        }
    }

    // Total appended-column count == the max out_pos + 1 across every
    // frame's specs (every spec contributes exactly one output column, at
    // its own out_pos, GLOBAL across the whole node -- see
    // R/lower.R::lower_window()'s doc comment).
    int n_appended = 0;
    for (int f = 0; f < frames.size(); ++f) {
        Rcpp::List frame = frames[f];
        Rcpp::List specs_r = frame["specs"];
        for (int s = 0; s < specs_r.size(); ++s) {
            Rcpp::List spec = specs_r[s];
            int out_pos = Rcpp::as<int>(spec["out_pos"]);
            n_appended = std::max(n_appended, out_pos + 1);
        }
    }

    std::vector<std::unique_ptr<cudf::column>> appended(static_cast<size_t>(n_appended));

    // Constant zero-valued INT32 key column, built lazily (at most once) --
    // only needed when `group_indices` is empty (an ungrouped window node,
    // section 3 of the design doc).
    std::unique_ptr<cudf::column> zero_key_col;
    auto ensure_zero_key = [&]() -> cudf::column_view {
        if (!zero_key_col) {
            cudf::numeric_scalar<int32_t> zero(0, true);
            zero_key_col = cudf::make_column_from_scalar(zero, nrow);
        }
        return zero_key_col->view();
    };

    for (int f = 0; f < frames.size(); ++f) {
        Rcpp::List frame = frames[f];
        Rcpp::IntegerVector order_idx_r = frame["order_idx"];
        Rcpp::LogicalVector order_desc_r = frame["order_desc"];
        Rcpp::List specs_r = frame["specs"];

        std::vector<WindowSpec> specs;
        specs.reserve(static_cast<size_t>(specs_r.size()));
        for (int i = 0; i < specs_r.size(); ++i) {
            specs.push_back(parse_window_spec(specs_r[i]));
        }

        // --- Step 1 (partial): assemble this frame's needed column set:
        // [group cols] ++ [order cols] ++ [unique value cols]. ---
        std::vector<cudf::column_view> group_col_views;
        if (group_indices.size() > 0) {
            for (int i = 0; i < group_indices.size(); ++i) {
                group_col_views.push_back(view.column(group_indices[i]));
            }
        } else {
            group_col_views.push_back(ensure_zero_key());
        }
        cudf::size_type n_group_cols = static_cast<cudf::size_type>(group_col_views.size());

        // Per order-col null precedence: nulls sort LAST regardless of
        // direction (matching real dplyr's own arrange()/order() convention
        // -- confirmed empirically against dplyr 1.2.1, including
        // arrange(desc(x)), which also places NA last, not first). cudf's
        // null_order is a physical-placement convention, not a "null is a
        // large/small value" one: null_order::AFTER means nulls compare as
        // larger than every value, so for ASCENDING that puts them last, but
        // for DESCENDING (largest first) that puts them FIRST instead --
        // achieving "last" in both directions means flipping null_order
        // together with column order (the same fix already applied in
        // src/ops_slice.cpp::gpu_slice_rank() for the ungrouped slice_min()/
        // slice_max() path). Getting this wrong here is silent, not a
        // compile/runtime error: a mismatched null_order still produces a
        // valid-looking permutation, it just physically misplaces the null
        // rows relative to what every downstream computation (the RANK
        // family's presorted scan especially) assumes -- caught empirically
        // while building Phase 5 W7's slice_min()/slice_max() grouped
        // desugar (min_rank(order_col, order_desc = TRUE) on a column with
        // nulls silently produced ranks offset by the group's null count,
        // e.g. 3,4,5 instead of 1,2,3, until this fix).
        std::vector<cudf::column_view> order_col_views;
        std::vector<cudf::order> order_col_orders;
        std::vector<cudf::null_order> order_col_null_precs;
        for (int i = 0; i < order_idx_r.size(); ++i) {
            int idx = order_idx_r[i];
            if (idx < 0 || idx >= n_input) {
                Rcpp::stop("gpu_window: order column index out of bounds: %d", idx);
            }
            order_col_views.push_back(view.column(idx));
            bool desc = (i < order_desc_r.size()) && static_cast<bool>(order_desc_r[i]);
            order_col_orders.push_back(desc ? cudf::order::DESCENDING : cudf::order::ASCENDING);
            order_col_null_precs.push_back(desc ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
        }
        cudf::size_type n_order_cols = static_cast<cudf::size_type>(order_col_views.size());

        // Dedup value columns referenced by this frame's specs: value_idx ->
        // position within needed_views/work_view.
        std::map<cudf::size_type, cudf::size_type> value_pos;
        std::vector<cudf::column_view> value_col_views;
        for (const auto& sp : specs) {
            if (!sp.has_value) continue;
            if (sp.value_idx < 0 || sp.value_idx >= n_input) {
                Rcpp::stop("gpu_window: value column index out of bounds: %d", sp.value_idx);
            }
            if (value_pos.find(sp.value_idx) == value_pos.end()) {
                value_pos[sp.value_idx] = n_group_cols + n_order_cols +
                    static_cast<cudf::size_type>(value_col_views.size());
                value_col_views.push_back(view.column(sp.value_idx));
            }
        }

        std::vector<cudf::column_view> needed_views;
        needed_views.reserve(group_col_views.size() + order_col_views.size() + value_col_views.size());
        needed_views.insert(needed_views.end(), group_col_views.begin(), group_col_views.end());
        needed_views.insert(needed_views.end(), order_col_views.begin(), order_col_views.end());
        needed_views.insert(needed_views.end(), value_col_views.begin(), value_col_views.end());
        cudf::table_view needed_view(needed_views);

        // --- Step 2: need_sort + stable perm. `group_indices.size() == 0`
        // means the NODE itself is ungrouped (no group columns supplied by
        // R at all) -- NOT "one substituted constant group", which is what
        // `n_group_cols`/`group_col_views` already handle above. ---
        bool need_sort = !(group_indices.size() == 0 && n_order_cols == 0);

        std::unique_ptr<cudf::column> perm;
        std::unique_ptr<cudf::table> gathered;
        cudf::table_view work_view;

        if (need_sort) {
            std::vector<cudf::column_view> sort_key_views;
            sort_key_views.insert(sort_key_views.end(), group_col_views.begin(), group_col_views.end());
            sort_key_views.insert(sort_key_views.end(), order_col_views.begin(), order_col_views.end());
            cudf::table_view sort_keys(sort_key_views);

            std::vector<cudf::order> col_orders(static_cast<size_t>(n_group_cols), cudf::order::ASCENDING);
            col_orders.insert(col_orders.end(), order_col_orders.begin(), order_col_orders.end());
            std::vector<cudf::null_order> null_precs(
                static_cast<size_t>(n_group_cols), cudf::null_order::AFTER);
            null_precs.insert(null_precs.end(), order_col_null_precs.begin(), order_col_null_precs.end());

            // STABLE is mandatory: cumsum()/lag()/row_number() must follow
            // ORIGINAL row order within a group.
            perm = cudf::stable_sorted_order(sort_keys, col_orders, null_precs);
            gathered = cudf::gather(needed_view, perm->view(), cudf::out_of_bounds_policy::DONT_CHECK);
            work_view = gathered->view();
        } else {
            work_view = needed_view;
        }

        // --- Step 4: groupby keys are `work`'s first n_group_cols columns
        // (the real group cols, or the constant zero column substituted
        // above). null_policy::INCLUDE + sorted::YES: NEVER sorted::NO here
        // (see the file-level comment / section 0 of the design doc). ---
        std::vector<cudf::column_view> keys_col_views;
        for (cudf::size_type i = 0; i < n_group_cols; ++i) {
            keys_col_views.push_back(work_view.column(i));
        }
        cudf::table_view keys_view(keys_col_views);

        std::vector<cudf::order> key_col_order(static_cast<size_t>(n_group_cols), cudf::order::ASCENDING);
        std::vector<cudf::null_order> key_null_prec(static_cast<size_t>(n_group_cols), cudf::null_order::AFTER);
        cudf::groupby::groupby gb(keys_view, cudf::null_policy::INCLUDE, cudf::sorted::YES,
                                  key_col_order, key_null_prec);

        // --- Step 5: compute, per spec, aligned with `work_view`'s row
        // order. Split into: the scan family (row_number()'s COUNT_ALL form
        // + cumsum/cummax/cummin/cumprod), the rank family (row_number(x)/
        // min_rank/dense_rank/percent_rank/cume_dist -- ALSO scan-based, see
        // the file-level comment), the shift family (lag()/lead()), and the
        // agg family (mean/sum/min/max/n/sd/var/first/last/nth). ---
        std::vector<int> scan_spec_indices;
        std::vector<int> rank_spec_indices;
        std::vector<int> shift_spec_indices;
        std::vector<int> agg_spec_indices;
        std::vector<int> rolling_spec_indices;
        for (size_t i = 0; i < specs.size(); ++i) {
            const WindowSpec& sp = specs[i];
            const std::string& fn = sp.fn;
            if (fn == "row_number" && !sp.has_value) {
                scan_spec_indices.push_back(static_cast<int>(i));
            } else if (is_cum_scan_fn(fn)) {
                scan_spec_indices.push_back(static_cast<int>(i));
            } else if (is_rank_fn(fn)) {
                // fn == "row_number" with a value column lands here (ranked
                // row_number(x)), not in scan_spec_indices above.
                rank_spec_indices.push_back(static_cast<int>(i));
            } else if (fn == "lag" || fn == "lead") {
                shift_spec_indices.push_back(static_cast<int>(i));
            } else if (is_agg_fn(fn) || is_nth_fn(fn)) {
                agg_spec_indices.push_back(static_cast<int>(i));
            } else if (is_rolling_fn(fn)) {
                rolling_spec_indices.push_back(static_cast<int>(i));
            } else {
                Rcpp::stop("gpu_window: unsupported window fn '%s'", fn.c_str());
            }
        }

        // Section 2.2's RANK-spec constraint, enforced defensively in C++
        // (R/window.R's extract_one_window_call() already enforces it when
        // building the spec): the presorted RANK scan branch assumes the
        // order-by column is already sorted within each group, which step 1
        // only guarantees when the ranked column IS this frame's sole order
        // column.
        for (int si : rank_spec_indices) {
            const WindowSpec& sp = specs[static_cast<size_t>(si)];
            if (order_idx_r.size() != 1 || sp.value_idx != order_idx_r[0]) {
                Rcpp::stop("gpu_window: internal: rank spec's value column must equal "
                           "the frame's sole order column (fn = '%s')", sp.fn.c_str());
            }
        }

        // Results, in `specs`/`work_view` order (NOT yet scattered back).
        std::vector<std::unique_ptr<cudf::column>> spec_results(specs.size());

        if (!scan_spec_indices.empty() || !rank_spec_indices.empty()) {
            struct ScanJob {
                int spec_index;
                int primary_request_idx;
                int indicator_request_idx;  // -1 if no sticky-NA fix needed
            };

            std::vector<cudf::groupby::scan_request> requests;
            std::vector<ScanJob> scan_jobs;
            // Sticky-NA fix (section 4 of the design doc, cum* numeric
            // family only): the indicator column must outlive gb.scan().
            std::vector<std::unique_ptr<cudf::column>> indicator_keepalive;

            for (int si : scan_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                ScanJob job{si, -1, -1};

                cudf::groupby::scan_request req;
                if (sp.fn == "row_number") {
                    // COUNT_ALL doesn't read `values` at all -- any column
                    // of the right row count works; the frame's own key
                    // column (position 0 of `work_view`) is always present.
                    req.values = keys_view.column(0);
                    req.aggregations.push_back(
                        cudf::make_count_aggregation<cudf::groupby_scan_aggregation>(
                            cudf::null_policy::INCLUDE));
                    job.primary_request_idx = static_cast<int>(requests.size());
                    requests.push_back(std::move(req));
                } else {
                    cudf::size_type pos = value_pos.at(sp.value_idx);
                    cudf::column_view value_view = work_view.column(pos);
                    req.values = value_view;
                    req.aggregations.push_back(make_cum_scan_agg(sp.fn));
                    job.primary_request_idx = static_cast<int>(requests.size());
                    requests.push_back(std::move(req));

                    if (value_view.has_nulls()) {
                        auto indicator_bool = cudf::is_null(value_view);
                        auto indicator_i8 = cudf::cast(indicator_bool->view(),
                                                        cudf::data_type{cudf::type_id::INT8});

                        cudf::groupby::scan_request ind_req;
                        ind_req.values = indicator_i8->view();
                        ind_req.aggregations.push_back(
                            cudf::make_max_aggregation<cudf::groupby_scan_aggregation>());
                        job.indicator_request_idx = static_cast<int>(requests.size());
                        requests.push_back(std::move(ind_req));

                        indicator_keepalive.push_back(std::move(indicator_i8));
                    }
                }
                scan_jobs.push_back(job);
            }

            for (int si : rank_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                ScanJob job{si, -1, -1};

                // The ranked column is this frame's sole order column (just
                // checked above), so its cudf::order/null_order are exactly
                // `order_col_orders[0]` / `order_col_null_precs[0]` -- the
                // SAME (possibly direction-flipped, nulls-last) null
                // precedence step 1's physical sort used for this column.
                cudf::size_type pos = value_pos.at(sp.value_idx);
                cudf::column_view value_view = work_view.column(pos);

                cudf::groupby::scan_request req;
                req.values = value_view;
                req.aggregations.push_back(
                    make_rank_scan_agg(sp.fn, order_col_orders[0], order_col_null_precs[0]));
                job.primary_request_idx = static_cast<int>(requests.size());
                requests.push_back(std::move(req));

                scan_jobs.push_back(job);
            }

            auto scan_pair = gb.scan(requests);
            auto& scan_results = scan_pair.second;

            for (const auto& job : scan_jobs) {
                std::unique_ptr<cudf::column> result =
                    std::move(scan_results[static_cast<size_t>(job.primary_request_idx)].results[0]);

                if (job.indicator_request_idx >= 0) {
                    std::unique_ptr<cudf::column>& indicator_max =
                        scan_results[static_cast<size_t>(job.indicator_request_idx)].results[0];

                    cudf::numeric_scalar<int8_t> zero_i8(0, true);
                    auto not_seen = cudf::binary_operation(
                        indicator_max->view(), zero_i8, cudf::binary_operator::EQUAL,
                        cudf::data_type{cudf::type_id::BOOL8});

                    auto mask_pair = cudf::bools_to_mask(not_seen->view());
                    result->set_null_mask(std::move(*mask_pair.first), mask_pair.second);
                }

                spec_results[static_cast<size_t>(job.spec_index)] = std::move(result);
            }
        }

        // percent_rank()'s (rank-1)/(n_valid-1) denominator: cudf's own
        // ONE_NORMALIZED implementation returns 0 (not NaN) for a group
        // whose valid (non-null) value count is exactly 1 (0/0 clamped to
        // 0 internally), but dplyr returns NaN for every valid row of such
        // a singleton-valid group (empirically verified against dplyr
        // 1.2.1, part of the W4 commit's E3 check: `percent_rank(c(5))` is
        // `NaN`, not `0`; a NULL row in that same group correctly stays NA
        // either way and is left untouched here via the `orig_valid` mask
        // below). Patched via a small side gb.aggregate() call (the
        // group's valid-value count + its row count, broadcast via
        // cudf::repeat() exactly like the agg family below), run only when
        // this frame actually has a percent_rank spec.
        for (int si : rank_spec_indices) {
            if (specs[static_cast<size_t>(si)].fn != "percent_rank") {
                continue;
            }

            cudf::size_type pos = value_pos.at(specs[static_cast<size_t>(si)].value_idx);
            cudf::column_view value_view = work_view.column(pos);

            std::vector<cudf::groupby::aggregation_request> pr_requests(2);
            pr_requests[0].values = value_view;
            pr_requests[0].aggregations.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
            pr_requests[1].values = keys_view.column(0);
            pr_requests[1].aggregations.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));

            auto pr_agg_pair = gb.aggregate(pr_requests);
            auto& pr_agg_results = pr_agg_pair.second;

            std::vector<cudf::column_view> pr_pre_repeat = {pr_agg_results[0].results[0]->view()};
            cudf::table_view pr_pre_repeat_tbl(pr_pre_repeat);
            auto pr_repeated = cudf::repeat(pr_pre_repeat_tbl, pr_agg_results[1].results[0]->view());

            cudf::numeric_scalar<int32_t> one_i32(1, true);
            auto is_singleton = cudf::binary_operation(
                pr_repeated->get_column(0).view(), one_i32, cudf::binary_operator::EQUAL,
                cudf::data_type{cudf::type_id::BOOL8});

            auto orig_valid = cudf::is_valid(spec_results[static_cast<size_t>(si)]->view());

            auto should_patch = cudf::binary_operation(
                is_singleton->view(), orig_valid->view(), cudf::binary_operator::LOGICAL_AND,
                cudf::data_type{cudf::type_id::BOOL8});

            cudf::numeric_scalar<double> nan_scalar(std::numeric_limits<double>::quiet_NaN(), true);
            spec_results[static_cast<size_t>(si)] = cudf::copy_if_else(
                nan_scalar, spec_results[static_cast<size_t>(si)]->view(), should_patch->view());
        }

        if (!agg_spec_indices.empty()) {
            // Phase 5 W4 "agg" family: mean()/sum()/min()/max()/n()/sd()/
            // var() (plain per-group aggregates) and first()/last()/nth()
            // (via make_nth_element_aggregation(), honoring an order_by
            // frame when this frame has order columns -- `work_view`'s
            // per-group row order already reflects that ordering, or, when
            // absent, the group's ORIGINAL row order, since step 1's sort is
            // stable). ONE gb.aggregate() call computes every spec in this
            // frame plus a COUNT_ALL request (the per-group row count),
            // then ONE cudf::repeat() call broadcasts every result back out
            // to `work_view`'s row count in lockstep -- see the file-level
            // comment and empirical check E4 (recorded in the W4 commit
            // message): a single groupby object (`gb`, built with
            // sorted::YES) always takes cudf's SORT aggregation path
            // (verified against groupby::dispatch_aggregation() in cudf's
            // own source: `_keys_are_sorted == sorted::YES` skips the hash
            // path unconditionally), so gb.aggregate()'s result row i is
            // exactly sorted-group i -- the same group order the COUNT_ALL
            // request itself enumerates, which is what makes the
            // repeat()-into-`work_view`-order step correct with NO further
            // permutation bookkeeping.
            struct AggJob {
                int spec_index;
                int request_idx;
            };

            std::vector<cudf::groupby::aggregation_request> requests;
            std::vector<AggJob> agg_jobs;

            cudf::groupby::aggregation_request count_req;
            count_req.values = keys_view.column(0);
            count_req.aggregations.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
            int count_request_idx = static_cast<int>(requests.size());
            requests.push_back(std::move(count_req));

            for (int si : agg_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                cudf::groupby::aggregation_request req;

                if (sp.fn == "n") {
                    // n() ignores its own (absent) value argument -- the
                    // exact same COUNT_ALL computation as `count_req`
                    // above, just under this spec's own request slot so
                    // the generic per-request extraction below stays
                    // uniform.
                    req.values = keys_view.column(0);
                    req.aggregations.push_back(
                        cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
                } else if (is_nth_fn(sp.fn)) {
                    if (!sp.has_value) {
                        Rcpp::stop("gpu_window: first()/last()/nth() require a value column");
                    }
                    cudf::size_type pos = value_pos.at(sp.value_idx);
                    req.values = work_view.column(pos);

                    // dplyr 1-based/negative `nth(x, k)` <-> cudf's own
                    // negative-indexing NTH_ELEMENT convention (verified
                    // empirically against dplyr 1.2.1): k > 0 -> cudf index
                    // k-1; k < 0 -> cudf index k UNCHANGED (dplyr's
                    // nth(x, -1) IS cudf's n=-1, both mean "last element");
                    // first()/last() are the fixed n=0/n=-1 special cases.
                    cudf::size_type n;
                    if (sp.fn == "first") {
                        n = 0;
                    } else if (sp.fn == "last") {
                        n = -1;
                    } else {
                        if (!sp.has_n) {
                            Rcpp::stop("gpu_window: nth() requires an element position");
                        }
                        n = sp.n > 0 ? sp.n - 1 : sp.n;
                    }
                    req.aggregations.push_back(
                        cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(
                            n, cudf::null_policy::INCLUDE));
                } else {
                    if (!sp.has_value) {
                        Rcpp::stop("gpu_window: '%s' requires a value column", sp.fn.c_str());
                    }
                    cudf::size_type pos = value_pos.at(sp.value_idx);
                    req.values = work_view.column(pos);
                    req.aggregations.push_back(cuplyr::get_groupby_agg(window_agg_cudf_name(sp.fn)));
                }

                agg_jobs.push_back(AggJob{si, static_cast<int>(requests.size())});
                requests.push_back(std::move(req));
            }

            auto agg_pair = gb.aggregate(requests);
            auto& agg_results = agg_pair.second;

            cudf::column_view counts_view =
                agg_results[static_cast<size_t>(count_request_idx)].results[0]->view();

            std::vector<cudf::column_view> pre_repeat_views;
            pre_repeat_views.reserve(agg_jobs.size());
            for (const auto& job : agg_jobs) {
                pre_repeat_views.push_back(agg_results[static_cast<size_t>(job.request_idx)].results[0]->view());
            }
            cudf::table_view pre_repeat_tbl(pre_repeat_views);

            auto repeated = cudf::repeat(pre_repeat_tbl, counts_view);
            std::vector<std::unique_ptr<cudf::column>> repeated_cols = repeated->release();

            for (size_t k = 0; k < agg_jobs.size(); ++k) {
                spec_results[static_cast<size_t>(agg_jobs[k].spec_index)] = std::move(repeated_cols[k]);
            }

            // na.rm=FALSE propagation: R's mean()/sum()/min()/max()/sd()/
            // var()/median() (Phase 6, task 6.2: median() verified
            // empirically against dplyr 1.2.1 to share this exact rule --
            // `group_by(g) |> mutate(m = median(x))` is NA for every row of
            // a group containing an NA `x`) all default to na.rm=FALSE, so
            // a group with ANY null value makes the WHOLE group's result NA
            // -- unlike cudf's own MEAN/SUM/MIN/MAX/STD/VARIANCE/MEDIAN
            // groupby aggregations, which always exclude nulls. n() is
            // unaffected (it counts ALL rows regardless of nulls, matching
            // dplyr exactly already); n_distinct() is also unaffected (NA
            // is its own distinct value via null_policy::INCLUDE baked
            // directly into the aggregation itself -- see
            // get_groupby_agg()'s own "n_distinct" branch docs -- not a
            // post-hoc propagation step); first()/last()/nth() are also
            // unaffected (na_rm=FALSE for those means "don't skip NA when
            // picking a position", which `null_policy::INCLUDE` above
            // already implements -- only an unrelated OTHER row's NA must
            // NOT affect these three, and it doesn't: they don't reduce
            // over the whole group).
            for (int si : agg_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                bool na_propagates = sp.fn == "mean" || sp.fn == "sum" || sp.fn == "min" ||
                                      sp.fn == "max" || sp.fn == "sd" || sp.fn == "var" ||
                                      sp.fn == "median";
                if (!na_propagates) {
                    continue;
                }

                cudf::size_type pos = value_pos.at(sp.value_idx);
                cudf::column_view value_view = work_view.column(pos);
                if (!value_view.has_nulls()) {
                    continue;  // nothing to propagate
                }

                auto null_indicator_bool = cudf::is_null(value_view);
                auto null_indicator_i8 = cudf::cast(null_indicator_bool->view(),
                                                     cudf::data_type{cudf::type_id::INT8});

                std::vector<cudf::groupby::aggregation_request> any_null_requests(1);
                any_null_requests[0].values = null_indicator_i8->view();
                any_null_requests[0].aggregations.push_back(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>());

                auto any_null_pair = gb.aggregate(any_null_requests);
                cudf::column_view any_null_per_group = any_null_pair.second[0].results[0]->view();

                std::vector<cudf::column_view> any_null_pre_repeat = {any_null_per_group};
                cudf::table_view any_null_pre_repeat_tbl(any_null_pre_repeat);
                // `counts_view` (this frame's per-group row count, computed
                // above) is reused here so the broadcast expands into the
                // exact same sorted-row blocks as every other agg-family
                // result.
                auto any_null_repeated = cudf::repeat(any_null_pre_repeat_tbl, counts_view);

                cudf::numeric_scalar<int8_t> zero_i8_b(0, true);
                auto has_any_null = cudf::binary_operation(
                    any_null_repeated->get_column(0).view(), zero_i8_b, cudf::binary_operator::NOT_EQUAL,
                    cudf::data_type{cudf::type_id::BOOL8});

                auto null_scalar = cudf::make_empty_scalar_like(spec_results[static_cast<size_t>(si)]->view());
                spec_results[static_cast<size_t>(si)] = cudf::copy_if_else(
                    *null_scalar, spec_results[static_cast<size_t>(si)]->view(), has_any_null->view());
            }
        }

        if (!shift_spec_indices.empty()) {
            std::vector<cudf::column_view> shift_value_views;
            std::vector<cudf::size_type> offsets;
            std::vector<std::unique_ptr<cudf::scalar>> fill_scalars;

            for (int si : shift_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                if (!sp.has_value) {
                    Rcpp::stop("gpu_window: lag()/lead() require a value column");
                }
                cudf::size_type pos = value_pos.at(sp.value_idx);
                shift_value_views.push_back(work_view.column(pos));

                cudf::size_type n = sp.has_n ? sp.n : 1;
                offsets.push_back(sp.fn == "lag" ? n : -n);

                SEXP default_sexp = sp.default_valid ? sp.default_value : R_NilValue;
                fill_scalars.push_back(build_scalar(sp.out_type, default_sexp, !sp.default_valid));
            }

            std::vector<std::reference_wrapper<cudf::scalar const>> fill_refs;
            fill_refs.reserve(fill_scalars.size());
            for (const auto& sc : fill_scalars) {
                fill_refs.push_back(std::cref(*sc));
            }

            cudf::table_view shift_values(shift_value_views);
            auto shift_pair = gb.shift(shift_values, offsets, fill_refs);
            std::vector<std::unique_ptr<cudf::column>> shifted = shift_pair.second->release();

            for (size_t k = 0; k < shift_spec_indices.size(); ++k) {
                int si = shift_spec_indices[k];
                spec_results[static_cast<size_t>(si)] = std::move(shifted[k]);
            }
        }

        if (!rolling_spec_indices.empty()) {
            // Rolling windows (Theo-requested benchmark task, see the
            // file-level design comment above ir_parse_roll_call() in
            // R/ir.R for the full empirical derivation of the min_periods/
            // na.rm scheme below). grouped_rolling_window() is used
            // UNIFORMLY whether or not this frame is actually grouped --
            // `keys_view` is either the real group columns or the
            // substituted constant zero-key column (same "one code path"
            // idiom `gb`/groupby above already uses). This is safe because
            // `work_view`'s row order is stable-sorted by GROUP COLS ONLY
            // (a rolling spec carries no order_cols, exactly like the scan
            // family -- n_order_cols == 0 for this frame), i.e. each
            // group's ORIGINAL relative row order is preserved -- exactly
            // grouped_rolling_window()'s documented "presorted by
            // group_key" requirement.
            for (int si : rolling_spec_indices) {
                const WindowSpec& sp = specs[static_cast<size_t>(si)];
                if (!sp.has_value) {
                    Rcpp::stop("gpu_window: '%s' requires a value column", sp.fn.c_str());
                }
                if (!sp.has_n || sp.n < 1) {
                    Rcpp::stop("gpu_window: '%s' requires a positive window width", sp.fn.c_str());
                }
                cudf::size_type pos = value_pos.at(sp.value_idx);
                cudf::column_view value_view = work_view.column(pos);
                cudf::size_type w = sp.n;

                // na.rm=FALSE: min_periods = w on a FIXED w-row window
                // already means "zero nulls tolerated" (w valid values
                // required out of exactly w physical rows) -- verified
                // empirically (see R/ir.R), no separate is-null-indicator
                // masking pass needed here (unlike the cum*/rank
                // sticky-NA fix elsewhere in this file).
                // na.rm=TRUE: min_periods = 1 (skip nulls when computing,
                // as low a valid-count bar as possible) plus a second
                // rolling COUNT_ALL(INCLUDE) pass below to force NA for the
                // leading, physically-incomplete (row-count < w) windows
                // that min_periods = 1 alone would wrongly compute a value
                // for using fewer than w rows.
                cudf::size_type min_periods = sp.na_rm ? 1 : w;
                auto agg = make_rolling_agg(sp.fn);
                auto result = cudf::grouped_rolling_window(keys_view, value_view, w, 0, min_periods, *agg);

                if (sp.na_rm) {
                    auto count_agg = cudf::make_count_aggregation<cudf::rolling_aggregation>(
                        cudf::null_policy::INCLUDE);
                    auto count_result = cudf::grouped_rolling_window(keys_view, value_view, w, 0, 1, *count_agg);

                    cudf::numeric_scalar<int32_t> w_scalar(w, true);
                    auto incomplete = cudf::binary_operation(
                        count_result->view(), w_scalar, cudf::binary_operator::LESS,
                        cudf::data_type{cudf::type_id::BOOL8});

                    auto null_scalar = cudf::make_empty_scalar_like(result->view());
                    result = cudf::copy_if_else(*null_scalar, result->view(), incomplete->view());
                }

                spec_results[static_cast<size_t>(si)] = std::move(result);
            }
        }

        // Cast each result to its spec's DECLARED output type if it
        // differs (the R schema is authoritative -- same idiom as
        // gpu_mutate_expr(), src/ops_expr.cpp).
        for (size_t i = 0; i < specs.size(); ++i) {
            cudf::data_type want = expr_type_from_str(specs[i].out_type);
            if (spec_results[i]->type().id() != want.id()) {
                spec_results[i] = cudf::cast(spec_results[i]->view(), want);
            }
        }

        auto results_tbl = std::make_unique<cudf::table>(std::move(spec_results));

        // --- Step 6: scatter back to original row order. `perm[i]` is the
        // ORIGINAL row index of the row at sorted position i; `target` is
        // only a shape/type template, so self-scattering is safe (perm is
        // a full permutation of [0,n), every destination row is written
        // exactly once). ---
        std::unique_ptr<cudf::table> final_frame_tbl;
        if (perm) {
            final_frame_tbl = cudf::scatter(results_tbl->view(), perm->view(), results_tbl->view());
        } else {
            final_frame_tbl = std::move(results_tbl);
        }

        std::vector<std::unique_ptr<cudf::column>> released = final_frame_tbl->release();
        for (size_t i = 0; i < specs.size(); ++i) {
            appended[static_cast<size_t>(specs[i].out_pos)] = std::move(released[i]);
        }
    }

    std::vector<std::unique_ptr<cudf::column>> final_columns;
    final_columns.reserve(static_cast<size_t>(n_input + n_appended));
    for (int i = 0; i < n_input; ++i) {
        final_columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    for (int i = 0; i < n_appended; ++i) {
        if (!appended[static_cast<size_t>(i)]) {
            Rcpp::stop("gpu_window: internal: appended column %d was never computed", i);
        }
        final_columns.push_back(std::move(appended[static_cast<size_t>(i)]));
    }

    auto result = std::make_unique<cudf::table>(std::move(final_columns));
    return make_gpu_table_xptr(std::move(result));
}
