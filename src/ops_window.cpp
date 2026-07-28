// src/ops_window.cpp
//
// GPU-native window functions (Phase 5, task W2): the "shift" family
// (lag()/lead()) and the "scan" family (cumsum()/cummax()/cummin()/
// cumprod(), plus bare row_number()'s no-order COUNT_ALL form). Every
// other window kind (ranked row_number(x)/min_rank()/dense_rank()/
// percent_rank()/cume_dist(), and the whole "agg" family --
// mean()/sum()/min()/max()/n()/sd()/var()/first()/last()/nth()) is W4
// work -- R/lower.R::validate_window_spec_w2() rejects those before this
// file is ever reached, so this file only has to implement the fn set
// above.
//
// This is the six-step algorithm from
// scratchpad/phase5_window_design.md section 2.2, implemented EXACTLY:
//
//   1. sort_keys = [group cols] ++ [order cols]; ASCENDING for every group
//      col, per-spec order for order cols; null_order::AFTER for every key.
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
// Empirical checks recorded in the W2 commit message: E1 (grouped cumsum
// on a shuffled 100k-row table matches dplyr exactly -- validates the
// whole perm/gather/scan/scatter round trip), E2 (grouped row_number()
// with duplicate + NA group keys), E5 (cudf group-scan NA IS non-sticky
// before the fix; the fix reproduces R's sticky semantics), E6 (lag()/
// lead() fill type + NA default). E3/E4 are W4's (rank/agg families).

#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "expr_eval.hpp"  // build_scalar(), expr_type_from_str()

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <climits>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

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

        std::vector<cudf::column_view> order_col_views;
        std::vector<cudf::order> order_col_orders;
        for (int i = 0; i < order_idx_r.size(); ++i) {
            int idx = order_idx_r[i];
            if (idx < 0 || idx >= n_input) {
                Rcpp::stop("gpu_window: order column index out of bounds: %d", idx);
            }
            order_col_views.push_back(view.column(idx));
            bool desc = (i < order_desc_r.size()) && static_cast<bool>(order_desc_r[i]);
            order_col_orders.push_back(desc ? cudf::order::DESCENDING : cudf::order::ASCENDING);
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
                static_cast<size_t>(n_group_cols + n_order_cols), cudf::null_order::AFTER);

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
        // order. Split into the scan family (row_number()'s COUNT_ALL form
        // + cumsum/cummax/cummin/cumprod) and the shift family
        // (lag()/lead()). ---
        std::vector<int> scan_spec_indices;
        std::vector<int> shift_spec_indices;
        for (size_t i = 0; i < specs.size(); ++i) {
            const std::string& fn = specs[i].fn;
            if (fn == "row_number" || is_cum_scan_fn(fn)) {
                scan_spec_indices.push_back(static_cast<int>(i));
            } else if (fn == "lag" || fn == "lead") {
                shift_spec_indices.push_back(static_cast<int>(i));
            } else {
                Rcpp::stop("gpu_window: unsupported window fn '%s' (W2 scope: the scan and "
                           "shift families only)", fn.c_str());
            }
        }

        // Results, in `specs`/`work_view` order (NOT yet scattered back).
        std::vector<std::unique_ptr<cudf::column>> spec_results(specs.size());

        if (!scan_spec_indices.empty()) {
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
