// src/ops_groupby.cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplyr {

std::unique_ptr<cudf::groupby_aggregation> get_groupby_agg(const std::string& agg_type) {
    if (agg_type == "sum") {
        return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "mean") {
        return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "min") {
        return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "max") {
        return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "n") {
        return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    } else if (agg_type == "std") {
        return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "variance") {
        return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "median") {
        // Phase 6, task 6.2: make_median_aggregation<groupby_aggregation>()
        // has a confirmed explicit instantiation in this environment's
        // compiled libcudf.so (verified via `nm -DC libcudf.so`, mirroring
        // the same due-diligence this file's roll_median()-exclusion
        // comment (src/ops_window.cpp) already did for the ROLLING
        // instantiation, which does NOT exist). Shares this file's ordinary
        // na.rm=FALSE whole-group-null propagation technique -- verified
        // empirically against dplyr 1.2.1 that median(), like mean()/sum()/
        // min()/max()/sd()/var(), silently returns NA for a group with any
        // NA (never errors, unlike quantile() below).
        return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    } else if (agg_type == "n_distinct") {
        // Phase 6, task 6.2: this ONE fixed-null_policy::INCLUDE overload
        // (NA counts as its own distinct value) is what makes n_distinct()
        // reusable from the window-aggregate path (src/ops_window.cpp),
        // which -- like every other window "agg"-family function -- has no
        // na.rm=/na_rm= argument support at all yet (see that file's own
        // is_agg_fn() list): gpu_summarise() itself does NOT call this
        // overload -- it needs BOTH null_policy directions (na.rm=TRUE
        // support), so it builds its own make_nunique_aggregation() call
        // directly instead (see the AggKind::NUNIQUE branch below).
        return cudf::make_nunique_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    } else {
        Rcpp::stop("Unknown aggregation type: " + agg_type);
    }
    return nullptr;
}

// Phase 6, task 6.1 (Fix A): does this aggregation kind need R's own
// na.rm = FALSE whole-group-null propagation (see the file-level rationale
// at gpu_summarise())? n() is deliberately excluded -- it counts rows
// regardless of nulls, matching dplyr's n() already. Phase 6, task 6.2:
// median() joins this list (verified empirically -- see get_groupby_agg()'s
// own comment); quantile()/n_distinct()/first()/last()/nth()/any()/all()
// are handled by entirely separate mechanisms (a hard error, cudf's own
// null_policy, or R's three-valued logic respectively -- see
// gpu_summarise()'s own docs) and never reach this predicate at all.
bool agg_na_propagates(const std::string& agg_type) {
    return agg_type == "sum" || agg_type == "mean" || agg_type == "min" ||
           agg_type == "max" || agg_type == "std" || agg_type == "variance" ||
           agg_type == "median";
}

// Phase 6, task 6.2: aggregation-kind classification for the functions that
// DON'T fit get_groupby_agg()'s "one simple aggregation + optional na.rm
// whole-group-null propagation" shape -- each needs its own request
// construction (quantile()'s probability, n_distinct()'s null_policy,
// first()/last()/nth()'s element position + null_policy) or, for any()/
// all(), an entirely custom post-hoc combination (cudf's groupby_aggregation
// has NO ANY/ALL instantiation at all in this environment -- verified via
// `nm -DC libcudf.so`: only aggregation/reduce_aggregation/
// segmented_reduce_aggregation exist for make_any_aggregation()/
// make_all_aggregation(), never groupby_aggregation).
enum class AggKind { SIMPLE, QUANTILE, NUNIQUE, NTH_ELEMENT, ANY, ALL };

AggKind classify_agg(const std::string& agg_type) {
    if (agg_type == "quantile") return AggKind::QUANTILE;
    if (agg_type == "n_distinct") return AggKind::NUNIQUE;
    if (agg_type == "first" || agg_type == "last" || agg_type == "nth") return AggKind::NTH_ELEMENT;
    if (agg_type == "any") return AggKind::ANY;
    if (agg_type == "all") return AggKind::ALL;
    return AggKind::SIMPLE;
}

// first()/last()/nth() -> cudf's NTH_ELEMENT position, mirroring
// src/ops_window.cpp's identical translation for the window path: dplyr's
// own 1-based/negative `nth(x, k)` convention IS cudf's own negative-index
// convention (`nth(x, -1)` == cudf's n = -1, both mean "last element"),
// only a positive k needs -1 (1-based -> 0-based). first()/last() are the
// fixed n = 0 / n = -1 special cases (agg_extra is unused/NA for these
// two).
cudf::size_type nth_element_position(const std::string& agg_type, double extra) {
    if (agg_type == "first") {
        return 0;
    }
    if (agg_type == "last") {
        return -1;
    }
    cudf::size_type n = static_cast<cudf::size_type>(extra);
    return n > 0 ? n - 1 : n;
}

// Phase 6, task 6.2: any()/all() with R's exact three-valued logic (verified
// empirically against dplyr 1.2.1: `any(c(TRUE, NA))` is `TRUE`,
// `any(c(FALSE, NA))` is `NA`, `all(c(FALSE, NA))` is `FALSE`,
// `all(c(TRUE, NA))` is `NA`). Composed from MIN/MAX on the BOOL8 value
// column (cudf's own MIN/MAX skip nulls already) plus a per-group
// "has any null" indicator (the same MAX-of-is_null() technique
// `null_out_groups_with_any_null()` uses above), since cudf has no
// ANY/ALL groupby aggregation to call directly.
//
// `reduced` is MAX(x) (any()) or MIN(x) (all()), both skip-null, so a
// group with ONLY null values comes back NULL here (not TRUE/FALSE) --
// `filled` replaces that with the identity value: `FALSE` for any()
// (matching `any(logical(0))`), `TRUE` for all() (matching
// `all(logical(0))`) -- correct for na_rm = TRUE, where a null-only group
// after "removing NA" is genuinely empty.
//
// For na_rm = FALSE (`has_null_i8` non-null), one extra correction is
// needed: a group with NO true value found (any()) or NO false value found
// (all()) but at least one NULL must come back NULL (R's "unknown"), not
// the identity-filled FALSE/TRUE -- `should_null_out` flags exactly those
// rows, and the final `copy_if_else` nulls them out (mirroring
// `null_out_groups_with_any_null()`'s own convention: the mask argument
// says WHERE to substitute the null scalar).
std::unique_ptr<cudf::column> combine_any_all(
    const cudf::column_view& reduced,
    const cudf::column_view* has_null_i8,
    bool is_any) {
    cudf::numeric_scalar<bool> identity_scalar(!is_any, true);
    auto reduced_is_null = cudf::is_null(reduced);
    auto filled = cudf::copy_if_else(identity_scalar, reduced, reduced_is_null->view());

    if (has_null_i8 == nullptr) {
        return filled;
    }

    cudf::numeric_scalar<int8_t> zero_i8(0, true);
    auto has_null_bool = cudf::binary_operation(
        *has_null_i8, zero_i8, cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8});

    std::unique_ptr<cudf::column> should_null_out;
    if (is_any) {
        auto not_filled = cudf::unary_operation(filled->view(), cudf::unary_operator::NOT);
        should_null_out = cudf::binary_operation(
            not_filled->view(), has_null_bool->view(), cudf::binary_operator::LOGICAL_AND,
            cudf::data_type{cudf::type_id::BOOL8});
    } else {
        should_null_out = cudf::binary_operation(
            filled->view(), has_null_bool->view(), cudf::binary_operator::LOGICAL_AND,
            cudf::data_type{cudf::type_id::BOOL8});
    }

    auto null_scalar = cudf::make_empty_scalar_like(filled->view());
    return cudf::copy_if_else(*null_scalar, filled->view(), should_null_out->view());
}

// Given a single aggregation's already-computed cudf result column and the
// ORIGINAL (pre-aggregation) value column it was computed from, null out
// every row of `agg_result` whose corresponding group had ANY null in
// `value_view` -- `any_null_per_group` is a per-group (same row count and
// order as `agg_result`) INT8 column, 0/1, already reduced via
// make_max_aggregation() over an is_null() indicator (see call sites).
// Mirrors src/ops_window.cpp's identical technique for the window-aggregate
// path (W4), just at group-row granularity instead of broadcast-to-row
// granularity (summarise()'s output rows already ARE the groups, so no
// cudf::repeat() step is needed here).
std::unique_ptr<cudf::column> null_out_groups_with_any_null(
    std::unique_ptr<cudf::column> agg_result,
    const cudf::column_view& any_null_per_group) {
    cudf::numeric_scalar<int8_t> zero_i8(0, true);
    auto has_any_null = cudf::binary_operation(
        any_null_per_group, zero_i8, cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8});

    auto null_scalar = cudf::make_empty_scalar_like(agg_result->view());
    return cudf::copy_if_else(*null_scalar, agg_result->view(), has_any_null->view());
}

// Phase 6, task 4: entirely-NA-group na.rm=TRUE identity values.
//
// Base R's own empty-vector reductions (verified empirically, see
// test-dplyr-summarise.R's own module doc): sum(numeric(0)) is 0,
// mean(numeric(0)) is NaN, min(numeric(0))/max(numeric(0)) are Inf/-Inf
// (both WITH a warning dplyr's summarise() itself surfaces, wrapped in its
// own condition machinery -- NOT replicated here, informational only, see
// this task's test file for the documented decision). sd()/var()/median()
// need NO fix: their NULL-for-an-all-excluded-group result already
// collects to NA, which IS base R's own sd(numeric(0))/var(numeric(0))/
// median(numeric(0)) value.
//
// `agg_type` restricted to "sum"/"mean"/"min"/"max" by the caller (see
// needs_identity_fix() below); na.rm=TRUE is a precondition (checked by the
// caller too -- na.rm=FALSE's whole-group-null propagation,
// agg_na_propagates(), is the entirely separate, already-existing fix for
// the opposite flag value).
bool needs_identity_fix(const std::string& agg_type, bool na_rm_flag,
                         const cudf::column_view& value_view) {
    if (!na_rm_flag) {
        return false;
    }
    if (agg_type == "sum" || agg_type == "mean") {
        return true;
    }
    if (agg_type == "min" || agg_type == "max") {
        // min()/max()'s identity (Inf/-Inf) can only be represented in a
        // FLOAT64 output column -- real dplyr promotes an INTEGER column's
        // result to double the moment any group needs this identity (a
        // data-dependent type decision cuplyr's parse-time/pre-execution
        // schema inference can't make; see test-dplyr-summarise.R's
        // "BOUNDARY" test). Only fix the case that's already FLOAT64 going
        // in, where no promotion is needed at all.
        return value_view.type().id() == cudf::type_id::FLOAT64;
    }
    return false;
}

// Build the identity scalar for `agg_type`, matching `out_type` (the
// aggregation's ACTUAL result column type, e.g. INT64 for sum() over an
// INT32 column, FLOAT64 for mean()/min()/max()) -- see needs_identity_fix()'s
// own docs for the value table.
std::unique_ptr<cudf::scalar> make_identity_scalar(const std::string& agg_type,
                                                    cudf::data_type out_type) {
    if (out_type.id() == cudf::type_id::INT64) {
        return std::make_unique<cudf::numeric_scalar<int64_t>>(0, true);
    }
    if (out_type.id() == cudf::type_id::INT32) {
        return std::make_unique<cudf::numeric_scalar<int32_t>>(0, true);
    }
    if (out_type.id() == cudf::type_id::FLOAT64) {
        double v;
        if (agg_type == "sum") {
            v = 0.0;
        } else if (agg_type == "mean") {
            v = std::nan("");
        } else if (agg_type == "min") {
            v = std::numeric_limits<double>::infinity();
        } else {
            // "max"
            v = -std::numeric_limits<double>::infinity();
        }
        return std::make_unique<cudf::numeric_scalar<double>>(v, true);
    }
    return nullptr;
}

// Given a SIMPLE aggregation's already-computed result column and a
// per-group "valid (non-null) input count" column (same row count/order as
// `agg_result`, built via a COUNT(EXCLUDE) request over the same input
// column), replace every row whose group had ZERO valid inputs with the
// identity scalar -- the entirely-NA-group na.rm=TRUE case. Mirrors
// null_out_groups_with_any_null()'s own copy_if_else()-on-a-mask
// convention, just substituting a real value instead of a null scalar.
std::unique_ptr<cudf::column> substitute_identity_for_empty_groups(
    std::unique_ptr<cudf::column> agg_result,
    const cudf::column_view& valid_count_per_group,
    const cudf::scalar& identity_scalar) {
    cudf::numeric_scalar<int32_t> zero_i32(0, true);
    auto is_empty = cudf::binary_operation(
        valid_count_per_group, zero_i32, cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8});

    return cudf::copy_if_else(identity_scalar, agg_result->view(), is_empty->view());
}

} // namespace cuplyr

// Phase 6, task 6.1 (Fix A): `na_rm` -- one flag per aggregation, same
// length/order as agg_col_indices/agg_types -- controls whether R's own
// na.rm = FALSE default (mean()/sum()/min()/max()/sd()/var() all default
// to it) is honored. cudf's own groupby MEAN/SUM/MIN/MAX/STD/VARIANCE
// aggregations always exclude nulls unconditionally, which matches R's
// na.rm = TRUE behavior but NOT its na.rm = FALSE default (a group with
// ANY null input value must produce a NULL result for the WHOLE group --
// verified empirically against dplyr 1.2.1). `na_rm[i] == FALSE` (R's
// default when the user didn't pass na.rm= at all) requests the extra
// whole-group-null propagation step below; `na_rm[i] == TRUE` (explicit
// na.rm = TRUE) skips it, using cudf's native behavior directly. n() is
// unaffected regardless of its na_rm entry (see agg_na_propagates()).
//
// This mirrors src/ops_window.cpp's identical rule for the window-aggregate
// path (W4) -- that comment block explicitly noted this exact gap
// ("R/summarise.R's existing mean()/sum()/min()/max() aggregations share
// the exact same gap -- tracked separately") as out of scope at the time;
// this is that follow-up fix.
// [[Rcpp::export]]
SEXP gpu_summarise(SEXP xptr, IntegerVector group_indices,
                   IntegerVector agg_col_indices, CharacterVector agg_types,
                   LogicalVector na_rm, NumericVector agg_extra) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int num_aggs = agg_col_indices.size();
    int num_groups = group_indices.size();

    if (na_rm.size() != num_aggs) {
        Rcpp::stop("na_rm length (%d) must match agg_col_indices length (%d)",
                   na_rm.size(), num_aggs);
    }
    if (agg_extra.size() != num_aggs) {
        Rcpp::stop("agg_extra length (%d) must match agg_col_indices length (%d)",
                   agg_extra.size(), num_aggs);
    }

    for (int i = 0; i < num_groups; ++i) {
        if (group_indices[i] < 0 || group_indices[i] >= view.num_columns()) {
            Rcpp::stop("Group column index out of bounds: " +
                       std::to_string(group_indices[i]));
        }
    }
    for (int i = 0; i < num_aggs; ++i) {
        if (agg_col_indices[i] < 0 || agg_col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Aggregation column index out of bounds: " +
                       std::to_string(agg_col_indices[i]));
        }
    }

    if (num_groups == 0) {
        std::vector<std::unique_ptr<cudf::column>> result_columns;

        for (int i = 0; i < num_aggs; ++i) {
            std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
            cudf::column_view col = view.column(agg_col_indices[i]);

            if (agg_type == "n") {
                // cudf::size_type (view.num_rows()) is already int32_t, so this
                // narrows nothing; INT32 also matches the grouped path's cudf
                // count aggregation (size_type) and dplyr's n() (integer),
                // keeping schema declaration ("n" -> INT32 in make_aggregation(),
                // R/ast.R) consistent with the actual GPU column in both the
                // grouped and ungrouped paths.
                int32_t count = view.num_rows();
                rmm::device_buffer data(sizeof(int32_t),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
                check_cuda(cudaMemcpy(data.data(), &count, sizeof(int32_t), cudaMemcpyHostToDevice),
                           "gpu_summarise count memcpy");
                result_columns.push_back(std::make_unique<cudf::column>(
                    cudf::data_type{cudf::type_id::INT32},
                    1,
                    std::move(data),
                    rmm::device_buffer{},
                    0
                ));
            } else {
                std::vector<int32_t> keys_data(view.num_rows(), 0);
                rmm::device_buffer keys_buf(keys_data.size() * sizeof(int32_t),
                                            rmm::cuda_stream_view(),
                                            rmm::mr::get_current_device_resource_ref());
                if (!keys_data.empty()) {
                    check_cuda(cudaMemcpy(keys_buf.data(), keys_data.data(),
                                           keys_data.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                               "gpu_summarise keys memcpy");
                }

                auto keys_col = std::make_unique<cudf::column>(
                    cudf::data_type{cudf::type_id::INT32},
                    view.num_rows(),
                    std::move(keys_buf),
                    rmm::device_buffer{},
                    0
                );

                std::vector<cudf::column_view> keys_views = { keys_col->view() };
                cudf::table_view keys_table(keys_views);

                // null_policy::INCLUDE: cudf::groupby's default
                // (null_policy::EXCLUDE) silently drops every row whose
                // group key is null, but dplyr's group_by()/summarise()
                // always treats NA as its own group like any other value
                // (verified empirically, see test-summarise.R's "keeps an
                // NA group key" test and test-dplyr-count.R's NA-key
                // tests -- add_count()'s join-back broadcast, R/mutate.R,
                // depends on this too).
                cudf::groupby::groupby gb(keys_table, cudf::null_policy::INCLUDE);

                AggKind kind = classify_agg(agg_type);

                // Phase 6, task 6.2: quantile()'s na.rm = FALSE (default)
                // with an actual NA present is a hard ERROR in base R
                // itself ("missing values and NaN's not allowed if 'na.rm'
                // is FALSE"), verified empirically against dplyr 1.2.1 --
                // NOT a NA-propagation like mean()/sum()/.../median()
                // above. A single "does this column have ANY null at all"
                // check is equivalent to dplyr's own per-group error
                // (real dplyr aborts the ENTIRE summarise() call the
                // moment any one group hits a NA, so there is no partial
                // result to preserve either way).
                if (kind == AggKind::QUANTILE && !static_cast<bool>(na_rm[i]) && col.has_nulls()) {
                    Rcpp::stop("quantile(): missing values and NaN's not allowed if na.rm = FALSE. "
                               "Use na.rm = TRUE to ignore missing values.");
                }

                if (kind == AggKind::ANY || kind == AggKind::ALL) {
                    bool is_any = (kind == AggKind::ANY);
                    bool na_rm_flag = static_cast<bool>(na_rm[i]);

                    std::vector<cudf::groupby::aggregation_request> requests(1);
                    requests[0].values = col;
                    requests[0].aggregations.push_back(
                        is_any ? cudf::make_max_aggregation<cudf::groupby_aggregation>()
                               : cudf::make_min_aggregation<cudf::groupby_aggregation>());

                    std::unique_ptr<cudf::column> null_indicator_i8;
                    if (!na_rm_flag) {
                        auto null_indicator_bool = cudf::is_null(col);
                        null_indicator_i8 = cudf::cast(null_indicator_bool->view(),
                                                        cudf::data_type{cudf::type_id::INT8});
                        cudf::groupby::aggregation_request null_req;
                        null_req.values = null_indicator_i8->view();
                        null_req.aggregations.push_back(
                            cudf::make_max_aggregation<cudf::groupby_aggregation>());
                        requests.push_back(std::move(null_req));
                    }

                    auto [result_keys, result_aggs] = gb.aggregate(requests);
                    cudf::column_view reduced_view = result_aggs[0].results[0]->view();
                    if (na_rm_flag) {
                        result_columns.push_back(combine_any_all(reduced_view, nullptr, is_any));
                    } else {
                        cudf::column_view has_null_view = result_aggs[1].results[0]->view();
                        result_columns.push_back(combine_any_all(reduced_view, &has_null_view, is_any));
                    }
                    continue;
                }

                // Phase 6, task 6.1 (Fix A): if this aggregation needs
                // na.rm=FALSE whole-group-null propagation, compute the
                // is_null-indicator MAX request in the SAME gb.aggregate()
                // call as the main aggregation -- one call guarantees both
                // results share the exact same (single, whole-table-as-one-
                // group) row, so no separate-call group-order assumption is
                // needed here.
                bool need_na_prop = kind == AggKind::SIMPLE && agg_na_propagates(agg_type) &&
                    !static_cast<bool>(na_rm[i]) && col.has_nulls();

                // Phase 6, task 4: entirely-NA-group na.rm=TRUE identity
                // values (sum()->0, mean()->NaN, min()->Inf, max()->-Inf) --
                // mutually exclusive with need_na_prop above (one requires
                // na_rm==FALSE, the other na_rm==TRUE), so both share the
                // same "extra request at index 1" slot below.
                bool need_identity_fix = kind == AggKind::SIMPLE &&
                    needs_identity_fix(agg_type, static_cast<bool>(na_rm[i]), col);

                std::unique_ptr<cudf::column> null_indicator_i8;
                std::vector<cudf::groupby::aggregation_request> requests;
                cudf::groupby::aggregation_request req;
                req.values = col;
                switch (kind) {
                    case AggKind::QUANTILE:
                        req.aggregations.push_back(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
                            {agg_extra[i]}, cudf::interpolation::LINEAR));
                        break;
                    case AggKind::NUNIQUE: {
                        cudf::null_policy np = static_cast<bool>(na_rm[i])
                            ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
                        req.aggregations.push_back(
                            cudf::make_nunique_aggregation<cudf::groupby_aggregation>(np));
                        break;
                    }
                    case AggKind::NTH_ELEMENT: {
                        cudf::size_type n = nth_element_position(agg_type, agg_extra[i]);
                        cudf::null_policy np = static_cast<bool>(na_rm[i])
                            ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
                        req.aggregations.push_back(
                            cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(n, np));
                        break;
                    }
                    default:
                        req.aggregations.push_back(get_groupby_agg(agg_type));
                        break;
                }
                requests.push_back(std::move(req));

                if (need_na_prop) {
                    auto null_indicator_bool = cudf::is_null(col);
                    null_indicator_i8 = cudf::cast(null_indicator_bool->view(),
                                                    cudf::data_type{cudf::type_id::INT8});
                    cudf::groupby::aggregation_request null_req;
                    null_req.values = null_indicator_i8->view();
                    null_req.aggregations.push_back(
                        cudf::make_max_aggregation<cudf::groupby_aggregation>());
                    requests.push_back(std::move(null_req));
                } else if (need_identity_fix) {
                    cudf::groupby::aggregation_request count_req;
                    count_req.values = col;
                    count_req.aggregations.push_back(
                        cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
                    requests.push_back(std::move(count_req));
                }

                auto [result_keys, result_aggs] = gb.aggregate(requests);

                std::unique_ptr<cudf::column> agg_result = std::move(result_aggs[0].results[0]);
                if (need_na_prop) {
                    agg_result = null_out_groups_with_any_null(
                        std::move(agg_result), result_aggs[1].results[0]->view());
                } else if (need_identity_fix) {
                    auto identity_scalar = make_identity_scalar(agg_type, agg_result->view().type());
                    agg_result = substitute_identity_for_empty_groups(
                        std::move(agg_result), result_aggs[1].results[0]->view(), *identity_scalar);
                }

                result_columns.push_back(std::move(agg_result));
            }
        }

        auto result = std::make_unique<cudf::table>(std::move(result_columns));
        return make_gpu_table_xptr(std::move(result));
    }

    std::vector<cudf::column_view> keys_views;
    for (int i = 0; i < num_groups; ++i) {
        keys_views.push_back(view.column(group_indices[i]));
    }
    cudf::table_view keys_table(keys_views);

    // null_policy::INCLUDE: see the comment on the other groupby::groupby
    // construction above -- dplyr's group_by()/summarise() always treats
    // NA as its own group; cudf's default (null_policy::EXCLUDE) would
    // silently drop every row whose group key is null.
    cudf::groupby::groupby gb(keys_table, cudf::null_policy::INCLUDE);

    std::vector<cudf::groupby::aggregation_request> requests;
    // Phase 6, task 6.1 (Fix A): for every aggregation needing na.rm=FALSE
    // whole-group-null propagation, an extra is_null-indicator MAX request
    // is appended to this SAME requests vector (na_prop_request_idx[i] ==
    // its index, -1 if not needed) so it's answered by the exact same
    // gb.aggregate() call as every other request below -- one call
    // guarantees identical group order across all results, with no
    // assumption needed about repeated aggregate() calls on the same `gb`
    // producing consistent ordering.
    std::vector<int> na_prop_request_idx(num_aggs, -1);
    // Each aggregation's OWN main request ends up at a DIFFERENT index than
    // `i` as soon as any earlier aggregation needed an extra na-prop
    // request interleaved into the same `requests` vector -- this tracks
    // the true request-vector index of aggregation i's main result
    // (`result_aggs[i]` would silently read a DIFFERENT aggregation's
    // result once indices have shifted; this bug was caught by
    // test-dplyr-summarise.R's mixed na.rm=TRUE/FALSE-on-the-same-column
    // oracle test before this fix).
    std::vector<int> main_request_idx(num_aggs, -1);
    // Phase 6, task 6.2: any()/all()'s own "does this group have any null"
    // companion request -- same interleave-into-the-shared-`requests`-
    // vector discipline as na_prop_request_idx above (and for the exact
    // same reason: `cudf::groupby::aggregate()`'s own docs explicitly warn
    // that "successive aggregate() calls may return results in different
    // orders", so any()/all()'s companion MUST be answered by this SAME
    // call, never a separate one).
    std::vector<int> any_all_null_request_idx(num_aggs, -1);
    // Phase 6, task 4: entirely-NA-group na.rm=TRUE identity values -- same
    // interleave-into-the-shared-`requests`-vector discipline as
    // na_prop_request_idx above, holding each aggregation's own extra
    // per-group "valid (non-null) count" COUNT(EXCLUDE) request (-1 if not
    // needed). Mutually exclusive with na_prop_request_idx per aggregation
    // (one requires na_rm==FALSE, the other na_rm==TRUE).
    std::vector<int> identity_fix_request_idx(num_aggs, -1);
    // Must outlive the gb.aggregate() call below (requests reference their views).
    std::vector<std::unique_ptr<cudf::column>> null_indicator_cols;

    for (int i = 0; i < num_aggs; ++i) {
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        cudf::column_view value_view = view.column(agg_col_indices[i]);
        AggKind kind = classify_agg(agg_type);
        bool na_rm_flag = static_cast<bool>(na_rm[i]);

        // Phase 6, task 6.2: quantile()'s na.rm = FALSE (default) with an
        // actual NA present is a hard ERROR in base R itself -- see the
        // identical check (and its own docs) in the num_groups == 0 branch
        // above.
        if (kind == AggKind::QUANTILE && !na_rm_flag && value_view.has_nulls()) {
            Rcpp::stop("quantile(): missing values and NaN's not allowed if na.rm = FALSE. "
                       "Use na.rm = TRUE to ignore missing values.");
        }

        cudf::groupby::aggregation_request req;
        req.values = value_view;
        switch (kind) {
            case AggKind::QUANTILE:
                req.aggregations.push_back(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
                    {agg_extra[i]}, cudf::interpolation::LINEAR));
                break;
            case AggKind::NUNIQUE: {
                cudf::null_policy np = na_rm_flag ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
                req.aggregations.push_back(
                    cudf::make_nunique_aggregation<cudf::groupby_aggregation>(np));
                break;
            }
            case AggKind::NTH_ELEMENT: {
                cudf::size_type n = nth_element_position(agg_type, agg_extra[i]);
                cudf::null_policy np = na_rm_flag ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
                req.aggregations.push_back(
                    cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(n, np));
                break;
            }
            case AggKind::ANY:
                req.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
                break;
            case AggKind::ALL:
                req.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
                break;
            default:
                req.aggregations.push_back(get_groupby_agg(agg_type));
                break;
        }
        main_request_idx[i] = static_cast<int>(requests.size());
        requests.push_back(std::move(req));

        if (kind == AggKind::ANY || kind == AggKind::ALL) {
            if (!na_rm_flag) {
                auto null_indicator_bool = cudf::is_null(value_view);
                null_indicator_cols.push_back(cudf::cast(
                    null_indicator_bool->view(), cudf::data_type{cudf::type_id::INT8}));

                cudf::groupby::aggregation_request null_req;
                null_req.values = null_indicator_cols.back()->view();
                null_req.aggregations.push_back(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>());
                any_all_null_request_idx[i] = static_cast<int>(requests.size());
                requests.push_back(std::move(null_req));
            }
            continue;  // any()/all() never need agg_na_propagates()'s own step
        }

        bool need_na_prop = kind == AggKind::SIMPLE && agg_na_propagates(agg_type) &&
            !na_rm_flag && value_view.has_nulls();
        if (need_na_prop) {
            auto null_indicator_bool = cudf::is_null(value_view);
            null_indicator_cols.push_back(cudf::cast(
                null_indicator_bool->view(), cudf::data_type{cudf::type_id::INT8}));

            cudf::groupby::aggregation_request null_req;
            null_req.values = null_indicator_cols.back()->view();
            null_req.aggregations.push_back(
                cudf::make_max_aggregation<cudf::groupby_aggregation>());
            na_prop_request_idx[i] = static_cast<int>(requests.size());
            requests.push_back(std::move(null_req));
            continue;
        }

        // Phase 6, task 4: entirely-NA-group na.rm=TRUE identity values --
        // see needs_identity_fix()'s own docs. Mutually exclusive with
        // need_na_prop above (already `continue`d past if so).
        if (kind == AggKind::SIMPLE && needs_identity_fix(agg_type, na_rm_flag, value_view)) {
            cudf::groupby::aggregation_request count_req;
            count_req.values = value_view;
            count_req.aggregations.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
            identity_fix_request_idx[i] = static_cast<int>(requests.size());
            requests.push_back(std::move(count_req));
        }
    }

    auto [result_keys, result_aggs] = gb.aggregate(requests);

    for (int i = 0; i < num_aggs; ++i) {
        if (na_prop_request_idx[i] < 0) {
            continue;
        }
        result_aggs[main_request_idx[i]].results[0] = null_out_groups_with_any_null(
            std::move(result_aggs[main_request_idx[i]].results[0]),
            result_aggs[na_prop_request_idx[i]].results[0]->view());
    }

    for (int i = 0; i < num_aggs; ++i) {
        if (identity_fix_request_idx[i] < 0) {
            continue;
        }
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        auto& main_result = result_aggs[main_request_idx[i]].results[0];
        auto identity_scalar = make_identity_scalar(agg_type, main_result->view().type());
        main_result = substitute_identity_for_empty_groups(
            std::move(main_result), result_aggs[identity_fix_request_idx[i]].results[0]->view(),
            *identity_scalar);
    }

    for (int i = 0; i < num_aggs; ++i) {
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        AggKind kind = classify_agg(agg_type);
        if (kind != AggKind::ANY && kind != AggKind::ALL) {
            continue;
        }
        bool is_any = (kind == AggKind::ANY);
        cudf::column_view reduced_view = result_aggs[main_request_idx[i]].results[0]->view();
        if (any_all_null_request_idx[i] < 0) {
            result_aggs[main_request_idx[i]].results[0] = combine_any_all(reduced_view, nullptr, is_any);
        } else {
            cudf::column_view has_null_view = result_aggs[any_all_null_request_idx[i]].results[0]->view();
            result_aggs[main_request_idx[i]].results[0] =
                combine_any_all(reduced_view, &has_null_view, is_any);
        }
    }

    std::vector<std::unique_ptr<cudf::column>> result_columns;

    for (int i = 0; i < result_keys->num_columns(); ++i) {
        result_columns.push_back(std::make_unique<cudf::column>(result_keys->get_column(i)));
    }

    for (int i = 0; i < num_aggs; ++i) {
        result_columns.push_back(std::move(result_aggs[main_request_idx[i]].results[0]));
    }

    auto result = std::make_unique<cudf::table>(std::move(result_columns));
    return make_gpu_table_xptr(std::move(result));
}
