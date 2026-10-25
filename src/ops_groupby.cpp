// src/ops_groupby.cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "int_sum.hpp"
#include "zero_sign.hpp"
#include "ts_agg.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cmath>
#include <limits>
#include <map>
#include <numeric>
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
// (the latter two WITH a warning, which R raises from the
// "identity_used" attribute gpu_summarise() returns -- see
// apply_minmax_identity()). sd()/var()/median() need NO fix: their
// NULL-for-an-all-excluded-group result already collects to NA, which IS
// base R's own sd(numeric(0))/var(numeric(0))/median(numeric(0)) value.
//
// `agg_type` restricted to "sum"/"mean"/"min"/"max" by the caller (see
// needs_identity_fix() below); na.rm=TRUE is a precondition (checked by the
// caller too -- na.rm=FALSE's whole-group-null propagation,
// agg_na_propagates(), is the entirely separate, already-existing fix for
// the opposite flag value).
//
// F07: min()/max() over FLOAT64 and INT32 (`identity_flag`, set by R for
// exactly those inputs and not for internal helper aggregations). Over INT32
// the identity can't live in the INT32 column, so apply_minmax_identity()
// promotes the WHOLE result column to FLOAT64 the moment any group is empty,
// which is what dplyr does. A request is only needed when the column has
// nulls at all: with no nulls and at least one row no group can be empty.
bool needs_identity_fix(const std::string& agg_type, bool na_rm_flag,
                         const cudf::column_view& value_view, bool identity_flag) {
    if (!na_rm_flag) {
        return false;
    }
    if (agg_type == "sum" || agg_type == "mean") {
        return true;
    }
    if (agg_type == "min" || agg_type == "max") {
        auto id = value_view.type().id();
        return identity_flag && value_view.has_nulls() &&
               (id == cudf::type_id::FLOAT64 || id == cudf::type_id::INT32);
    }
    return false;
}

// min()/max() the identity scalar can only represent when the input is one
// of these (everything else keeps its own type and never gets an identity).
bool minmax_has_identity(const std::string& agg_type, cudf::type_id id) {
    return (agg_type == "min" || agg_type == "max") &&
           (id == cudf::type_id::FLOAT64 || id == cudf::type_id::INT32);
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

// F07: min()/max() empty-group identity for FLOAT64/INT32 inputs. Returns
// true iff some group was empty (the caller reports it so R can warn with
// base R's own text). `valid_count_per_group` is nullptr when the whole
// input has zero rows (no group exists at all, e.g. a 0-row grouped
// summarise): nothing to substitute, but dplyr still evaluated the reduction
// on an empty chunk, so an INT32 result is typed double and the warning
// fires. Otherwise an INT32 result is promoted to FLOAT64 only when a group
// is actually empty; FLOAT64 results just get the +-Inf substituted.
bool apply_minmax_identity(std::unique_ptr<cudf::column>& result, bool is_min,
                           const cudf::column_view* valid_count_per_group) {
    bool input_empty = valid_count_per_group == nullptr;
    std::unique_ptr<cudf::column> is_empty;
    if (!input_empty) {
        cudf::numeric_scalar<int32_t> zero_i32(0, true);
        is_empty = cudf::binary_operation(
            *valid_count_per_group, zero_i32, cudf::binary_operator::EQUAL,
            cudf::data_type{cudf::type_id::BOOL8});
        auto any_empty = cudf::reduce(
            is_empty->view(), *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
            cudf::data_type{cudf::type_id::BOOL8});
        if (!static_cast<cudf::numeric_scalar<bool>*>(any_empty.get())->value()) {
            return false;
        }
    }

    if (result->type().id() == cudf::type_id::INT32) {
        result = cudf::cast(result->view(), cudf::data_type{cudf::type_id::FLOAT64});
    }
    if (!input_empty) {
        double v = std::numeric_limits<double>::infinity();
        if (!is_min) v = -v;
        cudf::numeric_scalar<double> identity(v, true);
        result = cudf::copy_if_else(identity, result->view(), is_empty->view());
    }
    return true;
}

// One-row result of a reduction over ZERO input rows (ungrouped summarise()
// of an empty table), using base R's empty-vector values: n() 0L, sum 0,
// mean NaN, min Inf / max -Inf (double, flagged via `identity_used`),
// median/sd/var/quantile NA_real_, n_distinct 0L, first/last/nth typed NA,
// any FALSE, all TRUE. Mirrors what dplyr returns when it evaluates each
// dot once on an empty vector.
std::unique_ptr<cudf::column> empty_input_reduction(const std::string& agg_type,
                                                    cudf::data_type in_type,
                                                    bool identity_flag,
                                                    bool& identity_used) {
    auto num = [](double v) {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<double>(v, true), 1);
    };
    auto null_of = [](cudf::data_type t) {
        return cudf::make_column_from_scalar(*cudf::make_default_constructed_scalar(t), 1);
    };
    if (agg_type == "n" || agg_type == "n_distinct") {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<int32_t>(0, true), 1);
    }
    if (agg_type == "sum") {
        auto id = in_type.id();
        if (id == cudf::type_id::INT32 || id == cudf::type_id::INT64 ||
            id == cudf::type_id::INT16 || id == cudf::type_id::INT8 ||
            id == cudf::type_id::BOOL8) {
            return cudf::make_column_from_scalar(cudf::numeric_scalar<int64_t>(0, true), 1);
        }
        return num(0.0);
    }
    if (agg_type == "mean") {
        return num(std::nan(""));
    }
    if (minmax_has_identity(agg_type, in_type.id()) && identity_flag) {
        identity_used = true;
        double inf = std::numeric_limits<double>::infinity();
        return num(agg_type == "min" ? inf : -inf);
    }
    if (agg_type == "median" || agg_type == "std" || agg_type == "variance" ||
        agg_type == "quantile") {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<double>(0.0, false), 1);
    }
    if (agg_type == "any") {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<bool>(false, true), 1);
    }
    if (agg_type == "all") {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<bool>(true, true), 1);
    }
    return null_of(in_type);
}

bool is_float_type(cudf::type_id id) {
    return id == cudf::type_id::FLOAT64 || id == cudf::type_id::FLOAT32;
}

// F22: what gpu_summarise() must know about a float value column. NaN
// arrives as a valid float (computed on the GPU: `0 / 0`, `Inf - Inf`, or
// uploaded: F55 keeps R's NaN), and R counts it as missing: na.rm = TRUE drops
// it, quantile() without it errors, median() is NA. cudf treats it as a number
// (MIN/MAX skip it, quantiles sort it last), so such aggregations read
// `nan_nulled`, the column with every NaN made null. One sum reduction settles
// it for almost every column: a NaN makes the sum NaN, so a finite sum means
// no NaN; only a non-finite one (or an overflow) runs the exact check.
struct FloatColumnInfo {
    bool has_nan = false;
    std::unique_ptr<cudf::column> nan_nulled;
};

FloatColumnInfo inspect_float_column(cudf::column_view col) {
    FloatColumnInfo info;
    if (col.size() == 0 || col.null_count() == col.size()) return info;
    auto total = cudf::reduce(col, *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                              cudf::data_type{cudf::type_id::FLOAT64});
    auto* sum = static_cast<cudf::numeric_scalar<double>*>(total.get());
    if (!sum->is_valid() || std::isfinite(sum->value())) return info;

    auto nan_flags = cudf::is_nan(col);
    auto any_nan = cudf::reduce(nan_flags->view(),
                                *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::BOOL8});
    auto* flag = static_cast<cudf::numeric_scalar<bool>*>(any_nan.get());
    info.has_nan = flag->is_valid() && flag->value();
    if (info.has_nan) {
        info.nan_nulled = std::make_unique<cudf::column>(col);
        auto mask = cudf::nans_to_nulls(col);
        info.nan_nulled->set_null_mask(std::move(*mask.first), mask.second);
    }
    return info;
}

// Does this aggregation read a float column with NaN as missing? Always under
// na.rm = TRUE. Under na.rm = FALSE R gives NaN (median() NA) where cudf's
// MIN/MAX/quantile would skip or sort past the NaN, so those read it as null
// too (NA, the accepted NaN-as-NA divergence; quantile() then errors like R).
// sd()/var() give NA for any NaN (R's use = "everything"), where cudf gives
// NaN, so they read it as null as well. sum/mean already give NaN from cudf
// as R does, and n_distinct/first/last/nth keep the value itself, so they read
// the column as is.
bool agg_reads_nan_as_null(const std::string& agg_type, bool na_rm) {
    if (agg_type == "n") return false;
    return na_rm || agg_type == "min" || agg_type == "max" || agg_type == "median" ||
           agg_type == "quantile" || agg_type == "std" || agg_type == "variance";
}

// F22: the QUANTILE aggregations of one gpu_summarise() call (median() of a
// float column is one too, at 0.5). All the probabilities asked of the same
// value column ride in ONE groupby request, so cudf sorts that column once
// however many there are (a reframe() of 101 percentiles is 101 aggregations
// of one column). Each aggregation is R's type 7, which cudf's LINEAR is not
// exactly: LINEAR interpolates (1 - f) * lo + f * hi even when f == 0 (0 * Inf
// = NaN, and 0 * 1e300 next to a far neighbour is noise), where R's
// stats:::quantile.default takes "x[lo]" at a whole position. And R's index
// 1 + (n - 1) * p is computed in two roundings, so it can be exactly whole where
// cudf's position p * (n - 1) keeps a 1e-16 fraction. So every quantile also
// asks for LOWER and HIGHER (and its request counts the valid values once), and
// take() recomputes R's index: a whole one names the element itself, otherwise
// the lower value is used wherever LOWER and HIGHER agree.
struct QuantileRequests {
    struct Slot { int req = -1; int pos = -1; int lo = -1; int hi = -1; int cnt = -1; double p = 0; };
    std::vector<Slot> slot;
    std::map<int, int> request_of_key;
    std::map<int, int> count_of_key;  // position of COUNT_VALID in the request
    // max(count - 1, 0) per request, shared by every probability (take()).
    std::map<int, std::unique_ptr<cudf::column>> n1_of_request;

    explicit QuantileRequests(int num_aggs) : slot(num_aggs) {}

    // `key` identifies the value column (equal keys share a request).
    void add(std::vector<cudf::groupby::aggregation_request>& requests, int i, int key,
             cudf::column_view values, double p) {
        auto found = request_of_key.find(key);
        int r;
        if (found == request_of_key.end()) {
            r = static_cast<int>(requests.size());
            requests.emplace_back();
            requests[r].values = values;
            request_of_key[key] = r;
            count_of_key[key] = static_cast<int>(requests[r].aggregations.size());
            requests[r].aggregations.push_back(
                cudf::make_count_aggregation<cudf::groupby_aggregation>(
                    cudf::null_policy::EXCLUDE));
        } else {
            r = found->second;
        }
        auto& aggs = requests[r].aggregations;
        Slot s;
        s.req = r;
        s.p = p;
        s.pos = static_cast<int>(aggs.size());
        aggs.push_back(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
            {p}, cudf::interpolation::LINEAR));
        s.lo = static_cast<int>(aggs.size());
        aggs.push_back(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
            {p}, cudf::interpolation::LOWER));
        s.hi = static_cast<int>(aggs.size());
        aggs.push_back(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
            {p}, cudf::interpolation::HIGHER));
        s.cnt = count_of_key.at(key);
        slot[i] = s;
    }

    bool has(int i) const { return slot[i].req >= 0; }

    std::unique_ptr<cudf::column> take(std::vector<cudf::groupby::aggregation_result>& results,
                                       int i) {
        const Slot& s = slot[i];
        const cudf::data_type f64{cudf::type_id::FLOAT64};
        const cudf::data_type b8{cudf::type_id::BOOL8};
        auto linear = std::move(results[s.req].results[s.pos]);
        // LOWER/HIGHER keep the value column's type (INT32 for an integer
        // column), LINEAR is FLOAT64: bring them to the result's type.
        auto as_result_type = [&](const std::unique_ptr<cudf::column>& c)
            -> std::unique_ptr<cudf::column> {
            if (c->type() == linear->type()) return nullptr;
            return cudf::cast(c->view(), linear->type());
        };
        auto lower_cast = as_result_type(results[s.req].results[s.lo]);
        auto higher_cast = as_result_type(results[s.req].results[s.hi]);
        cudf::column_view lower = lower_cast ? lower_cast->view() : results[s.req].results[s.lo]->view();
        cudf::column_view higher = higher_cast ? higher_cast->view() : results[s.req].results[s.hi]->view();
        // A null (an all-NA group under na.rm = TRUE) compares to null, which
        // copy_if_else treats as false: it keeps `linear`, itself null there.
        auto same = cudf::binary_operation(lower, higher, cudf::binary_operator::EQUAL, b8);
        auto picked = cudf::copy_if_else(lower, linear->view(), same->view());

        // The type-7 index of stats:::quantile.default, each step its own
        // operation: it rounds the product and the +1 separately, and a fused
        // expression could contract into an FMA. n1 = max(count - 1, 0) is the
        // same for every probability of the request.
        auto& n1 = n1_of_request[s.req];
        if (!n1) {
            auto n = cudf::cast(results[s.req].results[s.cnt]->view(), f64);
            auto minus1 = cudf::binary_operation(n->view(), cudf::numeric_scalar<double>(1.0, true),
                                                 cudf::binary_operator::SUB, f64);
            n1 = cudf::binary_operation(minus1->view(), cudf::numeric_scalar<double>(0.0, true),
                                        cudf::binary_operator::NULL_MAX, f64);
        }
        auto prod = cudf::binary_operation(n1->view(), cudf::numeric_scalar<double>(s.p, true),
                                           cudf::binary_operator::MUL, f64);
        auto idx = cudf::binary_operation(prod->view(), cudf::numeric_scalar<double>(1.0, true),
                                          cudf::binary_operator::ADD, f64);
        auto idx_floor = cudf::unary_operation(idx->view(), cudf::unary_operator::FLOOR);
        auto whole = cudf::binary_operation(idx->view(), idx_floor->view(),
                                            cudf::binary_operator::EQUAL, b8);
        // A whole index names the element at 0-based idx - 1: cudf's LOWER
        // (floor of the position) where that is floor(prod), else HIGHER.
        auto pos = cudf::binary_operation(idx->view(), cudf::numeric_scalar<double>(1.0, true),
                                          cudf::binary_operator::SUB, f64);
        auto prod_floor = cudf::unary_operation(prod->view(), cudf::unary_operator::FLOOR);
        auto on_lower = cudf::binary_operation(pos->view(), prod_floor->view(),
                                               cudf::binary_operator::EQUAL, b8);
        auto at_index = cudf::copy_if_else(lower, higher, on_lower->view());
        return cudf::copy_if_else(at_index->view(), picked->view(), whole->view());
    }
};

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
                   LogicalVector na_rm, NumericVector agg_extra,
                   LogicalVector identity, bool sort_keys) {
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
    if (identity.size() != num_aggs) {
        Rcpp::stop("identity length (%d) must match agg_col_indices length (%d)",
                   identity.size(), num_aggs);
    }

    // F07: per-aggregation "min()/max() hit the empty-group identity" report,
    // returned to the R side as the "identity_used" attribute of the result
    // pointer (the R side warns with base R's text and reconciles the
    // INT32 -> FLOAT64 promotion in the schema). `identity[i]` is TRUE only
    // for min()/max() over FLOAT64/INT32 input that is not an internal
    // helper aggregation.
    LogicalVector identity_used(num_aggs, false);
    // F34: per-aggregation "an integer/logical sum() left R's integer range
    // and was returned as FLOAT64" report (see narrow_int_sum()).
    LogicalVector sum_overflow(num_aggs, false);
    // F41: per-aggregation "a Date mean()/median() had a fractional result and
    // was floored" report (see ts_agg.hpp); the caller warns naming the column.
    LogicalVector ts_fraction(num_aggs, false);
    auto finish = [&](std::unique_ptr<cudf::table> result) {
        SEXP out = make_gpu_table_xptr(std::move(result));
        Rcpp::Shield<SEXP> keep(out);
        Rf_setAttrib(out, Rf_install("identity_used"), identity_used);
        Rf_setAttrib(out, Rf_install("sum_overflow"), sum_overflow);
        Rf_setAttrib(out, Rf_install("ts_fraction"), ts_fraction);
        return out;
    };
    // Applied to each aggregation's final column, whichever path made it:
    // an integer sum() narrows to INT32 (F34), and a double sum() is never
    // -0 (F67, unsign_zero_sum(): one row per group; a result of more than
    // 65536 groups is searched for a -0 first). A double mean() takes its
    // zero sign from its sum instead (split_mean() / finish_mean()).
    auto finish_sum = [&](int i, std::unique_ptr<cudf::column>& col) {
        std::string t = Rcpp::as<std::string>(agg_types[i]);
        if (is_int_sum(t, view.column(agg_col_indices[i]).type().id())) {
            sum_overflow[i] = narrow_int_sum(col);
        } else if (t == "sum") {
            unsign_zero_sum(col);
        }
    };

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

    // F34: dplyr's min()/max() of a logical column is an integer. A BOOL8
    // input is cast to INT32 up front so it takes the INT32 path (empty-group
    // identity, double promotion, warning) unchanged.
    std::vector<std::unique_ptr<cudf::column>> bool_minmax_cast(num_aggs);
    // F41: mean()/median() of a Date/POSIXct aggregate its FLOAT64 days /
    // microseconds (ts_agg.hpp); ts_finish() converts the result back.
    std::vector<std::unique_ptr<cudf::column>> ts_float_cast(num_aggs);
    std::vector<cudf::type_id> ts_target(num_aggs, cudf::type_id::EMPTY);
    auto agg_col = [&](int i) -> cudf::column_view {
        if (ts_float_cast[i]) return ts_float_cast[i]->view();
        return bool_minmax_cast[i] ? bool_minmax_cast[i]->view()
                                   : view.column(agg_col_indices[i]);
    };
    auto ts_finish = [&](int i, std::unique_ptr<cudf::column>& col) {
        if (ts_target[i] == cudf::type_id::EMPTY) return;
        bool frac = false;
        col = float64_to_ts(std::move(col), ts_target[i], frac);
        ts_fraction[i] = frac;
    };
    // F67: R's mean() of doubles is its sum (from +0, so never -0) divided by
    // n: -0 only when a negative sum rounds to zero on the division. cudf's
    // MEAN is SUM / COUNT_VALID but keeps the sum's -0, so a double mean() is
    // requested as [SUM, COUNT_VALID] and finish_mean() unsigns the sum and
    // divides, bit-identical to cudf's own MEAN otherwise. A Date/POSIXct or
    // integer mean can't be -0 and keeps MEAN.
    auto split_mean = [&](int i, cudf::column_view values) {
        return Rcpp::as<std::string>(agg_types[i]) == "mean" && !ts_float_cast[i] &&
               values.type().id() == cudf::type_id::FLOAT64;
    };
    auto add_mean_requests = [](cudf::groupby::aggregation_request& req) {
        req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        req.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE));
    };
    auto finish_mean = [](cudf::groupby::aggregation_result& r) {
        unsign_zero_sum(r.results[0]);
        r.results[0] = cudf::binary_operation(r.results[0]->view(), r.results[1]->view(),
                                              cudf::binary_operator::DIV,
                                              cudf::data_type{cudf::type_id::FLOAT64});
    };
    for (int i = 0; i < num_aggs; ++i) {
        std::string t = Rcpp::as<std::string>(agg_types[i]);
        if ((t == "min" || t == "max") &&
            view.column(agg_col_indices[i]).type().id() == cudf::type_id::BOOL8) {
            bool_minmax_cast[i] = cudf::cast(view.column(agg_col_indices[i]),
                                             cudf::data_type{cudf::type_id::INT32});
        }
        ts_target[i] = ts_agg_target(t, view.column(agg_col_indices[i]).type().id());
        if (ts_target[i] != cudf::type_id::EMPTY) {
            ts_float_cast[i] = ts_to_float64(view.column(agg_col_indices[i]));
        }
    }

    // F22: the column each aggregation reads (`eff[i]`): agg_col(i), or for a
    // float column holding NaN its NaN-as-null copy where R counts
    // NaN as missing (agg_reads_nan_as_null()). Each float column is inspected
    // at most once per call, and only when some aggregation of it cares.
    // median() of a float column goes through QuantileRequests at 0.5
    // (`median_q`): cudf's MEDIAN is LINEAR and gives NaN next to +-Inf.
    std::map<int, FloatColumnInfo> float_info;
    std::vector<cudf::column_view> eff(num_aggs);
    std::vector<char> median_q(num_aggs, 0);
    for (int i = 0; i < num_aggs; ++i) {
        std::string t = Rcpp::as<std::string>(agg_types[i]);
        eff[i] = agg_col(i);
        if (ts_float_cast[i] || !is_float_type(eff[i].type().id()) ||
            !agg_reads_nan_as_null(t, static_cast<bool>(na_rm[i]))) {
            continue;
        }
        median_q[i] = t == "median";
        auto it = float_info.find(agg_col_indices[i]);
        if (it == float_info.end()) {
            it = float_info.emplace(agg_col_indices[i], inspect_float_column(eff[i])).first;
        }
        if (it->second.has_nan) eff[i] = it->second.nan_nulled->view();
    }
    // A quantile slot's request key: the value column, so every quantile and
    // float median() of one column shares a request (they read the same
    // `eff`); a cast copy is its own.
    auto quantile_key = [&](int i) {
        return (ts_float_cast[i] || bool_minmax_cast[i]) ? -(i + 1) : agg_col_indices[i];
    };
    auto null_double = []() {
        return cudf::make_column_from_scalar(cudf::numeric_scalar<double>(0.0, false), 1);
    };

    // F07: ungrouped summarise() of a 0-row table crashed in cudf (a 0-group
    // groupby against the 1-row count column). Build the one-row result of
    // the empty-vector reductions directly.
    if (num_groups == 0 && view.num_rows() == 0) {
        std::vector<std::unique_ptr<cudf::column>> result_columns;
        for (int i = 0; i < num_aggs; ++i) {
            std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
            bool used = false;
            result_columns.push_back(empty_input_reduction(
                agg_type, agg_col(i).type(),
                static_cast<bool>(identity[i]), used));
            identity_used[i] = used;
            finish_sum(i, result_columns.back());
            ts_finish(i, result_columns.back());
        }
        return finish(std::make_unique<cudf::table>(std::move(result_columns)));
    }

    if (num_groups == 0) {
        std::vector<std::unique_ptr<cudf::column>> result_columns;

        // One group: the whole table, keyed by a constant (cudf's groupby needs a
        // keys table). F61: built on the device once per call (not a host vector
        // copied up per aggregation), declared sorted, and one groupby serves
        // every aggregation and the quantile batch. null_policy::INCLUDE:
        // cudf::groupby's default (EXCLUDE) silently drops every row whose group
        // key is null, but dplyr's group_by()/summarise() always treats NA as its
        // own group like any other value (verified empirically, see
        // test-summarise.R's "keeps an NA group key" test and
        // test-dplyr-count.R's NA-key tests -- add_count()'s join-back
        // broadcast, R/mutate.R, depends on this too). The key is never null
        // here, but the policy stays the grouped path's.
        std::unique_ptr<cudf::column> const_key;
        std::unique_ptr<cudf::groupby::groupby> const_gb;
        auto whole_table_groupby = [&]() -> cudf::groupby::groupby& {
            if (!const_gb) {
                const_key = cudf::make_column_from_scalar(
                    cudf::numeric_scalar<int32_t>(0, true), view.num_rows());
                const_gb = std::make_unique<cudf::groupby::groupby>(
                    cudf::table_view({const_key->view()}), cudf::null_policy::INCLUDE,
                    cudf::sorted::YES);
            }
            return *const_gb;
        };

        // F22: every quantile() of the call in one groupby request per value
        // column (see QuantileRequests); the loop below only places the results.
        std::vector<std::unique_ptr<cudf::column>> quantile_cols(num_aggs);
        {
            QuantileRequests quantiles(num_aggs);
            std::vector<cudf::groupby::aggregation_request> q_requests;
            for (int i = 0; i < num_aggs; ++i) {
                bool is_quantile =
                    classify_agg(Rcpp::as<std::string>(agg_types[i])) == AggKind::QUANTILE;
                if (!is_quantile && !median_q[i]) continue;
                cudf::column_view col = eff[i];
                bool has_na = !static_cast<bool>(na_rm[i]) && col.has_nulls();
                // Phase 6, task 6.2: quantile()'s na.rm = FALSE (default) with an
                // actual NA present is a hard ERROR in base R itself ("missing
                // values and NaN's not allowed if 'na.rm' is FALSE"), verified
                // against dplyr 1.2.1 -- not an NA propagation like mean()/sum().
                // "Any null at all" equals dplyr's per-group error: it aborts the
                // whole call the moment one group hits an NA.
                if (is_quantile && has_na) {
                    Rcpp::stop("quantile(): missing values and NaN's not allowed if na.rm = FALSE. "
                               "Use na.rm = TRUE to ignore missing values.");
                }
                // median()'s na.rm = FALSE with an NA is NA (the only group).
                if (has_na) {
                    quantile_cols[i] = null_double();
                    continue;
                }
                quantiles.add(q_requests, i, quantile_key(i), col,
                              median_q[i] ? 0.5 : agg_extra[i]);
            }
            if (!q_requests.empty()) {
                auto [result_keys, result_aggs] = whole_table_groupby().aggregate(q_requests);
                for (int i = 0; i < num_aggs; ++i) {
                    if (quantiles.has(i)) quantile_cols[i] = quantiles.take(result_aggs, i);
                }
            }
        }

        for (int i = 0; i < num_aggs; ++i) {
            std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
            cudf::column_view col = eff[i];

            if (quantile_cols[i]) {
                result_columns.push_back(std::move(quantile_cols[i]));
                continue;
            }

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
                cudf::groupby::groupby& gb = whole_table_groupby();

                AggKind kind = classify_agg(agg_type);

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
                    needs_identity_fix(agg_type, static_cast<bool>(na_rm[i]), col,
                                       static_cast<bool>(identity[i]));

                std::unique_ptr<cudf::column> null_indicator_i8;
                std::vector<cudf::groupby::aggregation_request> requests;
                cudf::groupby::aggregation_request req;
                req.values = col;
                switch (kind) {
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
                        if (split_mean(i, col)) {
                            add_mean_requests(req);
                        } else {
                            req.aggregations.push_back(get_groupby_agg(agg_type));
                        }
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
                if (split_mean(i, col)) finish_mean(result_aggs[0]);

                std::unique_ptr<cudf::column> agg_result = std::move(result_aggs[0].results[0]);
                if (need_na_prop) {
                    agg_result = null_out_groups_with_any_null(
                        std::move(agg_result), result_aggs[1].results[0]->view());
                } else if (need_identity_fix && minmax_has_identity(agg_type, col.type().id())) {
                    cudf::column_view counts = result_aggs[1].results[0]->view();
                    identity_used[i] = apply_minmax_identity(agg_result, agg_type == "min", &counts);
                } else if (need_identity_fix) {
                    auto identity_scalar = make_identity_scalar(agg_type, agg_result->view().type());
                    agg_result = substitute_identity_for_empty_groups(
                        std::move(agg_result), result_aggs[1].results[0]->view(), *identity_scalar);
                }

                finish_sum(i, agg_result);
                ts_finish(i, agg_result);
                result_columns.push_back(std::move(agg_result));
            }
        }

        return finish(std::make_unique<cudf::table>(std::move(result_columns)));
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
    QuantileRequests quantiles(num_aggs);

    for (int i = 0; i < num_aggs; ++i) {
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        cudf::column_view value_view = eff[i];
        AggKind kind = classify_agg(agg_type);
        bool na_rm_flag = static_cast<bool>(na_rm[i]);

        // Phase 6, task 6.2: quantile()'s na.rm = FALSE (default) with an
        // actual NA present is a hard ERROR in base R itself -- see the
        // identical check (and its own docs) in the num_groups == 0 branch
        // above.
        if (kind == AggKind::QUANTILE || median_q[i]) {
            bool has_na = !na_rm_flag && value_view.has_nulls();
            if (kind == AggKind::QUANTILE && has_na) {
                Rcpp::stop("quantile(): missing values and NaN's not allowed if na.rm = FALSE. "
                           "Use na.rm = TRUE to ignore missing values.");
            }
            // F22: shares one request (one sort) per value column.
            quantiles.add(requests, i, quantile_key(i), value_view,
                          median_q[i] ? 0.5 : agg_extra[i]);
            main_request_idx[i] = quantiles.slot[i].req;
            // median()'s na.rm = FALSE: a group with an NA is NA (the usual
            // na-prop step, applied to the quantile result below).
            if (has_na) {
                auto null_indicator_bool = cudf::is_null(value_view);
                null_indicator_cols.push_back(cudf::cast(
                    null_indicator_bool->view(), cudf::data_type{cudf::type_id::INT8}));
                cudf::groupby::aggregation_request null_req;
                null_req.values = null_indicator_cols.back()->view();
                null_req.aggregations.push_back(
                    cudf::make_max_aggregation<cudf::groupby_aggregation>());
                na_prop_request_idx[i] = static_cast<int>(requests.size());
                requests.push_back(std::move(null_req));
            }
            continue;
        }

        cudf::groupby::aggregation_request req;
        req.values = value_view;
        switch (kind) {
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
                if (split_mean(i, value_view)) {
                    add_mean_requests(req);
                } else {
                    req.aggregations.push_back(get_groupby_agg(agg_type));
                }
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
        if (kind == AggKind::SIMPLE &&
            needs_identity_fix(agg_type, na_rm_flag, value_view, static_cast<bool>(identity[i]))) {
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
        if (split_mean(i, eff[i])) finish_mean(result_aggs[main_request_idx[i]]);
    }

    // The quantile results leave their shared requests first, so everything
    // below that reads `results[0]` of a request sees only its own aggregation.
    std::vector<std::unique_ptr<cudf::column>> quantile_cols(num_aggs);
    for (int i = 0; i < num_aggs; ++i) {
        if (quantiles.has(i)) quantile_cols[i] = quantiles.take(result_aggs, i);
    }

    for (int i = 0; i < num_aggs; ++i) {
        if (na_prop_request_idx[i] < 0) {
            continue;
        }
        auto& target = quantile_cols[i] ? quantile_cols[i]
                                        : result_aggs[main_request_idx[i]].results[0];
        target = null_out_groups_with_any_null(
            std::move(target), result_aggs[na_prop_request_idx[i]].results[0]->view());
    }

    for (int i = 0; i < num_aggs; ++i) {
        if (identity_fix_request_idx[i] < 0) {
            continue;
        }
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        auto& main_result = result_aggs[main_request_idx[i]].results[0];
        if (minmax_has_identity(agg_type, agg_col(i).type().id())) {
            cudf::column_view counts = result_aggs[identity_fix_request_idx[i]].results[0]->view();
            identity_used[i] = apply_minmax_identity(main_result, agg_type == "min", &counts);
            continue;
        }
        auto identity_scalar = make_identity_scalar(agg_type, main_result->view().type());
        main_result = substitute_identity_for_empty_groups(
            std::move(main_result), result_aggs[identity_fix_request_idx[i]].results[0]->view(),
            *identity_scalar);
    }

    // F07: dplyr evaluates min()/max() once on an empty chunk when the input
    // has no rows, so a 0-row grouped result types an INT32 min()/max()
    // double (and warns) even though there is no group left to hold a value.
    if (view.num_rows() == 0) {
        for (int i = 0; i < num_aggs; ++i) {
            std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
            if (identity[i] &&
                minmax_has_identity(agg_type, agg_col(i).type().id())) {
                apply_minmax_identity(result_aggs[main_request_idx[i]].results[0],
                                      agg_type == "min", nullptr);
                identity_used[i] = true;
            }
        }
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
        result_columns.push_back(quantile_cols[i]
                                     ? std::move(quantile_cols[i])
                                     : std::move(result_aggs[main_request_idx[i]].results[0]));
        finish_sum(i, result_columns.back());
        ts_finish(i, result_columns.back());
    }

    auto result = std::make_unique<cudf::table>(std::move(result_columns));

    // F33: dplyr's group_by() order -- keys ascending, NA last. Bytewise
    // string compare is the C locale, factor codes give level order, BOOL8
    // gives FALSE < TRUE. The keys are distinct, so an unstable sort is
    // fine. Done on the k-row result, before R permutes [keys, aggs].
    if (sort_keys && result->num_rows() > 1) {
        cudf::table_view key_view = result->view().select(
            [&] { std::vector<cudf::size_type> k(num_groups); std::iota(k.begin(), k.end(), 0); return k; }());
        std::vector<cudf::order> order(num_groups, cudf::order::ASCENDING);
        std::vector<cudf::null_order> nulls(num_groups, cudf::null_order::AFTER);
        auto idx = cudf::sorted_order(key_view, order, nulls);
        result = cudf::gather(result->view(), idx->view(),
                              cudf::out_of_bounds_policy::DONT_CHECK);
    }

    return finish(std::move(result));
}

// Phase 9, task 1: grouping metadata API (group_data()/group_keys()/
// group_rows()/group_indices()/group_size(); n_groups() goes through
// R/group-by.R::count_groups(), which usually needs only gpu_group_count()
// at the end of this file).
//
// Representation chosen (cheapest that serves all these R-facing queries):
// a small DISTINCT-keys table (one row per group, in dplyr's own canonical
// order -- see below) plus two small host-side integer vectors: `offsets`
// (length num_groups + 1, the group boundary positions) and `perm` (length
// nrow, the ORIGINAL 1-based row index for every row, permuted into
// group-sorted order). `.rows` for group i is `perm[offsets[i]+1 ..
// offsets[i+1]]` (R-side 1-based slicing of `perm`, since `offsets` are
// 0-based boundary positions into it) -- group_data()/group_keys()/
// group_rows()/group_size() all derive from the distinct-keys
// table + offsets alone; only group_indices() needs `perm` itself (see
// R/group-by.R::compute_group_data()).
//
// Row order (verified empirically against dplyr 1.2.1, see this task's own
// test file): group_by()+group_data() emits groups in ASCENDING key order,
// NA sorted LAST, with each group's `.rows` in original-row order. This
// falls out for free from `cudf::groupby::groupby`'s own DEFAULT behavior
// (`sorted::NO`, no explicit `column_order`/`null_precedence`): its
// internal `sort_groupby_helper::key_sort_order()` (verified against cudf
// source, cpp/src/groupby/sort/sort_helper.cu) calls
// `stable_sorted_order(keys, {}, precedence)` where `{}` (empty
// column_order) means every column ASCENDING, and `precedence` defaults to
// `null_order::AFTER` for every column whenever `_null_precedence` (which
// we never set) is empty -- i.e. cudf's own "unsorted-keys" groupby path
// already IS dplyr's canonical group order, no manual pre-sort needed
// (contrast with src/ops_window.cpp's window families, which pre-sort
// manually via `stable_sorted_order()` themselves and construct their
// groupby with `sorted::YES` -- not needed here since we WANT this exact
// default behavior, not an identity-permutation optimization).
//
// null_policy::INCLUDE: same rationale as gpu_summarise() above -- dplyr's
// group_by() always treats NA as its own group, never drops NA-key rows.
// [[Rcpp::export]]
List gpu_group_info(SEXP xptr, IntegerVector group_col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_key_cols = group_col_indices.size();
    if (n_key_cols == 0) {
        Rcpp::stop("gpu_group_info() requires at least one group column index");
    }
    for (int i = 0; i < n_key_cols; ++i) {
        if (group_col_indices[i] < 0 || group_col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Group column index out of bounds: %d", group_col_indices[i]);
        }
    }

    std::vector<cudf::column_view> key_views;
    key_views.reserve(n_key_cols);
    for (int i = 0; i < n_key_cols; ++i) {
        key_views.push_back(view.column(group_col_indices[i]));
    }
    cudf::table_view keys_table(key_views);

    cudf::size_type nrow = view.num_rows();

    // Row-index column [0, 1, ..., nrow-1] as get_groups()'s "values" table
    // -- its gathered/grouped-order output IS the permutation (`perm`) this
    // function returns, built the same host-vector -> device_buffer ->
    // cudf::column idiom gpu_summarise() already uses above (never a
    // device-side Thrust call).
    std::vector<int32_t> row_idx_data(static_cast<size_t>(nrow));
    std::iota(row_idx_data.begin(), row_idx_data.end(), 0);

    rmm::device_buffer row_idx_buf(row_idx_data.size() * sizeof(int32_t),
                                    rmm::cuda_stream_view(),
                                    rmm::mr::get_current_device_resource_ref());
    if (!row_idx_data.empty()) {
        check_cuda(cudaMemcpy(row_idx_buf.data(), row_idx_data.data(),
                               row_idx_data.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "gpu_group_info row-index upload");
    }
    auto row_idx_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        nrow,
        std::move(row_idx_buf),
        rmm::device_buffer{},
        0
    );
    std::vector<cudf::column_view> row_idx_views = { row_idx_col->view() };
    cudf::table_view row_idx_table(row_idx_views);

    cudf::groupby::groupby gb(keys_table, cudf::null_policy::INCLUDE);
    auto groups_result = gb.get_groups(row_idx_table);

    std::vector<cudf::size_type> offsets_vec = groups_result.offsets;
    cudf::size_type num_groups = offsets_vec.empty()
        ? 0
        : static_cast<cudf::size_type>(offsets_vec.size()) - 1;

    // Distinct keys table: gather the FIRST row of each group out of the
    // (already sorted+grouped) per-row keys table -- `offsets[i]` is that
    // row's position for group i.
    std::unique_ptr<cudf::table> distinct_keys;
    if (num_groups > 0) {
        std::vector<int32_t> first_rows(static_cast<size_t>(num_groups));
        for (cudf::size_type i = 0; i < num_groups; ++i) {
            first_rows[static_cast<size_t>(i)] = static_cast<int32_t>(offsets_vec[static_cast<size_t>(i)]);
        }

        rmm::device_buffer first_rows_buf(first_rows.size() * sizeof(int32_t),
                                           rmm::cuda_stream_view(),
                                           rmm::mr::get_current_device_resource_ref());
        check_cuda(cudaMemcpy(first_rows_buf.data(), first_rows.data(),
                               first_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "gpu_group_info first-row-of-group upload");
        auto first_rows_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            static_cast<cudf::size_type>(first_rows.size()),
            std::move(first_rows_buf),
            rmm::device_buffer{},
            0
        );

        distinct_keys = cudf::gather(groups_result.keys->view(), first_rows_col->view(),
                                      cudf::out_of_bounds_policy::DONT_CHECK);
    } else {
        distinct_keys = cudf::empty_like(keys_table);
    }

    // perm: original row index (already 1-based for direct R use as
    // .rows/group_indices()) for every row, in group-sorted order --
    // segmented by `offsets` (still 0-based boundary positions) into each
    // group's own .rows on the R side.
    IntegerVector perm(nrow);
    if (nrow > 0) {
        cudf::column_view perm_view = groups_result.values->view().column(0);
        std::vector<int32_t> host_perm(static_cast<size_t>(nrow));
        check_cuda(cudaMemcpy(host_perm.data(), perm_view.data<int32_t>(),
                               static_cast<size_t>(nrow) * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "gpu_group_info perm download");
        for (cudf::size_type i = 0; i < nrow; ++i) {
            perm[i] = host_perm[static_cast<size_t>(i)] + 1;
        }
    }

    IntegerVector offsets(offsets_vec.begin(), offsets_vec.end());

    return List::create(
        Named("keys") = make_gpu_table_xptr(std::move(distinct_keys)),
        Named("offsets") = offsets,
        Named("perm") = perm
    );
}

// Number of groups gpu_group_info() would return (n_groups(), and the group
// count in a grouped table's printed header; R/group-by.R::count_groups()).
// Same groupby as gpu_group_info() (null_policy::INCLUDE, so NA is its own
// group and NaN keys stay separate from NA), but `get_groups()` is called
// with no values table, so there is no row-index upload and no n-row
// permutation download, and nothing is returned to R but the count.
// get_groups() itself still stable-sorts the keys, gathers an n-row sorted
// copy of them on the device, and copies all num_groups + 1 offsets to the
// host; only their count is used.
// [[Rcpp::export]]
int gpu_group_count(SEXP xptr, IntegerVector group_col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_key_cols = group_col_indices.size();
    if (n_key_cols == 0) {
        Rcpp::stop("gpu_group_count() requires at least one group column index");
    }
    std::vector<cudf::column_view> key_views;
    key_views.reserve(n_key_cols);
    for (int i = 0; i < n_key_cols; ++i) {
        if (group_col_indices[i] < 0 || group_col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Group column index out of bounds: %d", group_col_indices[i]);
        }
        key_views.push_back(view.column(group_col_indices[i]));
    }
    cudf::table_view keys_table(key_views);

    if (view.num_rows() == 0) {
        return 0;
    }

    cudf::groupby::groupby gb(keys_table, cudf::null_policy::INCLUDE);
    auto groups_result = gb.get_groups();
    return groups_result.offsets.empty()
        ? 0
        : static_cast<int>(groups_result.offsets.size()) - 1;
}

// The distinct group keys of a table (one row per observed group, in the
// same order as gpu_group_info()'s `keys`: ascending, NA last) and nothing
// else: no row-index upload, no n-row permutation download (F71's fixed-key
// capture needs only the keys; R/group-by.R::slice_group_keys()). The same
// groupby as gpu_group_info() (null_policy::INCLUDE).
// [[Rcpp::export]]
SEXP gpu_group_keys(SEXP xptr, IntegerVector group_col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_key_cols = group_col_indices.size();
    if (n_key_cols == 0) {
        Rcpp::stop("gpu_group_keys() requires at least one group column index");
    }
    std::vector<cudf::column_view> key_views;
    key_views.reserve(n_key_cols);
    for (int i = 0; i < n_key_cols; ++i) {
        if (group_col_indices[i] < 0 || group_col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Group column index out of bounds: %d", group_col_indices[i]);
        }
        key_views.push_back(view.column(group_col_indices[i]));
    }
    cudf::table_view keys_table(key_views);

    if (view.num_rows() == 0) {
        return make_gpu_table_xptr(cudf::empty_like(keys_table));
    }

    cudf::groupby::groupby gb(keys_table, cudf::null_policy::INCLUDE);
    auto groups_result = gb.get_groups();
    cudf::size_type num_groups = groups_result.offsets.empty()
        ? 0
        : static_cast<cudf::size_type>(groups_result.offsets.size()) - 1;
    if (num_groups == 0) {
        return make_gpu_table_xptr(cudf::empty_like(keys_table));
    }

    // the first row of each group out of the grouped per-row keys
    std::vector<int32_t> first_rows(static_cast<size_t>(num_groups));
    for (cudf::size_type i = 0; i < num_groups; ++i) {
        first_rows[static_cast<size_t>(i)] =
            static_cast<int32_t>(groups_result.offsets[static_cast<size_t>(i)]);
    }
    rmm::device_buffer first_rows_buf(first_rows.size() * sizeof(int32_t),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    check_cuda(cudaMemcpy(first_rows_buf.data(), first_rows.data(),
                           first_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
               "gpu_group_keys first-row-of-group upload");
    auto first_rows_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_groups,
        std::move(first_rows_buf),
        rmm::device_buffer{},
        0
    );
    return make_gpu_table_xptr(cudf::gather(groups_result.keys->view(), first_rows_col->view(),
                                             cudf::out_of_bounds_policy::DONT_CHECK));
}

// Per-column null counts of the given columns (F71: whether a key column
// holds an NA, one O(n) mask count per column, no groupby).
// [[Rcpp::export]]
IntegerVector gpu_null_counts(SEXP xptr, IntegerVector col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    IntegerVector out(col_indices.size());
    for (int i = 0; i < col_indices.size(); ++i) {
        if (col_indices[i] < 0 || col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Column index out of bounds: %d", col_indices[i]);
        }
        out[i] = static_cast<int>(view.column(col_indices[i]).null_count());
    }
    return out;
}

// The number of distinct key combinations of the given columns, equal to
// gpu_group_count() (NA is its own key, null_equality::EQUAL) but counted with
// a hash set instead of a sort-based groupby: F71 compares the group counts of
// a table and of a row subset of it to see whether a group was lost, which
// needs no ordering. 0 for a 0-row table.
// [[Rcpp::export]]
int gpu_distinct_key_count(SEXP xptr, IntegerVector group_col_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int n_key_cols = group_col_indices.size();
    if (n_key_cols == 0) {
        Rcpp::stop("gpu_distinct_key_count() requires at least one column index");
    }
    std::vector<cudf::column_view> key_views;
    key_views.reserve(n_key_cols);
    for (int i = 0; i < n_key_cols; ++i) {
        if (group_col_indices[i] < 0 || group_col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Group column index out of bounds: %d", group_col_indices[i]);
        }
        key_views.push_back(view.column(group_col_indices[i]));
    }
    cudf::table_view keys_table(key_views);

    if (view.num_rows() == 0) {
        return 0;
    }
    return static_cast<int>(cudf::distinct_count(keys_table, cudf::null_equality::EQUAL));
}
