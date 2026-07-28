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
    } else {
        Rcpp::stop("Unknown aggregation type: " + agg_type);
    }
    return nullptr;
}

// Phase 6, task 6.1 (Fix A): does this aggregation kind need R's own
// na.rm = FALSE whole-group-null propagation (see the file-level rationale
// at gpu_summarise())? n() is deliberately excluded -- it counts rows
// regardless of nulls, matching dplyr's n() already.
bool agg_na_propagates(const std::string& agg_type) {
    return agg_type == "sum" || agg_type == "mean" || agg_type == "min" ||
           agg_type == "max" || agg_type == "std" || agg_type == "variance";
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
                   LogicalVector na_rm) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int num_aggs = agg_col_indices.size();
    int num_groups = group_indices.size();

    if (na_rm.size() != num_aggs) {
        Rcpp::stop("na_rm length (%d) must match agg_col_indices length (%d)",
                   na_rm.size(), num_aggs);
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

                // Phase 6, task 6.1 (Fix A): if this aggregation needs
                // na.rm=FALSE whole-group-null propagation, compute the
                // is_null-indicator MAX request in the SAME gb.aggregate()
                // call as the main aggregation -- one call guarantees both
                // results share the exact same (single, whole-table-as-one-
                // group) row, so no separate-call group-order assumption is
                // needed here.
                bool need_na_prop = agg_na_propagates(agg_type) &&
                    !static_cast<bool>(na_rm[i]) && col.has_nulls();

                std::unique_ptr<cudf::column> null_indicator_i8;
                std::vector<cudf::groupby::aggregation_request> requests;
                cudf::groupby::aggregation_request req;
                req.values = col;
                req.aggregations.push_back(get_groupby_agg(agg_type));
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
                }

                auto [result_keys, result_aggs] = gb.aggregate(requests);

                std::unique_ptr<cudf::column> agg_result = std::move(result_aggs[0].results[0]);
                if (need_na_prop) {
                    agg_result = null_out_groups_with_any_null(
                        std::move(agg_result), result_aggs[1].results[0]->view());
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
    // Must outlive the gb.aggregate() call below (requests reference their views).
    std::vector<std::unique_ptr<cudf::column>> null_indicator_cols;

    for (int i = 0; i < num_aggs; ++i) {
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);
        cudf::column_view value_view = view.column(agg_col_indices[i]);

        cudf::groupby::aggregation_request req;
        req.values = value_view;
        req.aggregations.push_back(get_groupby_agg(agg_type));
        main_request_idx[i] = static_cast<int>(requests.size());
        requests.push_back(std::move(req));

        bool need_na_prop = agg_na_propagates(agg_type) &&
            !static_cast<bool>(na_rm[i]) && value_view.has_nulls();
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
