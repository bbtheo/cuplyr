// src/ops_groupby.cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplr {

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

} // namespace cuplr

// [[Rcpp::export]]
SEXP gpu_summarise(SEXP xptr, IntegerVector group_indices,
                   IntegerVector agg_col_indices, CharacterVector agg_types) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int num_aggs = agg_col_indices.size();
    int num_groups = group_indices.size();

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
                int64_t count = view.num_rows();
                rmm::device_buffer data(sizeof(int64_t),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
                check_cuda(cudaMemcpy(data.data(), &count, sizeof(int64_t), cudaMemcpyHostToDevice),
                           "gpu_summarise count memcpy");
                result_columns.push_back(std::make_unique<cudf::column>(
                    cudf::data_type{cudf::type_id::INT64},
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

                cudf::groupby::groupby gb(keys_table);

                std::vector<cudf::groupby::aggregation_request> requests;
                cudf::groupby::aggregation_request req;
                req.values = col;
                req.aggregations.push_back(get_groupby_agg(agg_type));
                requests.push_back(std::move(req));

                auto [result_keys, result_aggs] = gb.aggregate(requests);

                result_columns.push_back(std::move(result_aggs[0].results[0]));
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

    cudf::groupby::groupby gb(keys_table);

    std::vector<cudf::groupby::aggregation_request> requests;
    for (int i = 0; i < num_aggs; ++i) {
        std::string agg_type = Rcpp::as<std::string>(agg_types[i]);

        cudf::groupby::aggregation_request req;
        req.values = view.column(agg_col_indices[i]);
        req.aggregations.push_back(get_groupby_agg(agg_type));
        requests.push_back(std::move(req));
    }

    auto [result_keys, result_aggs] = gb.aggregate(requests);

    std::vector<std::unique_ptr<cudf::column>> result_columns;

    for (int i = 0; i < result_keys->num_columns(); ++i) {
        result_columns.push_back(std::make_unique<cudf::column>(result_keys->get_column(i)));
    }

    for (size_t i = 0; i < result_aggs.size(); ++i) {
        result_columns.push_back(std::move(result_aggs[i].results[0]));
    }

    auto result = std::make_unique<cudf::table>(std::move(result_columns));
    return make_gpu_table_xptr(std::move(result));
}
