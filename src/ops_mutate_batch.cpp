// src/ops_mutate_batch.cpp
// Batched mutate operations for fused lazy evaluation
#include "gpu_table.hpp"
#include "ops_common.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <map>
#include <set>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

namespace cuplyr {

// Convert R type string to cudf type
inline cudf::data_type get_cudf_type(const std::string& type_str) {
    if (type_str == "FLOAT64") return cudf::data_type{cudf::type_id::FLOAT64};
    if (type_str == "FLOAT32") return cudf::data_type{cudf::type_id::FLOAT32};
    if (type_str == "INT64") return cudf::data_type{cudf::type_id::INT64};
    if (type_str == "INT32") return cudf::data_type{cudf::type_id::INT32};
    if (type_str == "INT16") return cudf::data_type{cudf::type_id::INT16};
    if (type_str == "INT8") return cudf::data_type{cudf::type_id::INT8};
    if (type_str == "BOOL8") return cudf::data_type{cudf::type_id::BOOL8};
    if (type_str == "STRING") return cudf::data_type{cudf::type_id::STRING};
    // Default
    return cudf::data_type{cudf::type_id::FLOAT64};
}

} // namespace cuplyr

// [[Rcpp::export]]
SEXP gpu_mutate_batch(SEXP xptr, Rcpp::List expressions, Rcpp::List input_schema) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    Rcpp::CharacterVector col_names = input_schema["names"];
    Rcpp::CharacterVector col_types = input_schema["types"];

    // Track columns: name -> column_view
    // For computed columns, we keep the unique_ptr in owned_columns
    std::map<std::string, cudf::column_view> col_views;
    std::vector<std::unique_ptr<cudf::column>> owned_columns;
    std::map<std::string, size_t> output_col_indices;

    // Initialize with input columns (views, not copies)
    for (int i = 0; i < view.num_columns(); ++i) {
        col_views[Rcpp::as<std::string>(col_names[i])] = view.column(i);
    }

    // Process expressions in order (assumed toposorted by R code)
    for (int i = 0; i < expressions.size(); ++i) {
        Rcpp::List expr = expressions[i];
        std::string output_col = Rcpp::as<std::string>(expr["output_col"]);
        Rcpp::CharacterVector input_cols_r = expr["input_cols"];
        std::string op = Rcpp::as<std::string>(expr["op"]);
        std::string output_type_str = Rcpp::as<std::string>(expr["output_type"]);

        cudf::data_type output_type = get_cudf_type(output_type_str);
        std::unique_ptr<cudf::column> new_col;

        if (op == "copy") {
            // Deep copy with null mask preserved
            std::string src = Rcpp::as<std::string>(input_cols_r[0]);
            auto it = col_views.find(src);
            if (it == col_views.end()) {
                Rcpp::stop("Source column not found: %s", src.c_str());
            }
            new_col = std::make_unique<cudf::column>(it->second);
        } else if (expr.containsElementNamed("scalar") && !Rf_isNull(expr["scalar"])) {
            // Column op scalar
            double scalar_val = Rcpp::as<double>(expr["scalar"]);

            std::string col = Rcpp::as<std::string>(input_cols_r[0]);
            auto it = col_views.find(col);
            if (it == col_views.end()) {
                Rcpp::stop("Input column not found: %s", col.c_str());
            }

            // Create scalar with appropriate type
            auto scalar = cudf::make_numeric_scalar(output_type);
            if (output_type.id() == cudf::type_id::FLOAT64) {
                static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(scalar_val);
            } else if (output_type.id() == cudf::type_id::INT32) {
                static_cast<cudf::numeric_scalar<int32_t>*>(scalar.get())->set_value(
                    static_cast<int32_t>(scalar_val));
            } else if (output_type.id() == cudf::type_id::INT64) {
                static_cast<cudf::numeric_scalar<int64_t>*>(scalar.get())->set_value(
                    static_cast<int64_t>(scalar_val));
            } else if (output_type.id() == cudf::type_id::FLOAT32) {
                static_cast<cudf::numeric_scalar<float>*>(scalar.get())->set_value(
                    static_cast<float>(scalar_val));
            } else {
                // Default to double
                static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(scalar_val);
            }

            new_col = cudf::binary_operation(
                it->second, *scalar, get_arith_op(op), output_type
            );
        } else {
            // Column op column
            if (input_cols_r.size() < 2) {
                Rcpp::stop("Column-column operation requires two input columns");
            }
            std::string col1 = Rcpp::as<std::string>(input_cols_r[0]);
            std::string col2 = Rcpp::as<std::string>(input_cols_r[1]);

            auto it1 = col_views.find(col1);
            auto it2 = col_views.find(col2);
            if (it1 == col_views.end()) {
                Rcpp::stop("Input column not found: %s", col1.c_str());
            }
            if (it2 == col_views.end()) {
                Rcpp::stop("Input column not found: %s", col2.c_str());
            }

            new_col = cudf::binary_operation(
                it1->second, it2->second, get_arith_op(op), output_type
            );
        }

        // Update col_views for dependent expressions
        col_views[output_col] = new_col->view();

        // Store ownership
        owned_columns.push_back(std::move(new_col));
        output_col_indices[output_col] = owned_columns.size() - 1;
    }

    // Build final table
    // Strategy: keep input columns in order, replacing if output exists,
    // then append new columns that aren't replacements

    std::set<std::string> output_names;
    for (int i = 0; i < expressions.size(); ++i) {
        Rcpp::List expr = expressions[i];
        output_names.insert(Rcpp::as<std::string>(expr["output_col"]));
    }

    std::vector<std::unique_ptr<cudf::column>> final_columns;
    std::set<std::string> added_outputs;

    auto take_output_column = [&](const std::string& name) -> std::unique_ptr<cudf::column> {
        auto it = output_col_indices.find(name);
        if (it == output_col_indices.end()) {
            return nullptr;
        }
        size_t idx = it->second;
        if (idx >= owned_columns.size() || owned_columns[idx] == nullptr) {
            return nullptr;
        }
        return std::move(owned_columns[idx]);
    };

    // First, process input columns (replace or copy)
    for (int i = 0; i < view.num_columns(); ++i) {
        std::string name = Rcpp::as<std::string>(col_names[i]);

        if (output_names.count(name)) {
            // This column is being replaced - use the computed version
            auto replaced = take_output_column(name);
            if (replaced) {
                final_columns.push_back(std::move(replaced));
            } else {
                // Fallback to original if something went wrong
                final_columns.push_back(std::make_unique<cudf::column>(view.column(i)));
            }
            added_outputs.insert(name);
        } else {
            // Keep original (deep copy)
            final_columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
    }

    // Then append new columns that weren't replacements
    for (int i = 0; i < expressions.size(); ++i) {
        Rcpp::List expr = expressions[i];
        std::string output_col = Rcpp::as<std::string>(expr["output_col"]);

        if (!added_outputs.count(output_col)) {
            // New column, append it
            auto added = take_output_column(output_col);
            if (added) {
                final_columns.push_back(std::move(added));
            }
            added_outputs.insert(output_col);
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(final_columns));
    return make_gpu_table_xptr(std::move(result));
}
