// src/ops_bind.cpp
// GPU bind_rows and bind_cols implementations using cuDF
#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/unary.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <vector>
#include <string>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_bind_cols_impl(List table_ptrs) {
    using namespace cuplyr;

    if (table_ptrs.size() == 0) {
        Rcpp::stop("No tables provided to bind_cols");
    }

    // Get first table to establish row count
    Rcpp::XPtr<GpuTablePtr> first_ptr(table_ptrs[0]);
    cudf::table_view first_view = get_table_view(first_ptr);
    cudf::size_type expected_rows = first_view.num_rows();

    // Gather all columns from all tables
    std::vector<std::unique_ptr<cudf::column>> all_columns;

    for (int t = 0; t < table_ptrs.size(); ++t) {
        Rcpp::XPtr<GpuTablePtr> ptr(table_ptrs[t]);
        cudf::table_view view = get_table_view(ptr);

        // Verify row count matches
        if (view.num_rows() != expected_rows) {
            Rcpp::stop("All tables must have same number of rows. "
                       "Table 1 has %d rows, table %d has %d rows.",
                       expected_rows, t + 1, view.num_rows());
        }

        // Copy columns from this table
        for (cudf::size_type c = 0; c < view.num_columns(); ++c) {
            all_columns.push_back(
                std::make_unique<cudf::column>(view.column(c))
            );
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(all_columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_bind_rows_aligned(List table_ptrs) {
    // Assumes all tables have identical schema (same columns, same order, same types)
    // Schema alignment is done in R layer
    using namespace cuplyr;

    if (table_ptrs.size() == 0) {
        Rcpp::stop("No tables provided to bind_rows");
    }

    if (table_ptrs.size() == 1) {
        // Single table - just copy it
        Rcpp::XPtr<GpuTablePtr> ptr(table_ptrs[0]);
        cudf::table_view view = get_table_view(ptr);

        std::vector<std::unique_ptr<cudf::column>> columns;
        for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
        auto result = std::make_unique<cudf::table>(std::move(columns));
        return make_gpu_table_xptr(std::move(result));
    }

    // Build vector of table_views for concatenate
    std::vector<cudf::table_view> views;
    views.reserve(table_ptrs.size());

    for (int i = 0; i < table_ptrs.size(); ++i) {
        Rcpp::XPtr<GpuTablePtr> ptr(table_ptrs[i]);
        views.push_back(get_table_view(ptr));
    }

    // Verify schema compatibility
    cudf::size_type ncol = views[0].num_columns();
    for (size_t i = 1; i < views.size(); ++i) {
        if (views[i].num_columns() != ncol) {
            Rcpp::stop("Schema mismatch: table 1 has %d columns, table %d has %d columns. "
                       "Schema alignment should be done in R layer.",
                       ncol, static_cast<int>(i) + 1, views[i].num_columns());
        }
        // Type check
        for (cudf::size_type c = 0; c < ncol; ++c) {
            if (views[i].column(c).type().id() != views[0].column(c).type().id()) {
                Rcpp::stop("Type mismatch at column %d: table 1 has type %d, table %d has type %d. "
                           "Type coercion should be done in R layer.",
                           c, static_cast<int>(views[0].column(c).type().id()),
                           static_cast<int>(i) + 1, static_cast<int>(views[i].column(c).type().id()));
            }
        }
    }

    // Concatenate
    auto result = cudf::concatenate(views);
    return make_gpu_table_xptr(std::move(result));
}

// Create a null column of the specified type and size
// [[Rcpp::export]]
SEXP gpu_make_null_column(int nrows, std::string type_str) {
    using namespace cuplyr;

    if (nrows < 0) {
        Rcpp::stop("nrows must be non-negative");
    }

    cudf::size_type n = static_cast<cudf::size_type>(nrows);

    // Handle STRING type separately - it needs special construction
    if (type_str == "STRING") {
        // Create empty strings column with all nulls
        // For strings, we create an empty offsets column and set all null mask
        if (n == 0) {
            auto empty_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
            std::vector<std::unique_ptr<cudf::column>> cols;
            cols.push_back(std::move(empty_col));
            auto tbl = std::make_unique<cudf::table>(std::move(cols));
            return make_gpu_table_xptr(std::move(tbl));
        }

        // Create offsets: n+1 zeros (each string has zero length)
        std::vector<int32_t> offsets(n + 1, 0);
        rmm::device_buffer offsets_buf(offsets.data(), offsets.size() * sizeof(int32_t),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());

        auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            n + 1,
            std::move(offsets_buf),
            rmm::device_buffer{},
            0
        );

        // Empty char data
        rmm::device_buffer char_data(0, rmm::cuda_stream_view(),
                                     rmm::mr::get_current_device_resource_ref());

        // Create null mask - all nulls
        std::vector<uint8_t> validity(cudf::bitmask_allocation_size_bytes(n), 0x00);
        rmm::device_buffer null_mask(validity.data(), validity.size(),
                                     rmm::cuda_stream_view(),
                                     rmm::mr::get_current_device_resource_ref());

        std::vector<std::unique_ptr<cudf::column>> children;
        children.push_back(std::move(offsets_col));

        auto str_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::STRING},
            n,
            std::move(char_data),
            std::move(null_mask),
            n,  // null_count = all rows
            std::move(children)
        );

        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(std::move(str_col));
        auto tbl = std::make_unique<cudf::table>(std::move(cols));
        return make_gpu_table_xptr(std::move(tbl));
    }

    // For fixed-width types, use column factory
    cudf::data_type dtype;

    if (type_str == "FLOAT64") {
        dtype = cudf::data_type{cudf::type_id::FLOAT64};
    } else if (type_str == "FLOAT32") {
        dtype = cudf::data_type{cudf::type_id::FLOAT32};
    } else if (type_str == "INT32") {
        dtype = cudf::data_type{cudf::type_id::INT32};
    } else if (type_str == "INT64") {
        dtype = cudf::data_type{cudf::type_id::INT64};
    } else if (type_str == "INT16") {
        dtype = cudf::data_type{cudf::type_id::INT16};
    } else if (type_str == "INT8") {
        dtype = cudf::data_type{cudf::type_id::INT8};
    } else if (type_str == "BOOL8") {
        dtype = cudf::data_type{cudf::type_id::BOOL8};
    } else if (type_str == "TIMESTAMP_DAYS") {
        dtype = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    } else if (type_str == "TIMESTAMP_MICROSECONDS") {
        dtype = cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
    } else if (type_str == "TIMESTAMP_MILLISECONDS") {
        dtype = cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
    } else if (type_str == "TIMESTAMP_SECONDS") {
        dtype = cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS};
    } else if (type_str == "TIMESTAMP_NANOSECONDS") {
        dtype = cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
    } else {
        Rcpp::stop("Unsupported type for null column: %s", type_str.c_str());
    }

    std::unique_ptr<cudf::column> null_col;

    if (n == 0) {
        null_col = cudf::make_empty_column(dtype);
    } else {
        null_col = cudf::make_fixed_width_column(
            dtype, n, cudf::mask_state::ALL_NULL);
    }

    // Wrap in single-column table for R interop
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(null_col));
    auto tbl = std::make_unique<cudf::table>(std::move(cols));
    return make_gpu_table_xptr(std::move(tbl));
}

// Cast a column to a different type
// [[Rcpp::export]]
SEXP gpu_cast_column(SEXP xptr, int col_idx, std::string target_type) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds: %d", col_idx);
    }

    cudf::data_type target_dtype;
    if (target_type == "FLOAT64") {
        target_dtype = cudf::data_type{cudf::type_id::FLOAT64};
    } else if (target_type == "FLOAT32") {
        target_dtype = cudf::data_type{cudf::type_id::FLOAT32};
    } else if (target_type == "INT32") {
        target_dtype = cudf::data_type{cudf::type_id::INT32};
    } else if (target_type == "INT64") {
        target_dtype = cudf::data_type{cudf::type_id::INT64};
    } else if (target_type == "INT16") {
        target_dtype = cudf::data_type{cudf::type_id::INT16};
    } else if (target_type == "INT8") {
        target_dtype = cudf::data_type{cudf::type_id::INT8};
    } else if (target_type == "BOOL8") {
        target_dtype = cudf::data_type{cudf::type_id::BOOL8};
    } else {
        Rcpp::stop("Unsupported target type for casting: %s", target_type.c_str());
    }

    cudf::column_view source_col = view.column(col_idx);

    // If already the target type, just copy
    if (source_col.type().id() == target_dtype.id()) {
        std::vector<std::unique_ptr<cudf::column>> columns;
        for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
        auto result = std::make_unique<cudf::table>(std::move(columns));
        return make_gpu_table_xptr(std::move(result));
    }

    // Cast the column
    auto casted = cudf::cast(source_col, target_dtype);

    // Rebuild table with casted column
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
        if (i == col_idx) {
            columns.push_back(std::move(casted));
        } else {
            columns.push_back(std::make_unique<cudf::column>(view.column(i)));
        }
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// Get actual GPU type of a column (for debugging/validation)
// [[Rcpp::export]]
std::string gpu_column_type(SEXP xptr, int col_idx) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds: %d", col_idx);
    }

    cudf::type_id tid = view.column(col_idx).type().id();

    switch (tid) {
        case cudf::type_id::FLOAT64: return "FLOAT64";
        case cudf::type_id::FLOAT32: return "FLOAT32";
        case cudf::type_id::INT64: return "INT64";
        case cudf::type_id::INT32: return "INT32";
        case cudf::type_id::INT16: return "INT16";
        case cudf::type_id::INT8: return "INT8";
        case cudf::type_id::BOOL8: return "BOOL8";
        case cudf::type_id::STRING: return "STRING";
        case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
        case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
        case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
        case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
        case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
        default: return "UNKNOWN";
    }
}
