// src/transfer.cpp
#include "gpu_table.hpp"
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/unary.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cuda_runtime.h>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplr {

// Create GPU column from R numeric vector
std::unique_ptr<column> numeric_to_gpu(NumericVector x) {
    size_type n = x.size();

    // Allocate device memory
    rmm::device_buffer data(n * sizeof(double),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    // Copy from host to device
    cudaMemcpy(data.data(), &x[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Handle NAs by creating validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    // Check for NAs
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            // Clear bit i
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::FLOAT64},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R integer vector
std::unique_ptr<column> integer_to_gpu(IntegerVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    cudaMemcpy(data.data(), &x[0], n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Handle NAs
    rmm::device_buffer null_mask;
    size_type null_count = 0;
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);

    for (size_type i = 0; i < n; ++i) {
        if (IntegerVector::is_na(x[i])) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::INT32},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R character vector
std::unique_ptr<column> character_to_gpu(CharacterVector x) {
    size_type n = x.size();

    std::vector<std::string> strings(n);
    std::vector<bool> valids(n, true);

    for (size_type i = 0; i < n; ++i) {
        if (CharacterVector::is_na(x[i])) {
            valids[i] = false;
            strings[i] = "";
        } else {
            strings[i] = as<std::string>(x[i]);
        }
    }

    // Concatenate strings and create offsets
    std::string concatenated;
    std::vector<int32_t> offsets;
    offsets.push_back(0);

    for (const auto& s : strings) {
        concatenated += s;
        offsets.push_back(concatenated.size());
    }

    // Copy data to device
    rmm::device_buffer data(concatenated.size(),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource());
    cudaMemcpy(data.data(), concatenated.data(), concatenated.size(), cudaMemcpyHostToDevice);

    // Copy offsets to device
    rmm::device_buffer offsets_buf(offsets.data(), offsets.size() * sizeof(int32_t),
                                   rmm::cuda_stream_view(),
                                   rmm::mr::get_current_device_resource());

    // Create offsets column
    auto offsets_col = std::make_unique<column>(
        data_type{type_id::INT32},
        offsets.size(),
        std::move(offsets_buf),
        rmm::device_buffer{},
        0
    );

    // Create validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (!valids[i]) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource());
    }

    // Create STRING column with offsets as child
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(offsets_col));

    return std::make_unique<column>(
        data_type{type_id::STRING},
        n,
        std::move(data),
        std::move(null_mask),
        null_count,
        std::move(children)
    );
}

// Copy numeric column from GPU to R
NumericVector gpu_to_numeric(const cudf::column_view& col) {
    size_type n = col.size();
    NumericVector result(n);

    // Copy data from device to host
    cudaMemcpy(&result[0], col.data<double>(), n * sizeof(double), cudaMemcpyDeviceToHost);

    // Handle nulls
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);

        for (size_type i = 0; i < n; ++i) {
            if (!(validity[i / 8] & (1 << (i % 8)))) {
                result[i] = NA_REAL;
            }
        }
    }

    return result;
}

// Copy integer column from GPU to R
IntegerVector gpu_to_integer(const cudf::column_view& col) {
    size_type n = col.size();
    IntegerVector result(n);

    cudaMemcpy(&result[0], col.data<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);

        for (size_type i = 0; i < n; ++i) {
            if (!(validity[i / 8] & (1 << (i % 8)))) {
                result[i] = NA_INTEGER;
            }
        }
    }

    return result;
}

// Copy string column from GPU to R (simplified - assumes contiguous chars)
CharacterVector gpu_to_character(const cudf::column_view& col) {
    size_type n = col.size();
    CharacterVector result(n);

    // For strings, we need offsets and char data
    // This is a simplified implementation
    if (n == 0) return result;

    // Get offsets child column
    auto offsets_col = col.child(0);
    std::vector<int32_t> offsets(offsets_col.size());
    cudaMemcpy(offsets.data(), offsets_col.data<int32_t>(),
               offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Get char data
    size_t total_chars = offsets.back();
    std::vector<char> chars(total_chars);
    if (total_chars > 0) {
        cudaMemcpy(chars.data(), col.data<char>(), total_chars, cudaMemcpyDeviceToHost);
    }

    // Handle nulls
    std::vector<uint8_t> validity;
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        validity.resize(bitmask_allocation_size_bytes(n));
        cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost);
    }

    // Build R strings
    for (size_type i = 0; i < n; ++i) {
        if (!validity.empty() && !(validity[i / 8] & (1 << (i % 8)))) {
            result[i] = NA_STRING;
        } else {
            int32_t start = offsets[i];
            int32_t len = offsets[i + 1] - start;
            result[i] = std::string(chars.data() + start, len);
        }
    }

    return result;
}

} // namespace cuplr

// [[Rcpp::export]]
IntegerVector gpu_dim(SEXP xptr) {
    using namespace cuplr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    return IntegerVector::create(view.num_rows(), view.num_columns());
}

// [[Rcpp::export]]
List gpu_head(SEXP xptr, int n, CharacterVector col_names) {
    using namespace cuplr;
    using namespace cudf;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int nrow = std::min(static_cast<int>(view.num_rows()), n);
    int ncol = view.num_columns();

    List result(ncol);

    for (int i = 0; i < ncol; ++i) {
        column_view col = view.column(i);

        // Create a view of just the first n rows
        column_view head_col(col.type(), nrow, col.head(), col.null_mask(), col.null_count(), col.offset());

        switch (col.type().id()) {
            case type_id::FLOAT64:
                result[i] = gpu_to_numeric(head_col);
                break;
            case type_id::FLOAT32: {
                // Convert float to double for R
                std::vector<float> temp(nrow);
                cudaMemcpy(temp.data(), head_col.data<float>(), nrow * sizeof(float), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = temp[j];
                result[i] = rv;
                break;
            }
            case type_id::INT32:
            case type_id::BOOL8:
                result[i] = gpu_to_integer(head_col);
                break;
            case type_id::INT64: {
                // Convert int64 to double for R (may lose precision)
                std::vector<int64_t> temp(nrow);
                cudaMemcpy(temp.data(), head_col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                result[i] = rv;
                break;
            }
            case type_id::STRING: {
                // For strings, we need to be more careful with head
                // Just collect all for now and truncate in R
                result[i] = gpu_to_character(col);
                break;
            }
            default:
                // Unsupported type - return NAs
                result[i] = NumericVector(nrow, NA_REAL);
        }
    }

    result.names() = col_names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);

    return result;
}

// [[Rcpp::export]]
CharacterVector gpu_col_types(SEXP xptr) {
    using namespace cuplr;
    using namespace cudf;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int ncol = view.num_columns();
    CharacterVector result(ncol);

    for (int i = 0; i < ncol; ++i) {
        switch (view.column(i).type().id()) {
            case type_id::FLOAT64: result[i] = "dbl"; break;
            case type_id::FLOAT32: result[i] = "flt"; break;
            case type_id::INT64: result[i] = "i64"; break;
            case type_id::INT32: result[i] = "int"; break;
            case type_id::INT16: result[i] = "i16"; break;
            case type_id::INT8: result[i] = "i8"; break;
            case type_id::BOOL8: result[i] = "lgl"; break;
            case type_id::STRING: result[i] = "chr"; break;
            case type_id::TIMESTAMP_DAYS: result[i] = "date"; break;
            case type_id::TIMESTAMP_SECONDS:
            case type_id::TIMESTAMP_MILLISECONDS:
            case type_id::TIMESTAMP_MICROSECONDS:
            case type_id::TIMESTAMP_NANOSECONDS:
                result[i] = "dttm"; break;
            default: result[i] = "???"; break;
        }
    }

    return result;
}

// [[Rcpp::export]]
SEXP df_to_gpu(DataFrame df) {
    using namespace cuplr;

    int ncol = df.size();
    CharacterVector names = df.names();

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        switch (TYPEOF(col)) {
            case REALSXP:
                columns.push_back(numeric_to_gpu(col));
                break;
            case INTSXP:
                columns.push_back(integer_to_gpu(col));
                break;
            case STRSXP:
                columns.push_back(character_to_gpu(col));
                break;
            case LGLSXP:
                // Convert logical to integer then to BOOL8
                columns.push_back(integer_to_gpu(as<IntegerVector>(col)));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(tbl));
}

// Helper to get comparison operator
cudf::binary_operator get_compare_op(const std::string& op) {
    if (op == "==") return cudf::binary_operator::EQUAL;
    if (op == "!=") return cudf::binary_operator::NOT_EQUAL;
    if (op == ">")  return cudf::binary_operator::GREATER;
    if (op == ">=") return cudf::binary_operator::GREATER_EQUAL;
    if (op == "<")  return cudf::binary_operator::LESS;
    if (op == "<=") return cudf::binary_operator::LESS_EQUAL;
    Rcpp::stop("Unknown comparison operator: %s", op.c_str());
}

// Helper to get arithmetic operator
cudf::binary_operator get_arith_op(const std::string& op) {
    if (op == "+") return cudf::binary_operator::ADD;
    if (op == "-") return cudf::binary_operator::SUB;
    if (op == "*") return cudf::binary_operator::MUL;
    if (op == "/") return cudf::binary_operator::TRUE_DIV;
    if (op == "%/%") return cudf::binary_operator::FLOOR_DIV;
    if (op == "%%") return cudf::binary_operator::MOD;
    if (op == "^") return cudf::binary_operator::POW;
    Rcpp::stop("Unknown arithmetic operator: %s", op.c_str());
}

// [[Rcpp::export]]
SEXP gpu_filter_scalar(SEXP xptr, int col_idx, std::string op, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col = view.column(col_idx);

    // Create scalar for comparison
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    // Perform comparison to get boolean mask
    auto mask = cudf::binary_operation(
        col, *scalar, get_compare_op(op),
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply boolean mask to filter table
    auto result = cudf::apply_boolean_mask(view, mask->view());

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_filter_col(SEXP xptr, int col_idx, std::string op, int col_idx2) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns() ||
        col_idx2 < 0 || col_idx2 >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col1 = view.column(col_idx);
    cudf::column_view col2 = view.column(col_idx2);

    // Perform comparison to get boolean mask
    auto mask = cudf::binary_operation(
        col1, col2, get_compare_op(op),
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply boolean mask to filter table
    auto result = cudf::apply_boolean_mask(view, mask->view());

    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_scalar(SEXP xptr, int col_idx, std::string op, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col = view.column(col_idx);

    // Create scalar
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    // Perform binary operation
    auto new_col = cudf::binary_operation(
        col, *scalar, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    // Build new table with additional column
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    columns.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_mutate_binary_cols(SEXP xptr, int col_idx1, std::string op, int col_idx2) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (col_idx1 < 0 || col_idx1 >= view.num_columns() ||
        col_idx2 < 0 || col_idx2 >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds");
    }

    cudf::column_view col1 = view.column(col_idx1);
    cudf::column_view col2 = view.column(col_idx2);

    // Perform binary operation
    auto new_col = cudf::binary_operation(
        col1, col2, get_arith_op(op),
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    // Build new table with additional column
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(view.column(i)));
    }
    columns.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// [[Rcpp::export]]
SEXP gpu_select(SEXP xptr, IntegerVector col_indices) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Build vector of column views for selected columns
    std::vector<cudf::column_view> selected_views;
    for (int i = 0; i < col_indices.size(); ++i) {
        int idx = col_indices[i];
        if (idx < 0 || idx >= view.num_columns()) {
            Rcpp::stop("Column index out of bounds: %d", idx);
        }
        selected_views.push_back(view.column(idx));
    }

    // Create table view from selected columns
    cudf::table_view selected_view(selected_views);

    // Copy to new table
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < selected_view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(selected_view.column(i)));
    }

    auto result = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(result));
}

// Collect entire table back to R
// [[Rcpp::export]]
List gpu_collect(SEXP xptr, CharacterVector col_names) {
    using namespace cuplr;
    using namespace cudf;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int nrow = view.num_rows();
    int ncol = view.num_columns();

    List result(ncol);

    for (int i = 0; i < ncol; ++i) {
        column_view col = view.column(i);

        switch (col.type().id()) {
            case type_id::FLOAT64:
                result[i] = gpu_to_numeric(col);
                break;
            case type_id::FLOAT32: {
                std::vector<float> temp(nrow);
                cudaMemcpy(temp.data(), col.data<float>(), nrow * sizeof(float), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = temp[j];
                result[i] = rv;
                break;
            }
            case type_id::INT32:
            case type_id::BOOL8:
                result[i] = gpu_to_integer(col);
                break;
            case type_id::INT64: {
                std::vector<int64_t> temp(nrow);
                cudaMemcpy(temp.data(), col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost);
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                result[i] = rv;
                break;
            }
            case type_id::STRING:
                result[i] = gpu_to_character(col);
                break;
            default:
                result[i] = NumericVector(nrow, NA_REAL);
        }
    }

    result.names() = col_names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);

    return result;
}

// [[Rcpp::export]]
bool gpu_is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

// [[Rcpp::export]]
List gpu_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return List::create(
            Named("available") = false,
            Named("device_count") = 0
        );
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Get memory info
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    return List::create(
        Named("available") = true,
        Named("device_count") = device_count,
        Named("device_id") = device_id,
        Named("name") = std::string(prop.name),
        Named("compute_capability") = std::to_string(prop.major) + "." + std::to_string(prop.minor),
        Named("total_memory") = static_cast<double>(total_mem),
        Named("free_memory") = static_cast<double>(free_mem),
        Named("multiprocessors") = prop.multiProcessorCount
    );
}
