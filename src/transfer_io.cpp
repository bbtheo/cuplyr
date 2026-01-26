// src/transfer_io.cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <vector>

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
    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), &x[0], n * sizeof(double), cudaMemcpyHostToDevice),
                   "numeric_to_gpu memcpy");
    }

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

    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), &x[0], n * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "integer_to_gpu memcpy");
    }

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
                            rmm::mr::get_current_device_resource_ref());
    if (!concatenated.empty()) {
        check_cuda(cudaMemcpy(data.data(), concatenated.data(), concatenated.size(), cudaMemcpyHostToDevice),
                   "character_to_gpu memcpy data");
    }

    // Copy offsets to device
    rmm::device_buffer offsets_buf(offsets.data(), offsets.size() * sizeof(int32_t),
                                   rmm::cuda_stream_view(),
                                   rmm::mr::get_current_device_resource_ref());

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
                                       rmm::mr::get_current_device_resource_ref());
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

// Create GPU column from R logical vector
std::unique_ptr<column> logical_to_gpu(LogicalVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int8_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    std::vector<int8_t> bool_data(n);
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    size_type null_count = 0;

    for (size_type i = 0; i < n; ++i) {
        if (LogicalVector::is_na(x[i])) {
            bool_data[i] = 0;
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        } else {
            bool_data[i] = x[i] ? 1 : 0;
        }
    }

    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), bool_data.data(), n * sizeof(int8_t), cudaMemcpyHostToDevice),
                   "logical_to_gpu memcpy");
    }

    rmm::device_buffer null_mask;
    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::BOOL8},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R Date vector (days since epoch)
std::unique_ptr<column> date_to_gpu(NumericVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    std::vector<int32_t> days(n);
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    size_type null_count = 0;

    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            days[i] = 0;
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        } else {
            days[i] = static_cast<int32_t>(x[i]);
        }
    }

    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), days.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice),
                   "date_to_gpu memcpy");
    }

    rmm::device_buffer null_mask;
    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::TIMESTAMP_DAYS},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R POSIXct vector (seconds since epoch)
std::unique_ptr<column> posixct_to_gpu(NumericVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int64_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    std::vector<int64_t> micros(n);
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    size_type null_count = 0;

    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            micros[i] = 0;
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        } else {
            micros[i] = static_cast<int64_t>(std::llround(x[i] * 1e6));
        }
    }

    if (n > 0) {
        check_cuda(cudaMemcpy(data.data(), micros.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice),
                   "posixct_to_gpu memcpy");
    }

    rmm::device_buffer null_mask;
    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_view(),
                                       rmm::mr::get_current_device_resource_ref());
    }

    return std::make_unique<column>(
        data_type{type_id::TIMESTAMP_MICROSECONDS},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Copy numeric column from GPU to R
NumericVector gpu_to_numeric(const cudf::column_view& col) {
    size_type n = col.size();
    NumericVector result(n);

    if (n == 0) return result;

    check_cuda(cudaMemcpy(&result[0], col.data<double>(), n * sizeof(double), cudaMemcpyDeviceToHost),
               "gpu_to_numeric memcpy");

    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                   "gpu_to_numeric null_mask memcpy");

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

    if (n == 0) return result;

    check_cuda(cudaMemcpy(&result[0], col.data<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "gpu_to_integer memcpy");

    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n));
        check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                   "gpu_to_integer null_mask memcpy");

        for (size_type i = 0; i < n; ++i) {
            if (!(validity[i / 8] & (1 << (i % 8)))) {
                result[i] = NA_INTEGER;
            }
        }
    }

    return result;
}

// Copy bool column from GPU to R logical
LogicalVector gpu_to_logical(const cudf::column_view& col) {
    size_type n = col.size();
    LogicalVector result(n);

    if (n == 0) return result;

    std::vector<int8_t> temp(n);
    check_cuda(cudaMemcpy(temp.data(), col.data<int8_t>(), n * sizeof(int8_t), cudaMemcpyDeviceToHost),
               "gpu_to_logical memcpy");

    std::vector<uint8_t> validity;
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        validity.resize(bitmask_allocation_size_bytes(n));
        check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                   "gpu_to_logical null_mask memcpy");
    }

    for (size_type i = 0; i < n; ++i) {
        if (!validity.empty() && !(validity[i / 8] & (1 << (i % 8)))) {
            result[i] = NA_LOGICAL;
        } else {
            result[i] = temp[i] ? TRUE : FALSE;
        }
    }

    return result;
}

// Copy string column from GPU to R (simplified - assumes contiguous chars)
CharacterVector gpu_to_character(const cudf::column_view& col) {
    size_type n = col.size();
    CharacterVector result(n);

    if (n == 0) return result;

    auto offsets_col = col.child(0);
    std::vector<int32_t> offsets(offsets_col.size());
    check_cuda(cudaMemcpy(offsets.data(), offsets_col.data<int32_t>(),
                           offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "gpu_to_character offsets memcpy");

    size_t total_chars = offsets.back();
    std::vector<char> chars(total_chars);
    if (total_chars > 0) {
        check_cuda(cudaMemcpy(chars.data(), col.data<char>(), total_chars, cudaMemcpyDeviceToHost),
                   "gpu_to_character data memcpy");
    }

    std::vector<uint8_t> validity;
    if (col.null_count() > 0 && col.null_mask() != nullptr) {
        validity.resize(bitmask_allocation_size_bytes(n));
        check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                   "gpu_to_character null_mask memcpy");
    }

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

        column_view head_col(col.type(), nrow, col.head(), col.null_mask(), col.null_count(), col.offset());

        switch (col.type().id()) {
            case type_id::FLOAT64:
                result[i] = gpu_to_numeric(head_col);
                break;
            case type_id::FLOAT32: {
                std::vector<float> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), head_col.data<float>(), nrow * sizeof(float), cudaMemcpyDeviceToHost),
                           "gpu_head float memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = temp[j];
                result[i] = rv;
                break;
            }
            case type_id::INT32:
                result[i] = gpu_to_integer(head_col);
                break;
            case type_id::BOOL8:
                result[i] = gpu_to_logical(head_col);
                break;
            case type_id::INT64: {
                std::vector<int64_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), head_col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost),
                           "gpu_head int64 memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                result[i] = rv;
                break;
            }
            case type_id::TIMESTAMP_DAYS: {
                std::vector<int32_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), head_col.data<int32_t>(), nrow * sizeof(int32_t), cudaMemcpyDeviceToHost),
                           "gpu_head date memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                rv.attr("class") = "Date";
                result[i] = rv;
                break;
            }
            case type_id::TIMESTAMP_MICROSECONDS: {
                std::vector<int64_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), head_col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost),
                           "gpu_head posixct memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]) / 1e6;
                rv.attr("class") = CharacterVector::create("POSIXct", "POSIXt");
                rv.attr("tzone") = "UTC";
                result[i] = rv;
                break;
            }
            case type_id::STRING: {
                // For STRING columns, we need to handle head specially
                // since gpu_to_character expects the full column structure
                // Create a sliced view and collect only the first n strings
                CharacterVector str_result(nrow);
                if (nrow > 0) {
                    auto offsets_col = col.child(0);
                    // Get offsets for first nrow+1 elements (need nrow+1 for boundaries)
                    std::vector<int32_t> offsets(nrow + 1);
                    check_cuda(cudaMemcpy(offsets.data(), offsets_col.data<int32_t>(),
                                         (nrow + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost),
                               "gpu_head string offsets memcpy");

                    size_t total_chars = offsets[nrow] - offsets[0];
                    std::vector<char> chars(total_chars);
                    if (total_chars > 0) {
                        check_cuda(cudaMemcpy(chars.data(), col.data<char>() + offsets[0],
                                             total_chars, cudaMemcpyDeviceToHost),
                                   "gpu_head string data memcpy");
                    }

                    std::vector<uint8_t> validity;
                    if (col.null_count() > 0 && col.null_mask() != nullptr) {
                        validity.resize(bitmask_allocation_size_bytes(col.size()));
                        check_cuda(cudaMemcpy(validity.data(), col.null_mask(),
                                             validity.size(), cudaMemcpyDeviceToHost),
                                   "gpu_head string null_mask memcpy");
                    }

                    int32_t base_offset = offsets[0];
                    for (int j = 0; j < nrow; ++j) {
                        if (!validity.empty() && !(validity[j / 8] & (1 << (j % 8)))) {
                            str_result[j] = NA_STRING;
                        } else {
                            int32_t start = offsets[j] - base_offset;
                            int32_t len = offsets[j + 1] - offsets[j];
                            str_result[j] = std::string(chars.data() + start, len);
                        }
                    }
                }
                result[i] = str_result;
                break;
            }
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
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        bool is_date = Rf_inherits(col, "Date");
        bool is_posixct = Rf_inherits(col, "POSIXct");

        switch (TYPEOF(col)) {
            case REALSXP:
                if (is_date) {
                    columns.push_back(date_to_gpu(col));
                } else if (is_posixct) {
                    columns.push_back(posixct_to_gpu(col));
                } else {
                    columns.push_back(numeric_to_gpu(col));
                }
                break;
            case INTSXP:
                columns.push_back(integer_to_gpu(col));
                break;
            case STRSXP:
                columns.push_back(character_to_gpu(col));
                break;
            case LGLSXP:
                columns.push_back(logical_to_gpu(col));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(tbl));
}

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
                check_cuda(cudaMemcpy(temp.data(), col.data<float>(), nrow * sizeof(float), cudaMemcpyDeviceToHost),
                           "gpu_collect float memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = temp[j];
                result[i] = rv;
                break;
            }
            case type_id::INT32:
                result[i] = gpu_to_integer(col);
                break;
            case type_id::BOOL8:
                result[i] = gpu_to_logical(col);
                break;
            case type_id::INT64: {
                std::vector<int64_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost),
                           "gpu_collect int64 memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                result[i] = rv;
                break;
            }
            case type_id::TIMESTAMP_DAYS: {
                std::vector<int32_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), col.data<int32_t>(), nrow * sizeof(int32_t), cudaMemcpyDeviceToHost),
                           "gpu_collect date memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]);
                // Handle NAs
                if (col.null_count() > 0 && col.null_mask() != nullptr) {
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                               "gpu_collect date null_mask memcpy");
                    for (int j = 0; j < nrow; ++j) {
                        if (!(validity[j / 8] & (1 << (j % 8)))) {
                            rv[j] = NA_REAL;
                        }
                    }
                }
                rv.attr("class") = "Date";
                result[i] = rv;
                break;
            }
            case type_id::TIMESTAMP_MICROSECONDS: {
                std::vector<int64_t> temp(nrow);
                check_cuda(cudaMemcpy(temp.data(), col.data<int64_t>(), nrow * sizeof(int64_t), cudaMemcpyDeviceToHost),
                           "gpu_collect posixct memcpy");
                NumericVector rv(nrow);
                for (int j = 0; j < nrow; ++j) rv[j] = static_cast<double>(temp[j]) / 1e6;
                // Handle NAs
                if (col.null_count() > 0 && col.null_mask() != nullptr) {
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    check_cuda(cudaMemcpy(validity.data(), col.null_mask(), validity.size(), cudaMemcpyDeviceToHost),
                               "gpu_collect posixct null_mask memcpy");
                    for (int j = 0; j < nrow; ++j) {
                        if (!(validity[j / 8] & (1 << (j % 8)))) {
                            rv[j] = NA_REAL;
                        }
                    }
                }
                rv.attr("class") = CharacterVector::create("POSIXct", "POSIXt");
                rv.attr("tzone") = "UTC";
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
