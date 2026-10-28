// src/transfer_io.cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include "r_strings.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplyr {

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
        if (R_IsNA(x[i])) {
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

// Offsets of `strings` as a device INT32/INT64 column (libcudf's own width rule).
template <typename OffT>
static std::unique_ptr<column> make_offsets_column(const std::vector<std::string>& strings,
                                                   type_id id) {
    std::vector<OffT> offsets;
    offsets.reserve(strings.size() + 1);
    OffT acc = 0;
    offsets.push_back(acc);
    for (const auto& s : strings) {
        acc += static_cast<OffT>(s.size());
        offsets.push_back(acc);
    }
    rmm::device_buffer buf(offsets.data(), offsets.size() * sizeof(OffT),
                           rmm::cuda_stream_view(),
                           rmm::mr::get_current_device_resource_ref());
    return std::make_unique<column>(data_type{id}, static_cast<size_type>(offsets.size()),
                                    std::move(buf), rmm::device_buffer{}, 0);
}

// Create GPU column from R character vector. The offsets child is INT64 once
// the characters reach libcudf's large-strings threshold, INT32 below it (the
// rule in cudf/strings/detail/strings_children.cuh). `col_name` is only for the
// backstop error raised when the data needs INT64 but large strings are disabled.
std::unique_ptr<column> character_to_gpu(CharacterVector x, const char* col_name) {
    size_type n = x.size();

    std::vector<std::string> strings(n);
    std::vector<bool> valids(n, true);
    int64_t total = 0;

    for (size_type i = 0; i < n; ++i) {
        if (CharacterVector::is_na(x[i])) {
            valids[i] = false;
        } else {
            strings[i] = r_string_utf8(STRING_ELT(x, i));
            total += static_cast<int64_t>(strings[i].size());
        }
    }

    const bool large = total >= cudf::strings::get_offset64_threshold();
    if (large && !cudf::strings::is_large_strings_enabled()) {
        // Backstop: tbl_gpu() refuses with a classed R error before it uploads
        // anything (first_string_column_over()), so only the other callers
        // (a %in% haystack, a STRING literal, factor levels) get here.
        Rcpp::stop("column '%s' holds %lld bytes of characters, at or above libcudf's "
                   "large-strings threshold (%lld bytes), and large strings are disabled",
                   col_name ? col_name : "<strings>", static_cast<long long>(total),
                   static_cast<long long>(cudf::strings::get_offset64_threshold()));
    }

    // Concatenate the characters (size known, so one allocation)
    std::string concatenated;
    concatenated.reserve(total);
    for (const auto& s : strings) concatenated += s;

    // Copy data to device
    rmm::device_buffer data(concatenated.size(),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());
    if (!concatenated.empty()) {
        check_cuda(cudaMemcpy(data.data(), concatenated.data(), concatenated.size(), cudaMemcpyHostToDevice),
                   "character_to_gpu memcpy data");
    }
    concatenated = std::string();

    auto offsets_col = large ? make_offsets_column<int64_t>(strings, type_id::INT64)
                             : make_offsets_column<int32_t>(strings, type_id::INT32);

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

// Create GPU column from R Date vector (days since epoch). TIMESTAMP_DAYS holds
// whole days, so a fractional Date is floored (the calendar day R prints; a
// truncating cast gives the wrong day below zero) and `saw_fraction` is set so
// the R side can warn.
std::unique_ptr<column> date_to_gpu(NumericVector x, bool& saw_fraction) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_view(),
                            rmm::mr::get_current_device_resource_ref());

    std::vector<int32_t> days(n);
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    size_type null_count = 0;

    const double* xp = REAL(x);
    for (size_type i = 0; i < n; ++i) {
        const double v = xp[i];
        // NA, NaN and +-Inf are null: a TIMESTAMP_DAYS can't hold them, and
        // the int32 cast of a non-finite double is undefined behaviour.
        if (!std::isfinite(v)) {
            days[i] = 0;
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        } else {
            // Whole days (the common case) round-trip through the truncating
            // cast; only the rest pay for floor().
            int32_t t = static_cast<int32_t>(v);
            if (static_cast<double>(t) != v) {
                const double whole = std::floor(v);
                if (whole != v) saw_fraction = true;
                t = static_cast<int32_t>(whole);
            }
            days[i] = t;
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
        // NA, NaN and +-Inf are null (see date_to_gpu()); llround() of an
        // infinity is undefined.
        if (!std::isfinite(x[i])) {
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

// Validity of rows [0, n) of a column view, read from the null mask at bit
// offset() + i (a sliced view's mask is not rebased). One copy of the covering
// bytes; empty when the view has no nulls.
struct HostValidity {
    std::vector<uint8_t> bits;
    int shift = 0;

    explicit HostValidity(const cudf::column_view& col) {
        if (col.size() == 0 || col.null_mask() == nullptr || col.null_count() == 0) return;
        const size_t first = static_cast<size_t>(col.offset()) / 8;
        const size_t last = (static_cast<size_t>(col.offset()) + col.size() + 7) / 8;
        shift = col.offset() % 8;
        bits.resize(last - first);
        check_cuda(cudaMemcpy(bits.data(),
                              reinterpret_cast<const uint8_t*>(col.null_mask()) + first,
                              bits.size(), cudaMemcpyDeviceToHost),
                   "collect null_mask memcpy");
    }
    bool any() const { return !bits.empty(); }
    bool is_null(size_type i) const {
        const size_t b = static_cast<size_t>(i) + shift;
        return !(bits[b / 8] & (1 << (b % 8)));
    }
};

// Copy numeric column from GPU to R
NumericVector gpu_to_numeric(const cudf::column_view& col) {
    size_type n = col.size();
    NumericVector result(n);

    if (n == 0) return result;

    check_cuda(cudaMemcpy(&result[0], col.data<double>(), n * sizeof(double), cudaMemcpyDeviceToHost),
               "gpu_to_numeric memcpy");

    HostValidity v(col);
    if (v.any()) {
        for (size_type i = 0; i < n; ++i) {
            if (v.is_null(i)) result[i] = NA_REAL;
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

    HostValidity v(col);
    if (v.any()) {
        for (size_type i = 0; i < n; ++i) {
            if (v.is_null(i)) result[i] = NA_INTEGER;
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

    HostValidity v(col);
    for (size_type i = 0; i < n; ++i) {
        if (v.any() && v.is_null(i)) {
            result[i] = NA_LOGICAL;
        } else {
            result[i] = temp[i] ? TRUE : FALSE;
        }
    }

    return result;
}

// Fill `result` from a strings view whose offsets child is OffT wide: rows
// [offset(), offset() + n] of the offsets, chars from the first offset on.
template <typename OffT>
static void fill_strings(const cudf::column_view& col, CharacterVector& result) {
    const size_type n = col.size();
    cudf::strings_column_view sv(col);
    auto offsets_col = sv.offsets();
    std::vector<OffT> offsets(static_cast<size_t>(n) + 1);
    check_cuda(cudaMemcpy(offsets.data(), offsets_col.data<OffT>() + col.offset(),
                          offsets.size() * sizeof(OffT), cudaMemcpyDeviceToHost),
               "gpu_to_character offsets memcpy");

    const int64_t base = static_cast<int64_t>(offsets.front());
    const size_t total_chars = static_cast<size_t>(offsets.back() - offsets.front());
    std::vector<char> chars(total_chars);
    if (total_chars > 0) {
        check_cuda(cudaMemcpy(chars.data(), sv.chars_begin(cudf::get_default_stream()) + base,
                              total_chars, cudaMemcpyDeviceToHost),
                   "gpu_to_character data memcpy");
    }

    HostValidity v(col);
    for (size_type i = 0; i < n; ++i) {
        if (v.any() && v.is_null(i)) {
            result[i] = NA_STRING;
        } else {
            const int64_t start = static_cast<int64_t>(offsets[i]) - base;
            const int64_t len = static_cast<int64_t>(offsets[i + 1]) - static_cast<int64_t>(offsets[i]);
            // cudf caps one string at size_type, which is R's limit too
            if (len > std::numeric_limits<int>::max()) Rcpp::stop("a string is too long for R");
            SET_STRING_ELT(result, i, r_mk_utf8_char(chars.data() + start, static_cast<int>(len)));
        }
    }
}

// Copy string column (either offset width, possibly a sliced view) from GPU to R
CharacterVector gpu_to_character(const cudf::column_view& col) {
    size_type n = col.size();
    CharacterVector result(n);

    if (n == 0) return result;

    if (col.child(0).type().id() == type_id::INT64) {
        fill_strings<int64_t>(col, result);
    } else {
        fill_strings<int32_t>(col, result);
    }

    return result;
}

// Fixed-width column of T to an R double vector (NA for nulls), `scale` applied.
template <typename T>
static NumericVector fixed_to_numeric(const cudf::column_view& col, double scale) {
    const size_type n = col.size();
    std::vector<T> temp(n);
    if (n > 0) {
        check_cuda(cudaMemcpy(temp.data(), col.data<T>(), n * sizeof(T), cudaMemcpyDeviceToHost),
                   "collect memcpy");
    }
    NumericVector rv(n);
    for (size_type j = 0; j < n; ++j) rv[j] = static_cast<double>(temp[j]) / scale;
    HostValidity v(col);
    if (v.any()) {
        for (size_type j = 0; j < n; ++j) {
            if (v.is_null(j)) rv[j] = NA_REAL;
        }
    }
    return rv;
}

// One column to its R vector; the single converter behind gpu_collect(),
// gpu_collect_slice() and gpu_head().
static SEXP collect_column(const cudf::column_view& col) {
    switch (col.type().id()) {
        case type_id::FLOAT64:
            return gpu_to_numeric(col);
        case type_id::FLOAT32:
            return fixed_to_numeric<float>(col, 1.0);
        case type_id::INT32:
            return gpu_to_integer(col);
        case type_id::BOOL8:
            return gpu_to_logical(col);
        case type_id::INT64:
            return fixed_to_numeric<int64_t>(col, 1.0);
        case type_id::TIMESTAMP_DAYS: {
            NumericVector rv = fixed_to_numeric<int32_t>(col, 1.0);
            rv.attr("class") = "Date";
            return rv;
        }
        case type_id::TIMESTAMP_MICROSECONDS: {
            NumericVector rv = fixed_to_numeric<int64_t>(col, 1e6);
            rv.attr("class") = CharacterVector::create("POSIXct", "POSIXt");
            rv.attr("tzone") = "UTC";
            return rv;
        }
        case type_id::STRING:
            return gpu_to_character(col);
        default:
            return NumericVector(col.size(), NA_REAL);
    }
}

static List collect_view(const cudf::table_view& view, CharacterVector col_names) {
    int nrow = view.num_rows();
    int ncol = view.num_columns();

    List result(ncol);
    for (int i = 0; i < ncol; ++i) {
        result[i] = collect_column(view.column(i));
    }

    result.names() = col_names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);
    return result;
}

} // namespace cuplyr

// [[Rcpp::export]]
IntegerVector gpu_dim(SEXP xptr) {
    using namespace cuplyr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    return IntegerVector::create(view.num_rows(), view.num_columns());
}

// [[Rcpp::export]]
List gpu_head(SEXP xptr, int n, CharacterVector col_names) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int nrow = std::min(static_cast<int>(view.num_rows()), n);
    auto parts = cudf::slice(view, {0, nrow});
    return collect_view(parts[0], col_names);
}

// Internal test hook (F97): width in bits of a STRING column's offsets child
// (64L / 32L), NA for any other type. `col_idx` is 0-based.
// [[Rcpp::export]]
IntegerVector gpu_string_offsets_width(SEXP xptr, int col_idx) {
    using namespace cuplyr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    if (col_idx < 0 || col_idx >= view.num_columns()) Rcpp::stop("column index out of range");
    cudf::column_view col = view.column(col_idx);
    if (col.type().id() != cudf::type_id::STRING) return IntegerVector::create(NA_INTEGER);
    // A 0-row strings column from cudf can have no children (no offsets to
    // widen), so only a present INT64 offsets child counts as 64.
    if (col.num_children() == 0) return IntegerVector::create(32);
    return IntegerVector::create(col.child(0).type().id() == cudf::type_id::INT64 ? 64 : 32);
}

// [[Rcpp::export]]
CharacterVector gpu_col_types(SEXP xptr) {
    using namespace cuplyr;
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

// Large-strings pre-pass for tbl_gpu() (F97). The byte limit above which an
// upload needs INT64 offsets when libcudf has large strings disabled, -1 when
// they are enabled (the default: nothing more is done on the upload path).
// [[Rcpp::export]]
double large_strings_limit_if_disabled() {
    if (cudf::strings::is_large_strings_enabled()) return -1.0;
    return static_cast<double>(cudf::strings::get_offset64_threshold());
}

// The first character column of the list `cols` whose UTF-8 bytes (counted as
// character_to_gpu() does) reach `limit`: list(index = 1-based, bytes), or NULL.
// Runs before any column is uploaded, so a refusal allocates nothing on the device.
// [[Rcpp::export]]
SEXP first_string_column_over(List cols, double limit) {
    for (R_xlen_t j = 0; j < cols.size(); ++j) {
        SEXP col = cols[j];
        if (TYPEOF(col) != STRSXP) continue;
        double total = 0;
        for (R_xlen_t i = 0; i < Rf_xlength(col); ++i) {
            SEXP s = STRING_ELT(col, i);
            if (s != NA_STRING) total += static_cast<double>(cuplyr::r_string_utf8(s).size());
        }
        if (total >= limit) {
            return List::create(Named("index") = static_cast<int>(j + 1),
                                Named("bytes") = total);
        }
    }
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP df_to_gpu(DataFrame df) {
    using namespace cuplyr;

    int ncol = df.size();
    CharacterVector df_names = df.names();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);
    std::vector<int> fractional_dates;   // 1-based indices of floored Date columns

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        bool is_date = Rf_inherits(col, "Date");
        bool is_posixct = Rf_inherits(col, "POSIXct");

        switch (TYPEOF(col)) {
            case REALSXP:
                if (is_date) {
                    bool saw_fraction = false;
                    columns.push_back(date_to_gpu(col, saw_fraction));
                    if (saw_fraction) fractional_dates.push_back(i + 1);
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
                columns.push_back(character_to_gpu(col, CHAR(STRING_ELT(df_names, i))));
                break;
            case LGLSXP:
                columns.push_back(logical_to_gpu(col));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    // Keep the XPtr (it protects the pointer) alive while wrap() allocates.
    auto out = make_gpu_table_xptr(std::move(tbl));
    // tbl_gpu() warns for these and strips the attribute
    if (!fractional_dates.empty()) {
        Rcpp::Shield<SEXP> idx(Rcpp::wrap(fractional_dates));
        Rf_setAttrib(out, Rf_install("fractional_dates"), idx);
    }
    return out;
}

// [[Rcpp::export]]
List gpu_collect(SEXP xptr, CharacterVector col_names) {
    using namespace cuplyr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    return collect_view(get_table_view(ptr), col_names);
}

// Internal test hook (F97): gpu_collect() of the rows [begin, end) of a
// zero-copy cudf::slice() view (0-based, end exclusive).
// [[Rcpp::export]]
List gpu_collect_slice(SEXP xptr, CharacterVector col_names, int begin, int end) {
    using namespace cuplyr;
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);
    if (begin < 0 || end < begin || end > view.num_rows()) Rcpp::stop("slice bounds out of range");
    auto parts = cudf::slice(view, {begin, end});
    return collect_view(parts[0], col_names);
}
