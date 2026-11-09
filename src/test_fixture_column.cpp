// src/test_fixture_column.cpp
//
// Test hook (F105): gpu_test_make_column() builds a one-column GPU table of any
// libcudf type, so tests can reach the collect path with the column types that
// nothing in cuplyr uploads today (unsigned integers, decimals, durations, real
// dictionaries, nested types). `values` are decimal strings (NA = null), parsed
// here so UINT64 and DECIMAL128 values are exact. LIST rows are the two-element
// list [v, v + 1]; a STRUCT has one INT32 field. "DICTIONARY32_INT32" is a
// dictionary with INT32 keys. Not used by the package itself.

#include "gpu_table.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <Rcpp.h>

#include <memory>
#include <string>
#include <vector>

namespace cuplyr {
std::unique_ptr<cudf::column> character_to_gpu(Rcpp::CharacterVector x, const char* col_name);
}

namespace {

using int128 = __int128_t;

struct Parsed {
    std::vector<int128> vals;
    std::vector<uint8_t> validity;
    cudf::size_type null_count = 0;
};

Parsed parse_values(Rcpp::CharacterVector values) {
    const R_xlen_t n = values.size();
    Parsed p;
    p.vals.resize(n);
    p.validity.assign(cudf::bitmask_allocation_size_bytes(n), 0xFF);
    for (R_xlen_t i = 0; i < n; ++i) {
        if (Rcpp::CharacterVector::is_na(values[i])) {
            p.validity[i / 8] &= ~(1 << (i % 8));
            p.null_count++;
            continue;
        }
        std::string s = Rcpp::as<std::string>(values[i]);
        bool neg = !s.empty() && s[0] == '-';
        // unsigned, so the magnitude of the most negative value does not overflow
        unsigned __int128 v = 0;
        for (size_t k = neg ? 1 : 0; k < s.size(); ++k) {
            if (s[k] < '0' || s[k] > '9') Rcpp::stop("not an integer: %s", s.c_str());
            v = v * 10 + static_cast<unsigned>(s[k] - '0');
        }
        p.vals[i] = neg ? -static_cast<int128>(v) : static_cast<int128>(v);
    }
    return p;
}

rmm::device_buffer null_mask_of(const Parsed& p) {
    if (p.null_count == 0) return rmm::device_buffer{};
    return rmm::device_buffer(p.validity.data(), p.validity.size(), rmm::cuda_stream_view(),
                              rmm::mr::get_current_device_resource_ref());
}

template <typename T>
std::unique_ptr<cudf::column> fixed_column(cudf::data_type type, const Parsed& p) {
    const auto n = static_cast<cudf::size_type>(p.vals.size());
    std::vector<T> host(n);
    for (cudf::size_type i = 0; i < n; ++i) host[i] = static_cast<T>(p.vals[i]);
    rmm::device_buffer data = n == 0 ? rmm::device_buffer{}
                                     : rmm::device_buffer(host.data(), n * sizeof(T), rmm::cuda_stream_view(),
                                                          rmm::mr::get_current_device_resource_ref());
    return std::make_unique<cudf::column>(type, n, std::move(data), null_mask_of(p), p.null_count);
}

std::unique_ptr<cudf::column> list_column(const Parsed& p) {
    const auto n = static_cast<cudf::size_type>(p.vals.size());
    Parsed offsets;
    offsets.vals.resize(n + 1);
    for (cudf::size_type i = 0; i <= n; ++i) offsets.vals[i] = 2 * i;
    Parsed child;
    child.vals.resize(2 * n);
    for (cudf::size_type i = 0; i < n; ++i) {
        child.vals[2 * i] = p.vals[i];
        child.vals[2 * i + 1] = p.vals[i] + 1;
    }
    return cudf::make_lists_column(n, fixed_column<int32_t>(cudf::data_type{cudf::type_id::INT32}, offsets),
                                   fixed_column<int32_t>(cudf::data_type{cudf::type_id::INT32}, child),
                                   p.null_count, null_mask_of(p));
}

std::unique_ptr<cudf::column> struct_column(const Parsed& p) {
    const auto n = static_cast<cudf::size_type>(p.vals.size());
    Parsed field = p;
    field.validity.clear();
    field.null_count = 0;
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(fixed_column<int32_t>(cudf::data_type{cudf::type_id::INT32}, field));
    return cudf::make_structs_column(n, std::move(children), p.null_count, null_mask_of(p));
}

}  // namespace

// [[Rcpp::export]]
SEXP gpu_test_make_column(std::string type, Rcpp::CharacterVector values, int scale = 0) {
    using cudf::data_type;
    using cudf::type_id;

    std::unique_ptr<cudf::column> col;
    if (type == "DICTIONARY32") {
        auto strings = cuplyr::character_to_gpu(values, "x");
        col = cudf::dictionary::encode(strings->view());
    } else if (type == "DICTIONARY32_INT8_INDICES") {
        auto strings = cuplyr::character_to_gpu(values, "x");
        col = cudf::dictionary::encode(strings->view(), data_type{type_id::INT8});
    } else {
        const Parsed p = parse_values(values);
        if (type == "INT8") col = fixed_column<int8_t>(data_type{type_id::INT8}, p);
        else if (type == "INT16") col = fixed_column<int16_t>(data_type{type_id::INT16}, p);
        else if (type == "UINT8") col = fixed_column<uint8_t>(data_type{type_id::UINT8}, p);
        else if (type == "UINT16") col = fixed_column<uint16_t>(data_type{type_id::UINT16}, p);
        else if (type == "UINT32") col = fixed_column<uint32_t>(data_type{type_id::UINT32}, p);
        else if (type == "UINT64") col = fixed_column<uint64_t>(data_type{type_id::UINT64}, p);
        else if (type == "DECIMAL32") col = fixed_column<int32_t>(data_type{type_id::DECIMAL32, scale}, p);
        else if (type == "DECIMAL64") col = fixed_column<int64_t>(data_type{type_id::DECIMAL64, scale}, p);
        else if (type == "DECIMAL128") col = fixed_column<int128>(data_type{type_id::DECIMAL128, scale}, p);
        else if (type == "TIMESTAMP_SECONDS") col = fixed_column<int64_t>(data_type{type_id::TIMESTAMP_SECONDS}, p);
        else if (type == "TIMESTAMP_MILLISECONDS") col = fixed_column<int64_t>(data_type{type_id::TIMESTAMP_MILLISECONDS}, p);
        else if (type == "TIMESTAMP_NANOSECONDS") col = fixed_column<int64_t>(data_type{type_id::TIMESTAMP_NANOSECONDS}, p);
        else if (type == "DURATION_DAYS") col = fixed_column<int32_t>(data_type{type_id::DURATION_DAYS}, p);
        else if (type == "DURATION_SECONDS") col = fixed_column<int64_t>(data_type{type_id::DURATION_SECONDS}, p);
        else if (type == "DURATION_MILLISECONDS") col = fixed_column<int64_t>(data_type{type_id::DURATION_MILLISECONDS}, p);
        else if (type == "DURATION_MICROSECONDS") col = fixed_column<int64_t>(data_type{type_id::DURATION_MICROSECONDS}, p);
        else if (type == "DURATION_NANOSECONDS") col = fixed_column<int64_t>(data_type{type_id::DURATION_NANOSECONDS}, p);
        else if (type == "LIST") col = list_column(p);
        else if (type == "STRUCT") col = struct_column(p);
        else if (type == "DICTIONARY32_INT32") {
            auto keys = fixed_column<int32_t>(data_type{type_id::INT32}, p);
            col = cudf::dictionary::encode(keys->view());
        } else Rcpp::stop("gpu_test_make_column: unknown type %s", type.c_str());
    }

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));
    return cuplyr::make_gpu_table_xptr(std::make_unique<cudf::table>(std::move(cols)));
}
