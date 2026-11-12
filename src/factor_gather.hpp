// src/factor_gather.hpp
//
// Shared helpers for gathering a small per-level lookup table with a factor
// column's codes (1-based, nulls for NA) as the index. Used by
// gpu_remap_codes()/gpu_decode_factor() (ops_bind.cpp) and by the
// `factor_labels` expression handler (expr_eval.hpp, F50).
#ifndef CUPLYR_FACTOR_GATHER_HPP
#define CUPLYR_FACTOR_GATHER_HPP

#include "r_to_column.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <Rcpp.h>

#include <memory>
#include <vector>

namespace cuplyr {

// Convert a column of 1-based (possibly null) codes into a non-nullable,
// 0-based gather map suitable for cudf::gather(). cudf::gather() requires a
// non-nullable map, so null codes become 0 (an always-in-bounds placeholder);
// the caller reapplies the original null mask to the gathered result.
inline std::unique_ptr<cudf::column> sanitize_codes_for_gather(cudf::column_view const& codes_col) {
    cudf::numeric_scalar<int32_t> one(1, true);
    auto shifted = cudf::binary_operation(
        codes_col, one, cudf::binary_operator::SUB,
        cudf::data_type{cudf::type_id::INT32});

    if (!codes_col.has_nulls()) {
        return shifted;
    }

    cudf::numeric_scalar<int32_t> zero(0, true);
    return cudf::replace_nulls(shifted->view(), zero);
}

// Reapply `source`'s null mask onto `target` in place (used after gathering
// through a sanitized, non-nullable index column).
inline void reapply_null_mask(std::unique_ptr<cudf::column>& target,
                              cudf::column_view const& source) {
    if (!source.has_nulls()) {
        return;
    }
    rmm::device_buffer mask = cudf::copy_bitmask(source);
    target->set_null_mask(std::move(mask), source.null_count());
}

// Decode a column of 1-based factor codes to the label strings in `levels`
// (`levels[code - 1]`); null codes stay null. A zero-level factor has only NA
// codes, so there is nothing to gather from. Bounds-safe: every code outside
// [1, n_levels] (and every null) is clamped to the index n_levels, which
// NULLIFY turns into a null row, so no code can read past the levels column
// (a negative index would otherwise wrap to i + n).
inline std::unique_ptr<cudf::column> decode_codes_to_labels(cudf::column_view const& codes_col,
                                                            Rcpp::CharacterVector levels) {
    if (levels.size() == 0) {
        cudf::string_scalar null_str("", false);
        return cudf::make_column_from_scalar(null_str, codes_col.size());
    }

    auto levels_col = character_to_gpu(levels);
    auto const n_levels = static_cast<int32_t>(levels.size());

    cudf::numeric_scalar<int32_t> one(1, true);
    auto shifted = cudf::binary_operation(
        codes_col, one, cudf::binary_operator::SUB,
        cudf::data_type{cudf::type_id::INT32});
    cudf::numeric_scalar<int32_t> lo(0, true);
    cudf::numeric_scalar<int32_t> hi(n_levels - 1, true);
    cudf::numeric_scalar<int32_t> oob(n_levels, true);
    auto clamped = cudf::clamp(shifted->view(), lo, oob, hi, oob);
    auto map = clamped->has_nulls() ? cudf::replace_nulls(clamped->view(), oob)
                                    : std::move(clamped);

    std::vector<cudf::column_view> lookup_cols{levels_col->view()};
    cudf::table_view lookup_table(lookup_cols);

    auto gathered = cudf::gather(lookup_table, map->view(),
                                 cudf::out_of_bounds_policy::NULLIFY);
    auto gathered_cols = gathered->release();
    return std::move(gathered_cols[0]);
}

} // namespace cuplyr

#endif // CUPLYR_FACTOR_GATHER_HPP
