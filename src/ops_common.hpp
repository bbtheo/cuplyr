#ifndef CUPLYR_OPS_COMMON_HPP
#define CUPLYR_OPS_COMMON_HPP

#include <Rcpp.h>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cmath>
#include <memory>
#include <string>

namespace cuplyr {

inline cudf::binary_operator get_compare_op(const std::string& op) {
    if (op == "==") return cudf::binary_operator::EQUAL;
    if (op == "!=") return cudf::binary_operator::NOT_EQUAL;
    if (op == ">")  return cudf::binary_operator::GREATER;
    if (op == ">=") return cudf::binary_operator::GREATER_EQUAL;
    if (op == "<")  return cudf::binary_operator::LESS;
    if (op == "<=") return cudf::binary_operator::LESS_EQUAL;
    Rcpp::stop("Unknown comparison operator: %s", op.c_str());
}

// Shared by gpu_slice_head()/gpu_slice_tail()/gpu_slice_rank() (src/ops_slice.cpp):
// dplyr's slice_head()/slice_tail()/slice_min()/slice_max() all resolve their
// n=/prop= argument through the exact same rule (dplyr:::get_slice_size(),
// verified empirically against dplyr 1.2.1):
//   - type "n":    amount >= 0 -> clamp(0, floor(amount), nrow)
//                  amount <  0 -> clamp(0, ceiling(nrow + amount), nrow)
//   - type "prop": same, with `amount * nrow` substituted for `amount` in the
//                  floor/ceiling step (so a negative `prop` follows the same
//                  "ceiling(nrow + amount*nrow)" shape as a negative `n`).
// `nrow` can only be known once the input table actually exists (it may
// depend on upstream lazy ops), so this resolution happens here at lowering
// time rather than at R-side parse time -- see R/slice.R's module doc.
inline cudf::size_type compute_slice_size(double amount, bool is_prop, cudf::size_type nrow) {
    double raw = is_prop ? amount * static_cast<double>(nrow) : amount;
    double size = (amount >= 0)
        ? std::floor(raw)
        : std::ceil(static_cast<double>(nrow) + raw);

    if (size < 0) size = 0;
    if (size > static_cast<double>(nrow)) size = static_cast<double>(nrow);
    return static_cast<cudf::size_type>(size);
}

// F55: a floating column read with NaN as null (the evaluator's
// eval_ctx::nan_as_null() has a cheaper view route over the same mask; every
// other site uses this copy). If `col` is FLOAT64/
// FLOAT32 and holds a NaN (cudf::nans_to_nulls() grew the null count), returns
// a copy that owns the NaN-nulled mask; otherwise nullptr, so the caller keeps
// `col` (and a non-nullable column stays non-nullable). The data is untouched:
// only the mask differs, so the NaN value survives wherever the original column
// is gathered or returned.
inline std::unique_ptr<cudf::column> nan_nulled_copy(cudf::column_view col) {
    if (col.type().id() != cudf::type_id::FLOAT64 &&
        col.type().id() != cudf::type_id::FLOAT32) {
        return nullptr;
    }
    if (col.size() == 0) return nullptr;
    auto mask = cudf::nans_to_nulls(col);
    if (mask.second <= col.null_count()) return nullptr;
    auto copy = std::make_unique<cudf::column>(col);
    copy->set_null_mask(std::move(*mask.first), mask.second);
    return copy;
}

} // namespace cuplyr

#endif // CUPLYR_OPS_COMMON_HPP
