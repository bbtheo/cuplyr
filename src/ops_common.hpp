#ifndef CUPLYR_OPS_COMMON_HPP
#define CUPLYR_OPS_COMMON_HPP

#include <Rcpp.h>
#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>

#include <cmath>
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

} // namespace cuplyr

#endif // CUPLYR_OPS_COMMON_HPP
