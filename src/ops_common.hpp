#ifndef CUPLYR_OPS_COMMON_HPP
#define CUPLYR_OPS_COMMON_HPP

#include <Rcpp.h>
#include <cudf/binaryop.hpp>

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

} // namespace cuplyr

#endif // CUPLYR_OPS_COMMON_HPP
