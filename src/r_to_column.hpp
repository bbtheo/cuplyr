// src/r_to_column.hpp
//
// Declarations for the R-vector -> GPU column transfer helpers implemented
// in transfer_io.cpp. These already have external linkage there (plain
// functions inside `namespace cuplyr`, not `static`) -- this header just
// gives other translation units (expr_eval.hpp, for %in%'s haystack upload)
// a declaration to include instead of redeclaring them inline.
#ifndef CUPLYR_R_TO_COLUMN_HPP
#define CUPLYR_R_TO_COLUMN_HPP

#include <Rcpp.h>
#include <cudf/column/column.hpp>

#include <memory>

namespace cuplyr {

// Create GPU column from R numeric vector (FLOAT64, NA -> null)
std::unique_ptr<cudf::column> numeric_to_gpu(Rcpp::NumericVector x);

// Create GPU column from R integer vector (INT32, NA -> null)
std::unique_ptr<cudf::column> integer_to_gpu(Rcpp::IntegerVector x);

// Create GPU column from R character vector (STRING, NA -> null)
std::unique_ptr<cudf::column> character_to_gpu(Rcpp::CharacterVector x);

// Create GPU column from R logical vector (BOOL8, NA -> null)
std::unique_ptr<cudf::column> logical_to_gpu(Rcpp::LogicalVector x);

} // namespace cuplyr

#endif // CUPLYR_R_TO_COLUMN_HPP
