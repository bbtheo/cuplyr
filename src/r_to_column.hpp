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
std::unique_ptr<cudf::column> character_to_gpu(Rcpp::CharacterVector x, const char* col_name = nullptr);

// Create GPU column from R logical vector (BOOL8, NA -> null)
std::unique_ptr<cudf::column> logical_to_gpu(Rcpp::LogicalVector x);

// What a Date / POSIXct upload had to do to its values: floor a fraction (Date
// only), or null a +-Inf / a finite value the GPU type can't hold. NA and NaN
// are legitimate nulls and are not counted.
struct TimestampUpload {
    bool saw_fraction = false;
    long long infinite = 0;
    long long out_of_range = 0;
};

// Create GPU column from R Date (TIMESTAMP_DAYS, floored) / POSIXct
// (TIMESTAMP_MICROSECONDS, rounded) vector; `up` records what was lost.
std::unique_ptr<cudf::column> date_to_gpu(Rcpp::NumericVector x, TimestampUpload& up);
std::unique_ptr<cudf::column> posixct_to_gpu(Rcpp::NumericVector x, TimestampUpload& up);

} // namespace cuplyr

#endif // CUPLYR_R_TO_COLUMN_HPP
