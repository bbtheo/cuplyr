// src/ops_distinct.cpp
//
// GPU-native distinct(): cudf::stable_distinct() keeps the input row order
// (dplyr's contract: "first occurrence kept, original row order preserved")
// and, with duplicate_keep_option::KEEP_FIRST, matches dplyr's "first row of
// each key wins" rule exactly. null_equality::EQUAL and
// nan_equality::ALL_EQUAL are cudf's own defaults, and both happen to match
// dplyr/R semantics already (verified empirically against dplyr 1.2.1):
// two NA rows collapse to one, and two NaN rows collapse to one.
//
// The R side (R/distinct.R) always passes the *full* dedup key -- group
// columns (if any) prepended ahead of the user's requested key columns --
// and always gets back a table with every input column (stable_distinct()
// does not project). For the `.keep_all = FALSE` case, R follows up with a
// separate gpu_select() call to project down to just the key columns, in
// the exact order the user asked for; this file only implements the
// dedup step.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_distinct(SEXP xptr, IntegerVector key_indices) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    if (key_indices.size() == 0) {
        Rcpp::stop("distinct() requires at least one key column");
    }

    std::vector<cudf::size_type> keys;
    keys.reserve(key_indices.size());
    for (int i = 0; i < key_indices.size(); ++i) {
        int idx = key_indices[i];
        if (idx < 0 || idx >= view.num_columns()) {
            Rcpp::stop("Key column index out of bounds: %d (table has %d columns)",
                       idx, view.num_columns());
        }
        keys.push_back(static_cast<cudf::size_type>(idx));
    }

    std::unique_ptr<cudf::table> result = cudf::stable_distinct(
        view,
        keys,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL
    );

    return make_gpu_table_xptr(std::move(result));
}
