// src/ops_arrange.cpp
//
// Requires libcudf >= 22.02 for stable_sorted_order and gather APIs.
// Define CUPLYR_USE_UNSTABLE_SORT=1 to fall back to unstable sort if
// stable_sorted_order is unavailable in your cudf build.

#include "gpu_table.hpp"
#include "cuda_utils.hpp"

#include <cudf/sorting.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <climits>
#include <vector>

#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_arrange(SEXP xptr, IntegerVector col_indices, LogicalVector descending) {
    using namespace cuplyr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int num_keys = col_indices.size();

    // Validate lengths match
    if (descending.size() != num_keys) {
        Rcpp::stop("descending length (%d) must match col_indices length (%d)",
                   descending.size(), num_keys);
    }

    // Early return for empty sort specification
    if (num_keys == 0) {
        // Return copy of original table
        auto result = std::make_unique<cudf::table>(view);
        return make_gpu_table_xptr(std::move(result));
    }

    // Validate row count (int32 limit for indices)
    if (view.num_rows() > static_cast<cudf::size_type>(INT32_MAX)) {
        Rcpp::stop("Table has too many rows for arrange() (max ~2.1 billion)");
    }

    // Validate column indices and check for NA in descending
    for (int i = 0; i < num_keys; ++i) {
        if (col_indices[i] < 0 || col_indices[i] >= view.num_columns()) {
            Rcpp::stop("Sort column index out of bounds: %d (table has %d columns)",
                       col_indices[i], view.num_columns());
        }
        if (LogicalVector::is_na(descending[i])) {
            Rcpp::stop("NA values not allowed in descending specification");
        }
    }

    // Build key columns view (memory-efficient: just views, no copy)
    std::vector<cudf::column_view> key_views;
    key_views.reserve(num_keys);
    for (int i = 0; i < num_keys; ++i) {
        key_views.push_back(view.column(col_indices[i]));
    }
    cudf::table_view keys_table(key_views);

    // Build sort order specification
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;
    column_order.reserve(num_keys);
    null_precedence.reserve(num_keys);

    for (int i = 0; i < num_keys; ++i) {
        column_order.push_back(
            descending[i] ? cudf::order::DESCENDING : cudf::order::ASCENDING
        );
        // dplyr places NA LAST regardless of sort direction (verified
        // empirically against dplyr 1.2.1: arrange(desc(x)) still sorts NA
        // last, same as arrange(x)). cudf's null_order is a
        // physical-placement convention, not a value-magnitude one --
        // null_order::AFTER means "nulls sort after every non-null value in
        // THIS column's chosen order::", so for an ASCENDING column that
        // places nulls last, but for a DESCENDING column it would place
        // nulls FIRST (a real, previously-shipped bug: this file used
        // null_order::AFTER unconditionally, which put NAs first for
        // descending keys). Achieving "last" in both directions means
        // flipping null_order together with the direction -- the same
        // technique already used by the window rank kernel
        // (src/ops_window.cpp) and ops_slice.cpp's gpu_slice_rank().
        null_precedence.push_back(
            descending[i] ? cudf::null_order::BEFORE : cudf::null_order::AFTER
        );
    }

    // Phase 1: Compute sort order (memory: nrow * 4 bytes)
    // Use stable_sorted_order to match dplyr's stable sort semantics (ties preserve order)
#ifdef CUPLYR_USE_UNSTABLE_SORT
    // Fallback: unstable sort (faster but ties may reorder)
    std::unique_ptr<cudf::column> sort_indices = cudf::sorted_order(
        keys_table,
        column_order,
        null_precedence
    );
#else
    std::unique_ptr<cudf::column> sort_indices = cudf::stable_sorted_order(
        keys_table,
        column_order,
        null_precedence
    );
#endif

    // Phase 2: Gather full table using indices
    // cudf::gather creates a new table with rows reordered
    std::unique_ptr<cudf::table> sorted_table = cudf::gather(
        view,
        sort_indices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK
    );

    // sort_indices released here (RAII)

    return make_gpu_table_xptr(std::move(sorted_table));
}
