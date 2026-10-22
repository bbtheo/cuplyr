// src/gpu_table.hpp
//
// The payload behind every tbl_gpu$ptr. A table is a vector of SHARED, CONST
// columns: a verb that passes a column through unchanged (select, the
// untouched columns of mutate, bind_cols, ...) shares it instead of
// deep-copying, so two tables can point at the same device memory.
//
// Ownership rule (F57): a column that is part of a table is immutable. Ops
// build new columns (and mutate them -- set_null_mask() etc. -- only while
// they still own them as std::unique_ptr<cudf::column>), then hand them to a
// table through make_gpu_table_xptr(). Nothing may release()/move columns out
// of an input table or take a mutable_view of one; `const` makes that a
// compile error.
#ifndef CUPLYR_GPU_TABLE_HPP
#define CUPLYR_GPU_TABLE_HPP

#include <Rcpp.h>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>
#include <vector>

namespace cuplyr {

using SharedColumn = std::shared_ptr<const cudf::column>;

struct GpuTable {
    std::vector<SharedColumn> columns;
    // Set by a closing release scope (F58) on a table it created and nobody
    // returned: the columns are dropped, and any later use is an error.
    bool released = false;

    // Non-owning view; valid while this table (or any holder of its columns) lives.
    cudf::table_view view() const {
        std::vector<cudf::column_view> views;
        views.reserve(columns.size());
        for (const auto& c : columns) views.push_back(c->view());
        return cudf::table_view(views);
    }
};

// What the R external pointer holds.
using GpuTablePtr = std::shared_ptr<GpuTable>;

// Release scopes (F58, src/gpu_scope.cpp). While a GPU step's scope is the
// innermost one, every table make_gpu_table_xptr() creates is recorded in it;
// tables nobody returns are freed when the step's user-level call returns.
void gpu_scope_record(const GpuTablePtr& table);
// The tag every table XPtr carries, so a scope can tell them from other
// external pointers in a returned value.
SEXP gpu_table_xptr_tag();

// Custom destructor that ensures GPU cleanup
inline void release_gpu_table(GpuTablePtr* ptr) {
    if (ptr != nullptr) {
        // Reset drops this table's column references; a column's device memory
        // is freed when the last table sharing it goes.
        ptr->reset();
        delete ptr;
    }
}

inline SharedColumn share_column(std::unique_ptr<cudf::column> col) {
    return SharedColumn(std::move(col));
}

// Create an XPtr from shared columns (no copies).
inline Rcpp::XPtr<GpuTablePtr> make_gpu_table_xptr(std::vector<SharedColumn> columns) {
    auto table = std::make_shared<GpuTable>();
    table->columns = std::move(columns);
    gpu_scope_record(table);
    auto* sptr = new GpuTablePtr(std::move(table));
    return Rcpp::XPtr<GpuTablePtr>(sptr, true, gpu_table_xptr_tag());  // true = register destructor
}

// Create an XPtr from a freshly built table: its columns become shared.
inline Rcpp::XPtr<GpuTablePtr> make_gpu_table_xptr(std::unique_ptr<cudf::table> tbl) {
    std::vector<std::unique_ptr<cudf::column>> released = tbl->release();
    std::vector<SharedColumn> columns;
    columns.reserve(released.size());
    for (auto& c : released) columns.push_back(share_column(std::move(c)));
    return make_gpu_table_xptr(std::move(columns));
}

// Extract table_view from XPtr (non-owning view)
inline cudf::table_view get_table_view(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    if ((*xptr)->released) {
        Rcpp::stop("GPU table was released: it was an intermediate of a finished operation and is no longer valid");
    }
    return (*xptr)->view();
}

// The shared owner of input column `i`, to put into a result table.
inline SharedColumn get_shared_column(Rcpp::XPtr<GpuTablePtr> xptr, int i) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    if ((*xptr)->released) {
        Rcpp::stop("GPU table was released: it was an intermediate of a finished operation and is no longer valid");
    }
    const auto& cols = (*xptr)->columns;
    if (i < 0 || static_cast<size_t>(i) >= cols.size()) {
        Rcpp::stop("Column index out of bounds: %d", i);
    }
    return cols[i];
}

} // namespace cuplyr

#endif // CUPLYR_GPU_TABLE_HPP
