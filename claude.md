# cuplr – Notes for Coding Agents

GPU-backed dplyr API in R with C++/Rcpp bindings to libcudf.

## Quick Reference

### Core Data Structures
```r
# tbl_gpu structure (R/tbl-gpu.R)
list(
 ptr = <externalptr>,           # XPtr to cudf::table
 schema = list(
   names = c("col1", "col2"),   # column names
   types = c("FLOAT64", "INT32") # GPU type strings
 ),
 groups = c("col1"),            # group_by columns (can be empty)
 lazy_ops = list()              # reserved for future lazy eval
)
```

### Type Mappings
| R Type | GPU Type | Notes |
|--------|----------|-------|
| logical | BOOL8 | |
| integer | INT32 | |
| double | FLOAT64 | |
| character | STRING | |
| Date | TIMESTAMP_DAYS | |
| POSIXct | TIMESTAMP_MICROSECONDS | |
| factor | INT32 | codes only |
| integer64 | FLOAT64 | ⚠️ loses precision >2^53, warns |

## Implementing a New dplyr Verb

### Step 1: C++ Implementation (`src/ops_<verb>.cpp`)

```cpp
// Template for new GPU operation
#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include <cudf/...>  // operation-specific headers
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_<verb>(SEXP xptr, /* params */) {
    using namespace cuplr;

    // 1. Get table view
    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // 2. Validate inputs
    if (col_idx < 0 || col_idx >= view.num_columns()) {
        Rcpp::stop("Column index out of bounds: %d", col_idx);
    }

    // 3. Perform cuDF operation
    auto result = cudf::some_operation(view, ...);

    // 4. Return new table
    return make_gpu_table_xptr(std::move(result));
}
```

### Step 2: R Wrapper (`R/<verb>.R`)

```r
#' @export
#' @importFrom dplyr <verb>
<verb>.tbl_gpu <- function(.data, ...) {
 # 1. Capture expressions
 dots <- rlang::enquos(...)
 if (length(dots) == 0) return(.data)

 # 2. Parse expressions (verb-specific)
 for (i in seq_along(dots)) {
   expr <- dots[[i]]
   expr_text <- rlang::quo_text(expr)
   # ... parse and validate
 }

 # 3. Convert column names to 0-based indices
 col_idx <- match(col_name, .data$schema$names) - 1L

 # 4. Call C++ function
 new_ptr <- gpu_<verb>(.data$ptr, col_idx, ...)

 # 5. Return new tbl_gpu (preserves schema/groups unless modified)
 new_tbl_gpu(
   ptr = new_ptr,
   schema = .data$schema,  # or modified schema
   groups = .data$groups
 )
}
```

### Step 3: Update Exports

**NAMESPACE** (add these lines):
```
S3method(<verb>,tbl_gpu)
importFrom(dplyr,<verb>)
```

**Rcpp exports** (run or manually add):
```r
Rcpp::compileAttributes()
```

Or manually add to:
- `src/RcppExports.cpp`: function declaration + wrapper + CallEntries entry
- `R/RcppExports.R`: R wrapper function

### Step 4: Tests (`tests/testthat/test-<verb>.R`)

```r
test_that("<verb>() basic case works", {
 skip_if_no_gpu()

 df <- data.frame(x = c(1, 2, 3))
 gpu_df <- tbl_gpu(df)

 result <- gpu_df |>
   dplyr::<verb>(...) |>
   collect()

 expect_equal(result$x, expected)
})
```

## cuDF Header Quick Reference

| Operation | Header | Key Functions |
|-----------|--------|---------------|
| Sorting | `<cudf/sorting.hpp>` | `sorted_order`, `stable_sorted_order`, `sort` |
| Gathering/Scattering | `<cudf/copying.hpp>` | `gather`, `scatter`, `empty_like` |
| Filtering | `<cudf/stream_compaction.hpp>` | `apply_boolean_mask` |
| Binary ops | `<cudf/binaryop.hpp>` | `binary_operation` |
| Aggregation | `<cudf/aggregation.hpp>`, `<cudf/groupby.hpp>` | `groupby::aggregate` |
| Null handling | `<cudf/null_mask.hpp>` | `bitmask_allocation_size_bytes` |
| Scalars | `<cudf/scalar/scalar.hpp>`, `<cudf/scalar/scalar_factories.hpp>` | `make_numeric_scalar` |

## Source Map

| File | Purpose |
|------|---------|
| `src/gpu_table.hpp` | `GpuTablePtr`, `make_gpu_table_xptr()`, `get_table_view()` |
| `src/cuda_utils.hpp` | `check_cuda()` error helper |
| `src/ops_common.hpp` | `get_compare_op()`, `get_binary_op()` |
| `src/transfer_io.cpp` | `df_to_gpu()`, `gpu_collect()`, `gpu_head()`, `gpu_dim()` |
| `src/ops_filter.cpp` | `gpu_filter_scalar()`, `gpu_filter_col()`, `gpu_filter_mask()` |
| `src/ops_mutate.cpp` | `gpu_mutate_binary_*()`, `gpu_copy_column*()` |
| `src/ops_select.cpp` | `gpu_select()` |
| `src/ops_groupby.cpp` | `gpu_summarise()` |
| `src/ops_arrange.cpp` | `gpu_arrange()` |
| `R/tbl-gpu.R` | `tbl_gpu()`, `new_tbl_gpu()`, `is_tbl_gpu()` |
| `R/utils.R` | `gpu_type_from_r()`, `col_index()` |

## Common Patterns

### Expression Parsing (R side)
```r
# Capture unevaluated expressions
dots <- rlang::enquos(...)

# Get expression text
expr_text <- rlang::quo_text(expr)

# Get raw expression
raw_expr <- rlang::quo_get_expr(expr)

# Check if symbol (bare column name)
if (is.symbol(raw_expr)) col_name <- as.character(raw_expr)

# Check if call (function application)
if (is.call(raw_expr)) {
 fn_name <- as.character(raw_expr[[1]])  # e.g., "desc", "-"
 arg <- raw_expr[[2]]                     # first argument
}
```

### Column Index Conversion
```r
# R uses 1-based, C++ uses 0-based
col_idx_r <- match(col_name, .data$schema$names)  # 1-based, NA if not found
col_idx_cpp <- col_idx_r - 1L                      # 0-based for C++
```

### Validation Patterns (C++)
```cpp
// Bounds check
if (col_idx < 0 || col_idx >= view.num_columns()) {
    Rcpp::stop("Column index out of bounds: %d (table has %d columns)",
               col_idx, view.num_columns());
}

// NA check in LogicalVector
if (LogicalVector::is_na(value[i])) {
    Rcpp::stop("NA values not allowed in parameter");
}

// Row count limit (int32 for indices)
if (view.num_rows() > static_cast<cudf::size_type>(INT32_MAX)) {
    Rcpp::stop("Table too large (max ~2.1 billion rows)");
}
```

## Local Dev (pixi)

### Commands
| Command | When to Use |
|---------|-------------|
| `pixi run load-dev` | R-only changes, quick iteration |
| `pixi run install` | After C++ changes or export changes |
| `pixi run dev` | Stale artifacts, clean rebuild |
| `pixi run test` | After feature work (requires GPU) |
| `pixi run configure` | Only when CUDA/cudf paths change |

### Workflow
1. Edit code
2. `pixi run install` (if C++ changed) or `pixi run load-dev` (R only)
3. Test in R: `tbl_gpu(mtcars) |> filter(mpg > 20) |> collect()`
4. `pixi run test`

## Known Issues & Watchpoints

### Type Consistency
R schema types MUST match actual GPU column types. If changing type handling:
- Update `R/utils.R::gpu_type_from_r()`
- Update `src/transfer_io.cpp::df_to_gpu()`
- Update tests

### String Column Operations
String columns use offset-based storage (Apache Arrow format):
- `col.child(0)` = offsets column (int32, n+1 elements)
- `col.data<char>()` = concatenated character data
- When slicing (e.g., head), must slice both offsets and chars correctly

### Memory Semantics
- Most ops allocate new tables (immutable design)
- Peak memory for sorting: ~2x table size
- Use `gpu_memory_state()` to monitor
- GC frees GPU memory via pointer release

### Grouping Behavior
- `group_by()` only sets metadata (`$groups`), no GPU work
- Groups are applied during `summarise()`
- `arrange(.by_group=TRUE)` prepends group columns to sort keys
- Preserve `$groups` when returning new tbl_gpu unless ungrouping

## Testing

### Available Helpers (`tests/testthat/helper-*.R`)
```r
skip_if_no_gpu()           # Skip test if no GPU available
expect_valid_tbl_gpu(x)    # Check tbl_gpu structure
expect_data_on_gpu(x)      # Verify data is on GPU
gc_gpu()                   # Force GPU garbage collection
gpu_memory_snapshot()      # Get memory state for comparison
```

### Test Pattern
```r
test_that("operation handles edge case", {
 skip_if_no_gpu()

 # Setup
 df <- data.frame(...)
 gpu_df <- tbl_gpu(df)

 # Execute
 result <- gpu_df |> some_operation() |> collect()

 # Assert
 expect_equal(result$col, expected)
})
```
