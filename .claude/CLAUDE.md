# cuplyr – Notes for Coding Agents

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
  exec_mode = "eager",           # "eager" or "lazy"
  lazy_ops = list()              # AST for lazy evaluation
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
| integer64 | FLOAT64 | loses precision >2^53, warns |

## Source Map

### C++ Files
| File | Purpose |
|------|---------|
| `src/gpu_table.hpp` | `GpuTablePtr`, `make_gpu_table_xptr()`, `get_table_view()` |
| `src/cuda_utils.hpp` | `check_cuda()` error helper |
| `src/ops_common.hpp` | `get_compare_op()`, `get_binary_op()` |
| `src/transfer_io.cpp` | `df_to_gpu()`, `gpu_collect()`, `gpu_head()`, `gpu_dim()` |
| `src/ops_filter.cpp` | `gpu_filter_scalar()`, `gpu_filter_col()`, `gpu_filter_mask()` |
| `src/ops_filter_fused.cpp` | `gpu_filter_fused()` for multi-predicate AND masks |
| `src/ops_mutate.cpp` | `gpu_mutate_binary_*()`, `gpu_copy_column*()` |
| `src/ops_mutate_batch.cpp` | `gpu_mutate_batch()` for fused mutate expressions |
| `src/ops_select.cpp` | `gpu_select()` |
| `src/ops_groupby.cpp` | `gpu_summarise()` |
| `src/ops_arrange.cpp` | `gpu_arrange()` |
| `src/ops_compare.cpp` | comparison ops for summarise temp columns |
| `src/ops_join.cpp` | join logic with stable-sort for dplyr ordering |
| `src/ops_bind.cpp` | `gpu_bind_rows_aligned()`, `gpu_bind_cols_impl()` |
| `src/gpu_info.cpp` | device availability/info |

### R Files
| File | Purpose |
|------|---------|
| `R/tbl-gpu.R` | `tbl_gpu()`, `new_tbl_gpu()`, `is_tbl_gpu()`, `resolve_exec_mode()` |
| `R/utils.R` | `gpu_type_from_r()`, `col_index()` |
| `R/filter.R` | filter verb with boolean literal fast-path |
| `R/mutate.R` | mutate verb with left-associative chain support |
| `R/select.R` | select verb |
| `R/arrange.R` | arrange verb |
| `R/group-by.R` | `group_by()`, `ungroup()`, `group_vars()` |
| `R/summarise.R` | summarise/groupby verb |
| `R/join.R` | join verbs, `build_join_schema()`, `build_join_output_info()` |
| `R/bind.R` | `bind_rows()`, `bind_cols()` with schema unification |
| `R/collect.R` | pulls data to R, warns on INT64 precision loss |
| `R/compute.R` | `compute()`, `collapse()`, `as_lazy()`, `as_eager()`, `show_query()` |
| `R/ast.R` | AST node constructors (`ast_source`, `ast_filter`, etc.) |
| `R/optimizer.R` | AST optimization passes (projection, filter pushdown, fusion) |
| `R/lower.R` | `lower_and_execute()` AST to GPU execution |
| `R/gpu-memory.R` | memory reporting and GC helpers |
| `R/gpu.R` | `has_gpu()`, `gpu_details()` |
| `R/print.R` | `print.tbl_gpu()` method |

### Benchmark Files
| File | Purpose |
|------|---------|
| `benchmark/benchmark_memory.R` | lazy vs eager memory/time comparison |
| `benchmark/benchmark_filter_pushdown.R` | auto vs manual filter placement |

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

## Implementing a New dplyr Verb

### Step 1: C++ Implementation (`src/ops_<verb>.cpp`)

```cpp
#include "gpu_table.hpp"
#include "cuda_utils.hpp"
#include <cudf/...>  // operation-specific headers
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP gpu_<verb>(SEXP xptr, /* params */) {
    using namespace cuplyr;

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

## AST & Lazy Evaluation

### Lazy Mode Mechanics
- Lazy tables defer execution by building an AST in `$lazy_ops`
- Operations return new `tbl_gpu` with updated AST, no GPU work
- `collect()` triggers AST lowering and execution

### Optimizer Passes
The optimizer transforms the AST before execution. Pass order matters:

1. **Projection Pruning** (`push_down_projections`): Inserts `select` nodes to drop unused columns early
   - `build_join_output_info()` maps output columns back to left/right sources
   - Dead columns are dropped before expensive operations

2. **Mutate Fusion** (`fuse_mutates`): Combines consecutive mutate nodes
   - Guards: max 8 expressions, max 4 intermediates, max 3 reuses
   - Uses topological sort for dependent expressions

3. **Dead Column Pruning** (`prune_dead_columns`): Removes unused mutate outputs
   - Walks root to leaves tracking required columns
   - Empty mutate nodes are eliminated

4. **Filter Pushdown** (`push_down_filters`): Moves filters closer to data sources
   - Pushes across mutate when predicates don't depend on outputs
   - Join pushdown rules:
     - Inner join: left-only predicates -> left, right-only -> right
     - Left join: only left-only predicates pushed
     - Right join: only right-only predicates pushed
     - Full join: no side-only pushdown allowed

5. **Filter Reordering** (`reorder_filters`): Executes cheaper filters first
   - Collects consecutive filter chains
   - Sorts by `estimated_cost` field

6. **Filter Fusion** (`fuse_filters`): Marks filters for single-kernel AND-mask
   - Max 4 simple predicates for fusion

### Join-Specific Notes
- Lazy joins build `ast_join` with two inputs
- Source pointers attached via `set_ast_source_ptr()` before lowering
- Schema inference: `infer_schema.ast_join` uses `build_join_schema()` in `R/join.R`
- Right join: implemented via swapped left join + column reorder
- Ensure desired column names exist before calling `gpu_select`

### Optimizer Barriers
- Barrier reattachment uses a helper to avoid R copy-on-modify pitfalls
- Be careful when modifying AST nodes in place

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

## Known Issues & Sharp Edges

### Environment-Specific cuDF Issues
- **Header locations differ by version**: `bitmask_allocation_size_bytes` is in `cudf/null_mask.hpp`, NOT `cudf/bitmask.hpp`
- **Join headers**: Use `<cudf/join/join.hpp>`, not `<cudf/join.hpp>`
- **cuDF gather API**: This environment's `cudf::gather` has no `negative_index_policy` parameter
- **Avoid device-side Thrust** unless compiling with nvcc

### Type Consistency
R schema types MUST match actual GPU column types. If changing type handling:
- Update `R/utils.R::gpu_type_from_r()`
- Update `src/transfer_io.cpp::df_to_gpu()`
- Update tests for round-trip behavior

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

### Join Ordering & Unmatched Rows
- **cuDF join outputs are unordered**: We stable-sort join maps by left_map (then right_map) in `src/ops_join.cpp` to match dplyr
- **JoinNoMatch sentinel**: cuDF uses sentinel values; gather treats negatives as wraparound
- **Current fix**: Sanitize join maps on CPU (replace with `nrows`) before gather, use `out_of_bounds_policy::NULLIFY`
- **Right key dropping**: `keep = FALSE` drops right keys even when names differ

### Filter Parsing
- Boolean literal fast-path avoids `eval_tidy` without a data mask to prevent name-collision bugs

### Mutate Parsing
- Supports left-associative `+`/`-` chains (e.g., `a + b + c`) by lowering to sequential ops

### Bind Operations
- `bind_rows()` computes unified schema via `compute_unified_schema()`
- Type promotion hierarchy: BOOL8 < INT32 < INT64 < FLOAT64; STRING is widest
- Missing columns filled with nulls via `gpu_make_null_column()`
- `bind_cols()` uses `vctrs::vec_as_names()` for name repair when available
- Both operations materialize lazy tables before binding

## Debugging Build Failures

- If a cudf header can't be found, check pixi environment paths and `src/Makevars`
- Run `pixi run configure` after updating CUDA/cudf libs
- Use `rg --files -g '*bitmask*' $CONDA_PREFIX/include/cudf` to locate moved headers
- Join build errors: confirm `#include <cudf/join/join.hpp>` and avoid device-side Thrust unless compiling with nvcc
- GPU not detected is common in CI or local dev; tests use `skip_if_no_gpu()`
- Rcpp exports need regeneration after moving/adding functions: run `Rcpp::compileAttributes()` or `devtools::document()`

## Testing

### Available Helpers (`tests/testthat/helper-*.R`)
```r
skip_if_no_gpu()           # Skip test if no GPU available
expect_valid_tbl_gpu(x)    # Check tbl_gpu structure (allows exec_mode field)
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

### Lazy vs Eager Cross-Mode Testing
Use `tibble::as_tibble()` to avoid rownames/class mismatches when comparing:
```r
expect_equal(

  as_tibble(eager_result),
  as_tibble(lazy_result)
)
```

## Code Review Checklist

Before merging dev to master, verify:
- [ ] **Branch references**: Update Colab links back to `master` branch:
  - `README.md`: Change `blob/dev` to `blob/master` in Colab badge URLs
  - `notebooks/install_cuplyr.ipynb`: Change git clone from `-b dev` to remove the flag (or use `-b master`)
- [ ] **Type alignment**: `R/utils.R` and `src/transfer_io.cpp` agree on type mapping
- [ ] **Head/collect parity**: If a type is supported in `gpu_collect()`, ensure `gpu_head()` uses the same conversion path
- [ ] **Arrange semantics**: Stable sort support, NA ordering, group-prepend behavior
- [ ] **Rcpp exports**: If new C++ functions exist, `R/RcppExports.R` and `src/RcppExports.cpp` are updated
- [ ] **Tests**: New features have matching `tests/testthat/test-*.R` coverage with `skip_if_no_gpu()`
- [ ] **Joins**: Left-table order preserved, right-key dropping correct, pushdown rules followed
- [ ] **Binds**: Type promotion correct, null columns for missing, lazy tables materialized

## Development Mandates

**Test-first bugfixes (STRICT)**: When a bug is reported, always add a failing test that reproduces it *before* implementing the fix. If the test passes unexpectedly, revert the fix, confirm failure, then re-apply.

**Run tests after every feature (STRICT)**: After implementing any feature or fix, always run `pixi run test` before considering the work complete. Do not skip this step. If tests fail, fix the issue before moving on.

**Roxygen2 for exports (STRICT)**: NEVER edit the NAMESPACE file by hand. Always use `#' @export` roxygen tags on functions and run `devtools::document()` (or `pixi run load-dev` which triggers it) to regenerate NAMESPACE. The same applies to `@importFrom` directives. Manual NAMESPACE edits will be overwritten.

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
| Joins | `<cudf/join/join.hpp>` | `inner_join`, `left_join`, `full_join` |
| Concatenation | `<cudf/concatenate.hpp>` | `concatenate` (for bind_rows/bind_cols) |
