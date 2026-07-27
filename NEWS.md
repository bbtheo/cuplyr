# cuplyr (development version)

## Unified eager/lazy execution

* Every dplyr verb now builds a single AST node and routes through one shared execution path (`push_op()`/`push_join()` in `R/execute.R`, `lower_and_execute()` in `R/lower.R`) regardless of whether the table is eager or lazy. This collapsed a number of eager/lazy divergences that previously produced different results depending on execution mode; see the fixes below.

## Bug fixes

* `right_join()` now works at all; previously it errored on every call due to an internal argument-parsing mismatch.

* `full_join()` and `right_join()` now coalesce join key columns for unmatched rows instead of leaving them `NA`. For example, `full_join(x, y, by = "id")` now populates `id` from `y` for rows that only matched on the right side.

* `bind_rows()` now actually casts columns that get type-promoted to `STRING` (e.g. combining a numeric column in one table with a character column in another). Previously the promoted column kept its original GPU type while the schema claimed `STRING`, causing a type-mismatch failure.

* `mutate()` no longer silently promotes integer arithmetic to `FLOAT64`. `+`, `-`, `*`, `%%`, and `%/%` now preserve `INT32`/`INT64` when every input column and literal involved is integer or logical (matching R/dplyr, e.g. `TRUE + TRUE` is `2L`). Note that `INT32` results can now wrap around on overflow instead of silently promoting to a wider type, matching plain integer arithmetic semantics. `/` and `^` continue to always promote to `FLOAT64` (matching R: `1L / 2L` and `2L ^ 2L` are both doubles).

* `arrange(..., .by_group = TRUE)` now sorts prepended group columns in ascending order to match `dplyr::arrange.grouped_df()`, even when the user's own expression wraps a group column in `desc()` (e.g. `arrange(desc(g), x, .by_group = TRUE)` now sorts `g` ascending, `x` per the user's spec).

* `filter(TRUE)` is now a true no-op: it returns `.data` unchanged without doing GPU work, instead of round-tripping through a filter kernel.

* Lazy `summarise()` with `sd()`/`var()` no longer errors. Lazily evaluated aggregations now translate to the same cudf function names (`std`/`variance`) that the eager path always used.

* `summarise()` now accepts namespaced aggregation calls, e.g. `dplyr::n()` and `stats::sd(mpg)`, in both eager and lazy modes.

* `summarise(n = n())` now always returns an integer (`INT32`) column, matching `dplyr`. Previously, ungrouped `n()` silently returned a double due to a schema/GPU-column type mismatch.

* `summarise()` with an unnamed aggregation expression (e.g. `summarise(mean(mpg))`) now auto-names the output column from the expression text and warns, matching `mutate()`'s existing behavior for unnamed expressions. This had regressed to silently doing neither in either mode.

* Eager `select()` of a factor column now round-trips as a factor through `collect()`. Previously the eager path dropped factor level metadata, so `collect()` returned integer codes instead of factor labels.

## New features

* Added the `cuplyr.fallback` option to control whether `filter()`'s and `summarise()`'s CPU-evaluation fallback paths notify the caller. One of `"silent"` (default), `"warn"`, or `"error"`.

# cuplyr 0.1.1

## Bug fixes

* `collect()` now correctly restores factor columns with their original levels. Previously, factor columns were returned as integer codes instead of factors (#3).

* `names<-()` now validates the replacement value before assignment. It errors if the new names have wrong length, contain `NA` values, empty strings, or are not character (#4).

## Build system

* Fixed CCCL (CUDA Core Compute Libraries) header detection for RAPIDS 25.12+ in pixi/conda environments where headers are located in a `rapids/` subdirectory. The configure script now automatically detects `<cuda/stream_ref>` and related headers.

* Reordered include paths so CUDF/RMM headers take precedence over system CUDA headers, ensuring consistent CCCL versions.

## Documentation

* Added pkgdown documentation site with vignettes for getting started, complex analysis workflows, and query optimization.

# cuplyr 0.1.0

## Lazy evaluation

* `tbl_gpu()` gains a `lazy` argument to enable deferred execution. Operations build an AST (Abstract Syntax Tree) that is optimized and executed only when `collect()` or `compute()` is called. Set globally with `options(cuplyr.exec_mode = "lazy")` or `CUPLYR_EXEC_MODE=lazy` environment variable.

* Added an AST optimizer that applies multiple optimization passes before execution:
  - Projection pruning: push column selection close to data sources
  - Mutate fusion: combine consecutive mutate operations
  - Dead column pruning: remove unused intermediate columns
  - Filter pushdown: move filters earlier in the pipeline, including across joins
  - Filter reordering: execute cheaper filters first
  - Filter fusion: combine multiple filters into single GPU kernel

* `compute()` executes pending lazy operations and keeps the result on GPU.

* `collapse()` inserts an optimization barrier without executing.

* `as_lazy()` and `as_eager()` switch execution modes mid-pipeline.

* `is_lazy()` and `has_pending_ops()` check the current execution state.

* `show_query()` displays the pending operation tree for debugging.

## Join operations

* Added `inner_join()`, `left_join()`, `right_join()`, and `full_join()` for combining GPU tables (#2). Joins support automatic key detection (natural join), named vectors in `by` for different key names, `suffix` for column name conflicts, `keep` for retaining join keys, and `copy` to auto-transfer data frames to GPU.

## Bind operations

* Added `bind_rows()` for vertically combining GPU tables with automatic schema unification and type promotion.

* Added `bind_cols()` for horizontally combining GPU tables with `.name_repair` for duplicate column handling.

* Both bind functions automatically materialize lazy tables before binding.

# cuplyr 0.0.1

Initial release of cuplyr, a GPU-accelerated dplyr backend using NVIDIA's libcudf library.

## Core functionality

* `tbl_gpu()` transfers R data frames to GPU memory, returning a `tbl_gpu` object that works with dplyr verbs.

* `collect()` transfers GPU data back to R as a tibble.

* `as_tbl_gpu()` coerces data frames to GPU tables.

* `is_tbl_gpu()` tests if an object is a GPU table.

## dplyr verbs

* `filter()` supports scalar comparisons (`x > 5`, `x == "a"`) and column-to-column comparisons (`x > y`). Supports boolean vectors as filter masks.

* `select()` supports column selection by name, position, and tidyselect helpers.

* `mutate()` supports arithmetic operations (`+`, `-`, `*`, `/`, `^`) with scalars and between columns. Supports left-associative chains (e.g., `a + b + c`).

* `arrange()` sorts by one or more columns with `desc()` support for descending order. Supports `.by_group = TRUE` for grouped tables.

* `group_by()` sets grouping metadata for subsequent aggregation. `ungroup()` removes grouping.

* `summarise()` computes grouped aggregations with support for `sum()`, `mean()`, `min()`, `max()`, `n()`, `sd()`, and `var()`. Supports expressions inside aggregation functions (e.g., `sum(x > 0)`).

## Type support

* Supported R types: numeric (FLOAT64), integer (INT32), character (STRING), logical (BOOL8), Date (TIMESTAMP_DAYS), and POSIXct (TIMESTAMP_MICROSECONDS).

* factor columns are converted to INT32 codes.

* integer64 columns are converted to FLOAT64 with a warning about precision loss for values exceeding 2^53.

## GPU memory utilities

* `gpu_memory_usage()` estimates GPU memory footprint of a `tbl_gpu` object.

* `gpu_memory_state()` returns current GPU memory usage (total, free, used).

* `gpu_gc()` forces garbage collection to free GPU memory from unreferenced tables.

* `gpu_object_info()` returns detailed information about a GPU table.

* `verify_gpu_data()` confirms data resides on GPU, not in R memory.

* `gpu_size_comparison()` compares R object size vs GPU data size.

## GPU information

* `has_gpu()` checks if a compatible GPU is available.

* `gpu_details()` returns GPU device information (name, compute capability, memory).
