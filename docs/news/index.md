# Changelog

## cuplyr 0.1.1

### Bug fixes

- [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) now
  correctly restores factor columns with their original levels.
  Previously, factor columns were returned as integer codes instead of
  factors ([\#3](https://github.com/bbtheo/cuplyr/issues/3)).

- `names<-()` now validates the replacement value before assignment. It
  errors if the new names have wrong length, contain `NA` values, empty
  strings, or are not character
  ([\#4](https://github.com/bbtheo/cuplyr/issues/4)).

### Build system

- Fixed CCCL (CUDA Core Compute Libraries) header detection for RAPIDS
  25.12+ in pixi/conda environments where headers are located in a
  `rapids/` subdirectory. The configure script now automatically detects
  `<cuda/stream_ref>` and related headers.

- Reordered include paths so CUDF/RMM headers take precedence over
  system CUDA headers, ensuring consistent CCCL versions.

### Documentation

- Added pkgdown documentation site with vignettes for getting started,
  complex analysis workflows, and query optimization.

## cuplyr 0.1.0

### Lazy evaluation

- [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  gains a `lazy` argument to enable deferred execution. Operations build
  an AST (Abstract Syntax Tree) that is optimized and executed only when
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) or
  [`compute()`](https://dplyr.tidyverse.org/reference/compute.html) is
  called. Set globally with `options(cuplyr.exec_mode = "lazy")` or
  `CUPLYR_EXEC_MODE=lazy` environment variable.

- Added an AST optimizer that applies multiple optimization passes
  before execution:

  - Projection pruning: push column selection close to data sources
  - Mutate fusion: combine consecutive mutate operations
  - Dead column pruning: remove unused intermediate columns
  - Filter pushdown: move filters earlier in the pipeline, including
    across joins
  - Filter reordering: execute cheaper filters first
  - Filter fusion: combine multiple filters into single GPU kernel

- [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
  executes pending lazy operations and keeps the result on GPU.

- [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html)
  inserts an optimization barrier without executing.

- [`as_lazy()`](https://bbtheo.github.io/cuplyr/reference/as_lazy.md)
  and
  [`as_eager()`](https://bbtheo.github.io/cuplyr/reference/as_eager.md)
  switch execution modes mid-pipeline.

- [`is_lazy()`](https://bbtheo.github.io/cuplyr/reference/is_lazy.md)
  and
  [`has_pending_ops()`](https://bbtheo.github.io/cuplyr/reference/has_pending_ops.md)
  check the current execution state.

- [`show_query()`](https://bbtheo.github.io/cuplyr/reference/show_query.md)
  displays the pending operation tree for debugging.

### Join operations

- Added
  [`inner_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  [`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html),
  and
  [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  for combining GPU tables
  ([\#2](https://github.com/bbtheo/cuplyr/issues/2)). Joins support
  automatic key detection (natural join), named vectors in `by` for
  different key names, `suffix` for column name conflicts, `keep` for
  retaining join keys, and `copy` to auto-transfer data frames to GPU.

### Bind operations

- Added
  [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
  for vertically combining GPU tables with automatic schema unification
  and type promotion.

- Added
  [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
  for horizontally combining GPU tables with `.name_repair` for
  duplicate column handling.

- Both bind functions automatically materialize lazy tables before
  binding.

## cuplyr 0.0.1

Initial release of cuplyr, a GPU-accelerated dplyr backend using
NVIDIA’s libcudf library.

### Core functionality

- [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  transfers R data frames to GPU memory, returning a `tbl_gpu` object
  that works with dplyr verbs.

- [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
  transfers GPU data back to R as a tibble.

- [`as_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/as_tbl_gpu.md)
  coerces data frames to GPU tables.

- [`is_tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/is_tbl_gpu.md)
  tests if an object is a GPU table.

### dplyr verbs

- [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
  supports scalar comparisons (`x > 5`, `x == "a"`) and column-to-column
  comparisons (`x > y`). Supports boolean vectors as filter masks.

- [`select()`](https://dplyr.tidyverse.org/reference/select.html)
  supports column selection by name, position, and tidyselect helpers.

- [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  supports arithmetic operations (`+`, `-`, `*`, `/`, `^`) with scalars
  and between columns. Supports left-associative chains (e.g.,
  `a + b + c`).

- [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html)
  sorts by one or more columns with
  [`desc()`](https://dplyr.tidyverse.org/reference/desc.html) support
  for descending order. Supports `.by_group = TRUE` for grouped tables.

- [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  sets grouping metadata for subsequent aggregation.
  [`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html)
  removes grouping.

- [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  computes grouped aggregations with support for
  [`sum()`](https://rdrr.io/r/base/sum.html),
  [`mean()`](https://rdrr.io/r/base/mean.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html),
  [`max()`](https://rdrr.io/r/base/Extremes.html),
  [`n()`](https://dplyr.tidyverse.org/reference/context.html),
  [`sd()`](https://rdrr.io/r/stats/sd.html), and
  [`var()`](https://rdrr.io/r/stats/cor.html). Supports expressions
  inside aggregation functions (e.g., `sum(x > 0)`).

### Type support

- Supported R types: numeric (FLOAT64), integer (INT32), character
  (STRING), logical (BOOL8), Date (TIMESTAMP_DAYS), and POSIXct
  (TIMESTAMP_MICROSECONDS).

- factor columns are converted to INT32 codes.

- integer64 columns are converted to FLOAT64 with a warning about
  precision loss for values exceeding 2^53.

### GPU memory utilities

- [`gpu_memory_usage()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_usage.md)
  estimates GPU memory footprint of a `tbl_gpu` object.

- [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
  returns current GPU memory usage (total, free, used).

- [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
  forces garbage collection to free GPU memory from unreferenced tables.

- [`gpu_object_info()`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
  returns detailed information about a GPU table.

- [`verify_gpu_data()`](https://bbtheo.github.io/cuplyr/reference/verify_gpu_data.md)
  confirms data resides on GPU, not in R memory.

- [`gpu_size_comparison()`](https://bbtheo.github.io/cuplyr/reference/gpu_size_comparison.md)
  compares R object size vs GPU data size.

### GPU information

- [`has_gpu()`](https://bbtheo.github.io/cuplyr/reference/has_gpu.md)
  checks if a compatible GPU is available.

- [`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
  returns GPU device information (name, compute capability, memory).
