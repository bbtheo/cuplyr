# cuplyr (development version)

## Rolling (moving-window) functions: `roll_mean()`/`roll_sum()`/`roll_min()`/`roll_max()`/`roll_sd()`/`roll_median()` (Theo-requested benchmark task)

GPU-native rolling windows, added as a sixth window-function kind ("rolling")
alongside the existing shift/rank/scan/agg families (`R/window.R`'s
decomposition machinery, `src/ops_window.cpp`'s `gpu_window()`) -- no new
verb wiring was needed in `mutate()`/`filter()` at all, both pick these up
automatically through the same shared window-decomposition pass every other
window function already goes through.

* `roll_mean(x, n, na.rm = FALSE)`, `roll_sum()`, `roll_min()`, `roll_max()`,
  `roll_sd()` (RcppRoll/slider-adjacent naming) are GPU-native via
  `cudf::rolling_window()`/`grouped_rolling_window()` (right-aligned only
  this wave; `align=` is not a parameter on this surface). Grouped rolling
  (`group_by()`/`.by=`) works via the same frame machinery every other
  window function uses, with no new C++ primitive beyond the rolling calls
  themselves.
* `data.table::frollmean()`/`frollsum()`/`frollmin()`/`frollmax()`/`frollsd()`/
  `frollmedian()` are accepted as GPU-native aliases for the above (plus
  `roll_median()`, see below), as long as `align="right"` (the default),
  `fill=NA` (the default), and `adaptive=FALSE`/`partial=FALSE`/
  `give.names=FALSE` (all defaults) hold -- any other value is a hard
  `mutate()` error (no silent behavior change), matching this task's
  explicit right-aligned-only scope.
* `roll_median()` (and its `data.table::frollmedian()` alias) is registered
  and parses/types identically to its five siblings, but is **not**
  GPU-native: cudf 25.12 has no rolling MEDIAN aggregation instantiation at
  all in this environment (verified via `nm -DC libcudf.so`), so it
  transparently falls back to a CPU `roll_median()` implementation
  (`R/roll.R`, also directly exported/usable standalone) via the existing
  `mutate_window_fallback()` path -- still produces exactly the same result,
  just off-GPU.
* `na.rm = FALSE` (the default): any window containing an `NA` is `NA`
  ("poisoning") -- falls out of `cudf::rolling_window()`'s own
  `min_periods = n` requirement on a fixed `n`-row window with NO extra
  masking pass needed (verified empirically, a pleasant simplification vs.
  the cum*/rank families' sticky-NA fix). `na.rm = TRUE`: leading
  physically-incomplete windows (fewer than `n` rows available yet) are
  still `NA`, but `NA` *values* inside an otherwise-complete window are
  skipped -- verified bit-for-bit against `data.table::frollmean(...,
  na.rm = TRUE)`'s own output, since this is NOT simply "cudf's default
  null-skipping behavior" (that alone would incorrectly compute a value for
  the leading incomplete windows too).
* See `benchmark/benchmark_rolling.R` for a locally-measured cuplyr-vs-
  `data.table::froll*()` head-to-head across three data-size tiers
  (1e6/1e7/1e8 rows) and the original benchmark-post workload grid (mean/
  median/min/max/sum/sd x window 11/101/1001).

## NA-semantics parity fixes: `summarise()` `na.rm=` and `arrange(desc())` NA placement (Phase 6, task 6.1)

Two live correctness divergences from dplyr, both found during Phase 5, are fixed:

* **`summarise()` aggregations now honor R's own `na.rm = FALSE` default.** `mean()`/`sum()`/`min()`/`max()`/`sd()`/`var()` previously always used cudf's native NA-excluding behavior regardless of the user's call, silently diverging from dplyr whenever a group contained an `NA` (e.g. `summarise(m = mean(x))` returned the NA-excluded mean instead of `NA`). `summarise()` now propagates `NA` to the WHOLE group's result when any input value is `NA`, matching dplyr exactly (`src/ops_groupby.cpp`'s `gpu_summarise()`, using the same MAX(is_null) + `copy_if_else()` technique the window-aggregate path already used, W4).
  * `na.rm = TRUE` is now also a supported, parsed argument (e.g. `sum(x, na.rm = TRUE)`, including on IR sub-expressions like `sum(sqrt(x), na.rm = TRUE)`), opting back into cudf's native NA-excluding behavior explicitly.
  * `n()` is unaffected (it counts rows regardless of nulls, matching dplyr already).
  * `count()`/`tally()`'s own `wt=` weighted sum (routed through `summarise()`) now passes `na.rm = TRUE` explicitly, since it can no longer rely on `summarise()`'s own (buggy) default to exclude `NA` weights.
  * Known follow-up gap, not fixed here: `na.rm = TRUE` on a group that is ENTIRELY `NA` returns `NA` for `sum()`/`mean()`/`min()`/`max()`, where base R's own empty-vector reductions give `0`/`NaN`/`Inf`/`-Inf` respectively (cudf's null-excluding aggregation returns `NULL` for a wholly-excluded group) — tracked in `scratchpad/workflow_state.md`'s "Parked / discovered" section, with a pinned regression test (`test-dplyr-summarise.R`) marking the current (divergent) behavior for a future fix to flip.
* **`arrange(desc(x))` now places `NA` LAST**, matching dplyr exactly (previously placed `NA` FIRST for any descending sort key — a pre-existing divergence found during Phase 5's W7 grouped-slice work, `src/ops_arrange.cpp`'s `gpu_arrange()` flips `null_order` with sort direction now, the same technique already used by the window rank kernel (W7) and `ops_slice.cpp`'s `gpu_slice_rank()`). Applies to every column type (numeric, string, ...) and every position in a multi-key sort, including mixed ascending/descending keys and `.by_group = TRUE`. `test-arrange.R`'s old tests, which had pinned the previous (wrong) behavior as intended, are corrected.

## Window functions, grouped mutate/filter/slice, and `.by=` (Phase 5)

`mutate()`, `filter()`, `slice()`/`slice_head()`/`slice_tail()`/`slice_min()`/`slice_max()`/`slice_sample()`, and `summarise()` all gained GPU-native support for window functions, per-group computation, and on-the-fly (`.by=`) grouping — all lowered through a new expression-decomposition layer (`R/window.R`) and a single new C++ kernel (`src/ops_window.cpp`, `gpu_window()`) shared by every window-bearing call, grouped or not.

* **Window functions in `mutate()`/`filter()`**, arbitrarily nested with ordinary arithmetic/comparisons and with each other (`mutate(y = x - lag(x))`, `mutate(z = lag(cumsum(x)))`, `filter(x > mean(x))`):
  * `row_number()` / `row_number(x)`, `min_rank()`/`dense_rank()`/`percent_rank()`/`cume_dist()`, `ntile()`.
  * `lag()`/`lead()` (`n=`, `default=`, `order_by=`).
  * `cumsum()`/`cummax()`/`cummin()`/`cumprod()`/`cummean()`/`cumall()`/`cumany()`, matching R's own `NA`-propagation (`cumsum(c(1, 2, NA, 4))` is `1, 3, NA, NA`).
  * `consecutive_id()` (single-column).
  * `mean()`/`sum()`/`min()`/`max()`/`n()`/`sd()`/`var()` broadcast to every row of their group (or the whole table, ungrouped) — matching R's own `na.rm = FALSE` default (any `NA` in a group makes the *whole* group's result `NA`); `sum(x, na.rm = TRUE)` is also understood, excluding `NA` per group.
  * `first()`/`last()`/`nth()`, `order_by()`/`with_order()`.
  * Every one of these is evaluated per `group_by()` group when `.data` is grouped, with `mutate()`'s row order always preserved; a grouped `filter()` predicate is likewise evaluated against every row of the ORIGINAL (pre-filter) group before any row is dropped, and the result preserves original row order (never reordered into group-key order).
* **`.by=`** on `mutate()`/`filter()`/`summarise()`: on-the-fly grouping without `group_by()`, e.g. `mutate(df, y = mean(x), .by = g)`. Behaves like `group_by(g) |> mutate(...) |> ungroup()` for `mutate()`/`filter()`; `summarise(.by=)` has one genuine difference from `group_by()`-driven `summarise()` — it emits one row per distinct `.by` combination in FIRST-APPEARANCE order (the order each combination first appears in the data), not sorted/hash-groupby order. The result is always ungrouped; supplying `.by=` on an already-`group_by()`-grouped table is an error, matching dplyr's own message exactly.
* **Grouped `slice()` family** (`slice()`, `slice_head()`/`slice_tail()`, `slice_min()`/`slice_max()` with `with_ties=`/`na_rm=`), both via `group_by()` and `.by=`/`by=`. Grouped output reorders into group order (ascending group-key for `group_by()`, first-appearance order for `by=`/`.by=`); `slice()` itself is native only for a strictly increasing, duplicate-free positive index vector or an all-negative one — anything requiring row duplication or reordering (e.g. `slice(c(3, 1))`) still falls back to CPU evaluation.
* **`slice_sample()`** (`n=`/`prop=`, `replace=`, ungrouped/grouped/`by=`) is now GPU-native with exact `set.seed()` RNG parity: index generation runs on the CPU (consuming R's RNG stream call-for-call the same way `dplyr::slice_sample()` does internally), the actual row gather runs on the GPU. `weight_by=` still falls back to CPU evaluation.
* Two real bugs found and fixed along the way:
  * A rank-family window frame with a *descending* order and `NA`s in the ranked column silently mis-ranked every non-null row (offset by the group's own null count) — `src/ops_window.cpp` now flips null placement with sort direction, matching real dplyr's "`NA`s sort last regardless of direction" rule.
  * `cuplyr`'s own `arrange(desc(x))` was already found to place `NA`s FIRST where real dplyr places them LAST regardless of direction — a pre-existing, unrelated divergence surfaced by this work; fixed in Phase 6, task 6.1 (see this file's own "NA-semantics parity fixes" section, above).
* The narrow `mutate(x, n = n())`/`mutate(x, s = sum(<expr>))` broadcast that only existed to support `add_count()`/`add_tally()` has been removed now that the general window/aggregate machinery covers the same shapes (and richer ones, e.g. `n() + 1`) natively, with no join and no CPU round-trip.

## Transparent CPU fallback layer (Phase 2)

Every dplyr generic that doesn't yet have a GPU-native `tbl_gpu` method now works anyway, via a transparent CPU fallback: `R/fallback.R`'s `gpu_fallback()` materializes any pending lazy operations, `collect()`s to a tibble (restoring `group_by()` structure from the table's own grouping so grouped semantics apply exactly as on a real `grouped_df`), runs the real dplyr verb on the CPU, and re-uploads the result via `tbl_gpu()` — restoring grouping (read back from the verb's actual result, since some verbs change it), execution mode (a lazy input stays lazy), and factor levels.

* Newly working on `tbl_gpu`: `distinct()`, `slice()`/`slice_head()`/`slice_tail()`/`slice_min()`/`slice_max()`/`slice_sample()`, `rename()`/`rename_with()`, `relocate()`, `pull()`, `count()`/`tally()`/`add_count()`, `transmute()`, `reframe()`, `glimpse()`, `rowwise()`, `semi_join()`/`anti_join()`/`nest_join()`/`cross_join()`, `union()`/`union_all()`/`intersect()`/`setdiff()`/`setequal()`/`symdiff()`, `group_data()`/`group_keys()`/`group_indices()`/`group_size()`/`n_groups()`/`group_split()`/`group_map()`/`group_modify()`/`group_nest()`/`group_trim()`/`nest_by()`, `rows_insert()`/`rows_append()`/`rows_update()`/`rows_patch()`/`rows_upsert()`/`rows_delete()`, and the superseded `sample_n()`/`sample_frac()`. These are wired one phase at a time to GPU-native implementations going forward (see `scratchpad/todo.md`).
* Verbs whose result isn't a data frame at all (`pull()`'s vector, `group_split()`'s list), whose result is a data frame containing list-columns with no GPU representation (`nest_by()`, `group_nest()`, `nest_join()`), or that only report metadata about the grouping/data (`group_data()`, `group_keys()`, `group_indices()`, `group_size()`, `n_groups()`) are returned as plain R objects rather than re-uploaded.
* `rows_*(in_place = TRUE)` is rejected with a clear error for `tbl_gpu`, since there is no mutable in-place table to update — assign the returned value instead.
* Every fallback calls the existing `cuplyr_fallback_notify()` hook, so `options(cuplyr.fallback = "warn")`/`"error"` (introduced for `filter()`/`summarise()`'s CPU-eval paths) now also gates these verbs.
* Known gaps, not fixed by this layer because they aren't reached via S3 dispatch: `add_tally()`, `top_n()`, and `top_frac()` are plain (non-generic) functions in dplyr 1.2 that call `mutate()`/`filter()` internally with constructs (`n()` inside `mutate()`, `top_n_rank()` inside `filter()`) the GPU-native/CPU-eval paths for those verbs don't yet understand; `group_rows()` and `group_walk()` are likewise non-generic but work transparently since they only call `group_data()`/`group_map()`, which are now implemented.

## Unified eager/lazy execution

* Every dplyr verb now builds a single AST node and routes through one shared execution path (`push_op()`/`push_join()` in `R/execute.R`, `lower_and_execute()` in `R/lower.R`) regardless of whether the table is eager or lazy. This collapsed a number of eager/lazy divergences that previously produced different results depending on execution mode; see the fixes below.

## Expression engine

`filter()`, `mutate()`, and the expressions inside `summarise()`'s aggregation calls now all parse through one shared rlang-based expression IR (`R/ir.R`) lowered to `cudf::compute_column()` (`src/expr_eval.hpp`), replacing three separate special-cased parsers. This is a large capability jump for all three verbs:

* `filter()` now understands `&`, `|`, `!`, `xor()`, arbitrarily nested and combined with comparisons; `%in%` (including a set containing `NA` and an empty set); `is.na()`; `between()`; string-column comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) against a literal or another string column; and math on columns inside a predicate (`+ - * / %% %/% ^`, `sqrt`, `log`, `abs`, etc.) — the same surface `mutate()` supports. `&&`/`||` are now rejected with a message pointing at `&`/`|`, matching dplyr 1.1+.

* `mutate()` now understands arbitrarily nested arithmetic (`(x + y) * z - 1`), scalar-on-either-side expressions (`10 - x`), `%%`/`%/%`, math functions (`sqrt`, `log`/`log(x, base)`, `log2`, `log10`, `exp`, `abs`, `floor`, `ceiling`, `sin`, `cos`, `tan`, `round()`), comparisons and logical operators, `is.na()`/`between()`/`%in%`, `NA` literals (typed and bare), and later dots referencing earlier dots' output columns.

* `summarise()`'s aggregation calls (e.g. `sum(...)`, `mean(...)`) now accept any sub-expression the IR understands, not just a single comparison or arithmetic operator — e.g. `sum(carb > 3 & wt < 4)`, `mean(sqrt(hp))`, `sum(carb %% 2 == 0)` all now work, in both eager and lazy modes, grouped and ungrouped.

* Each of `filter()`/`mutate()`'s multi-expression calls now lowers to exactly one fused GPU kernel (one `compute_column()` call), regardless of how many predicates/expressions or how deeply nested — previously each operator in a chain was a separate kernel launch.

## Bug fixes

* Fixed a silent data-corruption bug in the lazy-mode projection-pushdown optimizer: `push_down_projections()` could drop a *replaced* (not newly-created) column from the upstream projection when a lazy `mutate()` overwrote an existing column it didn't itself read as an input. This broke `gpu_mutate_expr()`'s replace-in-place name matching, silently appending the new value at the end instead — rotating column values across the table with no error and no warning. Found via the T6 `mutate()` dplyr-oracle test suite; only genuinely new output columns can be pruned from the upstream requirement now.

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
