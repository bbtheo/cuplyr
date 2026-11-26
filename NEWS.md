# cuplyr 0.2.0

cuplyr 0.2.0 brings most of dplyr to the GPU. Nearly every dplyr verb now
works on a gibble, cuplyr's GPU table. Most run on the GPU, and the rest run
in dplyr on the CPU with the same result. Gibbles are lazy by default, so an
optimizer runs the whole pipeline at `collect()`. Also new: `join_by()`
joins, window and rolling functions, `across()`, `rowwise()`, set operations,
Parquet files on the GPU, exact `integer64` columns and a conda package.
cuplyr now needs RAPIDS libcudf 26.02 or later and a GPU with compute
capability 7.5.

## Breaking changes

* Gibbles are lazy by default. `tbl_gpu()` records each verb as a step, and
  `collect()`, `compute()` or `print()` run the plan; errors in a verb's
  arguments are still raised by that verb. For eager tables, set
  `options(cuplyr.exec_mode = "eager")` or pass `lazy = FALSE`.

* cuplyr needs RAPIDS libcudf 26.02 or later and is tested on 26.10. RAPIDS
  25.12 is no longer supported (#94).

* The minimum GPU is compute capability 7.5 (Turing). Every install route
  uses a CUDA 13 build of RAPIDS, which has no code for Volta (#150).

* Results that differed from dplyr now match it:
  - `summarise()` aggregates follow R's `na.rm = FALSE` default, so a group
    with an `NA` gives `NA`.
  - `arrange(desc(x))` puts `NA` last.
  - Grouped `summarise()` and `count()` return groups sorted by their keys,
    not in hash order.
  - Integer `+`, `-`, `*`, `%/%` and `%%` return integers, not doubles. An
    overflow gives `NA` with R's warning.
  - `NaN` stays `NaN`; it used to become `NA` on upload.
  - `integer64` columns are exact and collect to `bit64::integer64`; they used
    to be uploaded as doubles, which could corrupt their values.
  - `POSIXct` columns keep their time zone instead of coming back in UTC.
  - A gibble made from a `data.frame` collects to a `data.frame`, with its
    attributes.

* Uploading a column with no GPU type (`difftime`, `hms`, `ts`, `POSIXlt`, a
  list, a matrix with several columns, ...) raises an error that names the
  column, instead of losing its class.

* `print()` uses a tibble-style `# A gibble: n x m` header, R type labels and
  factor labels.

* `show_query()` is now a method for dplyr's generic. Its argument is `x`, and
  it prints the plan as R code and returns `x` invisibly.

* `split()`, `with()`, `subset()`, `t()`, `is.na()` and `anyNA()` of a gibble
  raise an error; they used to act on the object's internal fields.

## Installation and requirements

* cuplyr is available as a conda package, `r-cuplyr`, on the `bbtheo` channel
  on anaconda.org. It is a binary for RAPIDS 26.10 and CUDA 13 that needs an
  NVIDIA driver but no CUDA toolkit or compiler. Install it with conda, mamba
  or pixi; the README has the commands (#135).

* `install.sh` and `install_cuplyr()` build cuplyr against libcudf from pixi,
  conda or the system (`--method=` picks one). The conda route installs
  RAPIDS 26.10 for CUDA 13 with the CUDA compiler.

* `./configure` stops with a clear message when libcudf is older than 26.02,
  and prefers the CUDA toolkit of a conda or pixi environment (#38, #94).

* New `check_deps()` checks the driver, CUDA toolkit, libcudf and R packages,
  `verify_installation()` runs a small GPU round trip, and `diagnostics()`
  reports versions, the GPU, the memory resource and every option for a bug
  report, as text or JSON (#14, #49). `diagnose_loader()` helps when
  `library(cuplyr)` fails to find its libraries.

## New features

### Lazy plans

* The optimizer drops columns the result never reads, fuses consecutive
  `mutate()` calls and moves filters below mutates, selects and some joins.

* `explain()` and `show_query()` print the plan as R code without running it,
  including the columns a Parquet scan reads and the reason for each CPU
  fallback (#10). Printing a lazy gibble runs the plan and keeps nothing.

* A step whose result type depends on the data (an integer `sum()`, a join
  with `unmatched = "error"`) runs at the verb, as in eager mode;
  `options(cuplyr.plan_break = "warn")` reports it.

* Anything with no GPU form runs in dplyr on the CPU instead of failing.
  `options(cuplyr.fallback = "warn")` reports each fallback with its reason
  and the data moved; `"error"` forbids them.

### Verbs

* Newly on the GPU: `filter_out()`, `transmute()`, `rename()`,
  `rename_with()`, `relocate()`, `distinct()`, `pull()`, `count()`,
  `tally()`, `add_count()`, `add_tally()`, `reframe()`, the `slice()`
  family, `top_n()`, `sample_n()`, `group_split()`, `group_trim()`,
  `group_data()` and the other grouping functions, `rowwise()`, the
  `rows_*()` verbs and the `_at()`, `_all()` and `_if()` verbs.

* `group_map()`, `group_modify()`, `group_nest()`, `nest_by()`, `nest_join()`
  and `do()` run on the CPU.

* `.by` works in `mutate()`, `filter()`, `summarise()`, `reframe()` and the
  `slice()` family, and grouped verbs run per group on the GPU. `group_by()`
  takes computed keys and `.drop = FALSE`.

* `rowwise()` runs elementwise expressions and row aggregates such as
  `sum(c_across(a:e))` on the GPU. On 1e5 rows that takes milliseconds
  instead of about twenty seconds.

* `$`, `[[`, `[`, `[<-` for whole columns, `as.data.frame()`, `head()`,
  `rbind()`, `cbind()`, vctrs functions, `dplyr::bind_rows()` and
  `purrr::map_dfr()` work on gibbles.

* `gibble()`, `as_gibble()` and `is_gibble()` are aliases of `tbl_gpu()`,
  `as_tbl_gpu()` and `is_tbl_gpu()`.

### Expressions

* `filter()`, `mutate()` and `summarise()` share one expression engine that
  runs on the GPU:
  - arithmetic, comparisons, logic, `%in%`, `is.na()`, `is.nan()`,
    `between()`, `near()` and math functions, `round()` with R's rule;
  - `if_else()`, `case_when()`, `case_match()`, `coalesce()`, `na_if()`,
    `recode()` and dplyr 1.2's `recode_values()`, `replace_values()`,
    `replace_when()`, `when_any()` and `when_all()`;
  - window functions: `lag()`, `lead()`, the ranking and cumulative
    functions, `consecutive_id()`, `first()`, `last()`, `nth()`,
    `order_by()` and grouped aggregates such as `x - mean(x)`;
  - summaries: `median()`, `quantile()`, `n_distinct()`, `any()`, `all()`
    and the rest, of any expression and with arithmetic around them (#22);
  - base string functions: `paste()`, `substr()`, `nchar()`, `toupper()`,
    `trimws()`, `grepl()`, `sub()`, `gsub()`, `startsWith()`, ...;
  - factor comparisons with labels and `as.character()`;
  - `Date` and `POSIXct` literals, casts and `date + n`, lubridate's date
    parts (`year()`, `hour()`, `wday()`, ...) and fixed-unit `floor_date()`,
    `ceiling_date()` and `round_date()`.

* `across()`, `if_any()`, `if_all()`, `cur_group_id()` and `cur_column()` run
  on the GPU; `pick()` as a data frame and `cur_group()` run on the CPU.

* A function of your own named like one cuplyr runs on the GPU (`mean()`,
  `lag()`, ...) is the one called, as in dplyr.

### Joins and set operations

* The mutating joins, `semi_join()`, `anti_join()` and `cross_join()` run on
  the GPU, with `join_by()` equality, inequality, `between()`, `within()`,
  `overlaps()` and `closest()` conditions. `multiple`, `unmatched`,
  `relationship`, `na_matches`, `keep` and `suffix` follow dplyr.

* Keys of different types combine by vctrs' rules, an all-`NA` key included
  (#70).

* `union()`, `union_all()`, `intersect()`, `setdiff()`, `symdiff()` and
  `setequal()` run on the GPU. `bind_rows()` follows vctrs' type rules and
  binds factors to the union of their levels (#20, #70).

### Parquet

* New `read_parquet_gpu()` and `write_parquet_gpu()` read and write Parquet
  files on the GPU; only compressed bytes cross the PCIe bus.

* A lazy read reads only the columns the plan uses and skips row groups for
  simple filters on integer, `integer64`, `Date` and `POSIXct` columns.
  `options(cuplyr.parquet_pushdown = FALSE)` turns this off; use it for
  untrusted files, as malformed statistics can crash the session.

### Types

* `integer64` comparisons, sorting, grouping, joins and binds run exactly on
  the GPU; arithmetic and aggregates follow bit64 on the CPU.

* `NaN` counts as missing in `is.na()`, comparisons, `arrange()` and the
  `na.rm = TRUE` reducers, as in R.

* An infinite or out-of-range `Date` or `POSIXct` value warns on upload and
  becomes `NA`; it used to give `NA` or a wrong date silently.

* Columns keep their element names, attributes and one-column matrix shape
  by dplyr's rules (#21, #56).

* Tables with rows but no columns work like dplyr's (#19).

* Strings go to the GPU as UTF-8, and columns with more than 2 GiB of
  characters work.

### Rolling windows

* New `roll_mean()`, `roll_sum()`, `roll_min()`, `roll_max()` and `roll_sd()`
  run on the GPU in `mutate()` and `filter()`, grouped or not, in time that
  doesn't depend on the window width. `data.table::frollmean()` and its
  siblings work as aliases.

* New `roll_median()` matches `stats::median()` exactly and runs on the CPU.

### Options, errors and memory

* New `cuplyr_options()` lists every option with its value and where it came
  from; `?cuplyr-options` documents them. An invalid value warns once, with
  class `cuplyr_warning_invalid_option`, and the default is used.

* Errors and warnings have dplyr's class, message and call: missing columns,
  errors in `mutate()`, calls dplyr rejects, and the `min()` and `max()`
  warnings for empty groups (#13, #20, #24, #28, #30, #69).

* cuplyr installs CUDA's stream-ordered (`"async"`) memory resource when it
  loads, and each call frees its temporary GPU tables when it returns.
  `gpu_gc()` gives unused memory back to the device. Set
  `options(cuplyr.memory_resource = "pool")` before loading for RMM's pool, or
  `cuplyr.release_threshold` to release memory automatically
  (`vignette("memory")`).

* A step that runs out of GPU memory is retried once after garbage
  collection, and the final error says what helps (#13). New `gpu_sync()`
  waits for queued GPU work, for timing.

* New `vignette("performance")` and `vignette("memory")`, and
  `vignette("query-optimization")` matches the optimizer. Help-page examples
  run (#39).

## Performance

* Eager verb calls spend less time in R. Recording a lazy verb costs the same
  however long the plan is, and a column passed through a verb is shared
  instead of copied.

* Each `filter()` or `mutate()` call runs as one fused GPU kernel where it can,
  and computes a repeated subexpression once (#17).

* `arrange() |> head()` and `slice_min()`/`slice_max()`, grouped too, use a
  top-k selection instead of a full sort.

* `summarise(.by = )` runs as one grouped summary, and grouped `summarise()`
  needs less memory with many groups (#50).

* Faster `recode()` with many keys, `group_split()` with many groups and the
  `_all()` verbs on wide tables.

* Faster `round(x, digits)`, `quantile()` with many probabilities, very long
  `mutate()` calls and `summarise()` with `.drop = FALSE` (#75).

## Minor improvements and bug fixes

* `right_join()` works (it failed on every call) and returns dplyr's row
  order; `right_join()` and `full_join()` fill the keys of unmatched rows.

* Eager `select()` keeps factor columns as factors.

* `summarise()` accepts `dplyr::n()` and `stats::sd()`, returns `n()` as an
  integer, names an unnamed expression after its text, and works lazily with
  `sd()` and `var()`.

* `arrange(.by_group = TRUE)` sorts the group columns ascending, as dplyr
  does.

* `first()`, `last()` and `nth()` of a logical expression return a logical.

* `distinct()` and `count()` accept window functions in computed keys (#29).

* An unnamed `.data$x` in `distinct()`, `count()` or `group_by()` names the
  column `x`.

* `.env[[name]]` reads the variable `name` holds, and a column-free filter
  condition is evaluated once.

* `%in%` compares in the common type, as base R does (`1L %in% 1.5` is
  `FALSE`), and `sum(n())` and other windows over one value per group give
  dplyr's result.

* `semi_join()` and `anti_join()` keep the empty groups of a `.drop = FALSE`
  table.

* A `difftime` made on the CPU raises an error naming the column; native
  `difftime` columns are planned (#65).

* Lazy plans with chained joins, or a `mutate()` that overwrites a column,
  give the right result.

* `check_deps()` passes on a working pixi install and reads newer
  `nvidia-smi` output (#49).

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
