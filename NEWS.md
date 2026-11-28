# cuplyr (development version)

* `mean(x, na.rm = TRUE)` and `min()`/`max(x, na.rm = TRUE)` of a double in
  `mutate()` and `filter()`, `row_number(x)` and `ntile(x, n)` inside
  `order_by()`/`with_order()`, integer64 `%in%`, `first()`/`last()`/`nth()`,
  `n_distinct()`, `percent_rank()`, `cume_dist()` and `lag()`/`lead()` with a
  non-`NA` `default`, and a type-based `across(where(...))` after `n()`,
  `row_number()`, `ntile()`, `year()` or `roll_*()` in the same `mutate()`
  now run on the GPU instead of the CPU. `min()`/`max(x, na.rm = TRUE)` of an
  all-`NA` group gives `Inf`/`-Inf` with dplyr's warning, as in dplyr (#74).

* lubridate's `year()`, `month()`, `yday()` and `wday()` on a `tbl_gpu` now
  return doubles, as lubridate does, instead of integers (#74).

# cuplyr 0.2.0

cuplyr 0.2.0 brings most of dplyr to the GPU. Nearly every dplyr verb now
works on a `tbl_gpu`, cuplyr's GPU table. Most run on the GPU, and the rest
run in dplyr on the CPU with the same result. GPU tables are lazy by default,
so an optimizer runs the whole pipeline at `collect()`. Also new: `join_by()`
joins, window and rolling functions, `across()`, `rowwise()`, set operations,
Parquet files on the GPU, exact `integer64` columns and a conda package.
cuplyr now needs RAPIDS libcudf 26.02 or later and a GPU with compute
capability 7.5 or higher.

## Breaking changes

* GPU tables are lazy by default. `tbl_gpu()` returns a lazy table, each verb
  records a step, and `collect()`, `compute()` or `print()` run the plan.
  Errors in a verb's arguments are still raised by that verb, but dplyr's
  warnings, such as integer overflow or a many-to-many join, come when the
  plan runs, at every `print()` too. For eager tables, set
  `options(cuplyr.exec_mode = "eager")` or pass `lazy = FALSE`.

* On a lazy table with pending steps, `nrow()` and `dim()` run the plan,
  while `gpu_object_info()` and `gpu_memory_usage()` count the rows of the
  table it was made from and `verify_gpu_data()` can return `FALSE`. Call
  `compute()` first.

* cuplyr needs RAPIDS libcudf 26.02 or later and is tested on 26.10. RAPIDS
  25.12 is no longer supported (#94).

* The minimum GPU is compute capability 7.5 (Turing). The conda package and
  the pixi and conda install routes use a CUDA 13 build of RAPIDS, which has
  no code for Volta (#150).

* Results that differed from dplyr now match it:
  - Grouping follows dplyr. `summarise()` drops only the last grouping column
    by default (`.groups = "drop_last"`, with a message like dplyr's),
    `select()` keeps the grouping columns, joins keep `x`'s groups,
    `bind_rows()` keeps the first input's groups, `collect()` returns a
    `grouped_df`, and `tbl_gpu()` keeps the groups of a `grouped_df`. In 0.1.1
    each of these dropped the groups. `ungroup(x, col)` removes only `col`;
    0.1.1 removed every group.
  - `group_by()` evaluates its arguments as dplyr does. A tidyselect helper
    needs `pick()`, as in `group_by(pick(all_of(cols)))`, and a string such as
    `group_by("x")` makes one constant group; 0.1.1 took both as column names.
    `group_by()` with no arguments removes the grouping; 0.1.1 kept it.
  - `select(new = old)` renames the column; 0.1.1 kept the old name.
  - `summarise()` aggregates follow R's `na.rm = FALSE` default, so a group
    with an `NA` gives `NA`.
  - An ungrouped `summarise()` of zero rows, such as after a `filter()` that
    keeps none, returns one row, where `sum()` gives 0; 0.1.1 returned no
    rows.
  - `arrange(desc(x))` puts `NA` last, and `arrange(.by_group = TRUE)` sorts
    the group columns ascending, as dplyr does.
  - Grouped `summarise()` returns groups sorted by their keys, not in hash
    order, and keeps rows whose key is `NA` as a last group; 0.1.1 dropped
    those rows.
  - `full_join()` fills the keys of rows found only in `y`; 0.1.1 left them
    `NA`.
  - Integer `+`, `-` and `*` return integers, not doubles. An overflow gives
    `NA` with R's warning.
  - In `summarise()`, `sum()` of integers or logicals, such as `sum(x > 0)`,
    and an ungrouped `n()` return integers, not doubles, and `min()` and
    `max()` of a logical column return integers, not logicals. A `sum()` too
    large for an integer is a double, as in R.
  - `NaN` stays `NaN`; it used to become `NA` on upload.
  - `integer64` columns are exact and collect to `bit64::integer64`; they used
    to be uploaded as doubles, which corrupted them.
  - `POSIXct` columns keep their time zone instead of coming back in UTC.
  - A GPU table made from a `data.frame` collects to a `data.frame`, with its
    attributes.
  - Factor columns stay factors after `select()`, joins, `bind_rows()`,
    `bind_cols()`, a lazy `mutate()` and as `summarise()` group keys, and
    ordered factors stay ordered; 0.1.1 returned integer codes, or plain
    factors. Joins match factor keys by their labels; 0.1.1 matched the codes,
    which paired the wrong rows when the two tables' levels differed.

* Uploading a column with no GPU type (`difftime`, `hms`, `ts`, `POSIXlt`, a
  list, a matrix with several columns, ...) raises an error that names the
  column, instead of losing its class. So does a verb that makes a `difftime`
  on the CPU, such as `end - start` on dates. Returning such columns is an
  open issue (#65).

* `print()` uses a tibble-style `# A tbl_gpu: n x m` header, R type labels and
  factor labels.

* `show_query()` is now a method for dplyr's generic. Its argument is `x`, and
  it prints the plan as R code and returns `x` invisibly.

* `$`, `[[` and `length()` of a GPU table give its columns, as for a data
  frame, and `names<-` accepts `NA` and empty names. In 0.1.1, `$`, `[[` and
  `length()` gave the object's internal fields, such as `x$schema` and
  `x$ptr`, and `names<-` refused those names (#4). `split()`, `with()`,
  `subset()`, `t()`, `as.environment()`, `is.na()` and `anyNA()` of a GPU
  table raise an error; they used to act on the internal fields.

## Installation and requirements

* cuplyr is available as a conda package, `r-cuplyr`, on the `bbtheo` channel
  on anaconda.org. It is a binary for RAPIDS 26.10 and CUDA 13 that needs an
  NVIDIA driver for CUDA 13 but no CUDA toolkit or compiler. Install it with
  conda, mamba or pixi; the README has the commands (#135).

* `install.sh` and `install_cuplyr()` build cuplyr against libcudf from pixi,
  conda or the system (`--method=` picks one). The conda route installs
  RAPIDS 26.10 for CUDA 13 with the CUDA compiler, and the pixi environments
  carry the CUDA 13 build tools, so neither needs a system toolkit (#157).

* `./configure` stops with a clear message when libcudf is older than 26.02,
  and prefers the CUDA toolkit of a conda or pixi environment (#38, #94).

* New `check_deps()` checks the driver, CUDA toolkit, libcudf and R packages,
  `verify_installation()` runs a small GPU round trip, and `diagnostics()`
  reports versions, the GPU, the memory resource and every option for a bug
  report, as text or JSON. `diagnose_loader()` shows how cuplyr's shared
  library resolves libstdc++, libcudf and libcuda (#14, #49).

## New features

### Verbs

* Anything with no GPU form runs in dplyr on the CPU instead of failing.
  `options(cuplyr.fallback = "warn")` reports each fallback with its reason
  and the data moved; `"error"` forbids them.

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
  `sum(c_across(a:e))` on the GPU. On 1e5 rows that takes milliseconds,
  where dplyr takes about twenty seconds.

* `[`, `[<-` for whole columns, `as.data.frame()`, `as_tibble()`,
  `glimpse()`, `head()`, `rbind()`, `cbind()`, `vctrs::vec_slice()`,
  `vctrs::vec_size()`, `dplyr::bind_rows()` and `purrr::map_dfr()` work on
  GPU tables.

### Lazy plans

* The optimizer now also prunes unused columns below `summarise()`,
  `arrange()` and other barriers.

* `explain()` and `show_query()` print the plan as R code without running it,
  including the columns a Parquet scan reads and the reason for each CPU
  fallback (#10).

* Printing a lazy table runs the whole plan to show the first rows and keeps
  nothing; a plan over a Parquet file prints its schema unless it only selects
  or renames columns.

* A step whose result type or error depends on the data (an integer `sum()`,
  a join with `unmatched = "error"`) runs at the verb, as in eager mode;
  `options(cuplyr.plan_break = "warn")` reports it.

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
  `relationship`, `na_matches`, `keep` and `suffix` follow dplyr, and keys of
  different types combine by vctrs' rules, an all-`NA` key included (#70).

* `union()`, `union_all()`, `intersect()`, `setdiff()`, `symdiff()` and
  `setequal()` run on the GPU. `bind_rows()` follows vctrs' type rules and
  binds factors to the union of their levels. Binds and set operations return
  a lazy table when any input is lazy (#20, #70).

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
  by dplyr's rules, and tables with rows but no columns work (#19, #21, #56).

* Strings go to the GPU as UTF-8, and columns with more than 2 GiB of
  characters work.

### Rolling windows

* New `roll_mean()`, `roll_sum()`, `roll_min()`, `roll_max()` and `roll_sd()`
  run on the GPU in `mutate()` and `filter()`, grouped or not, in time that
  doesn't depend on the window width. `data.table::frollmean()` and its
  siblings work as aliases. New `roll_median()` matches `stats::median()`
  exactly and runs on the CPU.

### Options, errors and memory

* New `cuplyr_options()` lists every option with its value and where it came
  from; `?"cuplyr-options"` documents them. An invalid value warns once, with
  class `cuplyr_warning_invalid_option`, and the default is used.

* Errors and warnings have dplyr's class, message and call for missing
  columns, calls dplyr rejects and the `min()` and `max()` warnings for empty
  groups. Errors in `mutate()` carry dplyr's class and its `In argument:`
  context (#13, #20, #24, #28, #30, #69).

* cuplyr installs CUDA's stream-ordered (`"async"`) memory resource when it
  loads. It reserves a quarter of the free GPU memory, at most 1 GiB, and
  keeps the memory of freed tables for reuse, so other processes see it as
  used until `gpu_gc()` returns it to the device and reports it in
  `released_bytes`. Each call frees its temporary GPU tables when it returns.
  Before loading, set `options(cuplyr.memory_resource = "pool")` for RMM's
  pool, `cuplyr.pool_initial` to reserve less, or `cuplyr.release_threshold`
  to release memory automatically (`vignette("memory")`).

* `gpu_memory_state()` reports the memory resource, `pool_bytes` and
  `pool_used_bytes`. `used_bytes` counts the memory the resource holds, so it
  no longer falls when a table is freed, and the `freed_bytes` of `gpu_gc()`
  counts memory freed inside the resource.

* New `gpu_sync()` waits for queued GPU work, for timing. A step that runs out
  of GPU memory is retried once after garbage collection, and the final error
  says what helps (#13).

* Loading cuplyr sets the environment variable
  `LIBCUDF_USE_DEBUG_STREAM_POOL` unless it is set, so that Parquet reads
  don't slow later verbs (`?read_parquet_gpu`).

* New `vignette("performance")` and `vignette("memory")`, and
  `vignette("query-optimization")` matches the optimizer. Help-page examples
  run (#39).

## Performance

* Recording a `mutate()` on a lazy table no longer slows down as the plan
  grows, and a column passed through a verb is shared instead of copied.

* Each `filter()` or `mutate()` call runs as one fused GPU kernel where it can,
  and computes a repeated `round()`, `%in%` or string call once (#17).

* A lazy `arrange(x) |> head(n)` on one numeric or date key, and
  `slice_min()`/`slice_max()` (grouped too), use a top-k selection instead of
  a full sort.

## Minor improvements and bug fixes

* `right_join()` works (it failed on every call) and returns dplyr's row
  order and keys.

* `summarise()` accepts `dplyr::n()` and `stats::sd()`, names an unnamed
  expression after its text, and works lazily with `sd()` and `var()`.

* Lazy plans with chained joins, or a `mutate()` that overwrites a column,
  give the right result.

* The namespace loads with only its stated dependencies (it failed when utils
  was not attached, which `R CMD check` reports), `rowwise()`'s method takes
  `data` like the generic, and the C++ code no longer uses the APIs libcudf
  26.10 deprecates (#166).

* `?tbl_gpu` explains what a GPU table is and how to work with one: lazy by
  default, `collect()`, `compute()`, `explain()`, eager mode and the column
  types. The verbs' help pages say what runs on the GPU, what runs in dplyr
  on the CPU and where the result differs from dplyr, without internal
  development notes (#159).

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
