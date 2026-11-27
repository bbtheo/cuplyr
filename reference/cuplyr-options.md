# cuplyr options

Every cuplyr option is set with
[`options()`](https://rdrr.io/r/base/options.html) (scope one with
[`withr::local_options()`](https://withr.r-lib.org/reference/with_options.html))
and, for most, also with an environment variable. The option wins over
the environment variable, which wins over the default.
`cuplyr_options()` lists them all with the value in use and where it
came from.

## Usage

``` r
cuplyr_options(internal = FALSE)
```

## Arguments

- internal:

  Also list the internal switches (`cuplyr.roll_path`,
  `cuplyr.topk_path`, `cuplyr.check_schema_cache`).

## Value

`cuplyr_options()` returns a data frame of class `cuplyr_options` with
one row per option and the character columns `option`, `value` (the
value cuplyr uses; `NA` for an unset one), `source` (`"option"`, `"env"`
or `"default"`), `env_var` (`NA` when there is none) and `description`.
An invalid value is reported as the default, after its warning. Printing
it shows the same table.

## Options

- `cuplyr.exec_mode` (environment variable `CUPLYR_EXEC_MODE`): `"lazy"`
  (the default) or `"eager"`. Whether
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
  makes lazy tables or runs each verb at once; `tbl_gpu(lazy = )`
  overrides it. See
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- `cuplyr.fallback`: `"silent"` (the default), `"warn"` or `"error"`
  (`"warning"` means `"warn"`). What a CPU fallback does. See
  [cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md).

- `cuplyr.plan_break`: `"silent"` (the default) or `"warn"`. What a plan
  break does. See
  [cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md).

- `cuplyr.memory_resource` (`CUPLYR_MEMORY_RESOURCE`): `"async"` (the
  default), `"pool"`, or `"cuda"` (`"none"`; no memory resource). See
  [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md).

- `cuplyr.memory_pool` (`CUPLYR_MEMORY_POOL`): `TRUE` (the default) or
  `FALSE`; `0`, `"no"` and `"off"` also switch the resource off, and
  `1`, `"yes"` and `"on"` on.

- `cuplyr.pool_initial` (`CUPLYR_POOL_INITIAL`): the initial pool size
  in bytes, 0 or more. Unset: a quarter of the free GPU memory, at most
  1 GiB.

- `cuplyr.release_threshold` (`CUPLYR_RELEASE_THRESHOLD`): bytes the
  async resource keeps before returning memory to the driver, 0 or more.
  Unset (the default): no automatic release.

- `cuplyr.parquet_pushdown`: `TRUE` (the default) or `FALSE`. `FALSE`
  reads Parquet files without projection and predicate pushdown. See
  [`read_parquet_gpu()`](https://bbtheo.github.io/cuplyr/reference/read_parquet_gpu.md).

- `cuplyr.roll_median` (`CUPLYR_ROLL_MEDIAN`): `"exact"` (the default)
  or `"frollmedian"`, the engine of
  [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md).

The memory options are read once, when cuplyr is loaded: changing them
later has no effect until the next R session.

## Internal switches

Listed by `cuplyr_options(internal = TRUE)`; not for normal use.

- `cuplyr.roll_path` (`CUPLYR_ROLL_PATH`): `"auto"` (the default),
  `"scans"` or `"cudf"`; forces the path of rolling-window functions.

- `cuplyr.topk_path`: `"auto"` (the default), `"topk"` or `"sort"`;
  forces the route of
  [`slice_min()`](https://dplyr.tidyverse.org/reference/slice.html) and
  [`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html) (a
  top-k selection or a full sort).

- `cuplyr.check_schema_cache` (`CUPLYR_CHECK_SCHEMA_CACHE`): `TRUE` or
  `FALSE` (the default); checks the cached schema of every lazy table on
  use (the environment variable is `1`, `true` or `yes`).

## Invalid values

A value that is not allowed (from the option or the environment
variable) warns once per session and value, with class
`cuplyr_warning_invalid_option` (fields `option`, `value` and `source`).
The message names the allowed values, and the default is used. A set
option decides even when invalid: the environment variable is then not
consulted. An invalid value never makes `cuplyr_options()` or a verb
fail; a value that cannot even be inspected (a function, say) is
reported as invalid. When warnings are errors (`options(warn = 2)`), the
warning is raised as an error from the call that read the value, naming
the option or environment variable, and every later read raises it again
until the value is fixed, unless something in between caught the warning
and raised it again (a dplyr verb run on the CPU does). A warning counts
as given only when its signal returned (one caught by
`tryCatch(warning = )` is given again on the next read).

A blank option is not invalid. `""` means the default, without a warning
and without consulting the environment variable, for
`cuplyr.memory_resource`, `cuplyr.memory_pool`, `cuplyr.pool_initial`,
`cuplyr.roll_median` and `cuplyr.roll_path` (`NA` too for
`cuplyr.pool_initial`). For `cuplyr.exec_mode` and
`cuplyr.release_threshold`, `""` and `NA` count as unset, so the
environment variable decides, as `NA` does for `cuplyr.memory_resource`
and `cuplyr.memory_pool`. An environment variable that is empty or only
spaces counts as unset for every option: the default is used, without a
warning.

`cuplyr.memory_resource` reads the first element of a vector, as it
always did: `c("pool", "async")` means `"pool"`.

## Examples

``` r
cuplyr_options()
#> <cuplyr options>
#> cuplyr.exec_mode = lazy  (default, CUPLYR_EXEC_MODE)
#>     Execution mode of new tables: lazy plans (default) or eager, one verb at a time.
#> cuplyr.fallback = silent  (default)
#>     What a CPU fallback does: stay silent, warn, or error.
#> cuplyr.plan_break = silent  (default)
#>     What a plan break does: stay silent or warn.
#> cuplyr.memory_resource = async  (default, CUPLYR_MEMORY_RESOURCE)
#>     GPU memory resource: async (default), pool, or cuda (none). Read at load.
#> cuplyr.memory_pool = TRUE  (default, CUPLYR_MEMORY_POOL)
#>     Set FALSE (or 0, no, off) to use no memory resource. Read at load.
#> cuplyr.pool_initial = <unset>  (default, CUPLYR_POOL_INITIAL)
#>     Initial pool size in bytes (default: a quarter of free GPU memory, at most 1 GiB). Read at load.
#> cuplyr.release_threshold = <unset>  (default, CUPLYR_RELEASE_THRESHOLD)
#>     Bytes the async resource keeps before returning memory to the driver (unset: never). Read at load.
#> cuplyr.parquet_pushdown = TRUE  (default)
#>     Set FALSE to read Parquet files without projection and predicate pushdown.
#> cuplyr.roll_median = exact  (default, CUPLYR_ROLL_MEDIAN)
#>     Engine of roll_median(): exact (default) or frollmedian (data.table).

# Scope an option to a block
withr::with_options(list(cuplyr.fallback = "warn"), {
  # ... a pipeline that should say when it leaves the GPU
})
#> NULL
withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"), cuplyr_options()[1, ])
#> <cuplyr options>
#> cuplyr.exec_mode = eager  (env, CUPLYR_EXEC_MODE)
#>     Execution mode of new tables: lazy plans (default) or eager, one verb at a time.
```
