# Create an aggregation structure for summarise

Create an aggregation structure for summarise

## Usage

``` r
make_aggregation(
  output_col,
  input_col,
  fn,
  input_type = NULL,
  na_rm = FALSE,
  extra = NA_real_,
  identity = TRUE
)
```

## Arguments

- output_col:

  Output column name

- input_col:

  Input column name

- fn:

  Aggregation function name (sum, mean, min, max, n, sd, var, median,
  quantile, n_distinct, first, last, nth, any, all)

- input_type:

  Input column type

- na_rm:

  Logical scalar. `FALSE` (the default, matching R's own `na.rm = FALSE`
  default for
  [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html))
  means a group with ANY null input value must produce a NULL
  aggregation result for the WHOLE group – cudf's own groupby
  MEAN/SUM/MIN/MAX/STD/ VARIANCE/MEDIAN aggregations always exclude
  nulls, so `gpu_summarise()` (`src/ops_groupby.cpp`) applies an extra
  whole-group-null propagation step when this is `FALSE` and the input
  column actually has nulls (same technique as the window-aggregate
  path, `src/ops_window.cpp`). `TRUE` means "use cudf's native
  null-excluding behavior as-is" – no extra step. Meaningless for
  [`n()`](https://dplyr.tidyverse.org/reference/context.html) (counts
  rows regardless of nulls, matching dplyr already). For
  [`quantile()`](https://rdrr.io/r/stats/quantile.html), `na_rm = FALSE`
  with an actual null present is a hard ERROR (matching base R's own
  `quantile.default()`), not a NA propagation – see `gpu_summarise()`.
  For
  [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html),
  `na_rm` selects cudf's `null_policy` directly (`INCLUDE`/`EXCLUDE`)
  rather than a post-hoc propagation step – NA is its own distinct value
  when `na_rm = FALSE`, matching dplyr's
  [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
  exactly. For
  [`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html),
  `na_rm` also selects `null_policy` directly: `FALSE` (default) never
  skips a NA when picking the position (may return NA), `TRUE` skips NA
  values when picking. For
  [`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html),
  `na_rm` implements R's own three-valued logic (`any(c(TRUE, NA))` is
  `TRUE`, `any(c(FALSE, NA))` is `NA` when `na_rm = FALSE`, `FALSE` when
  `na_rm = TRUE`).

- extra:

  Numeric scalar, `NA_real_` unless `fn` needs an extra scalar parameter
  beyond the input column:
  [`quantile()`](https://rdrr.io/r/stats/quantile.html)'s probability
  (`[0, 1]`) or
  [`nth()`](https://dplyr.tidyverse.org/reference/nth.html)'s (possibly
  negative) element position.

- identity:

  Logical scalar. `TRUE` (the default) lets a
  [`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
  over an INT32/FLOAT64 column apply base R's empty-vector identity
  (`Inf`/`-Inf`, with dplyr's warning) for an empty group, which
  promotes an INT32 result column to FLOAT64 when it happens; see
  `gpu_summarise()` (`src/ops_groupby.cpp`). A STRING or Date/POSIXct
  one keeps its null value but is counted the same way, for dplyr's
  warning. `FALSE` would keep the result INT32 and never warn, but
  nothing calls it with `FALSE` today. Only meaningful for `min`/`max`
  over INT32/FLOAT64/STRING/timestamps; stored as `FALSE` for everything
  else.

## Value

An aggregation list structure
