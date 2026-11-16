# Summarise each group to any number of rows

[`dplyr::reframe()`](https://dplyr.tidyverse.org/reference/reframe.html)
for a `tbl_gpu`. The result is always ungrouped:
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) keys
come out sorted, `.by` keys in first-appearance order, and size-1
results are recycled to each group's size.

## Usage

``` r
# S3 method for class 'tbl_gpu'
reframe(.data, ..., .by = NULL)
```

## Arguments

- .data:

  A `tbl_gpu`.

- ...:

  Name-value pairs, as in
  [`dplyr::reframe()`](https://dplyr.tidyverse.org/reference/reframe.html).

- .by:

  Optional per-operation grouping, as in
  [`dplyr::reframe()`](https://dplyr.tidyverse.org/reference/reframe.html).

## Value

An ungrouped `tbl_gpu` (lazy if `.data` is lazy).

## Native shapes

The call runs on the GPU when every expression is named and is one of:

- `quantile(x, probs)` (`stats::`) with `probs` a column-free constant
  in \[0, 1\], a literal `na.rm`, `type` absent or 7;

- `range(x)` (`base::`) with a literal `na.rm`;

- [`n()`](https://dplyr.tidyverse.org/reference/context.html), or
  [`sum()`](https://rdrr.io/r/base/sum.html),
  [`mean()`](https://rdrr.io/r/base/mean.html),
  [`min()`](https://rdrr.io/r/base/Extremes.html),
  [`max()`](https://rdrr.io/r/base/Extremes.html),
  [`sd()`](https://rdrr.io/r/stats/sd.html),
  [`var()`](https://rdrr.io/r/stats/cor.html),
  [`median()`](https://rdrr.io/r/stats/median.html) of one argument with
  a literal `na.rm`;

- a column-free constant vector (`c(0.25, 0.75)`, `"lo"`), recycled.

The argument `x` must be an input column, or arithmetic
(`+ - * / %% %/%`, [`abs()`](https://rdrr.io/r/base/MathFun.html),
[`exp()`](https://rdrr.io/r/base/Log.html); `%%` and `%/%` take an
integer divisor or a whole-number double literal), comparisons, `& | !`
and [`is.na()`](https://rdrr.io/r/base/NA.html) over integer, double and
logical columns and literals, with no window or aggregate call inside.
Comparisons, `& | !` and [`is.na()`](https://rdrr.io/r/base/NA.html)
take only integer and logical columns, integer arithmetic on them, and
literals. [`quantile()`](https://rdrr.io/r/stats/quantile.html) takes
integers and doubles, [`median()`](https://rdrr.io/r/stats/median.html)
only doubles; [`min()`](https://rdrr.io/r/base/Extremes.html),
[`max()`](https://rdrr.io/r/base/Extremes.html) and
[`range()`](https://rdrr.io/r/base/range.html) also take a bare Date or
POSIXct column, and a bare string column under a C collation. Sizes
above 1 must agree, and no expression may read an earlier one's result.

Anything else runs dplyr's
[`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) on the
CPU (see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
One divergence remains on both paths:
[`quantile()`](https://rdrr.io/r/stats/quantile.html)'s element names
(`"10%"`) are dropped.
