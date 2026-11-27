# Rolling-window aggregates

Compute a right-aligned rolling (moving) aggregate over a numeric
vector: `n`-wide windows ending at (and including) each element. The
leading `n - 1` elements have no complete window and are always `NA`.
Matches
[`data.table::frollmean()`](https://rdrr.io/pkg/data.table/man/froll.html)/`frollsum()`/etc.'s
own `align = "right"` default exactly (see `Details`).

## Usage

``` r
roll_mean(x, n, na.rm = FALSE)

roll_sum(x, n, na.rm = FALSE)

roll_min(x, n, na.rm = FALSE)

roll_max(x, n, na.rm = FALSE)

roll_sd(x, n, na.rm = FALSE)

roll_median(x, n, na.rm = FALSE)
```

## Arguments

- x:

  A numeric vector.

- n:

  A single positive whole number: the window width.

- na.rm:

  Logical scalar, default `FALSE`. See Details.

## Value

A numeric vector the same length as `x`.

## Details

`na.rm = FALSE` (the default): any window containing at least one `NA`
produces `NA` for that position ("NA poisoning"). `na.rm = TRUE`: a
window is only ever `NA` because it doesn't yet have `n` physical
elements available (the leading `n - 1` positions, always). Once a
window has `n` elements, `NA`s within it are dropped before computing
(an all-`NA` window still yields `NA`). This is what
`data.table::frollmean(x, n, na.rm = TRUE)` does: the leading incomplete
positions stay `NA` even with `na.rm = TRUE`.

Inside
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
on a `tbl_gpu`, `roll_mean()`/`roll_sum()`/
`roll_min()`/`roll_max()`/`roll_sd()` (not `roll_median()`, see below)
run on the GPU, and per group on a grouped table; on an integer or
double column they take time that does not depend on the window width.
Called on an ordinary R vector, these functions run in R and give the
same values. The result types differ in one place: `roll_min()` and
`roll_max()` of an integer, logical, `Date` or `POSIXct` column keep
that type on the GPU, while the R functions return a double, and so do
dplyr on a data frame and a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) on a
`tbl_gpu` that runs on the CPU. The GPU library has no rolling median,
so a [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) or
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) with
`roll_median()` runs in dplyr on the CPU (see "Median engines").
`data.table::frollmean(x, n)` (and its siblings `frollsum()`,
`frollmin()`, `frollmax()` and `frollsd()`) run on the GPU inside
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
too, as long as `align`, `fill`, `adaptive`, `partial` and `give.names`
keep their defaults (`"right"`, `NA`, `FALSE`, `FALSE`, `FALSE`); any
other value runs in dplyr on the CPU.

**Median engines.** By default `roll_median()` uses a built-in exact
engine that returns exactly
[`stats::median()`](https://rdrr.io/r/stats/median.html) of each window
converted to double (so integer and logical input give double, as
[`median()`](https://rdrr.io/r/stats/median.html) of the window cast to
double would): the same NA/`na.rm` handling, `NaN` treated as missing,
`Inf`, and grouped windows. It runs in O(n log n) whatever the width
(roughly 0.2 s per 1e6 rows).
`options(cuplyr.roll_median = "frollmedian")` (or the environment
variable `CUPLYR_ROLL_MEDIAN`; the option wins) opts in to
[`data.table::frollmedian()`](https://rdrr.io/pkg/data.table/man/froll.html),
which needs data.table \>= 1.18.4 (Suggests). Its result equals
`data.table::frollmedian(x, n, na.rm = na.rm)` with `NaN` treated as
`NA` first, except that every position where data.table returns `Inf` or
`-Inf` takes the exact engine's value (so a midpoint of two finite
values that overflows comes out finite, as in
[`stats::median()`](https://rdrr.io/r/stats/median.html)). Everywhere
else it can differ from
[`stats::median()`](https://rdrr.io/r/stats/median.html) by one rounding
step (1 ulp) on windows with an even number of non-missing values. It is
faster on wide windows without `NA`s, but slower than the default with
`NA`s and `na.rm = TRUE` on wide windows (about 4x slower at width 1001
with 1% `NA`) and on thousands of small groups (several times slower).
If it is requested but data.table is missing or too old, the exact
engine is used and a message says so once per session. Any other value
of the option warns once (class `cuplyr_warning_invalid_option`) and the
exact engine is used; see
[cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md).
[`data.table::frollmedian()`](https://rdrr.io/pkg/data.table/man/froll.html)
written inside
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) keeps
data.table's own result.

The GPU computes these windows in one of two ways: with scans, whose
time does not depend on the window width, or with a direct rolling
kernel, which is faster on narrow windows (and also used for
`roll_min()` and `roll_max()` of dates, date-times and logicals). The
internal option `cuplyr.roll_path` (else the environment variable
`CUPLYR_ROLL_PATH`), one of `"auto"` (the default), `"scans"` or
`"cudf"`, forces one of them. It exists for tests and benchmarks and
changes speed, not results.
