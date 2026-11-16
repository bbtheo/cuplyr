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
(an all-`NA` window still yields `NA`). This `na.rm = TRUE` behavior was
verified empirically against `data.table::frollmean(x, n, na.rm = TRUE)`
(not assumed): the leading incomplete positions stay `NA` even though
na.rm is set.

Inside
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
on a `tbl_gpu`, `roll_mean()`/`roll_sum()`/
`roll_min()`/`roll_max()`/`roll_sd()` (not `roll_median()`, see below)
run GPU-natively, in time that does not depend on the window width
(segmented scans; cudf's own rolling kernel is kept for narrow windows,
where it is faster, and for min/max of timestamps and logicals) – these
R implementations are the CPU semantics reference and the engine for
standalone use. cudf 25.12 has no MEDIAN rolling aggregation, so
`roll_median()` always runs on the CPU (see "Median engines").
`data.table::frollmean(x, n)` (and its five siblings) work as equivalent
GPU-native aliases inside
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
too, as long as `align`/`fill`/`adaptive` are left at their defaults
(`"right"`, `NA`, `FALSE`) – any other value falls back / errors,
matching this task's explicit "right-aligned only" scope.

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

The internal option `cuplyr.roll_path` (else the environment variable
`CUPLYR_ROLL_PATH`), one of `"auto"` (the default), `"scans"` or
`"cudf"`, forces one of the two GPU paths. It exists for tests and
benchmarks and changes speed, not results.
