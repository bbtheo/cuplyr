# Summarise groups in a GPU table

Computes aggregations on groups defined by
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).
Operations are performed entirely on the GPU for maximum performance.

## Usage

``` r
# S3 method for class 'tbl_gpu'
summarise(.data, ..., .groups = NULL, .by = NULL)

# S3 method for class 'tbl_gpu'
summarize(.data, ..., .groups = NULL, .by = NULL)
```

## Arguments

- .data:

  A grouped `tbl_gpu` object created by
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).

- ...:

  Name-value pairs of summary functions. The name will be the name of
  the variable in the result. The value must be a single aggregation
  expression in the form `fun(column)`, where `column` can also be any
  expression the internal expression IR understands (see "Aggregation
  sub-expressions" below).

- .groups:

  Controls grouping structure of the result (Phase 6, task 3; matches
  [`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)'s
  own contract exactly, verified empirically against dplyr 1.2.1). One
  of:

  - `NULL` (default) – `"drop_last"` (see below), PLUS an informational
    message when the result is still grouped (see "The regroup message"
    below).

  - `"drop_last"` – drop the LAST grouping variable, keeping the rest
    (e.g. `group_by(a, b, c)` -\> result grouped by `a, b`); with only
    one grouping variable, the result is ungrouped. Same as the default,
    minus the message.

  - `"drop"` – the result is always ungrouped.

  - `"keep"` – every original grouping variable is retained.

  - `"rowwise"` – the summary runs on the GPU keeping every grouping
    variable, and the result is a rowwise `tbl_gpu` by those variables
    ([`dplyr::rowwise()`](https://dplyr.tidyverse.org/reference/rowwise.html));
    [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
    returns dplyr's own `rowwise_df`.

  On a rowwise `tbl_gpu` (see
  [`rowwise.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/rowwise.tbl_gpu.md))
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  computes one row per input row, the rowwise variables first.
  `.groups = NULL` (and `"keep"`) groups the result by those variables
  (with dplyr's message when `.groups` is not given), `"drop"` ungroups
  it, `"rowwise"` keeps it rowwise; `"drop_last"` is an error, as in
  dplyr. `.groups=` is only validated against an already-grouped `.data`
  – matching a genuine (if surprising) dplyr quirk verified empirically:
  any value other than `"rowwise"` (including a typo) is silently a
  no-op on an UNGROUPED `.data`, since there is nothing to
  drop/keep/peel from an empty group set.

  ### The regroup message

  When `.groups` is left `NULL` (the default) and the pre-summarise
  `.data` has MORE THAN ONE grouping variable, an informational message
  is emitted (loosely matching dplyr's own wording – this is
  informational, not a value under parity test). It is suppressed by
  `options(dplyr.summarise.inform = FALSE)` (an explicit `TRUE`/`FALSE`
  value for this option always wins outright); with the option unset,
  dplyr's own rule applies: the message only fires when the CALLING
  code's top environment is the global environment (i.e. top-level/
  interactive use) – it is silent when
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  is called from inside a function/package (as in a `testthat` test, or
  one verb calling another, e.g.
  [`tally.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tally.tbl_gpu.md)).

- .by:

  Optional on-the-fly grouping columns (tidyselect), GPU-native (Phase
  5, task W9). Unlike
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)'s
  `.by=` (which behave exactly like
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)),
  `summarise(.by=)` has one genuinely different rule verified
  empirically against dplyr 1.2.1: groups are emitted in
  FIRST-APPEARANCE order (the order each distinct `.by` combination
  first appears in `.data`), not sorted group-key order – see "`.by`
  group order" below. The result is always ungrouped – unlike a
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-driven
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  whose result's grouping now depends on `.groups=` (Phase 6, task 3);
  `.groups=` can't be combined with `.by=` at all (rejected by dplyr's
  own
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  generic). Supplying `.by` when `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr exactly.

## Value

A `tbl_gpu` object with one row per group containing the grouping
columns and computed aggregations.

## Details

### Supported aggregation functions

- `sum(x)` - Sum of values. Over an integer or logical `x` the result is
  an integer, as in dplyr; if it leaves R's integer range
  (`+-(2^31 - 1)`) dplyr returns a double, and so does `tbl_gpu`. Which
  one it is depends on the data, which a lazy plan (its schema is fixed
  before it runs) can't know, so a lazy
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  of an integer (INT32) `sum(x)` runs at the verb, like an eager one:
  the pending steps run first (a plan break, see `cuplyr.plan_break`)
  and the result is a lazy table with nothing pending. The same goes for
  a lazy
  [`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
  of an integer. A [`sum()`](https://rdrr.io/r/base/sum.html) of a
  logical `x` stays in the plan: it can only overflow with more than
  2^31 `TRUE` rows in a group, and then raises a
  `cuplyr_error_sum_overflow` error at
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

- `mean(x)` - Arithmetic mean

- `min(x)` - Minimum value

- `max(x)` - Maximum value

- [`n()`](https://dplyr.tidyverse.org/reference/context.html) - Count of
  rows in each group

- `sd(x)` - Standard deviation

- `var(x)` - Variance

- `median(x)` - Median value (Phase 6, task 6.2)

- `quantile(x, probs)` - A single quantile, `probs` a scalar in `[0, 1]`
  (Phase 6, task 6.2; only one probability per call is supported)

- `n_distinct(x)` - Count of distinct values, `NA` counted as its own
  distinct value by default (Phase 6, task 6.2)

- `first(x)` / `last(x)` / `nth(x, n)` - The first/last/nth value in the
  group's current row order (Phase 6, task 6.2)

- `any(x)` / `all(x)` - Logical reductions with R's own three-valued
  logic (Phase 6, task 6.2; `x` must be logical)

### Aggregation sub-expressions

`column` is not limited to a bare column name: any expression the same
expression IR that powers
[`filter.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)/[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
understands is computed as a hidden temporary column first, then
aggregated. This covers comparisons and logic (`sum(carb > 3 & wt < 4)`,
`any(carb > 3)`), arithmetic (`mean(hp / wt)`), and math functions
(`mean(sqrt(hp))`, `sum(carb %% 2 == 0)`), arbitrarily nested – the same
surface documented in
[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md).
A comparison/logical sub-expression is summed as `TRUE`/`FALSE` -\>
`1`/`0`, matching R's own coercion (`sum(c(TRUE, TRUE))` is `2L`) –
except inside
[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html),
which keep it logical. An expression shape the IR doesn't recognize
raises the same "Invalid aggregation expression" error as any other
unsupported aggregation shape.

`column` may also be (or contain) a WINDOW-function call (Phase 6, task
4), e.g. `last(cumsum(v))`, `sum(v - lag(v, default = 0))`,
`mean(cumsum(v) - lag(cumsum(v)))`: the window function is computed PER
GROUP first (over each group's own row order, using the same
window-decomposition engine
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
uses), and the aggregation then reduces that per-row window result –
matching dplyr's own two-step semantics exactly. Only ONE aggregation's
own argument tree is decomposed this way: combining TWO SEPARATE
aggregation calls at the dot level (`sum(cumsum(x)) - mean(y)`) is not
supported (the same "Invalid aggregation expression" error as any other
dot that isn't a bare `fn(single_arg)` shape).

### NA handling

[`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html)
all match R's own `na.rm = FALSE` default (Phase 6, tasks 6.1/6.2): if a
group contains ANY `NA` in the aggregated column, the WHOLE group's
result is `NA` for that aggregation – e.g. `summarise(m = mean(x))` is
`NA` for a group with even one `NA` value of `x`, matching real dplyr
exactly. Pass `na.rm = TRUE` explicitly (e.g. `sum(x, na.rm = TRUE)`) to
exclude `NA` values instead (cudf's own native aggregation behavior).
[`n()`](https://dplyr.tidyverse.org/reference/context.html) is
unaffected by `na.rm=` (it counts every row regardless of nulls, like
dplyr's own
[`n()`](https://dplyr.tidyverse.org/reference/context.html)).

When `na.rm = TRUE` and a group's aggregated column is ENTIRELY `NA`
(Phase 6, task 4), the result matches base R's own empty-vector
reduction identity, exactly like real dplyr:
[`sum()`](https://rdrr.io/r/base/sum.html) -\> `0`,
[`mean()`](https://rdrr.io/r/base/mean.html) -\> `NaN`,
[`min()`](https://rdrr.io/r/base/Extremes.html) -\> `Inf`,
[`max()`](https://rdrr.io/r/base/Extremes.html) -\> `-Inf` (with base
R's warning "no non-missing arguments to min; returning Inf" / "... max;
returning -Inf", as dplyr surfaces it, for double and integer input
alike; it is only raised when some group actually hits the identity),
[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/
[`median()`](https://rdrr.io/r/stats/median.html) -\> `NA` (their
natural result already, no special-casing needed).

Over an INTEGER column, dplyr promotes the WHOLE
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
result column to double the instant any group needs the identity, and
keeps it integer otherwise; cuplyr does the same. Because that type
depends on the data, a
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
with an integer
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
runs immediately even on a lazy table (pending operations are computed
first), and the result's schema reflects the real type. A 0-row input
also counts as an empty group: ungrouped,
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
returns one row of empty-vector reductions
([`n()`](https://dplyr.tidyverse.org/reference/context.html) 0,
[`mean()`](https://rdrr.io/r/base/mean.html) `NaN`,
[`sum()`](https://rdrr.io/r/base/sum.html) 0,
[`min()`](https://rdrr.io/r/base/Extremes.html) `Inf`, ...); grouped,
the result has no rows but an integer
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
is typed double, as in dplyr.

[`quantile()`](https://rdrr.io/r/stats/quantile.html) diverges from the
mean/sum/.../median family: matching base R's own `quantile.default()`
exactly, `na.rm = FALSE` (the default) with an actual `NA` present is a
hard ERROR ("missing values and NaN's not allowed if 'na.rm' is FALSE"),
not a `NA` result – pass `na.rm = TRUE` to compute the quantile over the
non-`NA` values instead.

[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
counts `NA` as its own distinct value when `na.rm = FALSE` (the default,
matching dplyr's own
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
exactly) – e.g. `n_distinct(c(1, 2, NA, 2))` is `3`. Pass `na.rm = TRUE`
to exclude `NA` from the distinct count instead.

[`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html)
use dplyr's own `na_rm=` spelling (underscore, not `na.rm`):
`na_rm = FALSE` (the default) never skips a `NA` when picking the
first/last/nth value (so the result can itself be `NA`); `na_rm = TRUE`
skips `NA` values, returning the first/last/nth non-`NA` value in the
group (or `NA` if the group has none).

[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html)
implement R's exact three-valued logic: `any(x)` is `TRUE` if any
element is `TRUE` (regardless of `NA`s present), `NA` if there's no
`TRUE` but at least one `NA` (with `na.rm = FALSE`, the default), else
`FALSE`; `all(x)` is `FALSE` if any element is `FALSE` (regardless of
`NA`s present), `NA` if there's no `FALSE` but at least one `NA` (with
`na.rm = FALSE`), else `TRUE`. `na.rm = TRUE` drops `NA` values first,
matching base R's
[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html)
exactly (an all-`NA` group with `na.rm = TRUE` is `FALSE` for
[`any()`](https://rdrr.io/r/base/any.html), `TRUE` for
[`all()`](https://rdrr.io/r/base/all.html), the empty-vector identity in
both cases).

### Ungrouped summarise

If `.data` is not grouped, summarise will compute aggregations over all
rows, returning a single-row table.

### No expressions

`summarise(.data)` with no expressions follows dplyr: ungrouped, it
returns a one-row table with no columns; grouped, it returns the
distinct group keys (one row per group, in
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)'s
ascending, `NA`-last order) and peels one grouping layer per `.groups`;
with `.by`, it returns the distinct `.by` combinations in
first-appearance order.

### Tables with no columns

cudf cannot hold rows without columns, so a zero-column `tbl_gpu` keeps
its row count on the R side. What that covers:
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`dim()`](https://rdrr.io/r/base/dim.html),
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)/[`as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html),
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html)/[`collapse()`](https://dplyr.tidyverse.org/reference/compute.html),
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)/[`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html),
zero-dot
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) (1
x 0),
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
with dots (a CPU fallback, so
[`n()`](https://dplyr.tidyverse.org/reference/context.html) counts
correctly), ungrouped
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_head()`](https://dplyr.tidyverse.org/reference/slice.html)/[`slice_tail()`](https://dplyr.tidyverse.org/reference/slice.html),
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)/[`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
of zero-column tables only, a literal-mask
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`dplyr_row_slice()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html),
and adding a column with `$<-`/`[[<-`. Every other verb, joins with a
zero-column side, vctrs operations (`vec_size()`, `vec_slice()`, ...),
and binding a zero-column table that has rows to one with columns raise
an error instead of returning a wrong row count (full support is F14).

### Redefining columns

An expression named like a grouping column replaces that key in place
(same position, the expression's type; the table stays grouped by that
name). An expression that mentions a name an earlier expression created
(or a name created twice) sees the new value, as in dplyr; such calls
run through the CPU fallback.

### Group order

`group_by(g) |> summarise(...)` returns groups in dplyr's
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
order: ascending, `NA` last, strings in the C locale, `FALSE < TRUE`,
factors in level order (`gpu_summarise()` sorts the small aggregated
result by the keys).
[`count()`](https://dplyr.tidyverse.org/reference/count.html)/[`tally()`](https://dplyr.tidyverse.org/reference/count.html)
inherit it.

### `.drop = FALSE`

On a `group_by(.drop = FALSE)` grouping with a factor key, or on a table
whose group keys are fixed (a `.drop = FALSE` table after
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html),
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html), ..., or a
`.preserve = TRUE` slice), every group of
[`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html)
gets a row, empty ones included, holding each expression evaluated on an
empty vector
([`n()`](https://dplyr.tidyverse.org/reference/context.html) 0,
[`sum()`](https://rdrr.io/r/base/sum.html) 0,
[`mean()`](https://rdrr.io/r/base/mean.html) `NaN`,
[`min()`](https://rdrr.io/r/base/Extremes.html) `Inf` with base R's
warning, ...); with no expressions the result is the keys of every
group. Such a call materializes a lazy table on the GPU (the row count
is data-dependent). The result is regrouped with the input's `.drop`.
Three shapes differ from dplyr or leave the GPU:
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
of a Date or POSIXct column is `NA` for an empty group (dplyr's
`Inf`/`-Inf` cannot be held by a GPU timestamp column; an uploaded
non-finite Date or POSIXct is `NA` as well, with the warning described
in [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)),
[`mean()`](https://rdrr.io/r/base/mean.html) of a Date or POSIXct column
is `NA` for an empty group where dplyr gives a `NaN` Date/POSIXct, and
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
of a character column runs the whole call on the CPU when a group is
empty.

### `.by` group order

Verified empirically against dplyr 1.2.1: `summarise(.by=)` emits one
row per distinct `.by` combination in the order that combination FIRST
appears in `.data` – e.g.
`summarise(tibble(g = c("b","a","b")), n = n(), .by = g)` returns `g` in
the order `c("b", "a")`, not `c("a", "b")`. This differs from
`group_by(g) |> summarise(...)`, whose groups are sorted. Implemented
via a desugar (`summarise_by_desugar()`): an ungrouped
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
marks each row's original position, a grouped summarise computes the
user's aggregations plus [`min()`](https://rdrr.io/r/base/Extremes.html)
of that position per group, and the result is sorted by that minimum
(each group's first-appearance position) before the helper column is
dropped.

## See also

[`group_by.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
for grouping data,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Simple aggregation over all rows
  total <- gpu_mtcars |>
    summarise(avg_mpg = mean(mpg)) |>
    collect()

  # Grouped aggregation
  by_cyl <- gpu_mtcars |>
    group_by(cyl) |>
    summarise(
      avg_mpg = mean(mpg),
      max_hp = max(hp),
      count = n()
    ) |>
    collect()

  # Multiple grouping columns
  by_cyl_gear <- gpu_mtcars |>
    group_by(cyl, gear) |>
    summarise(
      mean_mpg = mean(mpg),
      min_wt = min(wt)
    ) |>
    collect()
}
#> `summarise()` has regrouped the output by dropping the last grouping variable.
#> Summaries were computed grouped by cyl, gear.
#> Output is grouped by cyl.
#> Use `summarise(.groups = "drop_last")` to silence this message, or `.groups = "keep"`/`"drop"` to retain/drop every grouping level.
```
