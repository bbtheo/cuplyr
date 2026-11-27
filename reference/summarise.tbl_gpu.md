# Summarise each group down to one row

[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
computes one row per group of a GPU table (or one row in all, when it is
ungrouped), as
[`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
does. The aggregations listed below run on the GPU; any other expression
is evaluated by dplyr on the CPU (set
`options(cuplyr.fallback = "warn")` to be told when and why; see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).

## Usage

``` r
# S3 method for class 'tbl_gpu'
summarise(.data, ..., .groups = NULL, .by = NULL)

# S3 method for class 'tbl_gpu'
summarize(.data, ..., .groups = NULL, .by = NULL)
```

## Arguments

- .data:

  A `tbl_gpu`, usually grouped with
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html).

- ...:

  Name-value pairs of summary functions. The name will be the name of
  the variable in the result. The value is an aggregation expression in
  the form `fun(column)`, where `column` can also be an expression of
  columns (see "Aggregation sub-expressions" below), or arithmetic
  around such calls and the results of earlier expressions (see
  "Arithmetic around reducers" below).

- .groups:

  Grouping structure of the result, as in
  [`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).
  One of:

  - `NULL` (default): `"drop_last"`, with a message when the result is
    still grouped (see "The regroup message" below).

  - `"drop_last"`: drop the last grouping variable and keep the rest
    (`group_by(a, b, c)` gives a result grouped by `a, b`); with only
    one grouping variable, the result is ungrouped.

  - `"drop"`: the result is ungrouped.

  - `"keep"`: every grouping variable is kept.

  - `"rowwise"`: the summary runs on the GPU keeping every grouping
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
  dplyr. On an ungrouped `.data`, as in dplyr, `.groups` is not checked:
  any value other than `"rowwise"` (a typo included) is ignored.

  ### The regroup message

  When `.groups` is `NULL` and `.data` has more than one grouping
  variable,
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  tells you how the result is grouped, in words close to dplyr's.
  `options(dplyr.summarise.inform = FALSE)` silences it (an explicit
  `TRUE` or `FALSE` always wins). With the option unset, dplyr's rule
  applies: the message appears only when
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  is called from the global environment (top-level or interactive use),
  not from inside a function or package (a `testthat` test, or a verb
  such as
  [`tally.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tally.tbl_gpu.md)
  that calls
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)).

- .by:

  Optional grouping columns for this call only (tidyselect), run on the
  GPU. Groups come out in the order each `.by` combination first appears
  in `.data`, not sorted as after
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  (see "`.by` group order" below). The result is always ungrouped.
  `.groups` can't be combined with `.by`, and supplying `.by` when
  `.data` is already grouped is an error, as in dplyr.

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
  of an integer `sum(x)` runs at the verb, like an eager one: the
  pending steps run first (a plan break, see `cuplyr.plan_break`) and
  the result is a lazy table with nothing pending. The same goes for a
  lazy
  [`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
  of an integer. A [`sum()`](https://rdrr.io/r/base/sum.html) of a
  logical `x` stays in the plan: it can only overflow with more than
  2^31 `TRUE` rows in a group, and then raises a
  `cuplyr_error_sum_overflow` error at
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

- `mean(x)` - Arithmetic mean

- `min(x)` - Minimum value

- `max(x)` - Maximum value.
  [`min()`](https://rdrr.io/r/base/Extremes.html) and
  [`max()`](https://rdrr.io/r/base/Extremes.html) of a string column run
  on the GPU only when `Sys.getlocale("LC_COLLATE")` is `"C"` or
  `"POSIX"`: the GPU compares bytes, while R uses the session's
  collation, so under any other collation they run in dplyr on the CPU.

- [`n()`](https://dplyr.tidyverse.org/reference/context.html) - Count of
  rows in each group

- `sd(x)` - Standard deviation

- `var(x)` - Variance

- `median(x)` - Median value, of a double, `Date` or `POSIXct` column.
  The median of an integer or logical column is of the column's type or
  a double depending on each group's count, and runs in dplyr on the
  CPU.

- `quantile(x, probs)` - A single quantile, `probs` one number in
  `[0, 1]`, with R's default `type = 7`. Other forms run in dplyr on the
  CPU.

- `n_distinct(x)` - Count of distinct values, `NA` counted as its own
  distinct value by default

- `first(x)` / `last(x)` / `nth(x, n)` - The first/last/nth value in the
  group's current row order

- `any(x)` / `all(x)` - Logical reductions with R's three-valued logic

On the GPU, [`mean()`](https://rdrr.io/r/base/mean.html) and
[`median()`](https://rdrr.io/r/stats/median.html) of a `Date` are
floored to whole days with a `cuplyr_warning_fractional_date` warning,
where dplyr keeps the fractional `Date`.

Some argument types run in dplyr on the CPU: a factor column under any
aggregation but
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html);
an integer64 column under [`sum()`](https://rdrr.io/r/base/sum.html),
[`mean()`](https://rdrr.io/r/base/mean.html),
[`min()`](https://rdrr.io/r/base/Extremes.html),
[`max()`](https://rdrr.io/r/base/Extremes.html),
[`sd()`](https://rdrr.io/r/stats/sd.html),
[`var()`](https://rdrr.io/r/stats/cor.html),
[`median()`](https://rdrr.io/r/stats/median.html),
[`quantile()`](https://rdrr.io/r/stats/quantile.html),
[`any()`](https://rdrr.io/r/base/any.html) and
[`all()`](https://rdrr.io/r/base/all.html) (its
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html),
[`first()`](https://dplyr.tidyverse.org/reference/nth.html),
[`last()`](https://dplyr.tidyverse.org/reference/nth.html) and
[`nth()`](https://dplyr.tidyverse.org/reference/nth.html) run on the
GPU); and a `Date` or `POSIXct` column under any aggregation but
[`min()`](https://rdrr.io/r/base/Extremes.html),
[`max()`](https://rdrr.io/r/base/Extremes.html),
[`mean()`](https://rdrr.io/r/base/mean.html),
[`median()`](https://rdrr.io/r/stats/median.html),
[`first()`](https://dplyr.tidyverse.org/reference/nth.html),
[`last()`](https://dplyr.tidyverse.org/reference/nth.html),
[`nth()`](https://dplyr.tidyverse.org/reference/nth.html) and
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html).

### Aggregation sub-expressions

`column` is not limited to a bare column name: any expression that
[`filter.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
and
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
run on the GPU is computed as a hidden temporary column first, then
aggregated. This covers comparisons and logic (`sum(carb > 3 & wt < 4)`,
`any(carb > 3)`), arithmetic (`mean(hp / wt)`), and math functions
(`mean(sqrt(hp))`, `sum(carb %% 2 == 0)`), nested to any depth: the same
functions listed in
[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md).
A comparison or logical sub-expression is summed as `TRUE`/`FALSE` -\>
`1`/`0`, as in R (`sum(c(TRUE, TRUE))` is `2L`), and stays logical
inside
[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html).
Any other expression is evaluated by dplyr on the CPU.

`column` may also be (or contain) a window function call, e.g.
`last(cumsum(v))`, `sum(v - lag(v, default = 0))`,
`mean(cumsum(v) - lag(cumsum(v)))`: the window function is computed per
group first (over each group's own row order, with the same window
functions
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
runs on the GPU), and the aggregation then reduces that per-row result,
as dplyr does. This applies to the argument of one aggregation; several
aggregation calls in one expression are combined as described in the
next section.

### Arithmetic around reducers

An expression may combine reducer calls with arithmetic, comparisons,
math functions and constants, and may read the result of an earlier
expression: `mean(x) + 1`, `sum(x) / n()`, `max(x) - min(x)`,
`mean(x) - qnorm(.975) * sd(x) / sqrt(n())`, `sum(cumsum(x)) - mean(y)`,
`s = sum(x), m = s / n()`. Each reducer call is aggregated as above (a
repeated call once), and the rest is computed on the one-row-per-group
result with
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)'s
expressions, so the result has dplyr's types and `NA` values, with three
exceptions. Math with no real value (`sqrt(mean(x))` of a negative mean)
is `NaN` without R's "NaNs produced" warning. A power can differ from
R's in the last bit (`var(x)^0.5` of 9 is 2.9999999999999996, where R
computes 3 in extended precision), as in
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md).
An integer [`sum()`](https://rdrr.io/r/base/sum.html) that leaves the
integer range in some group is a double in every group, so `sum(x) + 1L`
gives a value in a group whose sum still is an integer where dplyr warns
about the overflow and gives `NA`.

These stay with dplyr (the fallback notice names the reason, see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)):
a function of the group's rows around a reducer (`lag(mean(x))`,
`cumsum(sum(x))`); `^` whose exponent is not a constant other than 0,
`NA` and an infinity (R's `NA^0` and `1^NA` are 1, the GPU's are `NA`);
`^` with an odd or non-integer exponent (`mean(x)^3`, `max(x)^0.5`,
`(-mean(x))^-1`) unless its base is
[`n()`](https://dplyr.tidyverse.org/reference/context.html),
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html),
[`sd()`](https://rdrr.io/r/stats/sd.html) or
[`var()`](https://rdrr.io/r/stats/cor.html) (R's power of a `-Inf` or
`-0`, which [`max()`](https://rdrr.io/r/base/Extremes.html) of an empty
group or `-mean(x)` of a zero mean gives, is not the GPU's: `(-Inf)^0.5`
is `NaN` in R, `Inf` on the GPU); `&`, `|` or `!` of anything but a
logical or an integer (R reads a double's `NaN` there as `NA`, and a
string is R's error);
[`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html) or
[`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
with a condition that is not a logical expression; a column of the input
outside a reducer (`mean(x) + x`, dplyr's "must be size 1" error); a
reducer that reads a name an earlier expression assigned
(`x = sum(x), t = sum(x)`); an expression that reads the name of a later
expression (`mean(x) + k, k = sum(y)`: dplyr still sees the variable `k`
there); [`count()`](https://dplyr.tidyverse.org/reference/count.html)
inside an expression; a name assigned twice. A reducer inside a quosure
spliced into the expression (`{{ stat }}`, `!!quo`) is computed in the
quosure's own environment.

### NA handling

[`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html)
follow R's `na.rm = FALSE` default: if a group holds any `NA` in the
aggregated column, that group's result is `NA` (`summarise(m = mean(x))`
is `NA` for a group with one `NA` in `x`), as in dplyr. Pass
`na.rm = TRUE` (e.g. `sum(x, na.rm = TRUE)`) to leave the `NA` values
out. [`n()`](https://dplyr.tidyverse.org/reference/context.html) counts
every row, `NA` or not, as dplyr's
[`n()`](https://dplyr.tidyverse.org/reference/context.html) does.

When `na.rm = TRUE` and a group's aggregated column is entirely `NA`,
the result is base R's value for an empty vector, as in dplyr:
[`sum()`](https://rdrr.io/r/base/sum.html) -\> `0`,
[`mean()`](https://rdrr.io/r/base/mean.html) -\> `NaN`,
[`min()`](https://rdrr.io/r/base/Extremes.html) -\> `Inf`,
[`max()`](https://rdrr.io/r/base/Extremes.html) -\> `-Inf` (with base
R's warning "no non-missing arguments to min; returning Inf" / "... max;
returning -Inf", as dplyr shows it, for double and integer input alike;
only when some group meets it),
[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html)
-\> `NA`.

Over an integer column, dplyr makes the whole
[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
result column a double as soon as any group needs that empty-vector
value, and keeps it integer otherwise; cuplyr does the same. Because
that type depends on the data, a
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

[`quantile()`](https://rdrr.io/r/stats/quantile.html) is different, as
in base R's [`quantile()`](https://rdrr.io/r/stats/quantile.html): with
`na.rm = FALSE` (the default), an `NA` in the group is an error
("missing values and NaN's not allowed if 'na.rm' is FALSE"), not an
`NA` result. Pass `na.rm = TRUE` to compute the quantile over the
non-`NA` values.

[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
counts `NA` as its own distinct value when `na.rm = FALSE` (the default,
as dplyr's
[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
does): `n_distinct(c(1, 2, NA, 2))` is `3`. Pass `na.rm = TRUE` to leave
`NA` out of the count.

[`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html)
use dplyr's `na_rm=` spelling (underscore, not `na.rm`): `na_rm = FALSE`
(the default) never skips an `NA` when picking the first/last/nth value
(so the result can itself be `NA`); `na_rm = TRUE` skips `NA` values,
returning the first/last/nth non-`NA` value in the group (or `NA` if the
group has none).

[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html)
follow R's three-valued logic: `any(x)` is `TRUE` if any element is
`TRUE` (whatever `NA`s there are), `NA` if there's no `TRUE` but at
least one `NA` (with `na.rm = FALSE`, the default), else `FALSE`;
`all(x)` is `FALSE` if any element is `FALSE`, `NA` if there's no
`FALSE` but at least one `NA` (with `na.rm = FALSE`), else `TRUE`.
`na.rm = TRUE` drops `NA` values first, as base R's
[`any()`](https://rdrr.io/r/base/any.html)/[`all()`](https://rdrr.io/r/base/all.html)
do (an all-`NA` group with `na.rm = TRUE` is `FALSE` for
[`any()`](https://rdrr.io/r/base/any.html) and `TRUE` for
[`all()`](https://rdrr.io/r/base/all.html)).

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

A GPU table can't hold rows without columns, so a zero-column `tbl_gpu`
keeps its row count in R. What that covers:
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`dim()`](https://rdrr.io/r/base/dim.html),
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)/[`as_tibble()`](https://tibble.tidyverse.org/reference/as_tibble.html),
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html)/[`collapse()`](https://dplyr.tidyverse.org/reference/compute.html),
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)/[`ungroup()`](https://dplyr.tidyverse.org/reference/group_by.html),
[`select()`](https://dplyr.tidyverse.org/reference/select.html)/[`relocate()`](https://dplyr.tidyverse.org/reference/relocate.html),
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html),
zero-dot
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) (1
x 0),
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
with dots (in dplyr on the CPU, so
[`n()`](https://dplyr.tidyverse.org/reference/context.html) counts
correctly),
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) and its
helpers,
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)/[`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md)
of zero-column tables only,
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)/[`filter_out()`](https://dplyr.tidyverse.org/reference/filter.html)
(dplyr runs the call on a table of the same height, so
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
and [`n()`](https://dplyr.tidyverse.org/reference/context.html) see its
rows),
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)
that leave no column (every column set to `NULL`),
[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
and joins with `by = character()` where a side has no columns, the set
operations of two zero-column tables, vctrs operations (`vec_size()`,
`vec_slice()`, ...),
[`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) with a
zero-column data frame result,
[`dplyr_row_slice()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html),
and adding a column with `$<-`/`[[<-`. Binding a zero-column table that
has rows to one with columns raises an error instead of returning a
wrong row count.

### Redefining columns

An expression named like a grouping column replaces that key in place
(same position, the expression's type; the table stays grouped by that
name). An expression that mentions a name an earlier expression created
(or a name created twice) sees the new value, as in dplyr. When that
expression is arithmetic around reducers and earlier results
(`m = s / n()`) it runs on the GPU (see above); a reducer that reads the
name (`x = sum(x), t = sum(x)`) or a repeated name runs in dplyr on the
CPU.

### Group order

`group_by(g) |> summarise(...)` returns groups in dplyr's
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
order: ascending, `NA` last, strings in the C locale, `FALSE < TRUE`,
factors in level order.
[`count()`](https://dplyr.tidyverse.org/reference/count.html) and
[`tally()`](https://dplyr.tidyverse.org/reference/count.html) give the
same order.

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

`summarise(.by = )` gives one row per distinct `.by` combination, in the
order that combination first appears in `.data`, as dplyr does:
`summarise(tibble(g = c("b", "a", "b")), n = n(), .by = g)` returns `g`
in the order `c("b", "a")`, not `c("a", "b")`.
`group_by(g) |> summarise(...)` sorts its groups instead. On the GPU
this is one grouped summary, and its peak GPU memory is that of the
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) form
plus up to 12 bytes a row.

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
