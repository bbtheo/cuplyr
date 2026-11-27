# Keep rows that match a condition

[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) keeps
the rows of a GPU table where every condition is `TRUE`, as
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html)
does. The conditions listed under "Supported expressions" run on the
GPU, within the limits stated there, with dplyr's rules for which rows
are kept and for `NA` (see "NA semantics" below). Other conditions run
in dplyr on the CPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
filter(.data, ..., .by = NULL, .preserve = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Logical expressions to filter by (see "Supported expressions" below).
  Multiple expressions, whether passed as separate dots or combined with
  `&` within one dot, are combined with AND (a row is kept only if every
  expression is `TRUE` for it).

- .by:

  Optional grouping columns for this call only (tidyselect), run on the
  GPU: `filter(df, x > mean(x), .by = g)` gives the same rows as
  `df |> group_by(g) |> filter(x > mean(x)) |> ungroup()`, and the
  result is ungrouped. Supplying `.by` when `.data` is already grouped
  (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

- .preserve:

  Whether to keep the groups the filter emptied. With `TRUE`, a
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-grouped
  result keeps the input's group keys (empty groups included) for
  [`group_data()`](https://dplyr.tidyverse.org/reference/group_data.html),
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) and
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
  like dplyr (see
  [`group_by.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
  for what carries them on). `FALSE` recomputes the groups from the
  remaining rows.

## Value

A `tbl_gpu` object containing only rows where all conditions are TRUE.
The GPU memory for the filtered result is newly allocated.

## Details

### Supported expressions

All the conditions of one
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) call are
combined into a single predicate and evaluated together on the GPU,
however many dots there are and however deeply they nest. These run on
the GPU:

- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` – against a literal,
  another column, an R variable, or an arithmetic expression
  (`(x - y) > 0`); for strings, see the string item below

- Logic: `&`, `|`, `!`, [`xor()`](https://rdrr.io/r/base/Logic.html),
  nested to any depth (`x > 2 & (y < 5 | z == 1)`); `&&`/`||` are an
  error that points at `&`/`|`

- [`is.na()`](https://rdrr.io/r/base/NA.html)

- `%in%`, including a set containing `NA` (as in R, `%in%` is never
  `NA`: an `NA` value is `TRUE` when the set contains `NA` and `FALSE`
  otherwise) and an empty set (`character(0)`,
  [`c()`](https://rdrr.io/r/base/c.html), etc.: `FALSE` for every row,
  `NA` rows included)

- `between(x, lo, hi)` (dplyr's definition: `x >= lo & x <= hi`, with
  its `NA` propagation), `near(x, y, tol = )` (`abs(x - y) < tol`, as
  [`dplyr::near()`](https://dplyr.tidyverse.org/reference/near.html))

- String comparisons, against a literal or another string column. `==`,
  `!=`, `%in%` and
  [`between()`](https://dplyr.tidyverse.org/reference/between.html)
  always run on the GPU. `<`, `<=`, `>` and `>=` compare strings in byte
  order on the GPU, while R uses the session's collation, so they run on
  the GPU only when `Sys.getlocale("LC_COLLATE")` is `"C"` or `"POSIX"`;
  under any other collation (`"en_US.UTF-8"`, say) the call runs in
  dplyr on the CPU. The same holds for
  [`min()`](https://rdrr.io/r/base/Extremes.html) and
  [`max()`](https://rdrr.io/r/base/Extremes.html) of a string column.

- Column-vs-column comparisons for every operator above

- Math on columns inside a predicate (`+ - * / %% %/% ^`, `sqrt`, `log`,
  `abs`, etc. – the same set
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  supports)

- [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html),
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
  [`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html),
  [`na_if()`](https://dplyr.tidyverse.org/reference/na_if.html),
  [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html),
  [`recode()`](https://dplyr.tidyverse.org/reference/recode.html)
  (scalar literal shapes),
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
  [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
  [`replace_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
  – the same conditional/vector functions
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  supports (see its own docs for the exact type-unification rules),
  usable anywhere inside a predicate, including as the whole predicate
  itself (e.g. `filter(case_when(x > 1 ~ TRUE, .default = FALSE))`)

- `when_any(...)`/`when_all(...)` – particularly useful inside
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) to
  combine several comma-separated conditions with `|` instead of `&`
  (e.g. `filter(when_any(x > 10, y < 0))`)

- The `.data`/`.env` pronouns, and ordinary R symbols (a bare name that
  matches a column resolves to that column; otherwise it is looked up as
  an R variable: columns shadow the environment, as in dplyr)

- Window and aggregate functions, anywhere inside a predicate, also next
  to plain column comparisons in the same dot
  (`filter(x > mean(x) & y < 5)`):
  [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)/`row_number(x)`,
  [`min_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`dense_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/
  [`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)/[`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html),
  [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/[`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html),
  [`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)/[`cumprod()`](https://rdrr.io/r/base/cumsum.html)/[`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html)/
  [`cumall()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumany()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html),
  [`ntile()`](https://dplyr.tidyverse.org/reference/ntile.html),
  [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`n()`](https://dplyr.tidyverse.org/reference/context.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html)/
  [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)/[`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html)
  – the same set
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
  supports (see its own docs for exact semantics/argument support),
  evaluated per group when `.data` is grouped (with
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) or
  `.by`, which behave the same). As in dplyr, a grouped filter keeps the
  original row order; it does not reorder rows by group.

Comparing a number with a string (`filter(mpg > "2")`) compares them as
strings, as R does; that call runs in dplyr on the CPU.

### Multi-dot semantics with window/aggregate predicates

Multiple dots (or multiple `&`-joined predicates within one dot) are
combined with AND, and, as in dplyr, every dot is evaluated against the
same input rows, never against the rows an earlier dot left. So
`filter(x > 0, cumsum(x) < 10)` computes `cumsum(x)` over every row of
the input, then combines that with `x > 0`; it does not keep `x > 0`
first and compute [`cumsum()`](https://rdrr.io/r/base/cumsum.html) over
the remaining rows (which would usually give a smaller cumulative sum
and a different result).

### NA semantics

[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) keeps a
row only if its combined predicate is `TRUE` (not `FALSE` and not `NA`),
as in dplyr:

- `filter(x > 5)` drops rows where `x` is `NA` (the comparison is `NA`,
  and `NA` isn't kept)

- `&`/`|` use R's three-valued logic: `NA & FALSE` is `FALSE`,
  `NA | TRUE` is `TRUE`, `NA & TRUE`/`NA | FALSE`/`NA & NA`/ `NA | NA`
  are all `NA`

- `filter(x > 2 | is.na(x))` is the idiom for "keep NA rows too", as in
  dplyr

- `%in%` never gives `NA` (see "Supported expressions" above), as R's
  `%in%` and [`match()`](https://rdrr.io/r/base/match.html)

- `filter(TRUE)` returns `.data` unchanged, with no GPU work;
  `filter(FALSE)` and `filter(NA)` both return zero rows

A floating `NaN` (e.g. a computed `0/0`) stays a value on the GPU, but
[`is.na()`](https://rdrr.io/r/base/NA.html), the six comparisons and
[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html) read
it as missing, as in R: `NaN > 1` is `NA` (the row is dropped) and
`is.na(NaN)` is `TRUE`.
[`is.nan()`](https://rdrr.io/r/base/is.finite.html) runs on the GPU too.

### Conditions that run on the CPU

A condition outside the list above (an R function with no GPU version,
for example) is handled in one of two ways. A dot that reads no table
column (a pre-computed logical vector, `rep(TRUE, n)`) is evaluated once
in R, with no data mask: a logical scalar is a constant predicate. On an
ungrouped
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) a
logical vector is a row mask: all the vectors of one call are sized
against the input table (a wrong size is dplyr's own error) and combined
with `&` with the call's other predicates, which see the same input
rows. On a grouped, `.by` or rowwise table, or in
[`filter_out()`](https://dplyr.tidyverse.org/reference/filter.html), a
logical vector runs the whole call in dplyr. A dot that reads a table
column but can't run on the GPU, or that visibly calls something
reaching the data mask by name (`exists("x")`, `get("ok")`), runs the
whole call in dplyr on the CPU (see `getOption("cuplyr.fallback")` below
to be notified when it does). The check reads the dot's syntax, so an
alias of such a function or a user function calling it is not seen and
has no data mask when the dot names no column. A lazy `.data` with
pending operations runs them early, on the GPU, when a logical vector
needs the row count (a plan break, see
`getOption("cuplyr.plan_break")`).

### `options(cuplyr.fallback = ...)`

Controls whether a CPU fallback (the one described above, and every
other verb that runs in dplyr on the CPU) notifies the caller. One of:

- `"silent"` (default) - no notification

- `"warn"` (or `"warning"`) - emits a warning naming the verb, the
  expression, the reason and the size collected

- `"error"` - [`stop()`](https://rdrr.io/r/base/stop.html)s instead of
  falling back

A lazy plan that only runs early on the GPU is a plan break, reported
under `options(cuplyr.plan_break = )` instead. See
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md).

### Performance

Filtering on the GPU is highly parallel. For the best performance, use
the conditions listed under "Supported expressions", which run on the
GPU, rather than ones that run in dplyr on the CPU.

## See also

[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
for creating new columns,
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for selecting columns,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Filter with single condition
  efficient_cars <- gpu_mtcars |>
    filter(mpg > 25)

  # Multiple conditions (combined with AND)
  result <- gpu_mtcars |>
    filter(mpg > 20) |>
    filter(cyl == 4) |>
    collect()

  # Compare two columns
  gpu_cars <- tbl_gpu(cars)
  fast_stops <- gpu_cars |>
    filter(dist < speed) |>
    collect()
}
```
