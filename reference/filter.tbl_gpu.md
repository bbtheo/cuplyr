# Filter rows of a GPU table

Selects rows from a GPU table where conditions are TRUE, similar to
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html).
Filtering is performed entirely on the GPU for maximum performance on
large datasets, and matches dplyr's row-keeping and three-valued-logic
(`NA`) semantics exactly (see "NA semantics" below).

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

  Logical expressions to filter by (see "Supported expressions" below
  for the full surface). Multiple expressions, whether passed as
  separate dots or combined with `&` within one dot, are always combined
  with AND (a row is kept only if every expression is TRUE for it).

- .by:

  Optional on-the-fly grouping columns (tidyselect), GPU-native (Phase
  5, task W6): `filter(df, x > mean(x), .by = g)` behaves identically to
  `df |> group_by(g) |> filter(x > mean(x)) |> ungroup()` – same
  window/aggregate-predicate machinery as
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)-driven
  filtering, and the result is always ungrouped. Supplying `.by` when
  `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr exactly.

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

[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) parses
its arguments through an internal expression IR (see `R/ir.R`) that
lowers every predicate in the call to a **single fused GPU kernel**
(`cudf::compute_column()` under the hood) – one kernel per
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) call, no
matter how many dots or how deeply the predicate is nested. The IR
understands:

- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=` – against a literal,
  another column, an environment variable, or an arithmetic expression
  (`(x - y) > 0`)

- Logic: `&`, `|`, `!`, [`xor()`](https://rdrr.io/r/base/Logic.html),
  arbitrarily nested (`x > 2 & (y < 5 | z == 1)`); `&&`/`||` are
  rejected with a message pointing at `&`/`|` (matching dplyr 1.1+)

- [`is.na()`](https://rdrr.io/r/base/NA.html)

- `%in%`, including a set containing `NA` (R's never-NA rule: a value
  that itself is `NA` is TRUE for `%in%` iff the set contains `NA`,
  never `NA` itself) and an empty set (`character(0)`,
  [`c()`](https://rdrr.io/r/base/c.html), etc. – always FALSE, for every
  row, including `NA` rows)

- `between(x, lo, hi)` (dplyr's definition: `x >= lo & x <= hi`,
  including its `NA` propagation), `near(x, y, tol = )`
  (`abs(x - y) < tol`, matching
  [`dplyr::near()`](https://dplyr.tidyverse.org/reference/near.html)
  exactly)

- String-column comparisons (`==`/`!=`/`<`/`<=`/`>`/`>=`), against a
  literal or another string column

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
  matches a column resolves to that column; otherwise it's evaluated as
  an environment variable – "columns shadow the environment")

- Window/aggregate functions (Phase 5, task W5), usable anywhere inside
  a predicate, including combined with plain column comparisons in the
  same dot (`filter(x > mean(x) & y < 5)`):
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
  evaluated per group when `.data` is grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) or
  the on-the-fly `.by=` argument, Phase 5 task W6 – identical semantics
  either way). A grouped filter – like dplyr's own – preserves the
  ORIGINAL row order in the result, it does not reorder into group-key
  order.

Comparing a column against a literal of an incompatible type (e.g. a
numeric column against a character literal) errors immediately, naming
the column and its type, rather than silently falling back or producing
a raw GPU-side type error.

### Multi-dot semantics with window/aggregate predicates

Multiple dots (or multiple `&`-joined predicates within one dot) are
always combined with AND, and – confirmed empirically against real
dplyr, since this is easy to get wrong – every dot's predicate is
evaluated against the SAME, ORIGINAL
(pre-[`filter()`](https://dplyr.tidyverse.org/reference/filter.html))
data, never against a progressively-narrowed intermediate result.
Concretely, `filter(x > 0, cumsum(x) < 10)` computes `cumsum(x)` over
every row of the original table, then ANDs that mask with `x > 0` – it
does NOT filter to `x > 0` first and then compute
[`cumsum()`](https://rdrr.io/r/base/cumsum.html) only over the surviving
rows (which would generally give a different, smaller cumulative sum and
therefore a different result). `cuplyr` reproduces this exactly: every
window-bearing dot's temp columns are computed against the table as it
stood when
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) was
called, before any dot's predicate (window-derived or plain) has removed
a single row.

### NA semantics

[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) keeps a
row iff its combined predicate is non-missing *and* TRUE – identical to
dplyr (`cudf::apply_boolean_mask()`'s "non-null AND true" rule matches
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html)'s
own row-keeping rule exactly). Concretely:

- `filter(x > 5)` drops rows where `x` is `NA` (the comparison is `NA`,
  and `NA` isn't kept), same as dplyr

- `&`/`|` use R's three-valued logic, not C's: `NA & FALSE` is `FALSE`,
  `NA | TRUE` is `TRUE`, `NA & TRUE`/`NA | FALSE`/`NA & NA`/ `NA | NA`
  are all `NA` (`cudf::ast_operator::NULL_LOGICAL_AND`/
  `NULL_LOGICAL_OR`, chosen specifically because they match)

- `filter(x > 2 | is.na(x))` is the idiom for "keep NA rows too" – works
  exactly as it does in dplyr

- `%in%` never produces `NA` (see "Supported expressions" above) – this
  is R's `%in%`/[`match()`](https://rdrr.io/r/base/match.html)
  semantics, not null-propagating

- `filter(TRUE)` is a no-op (returns `.data` unchanged, no GPU work);
  `filter(FALSE)` and `filter(NA)` both return a zero-row result (a bare
  `NA` predicate is `FALSE`-equivalent for row-keeping, same as dplyr)

A floating `NaN` (e.g. a computed `0/0`) stays a value on the GPU, but
[`is.na()`](https://rdrr.io/r/base/NA.html), the six comparisons and
[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html) read
it as missing, as in R: `NaN > 1` is `NA` (the row is dropped) and
`is.na(NaN)` is `TRUE`.
[`is.nan()`](https://rdrr.io/r/base/is.finite.html) runs on the GPU too.

### Not yet supported

- Arbitrary R functions with no cuDF equivalent – these fall back to CPU
  evaluation, see below

### Expressions the IR doesn't understand (CPU fallback)

An expression shape the IR doesn't recognize (see `R/ir.R`'s
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md))
is handled in one of two ways. A dot that reads no table column (a
pre-computed logical vector, `rep(TRUE, n)`) is evaluated once in R,
with no data mask: a logical scalar is a constant predicate. On an
ungrouped
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) a
logical vector is a row mask: all the vectors of one call are sized
against the input table (a wrong size is dplyr's own error) and combined
with `&` with the call's other predicates, which see the same input
rows. On a grouped, `.by` or rowwise table, or in
[`filter_out()`](https://dplyr.tidyverse.org/reference/filter.html), a
logical vector runs the whole call in dplyr. A dot that reads a table
column but doesn't parse (a rare shape given how much of dplyr's filter
surface the IR now covers), or that visibly calls something reaching the
data mask by name (`exists("x")`, `get("ok")`), runs the whole call in
dplyr on the CPU (see `getOption("cuplyr.fallback")` below to be
notified when it does). The check reads the dot's syntax, so an alias of
such a function or a user function calling it is not seen and has no
data mask when the dot names no column. A lazy `.data` with pending
operations runs them early, on the GPU, when a logical vector needs the
row count (a plan break, see `getOption("cuplyr.plan_break")`).

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

Filtering on GPU is highly parallel and can process billions of rows per
second. For best performance, prefer expressions the IR parses directly
into a GPU predicate (the entire "Supported expressions" list above)
over expressions that require the CPU fallback.

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
