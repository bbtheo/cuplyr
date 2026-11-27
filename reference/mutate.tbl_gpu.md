# Create, modify and delete columns

[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) adds new
columns or changes existing ones, as
[`dplyr::mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
does. The expressions listed under "Supported expressions" are computed
on the GPU, within the limits stated there. A call with any other
expression runs in dplyr on the CPU, and its result comes back to the
GPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
mutate(
  .data,
  ...,
  .by = NULL,
  .keep = c("all", "used", "unused", "none"),
  .before = NULL,
  .after = NULL
)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Name-value pairs of expressions. The name gives the column name (new
  or existing), and the value is an expression involving existing
  columns and/or scalar values (see "Supported expressions" below).

- .by:

  Optional grouping columns for this call only (tidyselect), run on the
  GPU: `mutate(df, y = mean(x), .by = g)` gives the same result as
  `df |> group_by(g) |> mutate(y = mean(x)) |> ungroup()`, with the rows
  in their original order. The result is always ungrouped. Supplying
  `.by` when `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, as in dplyr.

- .keep:

  One of `"all"` (default), `"used"`, `"unused"`, `"none"`; controls
  which of `.data`'s *other* columns (besides ones `...` creates or
  modifies, and group columns, both of which are always kept) survive in
  the output – see "`.keep` semantics" below.

- .before, .after:

  Optionally, control where new columns should appear, using the same
  tidyselect syntax as
  [`relocate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/relocate.tbl_gpu.md).
  Only new columns are moved; a modified existing column stays where it
  is.

## Value

A `tbl_gpu` object with the new or modified columns. If a column name
already exists, it is replaced. New columns are appended (subject to
`.before`/`.after`), and `.data`'s other columns are kept or dropped per
`.keep`.

## Details

### `.keep` semantics

As in dplyr, the columns created or modified by `...` (every dot's
output column, new or replaced) and every grouping column are always
kept, whatever `.keep` says. `.keep` only controls which other
(untouched, non-grouping) columns of `.data` are kept alongside them:

- `"all"` (default): every other column.

- `"used"`: other columns that are referenced as an *input* to any of
  `...`'s expressions (union across all dots).

- `"unused"`: other columns not referenced as an input to any expression
  – the complement of `"used"`.

- `"none"`: no other columns at all.

Column order: `.data`'s columns keep their order (modified columns don't
move), and new columns are appended at the end (or placed with
`.before`/`.after`); `.keep` then drops columns without changing that
order. This differs from
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html),
whose output order is the literal `...` dot order (see
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)'s
own docs).

### Supported expressions

A function runs on the GPU only when its name, in the expression's
environment, is the base, dplyr or lubridate function itself: a function
of your own with the same name (local, in an enclosing function or
global) runs on the CPU, as dplyr would call it, and `pkg::fn` is the
function of that package. The same holds for a function passed as a
value (`across(x, f)`, a list of functions held in a variable, `!!f`)
and for the functions and lambdas of the scoped verbs
([`mutate_at()`](https://dplyr.tidyverse.org/reference/mutate_all.html),
[`summarise_all()`](https://dplyr.tidyverse.org/reference/summarise_all.html),
[`filter_if()`](https://dplyr.tidyverse.org/reference/filter_all.html),
...): one that is, or calls, a function of yours with the name of a GPU
function runs on the CPU. These run on the GPU:

- Arithmetic: `+`, `-`, `*`, `/`, `^`, `%%`, `%/%`, unary `-`, nested to
  any depth (`(x + y) * z - 1`), with a scalar on either side
  (`mutate(y = 10 - x)`)

- Comparisons and logic: `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`,
  `!`, [`xor()`](https://rdrr.io/r/base/Logic.html) – these produce a
  logical column, e.g. `mutate(is_big = x > 100)`. Two strings compare
  with `==`, `!=`, `%in%` and
  [`between()`](https://dplyr.tidyverse.org/reference/between.html) on
  the GPU whatever the session's collation. `<`, `<=`, `>` and `>=` of
  strings compare bytes on the GPU, while R uses the session's
  collation, so they run on the GPU only when
  `Sys.getlocale("LC_COLLATE")` is `"C"` or `"POSIX"`, and in dplyr on
  the CPU otherwise. A string compared with a number runs on the CPU (R
  compares them as strings).

- [`is.na()`](https://rdrr.io/r/base/NA.html),
  [`is.nan()`](https://rdrr.io/r/base/is.finite.html),
  `between(x, lo, hi)`, `near(x, y, tol = )` (`abs(x - y) < tol`, as
  [`dplyr::near()`](https://dplyr.tidyverse.org/reference/near.html)
  defines it), `%in%` (including a bare `NA` on the right-hand side,
  e.g. `x %in% NA`, as `NA %in% NA` is `TRUE` in base R; an *empty* set
  is always `FALSE`)

- Math functions: [`sqrt()`](https://rdrr.io/r/base/MathFun.html),
  [`log()`](https://rdrr.io/r/base/Log.html) (including `log(x, base)`),
  [`log2()`](https://rdrr.io/r/base/Log.html),
  [`log10()`](https://rdrr.io/r/base/Log.html),
  [`exp()`](https://rdrr.io/r/base/Log.html),
  [`abs()`](https://rdrr.io/r/base/MathFun.html),
  [`floor()`](https://rdrr.io/r/base/Round.html),
  [`ceiling()`](https://rdrr.io/r/base/Round.html),
  [`sin()`](https://rdrr.io/r/base/Trig.html),
  [`cos()`](https://rdrr.io/r/base/Trig.html),
  [`tan()`](https://rdrr.io/r/base/Trig.html),
  `round(x)`/`round(x, digits)` (banker's rounding, as in R)

- String functions, among them
  [`nchar()`](https://rdrr.io/r/base/nchar.html),
  [`toupper()`](https://rdrr.io/r/base/chartr.html),
  [`tolower()`](https://rdrr.io/r/base/chartr.html),
  [`substr()`](https://rdrr.io/r/base/substr.html),
  [`trimws()`](https://rdrr.io/r/base/trimws.html),
  [`startsWith()`](https://rdrr.io/r/base/startsWith.html),
  [`endsWith()`](https://rdrr.io/r/base/startsWith.html);
  [`paste()`](https://rdrr.io/r/base/paste.html) and
  [`paste0()`](https://rdrr.io/r/base/paste.html) when every argument is
  a string (a number or a logical argument runs on the CPU), with a
  constant `sep` and no `collapse`; and
  [`grepl()`](https://rdrr.io/r/base/grep.html),
  [`sub()`](https://rdrr.io/r/base/grep.html) and
  [`gsub()`](https://rdrr.io/r/base/grep.html) with a constant pattern
  (`ignore.case`, `perl` and `useBytes` run on the CPU)

- Dates and date-times: a `Date` plus or minus a whole number of days,
  lubridate's `year()`, `month()`, `day()`, `hour()`, `minute()`,
  `second()`, `quarter()`, `yday()`, `wday()`, `floor_date()`,
  `ceiling_date()` and `round_date()`, and
  [`as.Date()`](https://rdrr.io/r/base/as.Date.html)/[`as.POSIXct()`](https://rdrr.io/r/base/as.POSIXlt.html),
  for the arguments the GPU supports; other forms run on the CPU

- A factor column compared with a label (`f == "a"`,
  `f %in% c("a", "b")`, and `<`, `>`, ... for an ordered factor), and
  [`as.character()`](https://rdrr.io/r/base/character.html),
  [`as.integer()`](https://rdrr.io/r/base/integer.html) or
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html) of a factor
  column

- `if_else(cond, yes, no, missing = NULL)` – strict-but-promoting type
  unification across `yes`/`no`/`missing` (int/double/logical mix
  freely, like dplyr's vctrs-based common type; `STRING` only unifies
  with `STRING`). An `NA` condition produces `NA` (or `missing`, if
  supplied) in the output, as in dplyr (this is not the same as treating
  an `NA` condition like `FALSE`)

- `case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` –
  first-match-wins; an `NA` condition is treated as no match (falls
  through to the next clause, or `.default`); unmatched rows become a
  typed `NA` when `.default` isn't supplied; the same value-type
  unification rule as
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)
  applies across every `val_i` plus `.default`. A call with `.ptype=`,
  `.size=` or `.unmatched=` runs on the CPU.

- `coalesce(...)` – first non-NA value per row, left to right, across
  any number of columns/scalars; same value-type unification rule as
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)

- `na_if(x, y)` – `x` with `NA` wherever `x == y`; the output always
  keeps `x`'s own type (unlike
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)/[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html),
  `y` is never promoted into the result, only compared against)

- `case_match(.x, old_values ~ new_value, ..., .default = val)` –
  matches `.x` by value against each clause's (possibly multi-element)
  left-hand side, first match wins; `NA` on a clause's left-hand side
  matches `.x`'s own `NA` elements (base R's `NA %in% NA` is `TRUE`);
  unmatched rows become `.default` (typed `NA` if not supplied), with
  the same value-type unification rule as
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
  across every `new_value` plus `.default`. dplyr 1.2 deprecates it in
  favor of
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)/[`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html);
  it still runs on the GPU. A call with `.ptype=` runs on the CPU.

- `recode(.x, old = new, ..., .default =, .missing =)` – on the GPU only
  for a bare character/integer/double column `.x` and scalar literal
  replacements of one shared type
  ([`dplyr::recode`](https://dplyr.tidyverse.org/reference/recode.html),
  including `!!!` lookups); duplicate keys, factor `.x`,
  [`recode_factor()`](https://dplyr.tidyverse.org/reference/recode.html),
  and a result type different from `.x` without `.default` run on the
  CPU.

- `recode_values(x, old_values ~ new_value, ..., default = val)` – the
  non-deprecated equivalent of
  [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)
  above (the same semantics, `default` instead of `.default`). The
  `from=`/`to=` lookup-table form and `unmatched=`/`ptype=` run on the
  CPU.

- `replace_values(x, old_values ~ new_value, ...)` – like
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
  but type stable: the output always keeps `x`'s own type (every
  `new_value` is cast to it, never unified/promoted), and unmatched rows
  keep their original `x` value (there is no `default=`). The
  `from=`/`to=` form runs on the CPU.

- `replace_when(x, cond1 ~ val1, cond2 ~ val2, ...)` – like
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
  but with the type-stable, "unmatched rows keep `x`" behavior of
  [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)
  (an `NA` condition is treated as no match, as in
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html))

- `when_all(..., na_rm = FALSE)` / `when_any(..., na_rm = FALSE)` –
  elementwise `&`/`|` generalized to any number of logical inputs
  (`when_any(x, y, z)` is `x | y | z`). `na_rm = TRUE` treats `NA` as
  each operator's identity element (`FALSE` for
  [`when_any()`](https://dplyr.tidyverse.org/reference/when-any-all.html),
  `TRUE` for
  [`when_all()`](https://dplyr.tidyverse.org/reference/when-any-all.html))
  rather than propagating it – e.g. `when_any(NA, NA, na_rm = TRUE)` is
  `FALSE`, not `NA`. A call with `size=` runs on the CPU.

- Plain column copies (`mutate(y = x)`)

- A later dot in the same
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) call
  referencing an earlier dot's output column
  (`mutate(a = x + y, b = a * 2)`), also when it is a window function's
  output (`mutate(a = cumsum(x), b = lag(a))`)

- The `.data`/`.env` pronouns, and ordinary R symbols (a bare name that
  matches a column resolves to that column; otherwise it is looked up as
  an R variable: columns shadow the environment, as in dplyr)

- Window functions, on ungrouped and grouped
  ([`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  tables, nested to any depth with any of the above
  (`mutate(y = x - lag(x))`, `mutate(z = lag(cumsum(x)))`,
  `mutate(centered = x - mean(x))`):

  - [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
    (sequential position) / `row_number(x)` (ranked position, ties
    broken by original position) – per-group under
    [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)

  - `lag(x, n = 1, default = NA, order_by = NULL)` / `lead(...)` – a
    shift that keeps the row order; `order_by=` (a bare column,
    `desc(column)`, or `-column`) makes the shift follow that ordering
    instead of the table's/group's current row order

  - [`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)/[`cumprod()`](https://rdrr.io/r/base/cumsum.html)
    – with R's `NA` propagation (`cumsum(c(1, 2, NA, 4))` is
    `1, 3, NA, NA`: once an NA is seen, every later cumulative value in
    that group is NA too). The
    [`cumsum()`](https://rdrr.io/r/base/cumsum.html) of an integer or
    logical is an integer that is `NA` from the first overflow on (with
    R's warning), as in R

  - [`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html)
    (`cumsum(x) / row_number()`, accumulated in double),
    [`cumall()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumany()`](https://dplyr.tidyverse.org/reference/cumall.html),
    and
    [`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html)
    of one column (a group id that increases whenever the value changes;
    adjacent `NA`s count as unchanged)

  - [`min_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`dense_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)/[`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    – ties and `NA` as in dplyr (`NA` input -\> `NA` rank;
    [`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    is `(min_rank(x) - 1) / (n_non_NA - 1)`, `NaN` for a group with
    exactly one non-`NA` value, as dplyr's `NaN`, not `NA`;
    [`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    is `rank(x, ties = "max") / n_non_NA`)

  - [`roll_mean()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    [`roll_sum()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    [`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
    and
    [`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    and data.table's `frollmean()`, `frollsum()`, `frollmin()`,
    `frollmax()` and `frollsd()` with their default `align`, `fill`,
    `adaptive`, `partial` and `give.names`;
    [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
    runs on the CPU.
    [`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md)
    of an integer, logical, `Date` or `POSIXct` column keep its type,
    where dplyr returns a double

  - [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`n()`](https://dplyr.tidyverse.org/reference/context.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/
    [`median()`](https://rdrr.io/r/stats/median.html)/[`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
    – one aggregate broadcast to every row of its group (or the whole
    table, ungrouped);
    [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`median()`](https://rdrr.io/r/stats/median.html)
    follow R's `na.rm = FALSE` default (any `NA` in the group makes that
    group's result `NA`);
    [`n()`](https://dplyr.tidyverse.org/reference/context.html) always
    counts every row regardless of `NA`s, and
    [`n_distinct()`](https://dplyr.tidyverse.org/reference/n_distinct.html)
    counts `NA` as a value.
    [`median()`](https://rdrr.io/r/stats/median.html) runs on the GPU
    for a double, `Date` or `POSIXct` column (of an integer or logical
    one, on the CPU), and
    [`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)
    of a string column under the collation rule of the comparisons
    above. On the GPU, [`mean()`](https://rdrr.io/r/base/mean.html) and
    [`median()`](https://rdrr.io/r/stats/median.html) of a `Date` are
    floored to whole days with a `cuplyr_warning_fractional_date`
    warning, where dplyr keeps the fractional `Date`.
    [`sum()`](https://rdrr.io/r/base/sum.html) of an integer or logical
    is an integer, as in dplyr. When a sum leaves the integer range
    dplyr returns a double (for the whole column, and for any expression
    on it): the call is then rerun by dplyr on the CPU. A lazy table
    can't plan for that (the type depends on the data), so a lazy
    [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) with
    an integer [`sum()`](https://rdrr.io/r/base/sum.html) runs at the
    verb like an eager one (a plan break first runs the pending steps)
    and returns a lazy table with nothing pending; a logical
    [`sum()`](https://rdrr.io/r/base/sum.html) stays in the plan (it
    overflows only past 2^31 `TRUE` rows, with a
    `cuplyr_error_sum_overflow` at
    [`collect()`](https://dplyr.tidyverse.org/reference/compute.html))

  - `first(x, order_by = NULL)` / `last(x, order_by = NULL)` /
    `nth(x, n, order_by = NULL)` – the first/last/`n`th row of the group
    in its current row order, or in `order_by`'s order when supplied;
    negative `n` counts from the end (`nth(x, -1)` == `last(x)`); an
    out-of-range `n` gives `NA`. A call with `default=` or `na_rm=` runs
    on the CPU.

  - `ntile(x = row_number(), n)`

  - `order_by(order_by, call)` / `with_order(order_by, fun, x, ...)` –
    as in dplyr, only the first argument of `call` is put in
    `order_by`'s order (stable, `NA` last); the other arguments are
    evaluated in row order, and the call stays on the GPU only when each
    of them is a non-`NULL` scalar that reads no column. On the GPU: a
    window call wrapped directly
    ([`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)/
    [`cumprod()`](https://rdrr.io/r/base/cumsum.html)/[`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/[`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/`roll_*()`
    except
    [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    and
    [`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumall()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumany()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html)),
    the rank functions that depend only on the values
    ([`min_rank()`](https://dplyr.tidyverse.org/reference/row_number.html),
    [`dense_rank()`](https://dplyr.tidyverse.org/reference/row_number.html),
    ... are unaffected by the order), and an elementwise call whose
    other arguments are scalars (`order_by(o, cumsum(x) * 2)` is
    `cumsum(x) * 2`: the window inside the first argument runs in row
    order, as in dplyr). Everything else runs the whole call on the CPU,
    which gives dplyr's value or error:
    [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    an elementwise call with another column argument
    (`order_by(o, cumsum(x) + lag(x))`), a `NULL` or empty extra
    argument (`lag(x, default = NULL)`), `row_number(x)`/`ntile(x, n)`
    (ties broken by the order),
    [`first()`](https://dplyr.tidyverse.org/reference/nth.html)/[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html),
    aggregates, a nested
    [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)
    as the wrapped call itself (`order_by(o, order_by(i, cumsum(x)))`;
    nested inside the first argument it stays on the GPU), and shapes
    dplyr rejects (`order_by(o, row_number())`)

  Every one of these is evaluated within each
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  group when `.data` is grouped, as in dplyr: the rows keep their order
  in the output (unlike grouped
  [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)).

### Column replacement behavior

When the output column name matches an existing column, the existing
column is replaced in-place (preserving column order). For example,
`mutate(x = x + 1)` will modify `x` rather than creating a duplicate.

### Result type (type promotion)

The output column's type follows from the operation and its inputs (the
column types and, for a column-scalar expression, the R literal's own
type), as in R:

- A plain column copy (`mutate(y = x)`) preserves the source column's
  type exactly.

- Arithmetic on a string column is an error, as in R. Arithmetic on a
  factor column runs in dplyr on the CPU (dplyr warns and gives `NA`). A
  `Date` plus or minus a whole number of days stays on the GPU; other
  date and date-time arithmetic (`t + 3600` on a `POSIXct`, the
  difference of two dates, ...) runs in dplyr on the CPU.

- `/` and `^` always give a double, as in R (`1L / 2L` and `2L ^ 2L` are
  both doubles).

- `+`, `-`, `*`, `%%`, `%/%` keep integers: the result is an integer
  when every input column and R literal involved is integer or logical
  (`TRUE + TRUE` is `2L`, as in R). Like R, an integer `+`, `-` or `*`
  whose result leaves `.Machine$integer.max` is `NA` with the warning
  "NAs produced by integer overflow" (once per call; at
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) for
  a lazy table), and `%/%`/`%%` by `0L` are `NA`. If any input is a
  double, the result is a double; otherwise, if any input is an
  integer64, the result is an integer64. A bare R numeric literal (e.g.
  the `1` in `mutate(y = int_col + 1)`) is a *double* in R, so it makes
  the result a double – write `1L` to keep an integer result.

- Comparisons, logic, [`is.na()`](https://rdrr.io/r/base/NA.html),
  [`between()`](https://dplyr.tidyverse.org/reference/between.html),
  `%in%` all produce a logical column.

- [`sqrt()`](https://rdrr.io/r/base/MathFun.html),
  [`log()`](https://rdrr.io/r/base/Log.html),
  [`exp()`](https://rdrr.io/r/base/Log.html),
  [`floor()`](https://rdrr.io/r/base/Round.html),
  [`ceiling()`](https://rdrr.io/r/base/Round.html),
  [`sin()`](https://rdrr.io/r/base/Trig.html),
  [`cos()`](https://rdrr.io/r/base/Trig.html),
  [`tan()`](https://rdrr.io/r/base/Trig.html) always give a double, as
  in R (`floor(1L)` is a double), and so does
  [`round()`](https://rdrr.io/r/base/Round.html) (`typeof(round(1L))` is
  also a double in R, unlike
  [`abs()`](https://rdrr.io/r/base/MathFun.html) below).

- [`abs()`](https://rdrr.io/r/base/MathFun.html) preserves the
  argument's type, except a logical column, which becomes an integer (as
  in R: `typeof(abs(TRUE))` is `"integer"`, not `"logical"`).

- A bare `NA` (untyped) is logical when it is an entire mutate
  expression by itself (`mutate(y = NA)`, as `typeof(NA)`); as an
  operand inside an arithmetic expression (`mutate(y = int_col + NA)`)
  it takes the other operand's type (as in R: `1L + NA` is an integer,
  not a double). A typed NA literal (`NA_real_`, `NA_integer_`, ...)
  always keeps its own type.

### Runs on the CPU

An expression that is not in the list above, or that uses a form the
list sends to the CPU, runs the whole
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) call in
dplyr on the CPU (set `options(cuplyr.fallback = "warn")` to be told
when; see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
So does an argument type for which the GPU would not give R's result,
for example [`cummax()`](https://rdrr.io/r/base/cumsum.html) of a
logical column, or [`sum()`](https://rdrr.io/r/base/sum.html),
[`mean()`](https://rdrr.io/r/base/mean.html),
[`min()`](https://rdrr.io/r/base/Extremes.html) or
[`max()`](https://rdrr.io/r/base/Extremes.html) of an integer64 column.
Among the other cases:

- [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)/[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
  with a general expression as the ordering argument (only a bare
  column, `desc(column)`, or `-column` is on the GPU, with `-` for
  numeric/logical columns and
  [`desc()`](https://dplyr.tidyverse.org/reference/desc.html) of a
  string only under a C collation: `order_by(a + b, cumsum(x))` runs on
  the CPU); see the
  [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)
  item above for what the wrapped call may be on the GPU

- `consecutive_id(a, b, ...)` of more than one column

- Sliding-window functions other than the `roll_*()` family
  (`slider::slide_*()` and similar)

- The arguments named in the items above:
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)'s
  and
  [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)'s
  `.ptype=`/`.size=`/`.unmatched=`,
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)'s
  and
  [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)'s
  `from=`/`to=` and `unmatched=`/`ptype=`, and
  [`when_all()`](https://dplyr.tidyverse.org/reference/when-any-all.html)'s
  and
  [`when_any()`](https://dplyr.tidyverse.org/reference/when-any-all.html)'s
  `size=`

### Performance

GPU arithmetic is highly parallel; memory bandwidth is usually the
limiting factor, not compute.

## See also

[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md)
for filtering rows,
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for selecting columns,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Add a new column
  result <- gpu_mtcars |>
    mutate(kpl = mpg * 0.425) |>
    collect()

  # Modify an existing column
  adjusted <- gpu_mtcars |>
    mutate(mpg = mpg + 5) |>
    collect()

  # Combine two columns
  gpu_cars <- tbl_gpu(cars)
  result <- gpu_cars |>
    mutate(ratio = dist / speed) |>
    collect()

  # Chain multiple mutations
  result <- gpu_mtcars |>
    mutate(power_weight = hp / wt) |>
    mutate(efficiency = mpg * power_weight) |>
    collect()
}
```
