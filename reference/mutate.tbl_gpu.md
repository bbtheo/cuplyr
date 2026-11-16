# Create or modify columns in a GPU table

Adds new columns or modifies existing columns in a GPU table using
expressions, similar to
[`dplyr::mutate()`](https://dplyr.tidyverse.org/reference/mutate.html).
All computations are performed on the GPU for maximum performance.

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

  Optional on-the-fly grouping columns (tidyselect), GPU-native (Phase
  5, task W6): `mutate(df, y = mean(x), .by = g)` behaves identically to
  `df |> group_by(g) |> mutate(y = mean(x)) |> ungroup()` – same
  window-function machinery, same row-order-preservation contract, but
  the *result* is always ungrouped (`.by` never leaves grouping metadata
  behind, unlike
  [`dplyr::group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)).
  Supplying `.by` when `.data` is already grouped (via
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  is an error, matching dplyr exactly.

- .keep:

  One of `"all"` (default), `"used"`, `"unused"`, `"none"`; controls
  which of `.data`'s *other* columns (besides ones `...` creates or
  modifies, and group columns, both of which are always kept) survive in
  the output – see "`.keep` semantics" below.

- .before, .after:

  Optionally, control where new columns should appear, using the same
  tidyselect syntax as
  [`relocate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/relocate.tbl_gpu.md).
  Only genuinely *new* columns are repositioned; a modified existing
  column stays at its original position.

## Value

A `tbl_gpu` object with the new or modified columns. If a column name
already exists, it is replaced. New columns are appended (subject to
`.before`/`.after`), and `.data`'s other columns are kept or dropped per
`.keep`.

## Details

### `.keep` semantics

Verified empirically against dplyr 1.2.1 (dplyr's own docs understate
the exact rule): the columns created or modified by `...` (i.e. every
dot's output column, whether new or an existing name being replaced) and
every group column are **always** kept, regardless of `.keep`. `.keep`
only controls which *other* (untouched, non-group) columns of `.data`
survive alongside them:

- `"all"` (default): every other column.

- `"used"`: other columns that are referenced as an *input* to any of
  `...`'s expressions (union across all dots).

- `"unused"`: other columns NOT referenced as an input to any expression
  – the complement of `"used"`.

- `"none"`: no other columns at all.

Column order: `.data`'s original columns keep their original relative
order (modified columns don't move), and genuinely new columns are
appended at the end (or repositioned via `.before`/`.after`); the final
set is then filtered down according to `.keep`, preserving that same
relative order. This differs from
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html),
whose output order is the literal `...` dot order (see
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)'s
own docs).

### Supported expressions

[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) parses
its arguments through the same expression IR that
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) uses
(see `R/ir.R`), lowered to `cudf::compute_column()` under the hood. A
function is native only when its name resolves, in the expression's
environment, to the base/dplyr/lubridate function itself: a function of
your own with the same name (local, in an enclosing function or global)
runs on the CPU, as dplyr would call it, and `pkg::fn` is the function
of that package. The same holds for a function passed as a value
(`across(x, f)`, a list of functions held in a variable, `!!f`) and for
the functions and lambdas of the scoped verbs
([`mutate_at()`](https://dplyr.tidyverse.org/reference/mutate_all.html),
[`summarise_all()`](https://dplyr.tidyverse.org/reference/summarise_all.html),
[`filter_if()`](https://dplyr.tidyverse.org/reference/filter_all.html),
...): one that is, or calls, a function of yours with a native
function's name runs on the CPU. The IR understands:

- Arithmetic: `+`, `-`, `*`, `/`, `^`, `%%`, `%/%`, unary `-`,
  arbitrarily nested (`(x + y) * z - 1`), and scalar-on-either-side
  (`mutate(y = 10 - x)`)

- Comparisons and logic: `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`,
  `!`, [`xor()`](https://rdrr.io/r/base/Logic.html) – these produce a
  `BOOL8` column, e.g. `mutate(is_big = x > 100)`

- [`is.na()`](https://rdrr.io/r/base/NA.html), `between(x, lo, hi)`,
  `near(x, y, tol = )` (`abs(x - y) < tol`, matching
  [`dplyr::near()`](https://dplyr.tidyverse.org/reference/near.html)'s
  own definition exactly), `%in%` (including a bare `NA` on the
  right-hand side, e.g. `x %in% NA`, matching `NA %in% NA` being `TRUE`
  in base R – not the same as an *empty* set, which is always `FALSE`)

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
  `round(x)`/`round(x, digits)` (banker's rounding, matching R)

- `if_else(cond, yes, no, missing = NULL)` – strict-but-promoting type
  unification across `yes`/`no`/`missing` (int/double/logical mix
  freely, like dplyr's vctrs-based common type; `STRING` only unifies
  with `STRING`). A `NA` condition produces `NA` (or `missing`, if
  supplied) in the output, matching dplyr exactly (this is NOT the same
  as treating a `NA` condition like `FALSE`)

- `case_when(cond1 ~ val1, cond2 ~ val2, ..., .default = val)` –
  first-match-wins; a `NA` condition is treated as no match (falls
  through to the next clause, or `.default`); unmatched rows become a
  typed `NA` when `.default` isn't supplied; the same value-type
  unification rule as
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)
  applies across every `val_i` plus `.default`.
  `.ptype=`/`.size=`/`.unmatched=` are not yet supported.

- `coalesce(...)` – first non-NA value per row, left to right, across
  any number of columns/scalars; same value-type unification rule as
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)

- `na_if(x, y)` – `x` with `NA` wherever `x == y`; the output always
  keeps `x`'s own type exactly (unlike
  [`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)/[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html),
  `y` is never promoted into the result, only compared against)

- `case_match(.x, old_values ~ new_value, ..., .default = val)` –
  matches `.x` by VALUE against each clause's (possibly multi-element)
  left-hand side, first match wins; `NA` on a clause's left-hand side
  matches `.x`'s own `NA` elements (base R's `NA %in% NA` is `TRUE`);
  unmatched rows become `.default` (typed `NA` if not supplied), with
  the same value-type unification rule as
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
  across every `new_value` plus `.default`. Deprecated upstream in dplyr
  1.2 in favor of
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)/[`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
  but still supported here. `.ptype=` is not yet supported.

- `recode(.x, old = new, ..., .default =, .missing =)` – native only for
  a bare character/integer/double column `.x` and scalar literal
  replacements of one shared type
  ([`dplyr::recode`](https://dplyr.tidyverse.org/reference/recode.html),
  including `!!!` lookups); duplicate keys, factor `.x`,
  [`recode_factor()`](https://dplyr.tidyverse.org/reference/recode.html),
  and a result type different from `.x` without `.default` run on the
  CPU.

- `recode_values(x, old_values ~ new_value, ..., default = val)` – the
  non-deprecated equivalent of
  [`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)
  above (identical semantics, `default` instead of `.default`). The
  `from=`/`to=` vector-lookup-table interface and `unmatched=`/`ptype=`
  are not yet supported.

- `replace_values(x, old_values ~ new_value, ...)` – like
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html),
  but TYPE STABLE: the output always keeps `x`'s own type (every
  `new_value` is cast to it, never unified/promoted), and unmatched rows
  keep their original `x` value (there is no `default=`). The
  `from=`/`to=` interface is not yet supported.

- `replace_when(x, cond1 ~ val1, cond2 ~ val2, ...)` – like
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html),
  but boolean-condition clauses feeding the same type-stable-to-`x` /
  "unmatched rows keep `x`" behavior as
  [`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)
  (a `NA` condition is treated as no match, same as
  [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html))

- `when_all(..., na_rm = FALSE)` / `when_any(..., na_rm = FALSE)` –
  elementwise `&`/`|` generalized to any number of logical inputs
  (`when_any(x, y, z)` is `x | y | z`). `na_rm = TRUE` treats `NA` as
  each operator's identity element (`FALSE` for
  [`when_any()`](https://dplyr.tidyverse.org/reference/when-any-all.html),
  `TRUE` for
  [`when_all()`](https://dplyr.tidyverse.org/reference/when-any-all.html))
  rather than propagating it – e.g. `when_any(NA, NA, na_rm = TRUE)` is
  `FALSE`, not `NA`. `size=` is not yet supported.

- Plain column copies (`mutate(y = x)`)

- A later dot in the same
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) call
  referencing an earlier dot's output column
  (`mutate(a = x + y, b = a * 2)`), INCLUDING one that's itself a window
  function's output (`mutate(a = cumsum(x), b = lag(a))`)

- The `.data`/`.env` pronouns, and ordinary R symbols (a bare name that
  matches a column resolves to that column; otherwise it's evaluated as
  an environment variable – "columns shadow the environment")

- Window functions, GPU-native for both ungrouped and grouped
  ([`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html))
  tables, arbitrarily nested with any of the above
  (`mutate(y = x - lag(x))`, `mutate(z = lag(cumsum(x)))`,
  `mutate(centered = x - mean(x))`):

  - [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
    (sequential position) / `row_number(x)` (ranked position, ties
    broken by original position) – per-group under
    [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)

  - `lag(x, n = 1, default = NA, order_by = NULL)` / `lead(...)` –
    row-order-preserving shift; `order_by=` (a bare column,
    `desc(column)`, or `-column`) re-frames the shift to follow that
    ordering instead of the table's/group's current row order

  - [`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)/[`cumprod()`](https://rdrr.io/r/base/cumsum.html)
    – matching R's own NA-propagation (`cumsum(c(1, 2, NA, 4))` is
    `1, 3, NA, NA`: once an NA is seen, every later cumulative value in
    that group is NA too). The
    [`cumsum()`](https://rdrr.io/r/base/cumsum.html) of an integer or
    logical is an integer that is `NA` from the first overflow on (with
    R's warning), as in R

  - [`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html)
    (`cumsum(x) / row_number()`, accumulated in double),
    [`cumall()`](https://dplyr.tidyverse.org/reference/cumall.html)/
    [`cumany()`](https://dplyr.tidyverse.org/reference/cumall.html)
    (tri-state-encoded
    [`cummin()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)),
    and
    [`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html)
    (single-column only; a run-length group id that increments whenever
    the input value changes, treating adjacent `NA`s as unchanged) – all
    built from the primitives above

  - [`min_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`dense_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)/[`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    – ties/`NA` handling matches dplyr exactly (`NA` input -\> `NA`
    rank;
    [`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    is `(min_rank(x) - 1) / (n_non_NA - 1)`, `NaN` for a group with
    exactly one non-`NA` value, matching dplyr's own `NaN`, not `NA`;
    [`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html)
    is `rank(x, ties = "max") / n_non_NA`)

  - [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`n()`](https://dplyr.tidyverse.org/reference/context.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)
    – one aggregate broadcast to every row of its group (or the whole
    table, ungrouped);
    [`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)
    match R's own `na.rm = FALSE` default (ANY `NA` in the group makes
    the WHOLE group's result `NA`);
    [`n()`](https://dplyr.tidyverse.org/reference/context.html) always
    counts every row regardless of `NA`s.
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
    out-of-range `n` gives `NA`; `default=`/`na_rm=` are not yet
    supported

  - `ntile(x = row_number(), n)` – desugars into
    [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)/
    [`sum()`](https://rdrr.io/r/base/sum.html), so it's native as a
    consequence of those being native

  - `order_by(order_by, call)` / `with_order(order_by, fun, x, ...)` –
    like dplyr, only the FIRST argument of `call` is put in `order_by`'s
    order (stable, `NA` last); the other arguments are evaluated in row
    order, and the call stays native only when each of them is a
    non-`NULL` scalar that reads no column. Native: a window call
    wrapped directly
    ([`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/[`cummin()`](https://rdrr.io/r/base/cumsum.html)/
    [`cumprod()`](https://rdrr.io/r/base/cumsum.html)/[`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/[`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/`roll_*()`
    except
    [`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
    and the calls cuplyr desugars into several windows,
    [`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumall()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`cumany()`](https://dplyr.tidyverse.org/reference/cumall.html)/[`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html),
    each part in `order_by`'s order), the value-only rank functions
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
    nested inside the first argument it stays native), and shapes dplyr
    rejects (`order_by(o, row_number())`)

  Every one of these is evaluated within each
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  group when `.data` is grouped (matching `dplyr`'s own grouped-mutate
  semantics: row order is always preserved in the output, unlike grouped
  [`slice()`](https://dplyr.tidyverse.org/reference/slice.html)).

### Column replacement behavior

When the output column name matches an existing column, the existing
column is replaced in-place (preserving column order). For example,
`mutate(x = x + 1)` will modify `x` rather than creating a duplicate.

### Result type (type promotion)

The output column's GPU type is inferred from the operation and its
inputs (column type(s) and, for a column-scalar expression, the R
literal's own type), applied in this order:

- A plain column copy (`mutate(y = x)`) preserves the source column's
  type exactly.

- Arithmetic on `STRING`, factor (`DICTIONARY32`), or `Date`/ `POSIXct`
  (`TIMESTAMP_*`) columns is an error: cuplyr does not support
  arithmetic on those types.

- `/` and `^` always promote to `FLOAT64` (matching R/dplyr: `1L / 2L`
  and `2L ^ 2L` are both doubles).

- `+`, `-`, `*`, `%%`, `%/%` preserve integer types: the result is
  `INT32` when every input column and R literal involved is integer or
  logical (`TRUE + TRUE` is `2L`, matching R). Like R, an integer `+`,
  `-` or `*` whose result leaves `.Machine$integer.max` is `NA` with the
  warning "NAs produced by integer overflow" (once per call; at
  [`collect()`](https://dplyr.tidyverse.org/reference/compute.html) for
  a lazy table), and `%/%`/`%%` by `0L` are `NA`. If any input is
  `FLOAT64`/`FLOAT32`, the result is `FLOAT64`; otherwise if any input
  is `INT64`, the result is `INT64`. A bare R numeric literal (e.g. the
  `1` in `mutate(y = int_col + 1)`) is a *double* in R, so it promotes
  the result to `FLOAT64` – write `1L` to keep an integer result.

- Comparisons, logic, [`is.na()`](https://rdrr.io/r/base/NA.html),
  [`between()`](https://dplyr.tidyverse.org/reference/between.html),
  `%in%` all produce `BOOL8`.

- [`sqrt()`](https://rdrr.io/r/base/MathFun.html),
  [`log()`](https://rdrr.io/r/base/Log.html),
  [`exp()`](https://rdrr.io/r/base/Log.html),
  [`floor()`](https://rdrr.io/r/base/Round.html),
  [`ceiling()`](https://rdrr.io/r/base/Round.html),
  [`sin()`](https://rdrr.io/r/base/Trig.html),
  [`cos()`](https://rdrr.io/r/base/Trig.html),
  [`tan()`](https://rdrr.io/r/base/Trig.html) always promote to
  `FLOAT64` (matching R, e.g. `floor(1L)` is a double) – as does
  [`round()`](https://rdrr.io/r/base/Round.html), for the same reason
  (`typeof(round(1L))` is also a double in R, unlike
  [`abs()`](https://rdrr.io/r/base/MathFun.html) below).

- [`abs()`](https://rdrr.io/r/base/MathFun.html) preserves the
  argument's type, except a logical column, which promotes to `INT32`
  (matching R: `typeof(abs(TRUE))` is `"integer"`, not `"logical"`).

- A bare `NA` (untyped) is `BOOL8` when it's an entire mutate expression
  by itself (`mutate(y = NA)`, matching `typeof(NA)`); as an operand
  inside an arithmetic expression (`mutate(y = int_col + NA)`) it's a
  type no-op instead, adopting the other operand's type (matching R:
  `1L + NA` is an integer, not a double). A typed NA literal
  (`NA_real_`, `NA_integer_`, ...) always keeps its own type.

### Not yet supported

- [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)/[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
  with a general expression as the ordering argument (only a bare
  column, `desc(column)`, or `-column` is native, with `-` for
  numeric/logical columns and
  [`desc()`](https://dplyr.tidyverse.org/reference/desc.html) of a
  string only under a C collation: `order_by(a + b, cumsum(x))` runs on
  the CPU); see the
  [`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)
  item above for what the wrapped call may be natively

- Multi-column `consecutive_id(a, b, ...)` – single-column only

- Rolling/sliding-window functions (`slider::slide_*()` and similar) –
  planned for a later phase

- String manipulation (concatenation, case conversion, substr, regex,
  ...) – only string *comparison* is supported

- [`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)'s/[`case_match()`](https://dplyr.tidyverse.org/reference/case_match.html)'s
  `.ptype=`/`.size=`/`.unmatched=` arguments,
  [`recode_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)'s/[`replace_values()`](https://dplyr.tidyverse.org/reference/recode-and-replace-values.html)'s
  `from=`/`to=` vector-lookup-table interface and `unmatched=`/`ptype=`,
  and
  [`when_all()`](https://dplyr.tidyverse.org/reference/when-any-all.html)'s/[`when_any()`](https://dplyr.tidyverse.org/reference/when-any-all.html)'s
  `size=` – planned for a later phase

An expression shape the IR doesn't recognize at all (no window function,
no aggregate, not one of the arithmetic/comparison/string-comparison
shapes listed above) is a hard error (unlike
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html), which
falls back to CPU evaluation for some unrecognized shapes) – a general
CPU fallback for
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) covering
every unrecognized shape (not just window/ aggregate ones) is planned
for a later phase.

### Performance

GPU arithmetic operations are highly vectorized and can process billions
of elements per second. Memory bandwidth is typically the limiting
factor, not compute.

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
