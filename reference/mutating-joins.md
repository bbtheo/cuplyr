# Mutating joins for `tbl_gpu` (`left_join()`, `inner_join()`, `full_join()`, `right_join()`)

Equi and non-equi joins on the GPU, with dplyr's `by=`/`keep=`/
`na_matches=`/`multiple=`/`unmatched=`/`relationship=` semantics. Row
order, key coalescing, suffixing and every cardinality check follow
[`dplyr::left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
and friends. See "Details" for what runs on the GPU and what runs in
dplyr on the CPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
left_join(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL,
  na_matches = "na",
  multiple = "all",
  unmatched = "drop",
  relationship = NULL
)

# S3 method for class 'tbl_gpu'
inner_join(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL,
  na_matches = "na",
  multiple = "all",
  unmatched = "drop",
  relationship = NULL
)

# S3 method for class 'tbl_gpu'
full_join(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL,
  na_matches = "na",
  multiple = "all",
  relationship = NULL
)

# S3 method for class 'tbl_gpu'
right_join(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL,
  na_matches = "na",
  multiple = "all",
  unmatched = "drop",
  relationship = NULL
)
```

## Arguments

- x, y:

  `tbl_gpu` objects to join. If `y` is not a `tbl_gpu`, set
  `copy = TRUE` to upload it automatically (`tbl_gpu(y)`).

- by:

  Join specification: `NULL` (natural join on every common column name),
  an unnamed character vector (`c("a", "b")`, matched by identical name
  on both sides), a named character vector (`c("lx" = "ry")`, for
  differently-named keys), or a
  [`dplyr::join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
  spec. Key columns whose types differ are cast to their common type for
  matching, as in dplyr (integer/double/logical, factor/character,
  Date/POSIXct), and a pair with no common type raises
  `dplyr_error_join_incompatible_type` when the verb is called.
  `by = character()` (deprecated in dplyr 1.1.0, warns once) is a cross
  join with the usual `multiple=`/`relationship=`/ `unmatched=` rules.
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
  conditions run on the GPU: equality (`a`, `a == b`), inequality (`>=`,
  `>`, `<=`, `<`, and the
  [`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/`overlaps()`
  helpers built from them), one `closest()` rolling condition next to
  any number of `==` conditions, and inequalities between two string
  columns. Two `closest()` conditions, `closest()` next to another
  inequality, and factor/logical/Date-vs-POSIXct rolling keys run in
  dplyr on the CPU (see `options(cuplyr.fallback = )` to be told), with
  the same result as dplyr.

- copy:

  Whether to upload a non-`tbl_gpu` `y` via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- suffix:

  Length-2 character vector used to disambiguate colliding non-key
  column names (default `c(".x", ".y")`). As in dplyr, an already-taken
  suffixed name gets suffixed again (`val.y.y`) rather than silently
  colliding. A `suffix` with an empty element (`c("", "")`) that still
  leaves two columns with the same name runs in dplyr on the CPU, which
  gives dplyr's result (the right column replaces the left one, and the
  output has one column fewer).

- ...:

  Not used; dplyr itself requires these to be empty.

- keep:

  Whether to keep both sides' join key columns in the output. `NULL`
  (the default) resolves per condition, as in dplyr: an equi condition's
  key columns coalesce to one (the right-side copy is dropped), while a
  non-equi condition's columns are always kept on both sides (there is
  no single coalesced value for an inequality). Explicit `keep = FALSE`
  is only allowed when every condition is equi (otherwise it is dplyr's
  error); explicit `keep = TRUE` keeps and suffixes every key column on
  both sides.

- na_matches:

  `"na"` (default): an `NA` key matches another `NA` key, like
  [`base::merge()`](https://rdrr.io/r/base/merge.html). `"never"`: `NA`
  keys never match anything, including another `NA`.

- multiple:

  How to resolve a left/right row matching more than one row on the
  other side: `"all"` (default, every match kept), `"any"`, `"first"`,
  or `"last"`, all on the GPU (the undocumented `"error"` and
  `"warning"` run in dplyr on the CPU). The default `"all"` costs
  nothing extra. Under right/full joins a y row that no kept pair uses
  is listed once as unmatched.

- unmatched:

  How to handle rows with no match: `"drop"` (default) or `"error"`
  (raises dplyr's `dplyr_error_join_matches_nothing`/
  `dplyr_error_join_matches_remaining` conditions, including the
  offending row number).
  [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  has no `unmatched=` parameter, as in dplyr (it never drops a row).

- relationship:

  Optional cardinality assertion: `NULL` (default – emits dplyr's own
  many-to-many advisory warning whenever *both* sides have a genuine
  duplicate, for equality joins called directly; never for
  inequality/overlap joins or indirect calls, as in dplyr),
  `"one-to-one"`, `"one-to-many"`, `"many-to-one"`, or `"many-to-many"`
  (silences the advisory warning). A violation raises dplyr's
  `dplyr_error_join_relationship_*` condition, with its class vector and
  first-offending-row bullet.

## Value

A `tbl_gpu` with dplyr's own row/column contract for the given join
type.
[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)'s
row order is x-matched rows (in `x`'s own order) followed by any
unmatched `y` rows appended last.

## Details

### On the GPU

- Equi joins (`join_by(a)`, `by = "col"`, `by = c(a = "b")`): a hash
  join, with the rows put in dplyr's order.

- Non-equi and mixed
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
  conditions (`>=`, `>`, `<=`, `<`, and their
  [`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/`overlaps()`
  forms).

- `na_matches=`, `multiple=`, `unmatched=` and `relationship=`, with
  dplyr's condition classes and message text.

- `join_by(closest(...))` rolling joins: exactly one `closest()`
  condition (`>=`, `>`, `<=`, `<`) plus `==` conditions, on integer,
  double, Date, POSIXct or string rolling keys. Each row of `x` finds
  its closest rows of `y` without building the whole inequality join.
  Every y row tied at the closest value is returned, and
  `multiple=`/`unmatched=`/`relationship=` act on those matches, as in
  dplyr.

- A non-equi condition comparing two string columns: strings compare in
  byte (C locale) order.

### In dplyr on the CPU

These, among others, run in dplyr and the result comes back to the GPU
(with a notice under `options(cuplyr.fallback = "warn")`):

- Two `closest()` conditions, `closest()` next to another inequality,
  and rolling keys that are factors, logicals or a Date against a
  POSIXct.

- A non-equi condition comparing a string with a non-string column.

- Factor keys whose levels differ between `x` and `y`.

- A `Date` key joined to a `POSIXct` key in a zone other than UTC (dplyr
  compares the date as local midnight of that zone).

- `suffix = c("", ...)` when it would still produce a genuine duplicate
  output column name.

- [`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)
  (not one of the four verbs documented here) always runs on the CPU:
  its result holds a list column, which a GPU table can't hold.
