# Mutating joins for `tbl_gpu` (`left_join()`, `inner_join()`, `full_join()`, `right_join()`)

GPU-native equi and non-equi joins mirroring
[`dplyr::left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)'s
own `by=`/`keep=`/`na_matches=`/`multiple=`/`unmatched=`/`relationship=`
semantics (Phase 7). Row order, key coalescing, suffixing, and every
cardinality check reproduce dplyr's own contract exactly – see "Details"
below for what runs natively on the GPU versus what transparently falls
back to CPU (`dplyr`) evaluation.

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
  supports both equi conditions (`a`, `a == b`) and non-equi conditions
  (`>=`, `>`, `<=`, `<`, and the
  [`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/`overlaps()`
  helpers that pre-expand to pairs of them) – all GPU-native (Phase 7
  J6/J7). A `closest()` rolling condition (one per call, next to any
  number of `==` conditions) and non-equi conditions comparing two
  `STRING` columns are GPU-native too (F24). Two `closest()` conditions,
  `closest()` next to another inequality, and
  factor/logical/Date-vs-POSIXct rolling keys transparently fall back to
  CPU evaluation (a `cuplyr.fallback` notification fires; see
  `options(cuplyr.fallback = )`), still with a result that matches
  `dplyr` exactly.

- copy:

  Whether to upload a non-`tbl_gpu` `y` via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- suffix:

  Length-2 character vector used to disambiguate colliding non-key
  column names (default `c(".x", ".y")`). Ported from
  `dplyr:::add_suffixes()`: an already-taken suffixed name gets suffixed
  again (`val.y.y`) rather than silently colliding. `suffix = c("", "")`
  (or any element set to `""`) that still collides with a genuine
  duplicate name reproduces dplyr's name-keyed column-overwrite
  semantics via a CPU fallback – this package's fixed-column-count
  gather has no native way to express "the right column's data silently
  replaces the left's, and the output has one fewer column than the
  naive merge".

- ...:

  Not used; dplyr itself requires these to be empty.

- keep:

  Whether to keep both sides' join key columns in the output. `NULL`
  (the default) resolves *per condition*, exactly like
  `dplyr:::join_cols()`: an equi condition's key columns coalesce to one
  (the right-side copy is dropped), while a non-equi condition's columns
  are always kept on both sides (there is no single "coalesced" value
  for an inequality). Explicit `keep = FALSE` is only legal when every
  condition is equi (errors with dplyr's own text otherwise); explicit
  `keep = TRUE` keeps and suffixes every column on both sides uniformly.

- na_matches:

  `"na"` (default): an `NA` key matches another `NA` key, like
  [`base::merge()`](https://rdrr.io/r/base/merge.html). `"never"`: `NA`
  keys never match anything, including another `NA`
  (`cudf::null_equality::UNEQUAL` on the GPU).

- multiple:

  How to resolve a left/right row matching more than one row on the
  other side: `"all"` (default, every match kept), `"any"`, `"first"`,
  or `"last"` (the undocumented `"error"` and `"warning"` run on the CPU
  fallback). Filtering runs device-side against the already-sorted join
  maps (Phase 7 J5) – `"all"`, the default used by every performance
  benchmark, costs nothing extra. Under right/full joins a y row that no
  kept pair uses is listed once as unmatched.

- unmatched:

  How to handle rows with no match: `"drop"` (default) or `"error"`
  (raises dplyr's exact `dplyr_error_join_matches_nothing`/
  `dplyr_error_join_matches_remaining` conditions, including the
  offending row number).
  [`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
  has no `unmatched=` parameter at all, matching dplyr's own signature
  (it never drops a row).

- relationship:

  Optional cardinality assertion: `NULL` (default – emits dplyr's own
  many-to-many advisory warning whenever *both* sides have a genuine
  duplicate, for equality joins called directly; never for
  inequality/overlap joins or indirect calls, as in dplyr),
  `"one-to-one"`, `"one-to-many"`, `"many-to-one"`, or `"many-to-many"`
  (silences the advisory warning). A violation raises dplyr's exact
  `dplyr_error_join_relationship_*` condition, including the class
  vector and first-offending-row bullet. Known limitation: with the
  default `relationship`, a lazy pipeline whose
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) the
  optimizer pushes below the join may not emit the advisory many-to-many
  warning, or may report different row numbers in it (the rows returned
  are identical).

## Value

A `tbl_gpu` with dplyr's own row/column contract for the given join
type.
[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)'s
row order is x-matched rows (in `x`'s own order) followed by any
unmatched `y` rows appended last.

## Details

### GPU-native coverage (Phase 7)

- Equi joins (`join_by(a)`, `by = "col"`, `by = c(a = "b")`): a hash
  join
  (`cudf::inner_join()`/[`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)),
  stable-sorted to reproduce dplyr's row order (Phase 7 J1/J4/J6).

- Non-equi/mixed
  [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
  conditions (`>=`, `>`, `<=`, `<`, and their
  [`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/`overlaps()`
  derivations): `cudf::mixed_join()` when at least one equi condition is
  present, `cudf::conditional_join()` otherwise (Phase 7 J7).

- `na_matches=`, `multiple=`, `unmatched=`, `relationship=` are fully
  GPU-native/checked, with dplyr's exact condition classes and message
  text (Phase 7 J1/J5).

- `join_by(closest(...))` rolling joins (F24): exactly one `closest()`
  condition (`>=`, `>`, `<=`, `<`) plus `==` conditions, on integer,
  double, Date or POSIXct (or `STRING`) rolling keys: y is sorted by
  (equi keys, rolling key) and each x row finds its closest rows by
  binary search (`gpu_rolling_join()`), so a rolling join never
  materialises the inequality join. Every y row tied at the closest
  value is returned, and `multiple=`/`unmatched=`/`relationship=` act on
  those matches, as in dplyr.

- A non-equi condition comparing two `STRING` columns (F24): both sides
  are replaced by INT32 codes of one shared bytewise (C-locale) order
  before the AST evaluator runs.

### Falls back to CPU (transparent, notified)

- Two `closest()` conditions, `closest()` next to another inequality,
  and rolling keys that are factors, logicals or a Date against a
  POSIXct.

- A non-equi condition comparing a `STRING` with a non-`STRING` column.

- `suffix = c("", ...)` when it would still produce a genuine duplicate
  output column name.

- [`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)
  stays CPU-only entirely (needs list-column support, Phase 11) – it is
  not one of the four verbs documented here.
