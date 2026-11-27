# Compute a join's output column names/types/origin, mirroring `dplyr:::join_cols()` exactly

Ported directly from dplyr 1.2.1's own `join_cols()` (verified via
`body(dplyr:::join_cols)`), specialized to this package's
always-resolved `keep` (never `NULL` here – every join verb calls
[`resolve_join_keep()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_keep.md)
first, which returns `FALSE` for both the real `NULL` default AND a
legal explicit `FALSE`; see its own roxygen for why those two collapse
to one value here).

## Usage

``` r
build_join_output_info(
  left_schema,
  right_schema,
  join_spec,
  suffix = c(".x", ".y"),
  keep = FALSE
)
```

## Details

`keep = TRUE`: every column on both sides is suffixed uniformly via
[`add_suffixes()`](https://bbtheo.github.io/cuplyr/reference/add_suffixes.md)
(no column is "ignored"), and nothing is dropped from the right table –
reproduces e.g. `join_by(a)` + `keep = TRUE` giving `a.x`/`a.y` for BOTH
copies of the shared key.

`keep != TRUE` (the `NULL`-default/legal-`FALSE` case): PER CONDITION,
mirroring `dplyr:::join_cols()`'s own `is_null(keep)` branch exactly
(this branch, not a whole-join TRUE/FALSE collapse, governs the real
`keep = NULL` default in dplyr 1.2.1: `join_by(c == d, a >= b)` drops
`d` but keeps both `a` and `b`). `is_eq <- join_spec$op == "=="` selects
which condition PAIRS collapse to one coalesced column (the join KEY
columns for those pairs specifically are excluded from suffixing – they
keep their bare original name – and the right table's key column for
that pair is dropped from the output) versus which pairs are kept
verbatim on BOTH sides, suffixed like any other colliding column (a
genuinely non-equi condition, e.g. `join_by(a >= b)`, has no single
shared key value to collapse to – there IS no "coalesced" column for
it). An all-equi spec (`is_eq` all `TRUE`) collapses to the
`keep = FALSE` output exactly; an all-non-equi spec (`is_eq` all
`FALSE`) collapses to byte-identical output to the `keep = TRUE` branch
above (every column suffixed against the full cross set, nothing
dropped) – verified algebraically: `x_ignore`/`y_ignore` are both
`character(0)`, so `y_aux` becomes all of `right_names` and every column
check is unconditional, exactly reproducing
`add_suffixes(left_names, right_names, ...)`/`add_suffixes(right_names, left_names, ...)`.
Only a MIXED equi/non-equi spec actually needs this per-condition
machinery to diverge from either extreme.
