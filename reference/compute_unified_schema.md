# Compute unified schema from multiple tables

Phase 11 L4: shares
[`resolve_common_col_type()`](https://bbtheo.github.io/cuplyr/reference/resolve_common_col_type.md)'s
type ladder with the set-op verbs (R/sets.R) – the old
`promote_types()`'s unconditional "STRING is widest" rule is RETIRED; a
numeric-vs-character (or any other non-ladder) mismatch is now a hard
error matching real dplyr/vctrs' `vctrs_error_incompatible_type` text
([`abort_bind_rows_incompatible()`](https://bbtheo.github.io/cuplyr/reference/abort_bind_rows_incompatible.md)),
not a silent STRING coercion.

## Usage

``` r
compute_unified_schema(
  tables,
  positions = seq_along(tables),
  call = rlang::caller_env()
)
```

## Arguments

- tables:

  List of tbl_gpu objects

- positions:

  Each table's position among the user's inputs (for the error's `..N`
  labels)

- call:

  The user's call, for errors

## Value

List with `names`, `types`, and `factor_levels` (`NULL` if no column in
the union ends up `DICTIONARY32`) for the unified schema

## Details

Factor columns (Phase 11 L2, Bug 2/3, unchanged by L4): a column present
as `DICTIONARY32` in every table that has it unifies to `DICTIONARY32`
with `factor_levels` set to the LEVEL UNION across every occurrence
([`base::union()`](https://rdrr.io/r/base/sets.html), first-occurrence
order – verified empirically against
`vctrs::vec_c(factor("a"), factor("b"))` -\> levels `c("a", "b")`). A
column that's `DICTIONARY32` in some tables and `STRING` in at least one
other unifies to plain `STRING` with no `factor_levels` entry at all
(dplyr's own factor + character -\> character contract) –
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md)
decodes each factor occurrence to its labels via `gpu_decode_factor()`
([`cast_column()`](https://bbtheo.github.io/cuplyr/reference/cast_column.md),
below). A factor column against any OTHER type (numeric, timestamp, ...)
is now a hard error too (real dplyr's own behavior – verified
empirically), where it used to silently fall through `promote_types()`'s
STRING-is-widest rule.
