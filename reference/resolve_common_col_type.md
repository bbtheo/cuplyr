# Resolve the common GPU type for one column across a bind/set operation

The type-compatibility ladder shared by
[`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)
(R/sets.R) and
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md),
so the set-op verbs and
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
use EXACTLY one promotion rule. A numeric column against a character
column of the same name is not coerced to STRING: as in real dplyr
([`vctrs::vec_ptype2()`](https://vctrs.r-lib.org/reference/vec_ptype2.html)-backed
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)),
it's a hard `vctrs_error_incompatible_type`, see
[`abort_bind_rows_incompatible()`](https://bbtheo.github.io/cuplyr/reference/abort_bind_rows_incompatible.md)
below.

## Usage

``` r
resolve_common_col_type(
  tx,
  ty,
  fx = FALSE,
  fy = FALSE,
  lvls_x = NULL,
  lvls_y = NULL
)
```

## Arguments

- tx, ty:

  GPU type strings for the two occurrences of this column
  ([`unname()`](https://rdrr.io/r/base/unname.html)d already by the
  caller)

- fx, fy:

  Logical: is this occurrence `DICTIONARY32` (a factor)?

- lvls_x, lvls_y:

  Level vectors for `tx`/`ty` when `fx`/`fy` is `TRUE` (`NULL`
  otherwise)

## Value

A list with `$ok` (logical) and, when `TRUE`, `$type` (promoted type
string) and `$factor_levels` (level vector, or `NULL` when the result
isn't `DICTIONARY32`)

## Details

Four cases, all pinned via oracle-verified `dplyr`/`vctrs` behavior:

- Both factor (`DICTIONARY32`): union of both level sets (or a no-op if
  identical) – `vctrs::vec_c(factor, factor)`'s contract.

- Exactly one factor, the other `STRING`: decodes to `STRING` (dplyr's
  factor + character -\> character contract). Exactly one factor, the
  other anything else (numeric, timestamp, ...): a hard incompatibility
  (`$ok = FALSE`) – real
  [`vctrs::vec_ptype2()`](https://vctrs.r-lib.org/reference/vec_ptype2.html)
  refuses this too.

- Identical types: a no-op.

- Both in the numeric ladder (`BOOL8 < INT32 < INT64 < FLOAT64`, except
  that INT64 (integer64) and FLOAT64 have no common type) or both in the
  timestamp ladder (`TIMESTAMP_DAYS < ... < TIMESTAMP_NANOSECONDS`):
  promotes to the wider member. The timestamp cast (Date \<-\> POSIXct,
  etc.) is native (`gpu_cast_column()`, `src/ops_bind.cpp`, handles the
  TIMESTAMP\_\* targets), so this ladder is native for both callers.

- Anything else (e.g. numeric vs. character, factor vs. numeric, numeric
  vs. timestamp): `$ok = FALSE` – the caller reports its own
  verb-specific incompatibility error (sets.R's
  `"`x`and`y` are not compatible."` bulleted form vs. bind.R's
  `vctrs`-style
  `"Can't combine ...`"`form, see`abort_bind_rows_incompatible()\`).
