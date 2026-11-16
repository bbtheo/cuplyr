# Resolve the common GPU type for one column across a bind/set operation (Phase 11 L4)

The type-compatibility ladder extracted out of
[`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)
(R/sets.R, Phase 8) so
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)
and the set-op verbs share EXACTLY one promotion rule, instead of
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
having its own looser one (the old `promote_types()`'s unconditional
"STRING is widest" rule, which silently coerced e.g. a numeric column
against a character column of the same name – a real bug: real dplyr's
own
[`vctrs::vec_ptype2()`](https://vctrs.r-lib.org/reference/vec_ptype2.html)-backed
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
has never allowed this, it's a hard `vctrs_error_incompatible_type`, see
[`abort_bind_rows_incompatible()`](https://bbtheo.github.io/cuplyr/reference/abort_bind_rows_incompatible.md)
below).

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
  etc.) is genuinely native as of Phase 11 L3 (`gpu_cast_column()`,
  `src/ops_bind.cpp`, gained the 5 TIMESTAMP\_\* targets) – this ladder
  used to require
  [`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)'s
  own caller to set a `$needs_fallback` flag for exactly this case (no
  native `TIMESTAMP_DAYS` -\> `TIMESTAMP_MICROSECONDS` cast existed
  yet); that flag is retired, this ladder is now unconditionally native
  for both callers.

- Anything else (e.g. numeric vs. character, factor vs. numeric, numeric
  vs. timestamp): `$ok = FALSE` – the caller reports its own
  verb-specific incompatibility error (sets.R's
  `"`x`and`y` are not compatible."` bulleted form vs. bind.R's
  `vctrs`-style
  `"Can't combine ...`"`form, see`abort_bind_rows_incompatible()\`).
