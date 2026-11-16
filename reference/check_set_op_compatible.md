# Check column-set/type compatibility for a set operation, dplyr-style

Mirrors `dplyr:::is_compatible()` (verified empirically against dplyr
1.2.1 + vctrs `vec_ptype2()`): same column count, same column name SET
(order tolerated), and a per-column common type, via
[`resolve_common_col_type()`](https://bbtheo.github.io/cuplyr/reference/resolve_common_col_type.md)
(R/bind.R). As of Phase 11 L4 this is EXACTLY the same ladder
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
own
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)
uses – the two used to diverge (this function reimplemented vctrs'
stricter common-type ladder from scratch because
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
old `promote_types()` treated `STRING` as a universal widest type,
silently coercing a numeric/date column against a character one, which
real vctrs' `vec_ptype2()` REFUSES – verified: `vec_ptype2(1L, "a")`
errors). `promote_types()` is gone; both verbs now share one strict
ladder.

## Usage

``` r
check_set_op_compatible(x_schema, y_schema)
```

## Arguments

- x_schema, y_schema:

  Schemas
  ([`current_schema()`](https://bbtheo.github.io/cuplyr/reference/current_schema.md))
  of `x`/`y`.

## Value

A list with `$target_schema` (unified schema, x's names/order) and
`$needs_fallback` (logical: `TRUE` only for duplicated column names).

## Details

Factor columns get a four-way split (verified against real
`vec_ptype2()` behavior for factors, see inline comments): both-factor
with IDENTICAL levels is native (same underlying INT32 codes, safe to
compare/concatenate directly); both-factor with DIFFERING levels is ALSO
native as of Phase 11 L2 (`gpu_remap_codes()`, `src/ops_bind.cpp`,
remaps each side's codes onto the level UNION, mirroring
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
own Bug 2 fix); factor-vs-character is ALSO native as of Phase 11 L2
(`gpu_decode_factor()` decodes the factor side to its labels, mirroring
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
own Bug 3 fix) – both used to route through `$needs_fallback` before
these two GPU primitives existed. Factor vs. anything else (numeric,
date, ...) is a genuine vctrs error, reproduced here directly.

Phase 11 L4: the per-column ladder itself is now
[`resolve_common_col_type()`](https://bbtheo.github.io/cuplyr/reference/resolve_common_col_type.md)
(R/bind.R), shared verbatim with
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
own
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)
– this function now only adds the column-count/name-SET checks and its
own verb-specific ("`x` and `y` are not compatible.", bulleted) error
formatting on top. The `Date`-vs-`POSIXct` timestamp promotion that used
to require `$needs_fallback` (no native `TIMESTAMP_DAYS` -\>
`TIMESTAMP_MICROSECONDS` cast existed yet) is now unconditionally native
as of Phase 11 L3 (`gpu_cast_column()` gained the 5 timestamp targets) –
`$needs_fallback` is now ALWAYS `FALSE` and kept only for call-site
compatibility (every caller in this file still checks it).
