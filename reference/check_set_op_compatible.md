# Check column-set/type compatibility for a set operation, dplyr-style

Mirrors `dplyr:::is_compatible()` (checked against dplyr 1.2.1 + vctrs
`vec_ptype2()`): same column count, same column name set (order
tolerated), and a per-column common type, via
[`resolve_common_col_type()`](https://bbtheo.github.io/cuplyr/reference/resolve_common_col_type.md)
(R/bind.R), the same ladder
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)'s
own
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)
uses. That ladder follows vctrs' `vec_ptype2()`, which refuses a
numeric/date column against a character one (`vec_ptype2(1L, "a")`
errors), so neither verb coerces such a pair to a string.

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

A list with `$target_schema` (unified schema, x's names/order),
`$needs_fallback` (logical: `TRUE` for duplicated column names, or a
`Date` next to a non-UTC `POSIXct`) and `$fallback_reason`.

## Details

Factor columns get a four-way split (as `vec_ptype2()` treats factors,
see inline comments): both-factor with identical levels runs on the GPU
as is (same underlying INT32 codes, safe to compare/concatenate
directly); both-factor with differing levels runs on the GPU through
`gpu_remap_codes()` (`src/ops_bind.cpp`), which remaps each side's codes
onto the level union, as
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
does; factor-vs-character runs on the GPU through `gpu_decode_factor()`,
which decodes the factor side to its labels, as
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
does. Factor vs. anything else (numeric, date, ...) is a genuine vctrs
error, reproduced here directly.

This function adds the column-count/name-set checks and its own
verb-specific ("`x` and `y` are not compatible.", bulleted) error
formatting on top of the shared ladder. A `Date`-vs-`POSIXct` column
promotes to `POSIXct` on the GPU (`gpu_cast_column()`'s timestamp
targets), except next to a non-UTC `POSIXct`, where the `Date` becomes
local midnight of that zone: `$needs_fallback` is then `TRUE`, as it is
for duplicated column names.
