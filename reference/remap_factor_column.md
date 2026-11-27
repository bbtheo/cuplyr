# Remap a factor column's physical codes onto a different level ordering

Used by
[`align_to_schema()`](https://bbtheo.github.io/cuplyr/reference/align_to_schema.md)
when unifying two `DICTIONARY32` columns under the same name whose level
VECTORS differ: dplyr's own
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)/[`vctrs::vec_c()`](https://vctrs.r-lib.org/reference/vec_c.html)
contract for two factors is a LEVEL UNION, not a raw code concatenation
– naively concatenating this table's codes as-is would silently
misinterpret them once the union's code assignment differs from this
table's own. `match(current_levels, target_levels)` gives, for each of
this table's OLD (1-based) levels in order, its NEW 1-based code in
`target_levels` (always found – `target_levels` is always a superset,
see
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md));
`gpu_remap_codes()` (`src/ops_bind.cpp`) gathers using the column's own
(null-safe) codes as the index.

## Usage

``` r
remap_factor_column(tbl, col_name, current_levels, target_levels)
```

## Arguments

- tbl:

  A tbl_gpu object

- col_name:

  Name of the factor column to remap

- current_levels:

  This table's own level vector for `col_name`

- target_levels:

  The unified level vector to remap onto

## Value

A tbl_gpu with `col_name`'s codes remapped (schema/types/groups
otherwise unchanged)
