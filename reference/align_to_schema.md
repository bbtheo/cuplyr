# Align a table to a target schema

Align a table to a target schema

## Usage

``` r
align_to_schema(tbl, target_schema)
```

## Arguments

- tbl:

  A tbl_gpu object

- target_schema:

  List with `names`, `types`, and optionally `factor_levels` (the
  unified level set for any `DICTIONARY32` column, see
  [`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md))

## Value

A tbl_gpu aligned to the target schema
