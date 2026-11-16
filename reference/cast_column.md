# Cast a column to a different type

Cast a column to a different type

## Usage

``` r
cast_column(tbl, col_name, target_type, source_levels = NULL)
```

## Arguments

- tbl:

  A tbl_gpu object

- col_name:

  Name of column to cast

- target_type:

  Target type string

- source_levels:

  For a `DICTIONARY32` source column being cast to `STRING` (dplyr's
  factor + character -\> character contract, Bug 3 of Phase 11 L2): this
  column's OWN level vector, decoded via `gpu_decode_factor()` instead
  of `gpu_cast_to_string()`'s numeric-only path (which only ever sees
  the PHYSICAL INT32 column and would silently emit its raw codes as
  strings, e.g. `"2"` instead of `"m"`). `NULL`/unused for every other
  source type.

## Value

A tbl_gpu with the column cast to the target type
