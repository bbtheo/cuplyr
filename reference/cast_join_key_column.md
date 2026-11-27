# Cast one join key column to its common type

A factor key meeting a character key is decoded to its labels with
`gpu_decode_factor()` – never `gpu_cast_column()`, which would only see
the physical INT32 codes. Everything else is a numeric/timestamp ladder
step handled by `gpu_cast_column()`. The column is replaced in place.

## Usage

``` r
cast_join_key_column(ptr, idx, from_type, to_type, levels)
```
