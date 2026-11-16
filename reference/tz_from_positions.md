# Rebuild `schema$tz` from per-position zones

The inverse of
[`tz_by_position()`](https://bbtheo.github.io/cuplyr/reference/tz_by_position.md).

## Usage

``` r
tz_from_positions(names, types, by_pos)
```

## Arguments

- names, types:

  Output column names and GPU types

- by_pos:

  Character vector the length of `names` (`"UTC"` for UTC and
  non-timestamp columns)

## Value

A named character vector, or `NULL` when every column is UTC
