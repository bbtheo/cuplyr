# Rebuild `schema$factor_levels` from per-position level vectors

The inverse of
[`levels_by_position()`](https://bbtheo.github.io/cuplyr/reference/levels_by_position.md):
entries in column order, named by the output column names (which may
repeat). When they repeat, the list is tagged `by_position` so
[`levels_by_position()`](https://bbtheo.github.io/cuplyr/reference/levels_by_position.md)
can trust its order.

## Usage

``` r
levels_from_positions(names, by_pos)
```

## Arguments

- names:

  Output column names

- by_pos:

  List the length of `names` (`NULL` for a non-factor column)

## Value

A named list, or `NULL` when no column is a factor
