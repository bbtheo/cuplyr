# Factor levels by column position

`schema$factor_levels` is a list named by column. When a table has
duplicated column names (positional select()/relocate()/rename()) a name
no longer identifies a column, so an entry belongs to the n-th
`DICTIONARY32` column carrying its name: entries are kept in column
order, and a duplicated name may repeat in the list. Unique names keep
the plain by-name lookup.

## Usage

``` r
levels_by_position(schema)
```

## Arguments

- schema:

  A schema with `names`, `types` and `factor_levels`

## Value

A list the length of `schema$names`: the level vector of a factor
column, `NULL` for any other

## Details

Only
[`levels_from_positions()`](https://bbtheo.github.io/cuplyr/reference/levels_from_positions.md)
lays a list out in column order, and it tags the list
(`attr(, "by_position")`) when the names repeat. A list for duplicated
names without that tag was written by name somewhere, so its order says
nothing about which column an entry belongs to: that is an error rather
than a guess. Every writer that can produce duplicated names (`names<-`,
positional nodes in
[`propagate_factor_levels()`](https://bbtheo.github.io/cuplyr/reference/propagate_factor_levels.md),
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md))
goes through
[`levels_from_positions()`](https://bbtheo.github.io/cuplyr/reference/levels_from_positions.md).
