# Create a reframe AST node

The LONG step of a native
[`reframe()`](https://dplyr.tidyverse.org/reference/reframe.html) (F22,
`R/reframe.R`): reshapes the wide one-row-per-group table `input` into
`k` rows per group.

## Usage

``` r
ast_reframe(input, outputs, k, first_k_name = NA_character_)
```

## Arguments

- input:

  Input AST node (the wide table)

- outputs:

  The output columns in order, each `list(name, kind, ...)`:
  `kind = "rep"` (`src` = one wide column, every row repeated `k` times;
  the group keys and size-1 results), `"interleave"` (`src` = `k` wide
  columns interleaved row by row; a size-`k` result) or `"literal"`
  (`value`, an R vector of length 1 or `k`, tiled)

- k:

  Rows per group

- first_k_name:

  Name of the first output of size `k` (NA when `k == 1`): the column in
  dplyr's "Can't recycle" error for an empty table

## Value

An ast_reframe node
