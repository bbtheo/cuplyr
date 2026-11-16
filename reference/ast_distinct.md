# Create a distinct AST node

Create a distinct AST node

## Usage

``` r
ast_distinct(input, key_cols, keep_all = FALSE, key_positions = NULL)
```

## Arguments

- input:

  Input AST node (any dot expressions that compute new key columns, e.g.
  `distinct(df, z = x + 1)`, are already lowered into an `ast_mutate`
  wrapping this node by
  [`distinct.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/distinct.tbl_gpu.md)
  – this node only ever does the dedup step)

- key_cols:

  Character vector of column names to dedup on, in the exact order the
  `.keep_all = FALSE` output should have (group columns, if any, already
  prepended by the caller)

- keep_all:

  Logical, `TRUE` to keep every column of `input` (dplyr's
  `.keep_all = TRUE`), `FALSE` to project down to just `key_cols`

- key_positions:

  `NULL` (key on `key_cols` by name), or the integer positions of the
  key columns in `input`'s schema: used instead of the names when the
  input has duplicated column names. Such a node is never pruned through
  by name.

## Value

An ast_distinct node
