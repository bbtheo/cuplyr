# Restrict factor level metadata to columns that survive an op

`schema$factor_levels` is always recomputed from the *input* schema's
factor_levels, filtered to the names present in the node's output
schema. So an eager
[`select()`](https://dplyr.tidyverse.org/reference/select.html) of a
factor column keeps its `factor_levels`, and
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
returns a factor rather than integer codes.

## Usage

``` r
propagate_factor_levels(node, input_schema, surviving_names)
```

## Arguments

- node:

  The AST node being applied (used to translate factor_levels keys
  through a `rename` node's name map, see
  [`ast_rename()`](https://bbtheo.github.io/cuplyr/reference/ast_rename.md),
  `R/ast.R` – without this, a renamed factor column's levels would be
  silently dropped here since its key in `input_schema$factor_levels` is
  the OLD name, which no longer appears in `surviving_names`)

- input_schema:

  The schema of the op's input (pre-op)

- surviving_names:

  Character vector of column names in the new schema

## Value

A named list of factor levels, or `NULL` if none survive
