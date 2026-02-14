# Prune unused mutate outputs to reduce intermediate width

Walks the AST from root to leaves, tracking columns required downstream.
Any mutate expression whose output is not required is dropped; empty
mutate nodes are removed entirely.

## Usage

``` r
prune_dead_columns(ast, required_cols = NULL, group_cols = character())
```

## Arguments

- ast:

  Root AST node

- required_cols:

  Columns required by parent (NULL = all output cols)

- group_cols:

  Group columns that must be preserved

## Value

AST with unused mutate outputs pruned
