# Push down column projections to reduce data width early

Push down column projections to reduce data width early

## Usage

``` r
push_down_projections(ast, required_cols = NULL, group_cols = character())
```

## Arguments

- ast:

  Root AST node

- required_cols:

  Columns required by parent (NULL = all output cols)

- group_cols:

  Group columns that must be preserved

## Value

Optimized AST with select nodes inserted
