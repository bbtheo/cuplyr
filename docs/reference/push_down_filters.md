# Push filters below mutates when predicates do not depend on mutate outputs

Push filters below mutates when predicates do not depend on mutate
outputs

## Usage

``` r
push_down_filters(ast)
```

## Arguments

- ast:

  Root AST node

## Value

AST with filters pushed down across mutates where safe
