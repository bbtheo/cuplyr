# Create a join AST node

Create a join AST node

## Usage

``` r
ast_join(
  type,
  left,
  right,
  by,
  keep = FALSE,
  suffix = c(".x", ".y"),
  na_matches = "na"
)
```

## Arguments

- type:

  Join type: "inner", "left", "right", "full"

- left:

  Left input AST node

- right:

  Right input AST node

- by:

  Join specification list(left = , right = )

- keep:

  Logical, keep both key columns when names match

- suffix:

  Character vector of length 2

- na_matches:

  Character, "na" or "never"

## Value

An ast_join node
