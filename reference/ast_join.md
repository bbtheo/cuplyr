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
  na_matches = "na",
  multiple = "all",
  unmatched = "drop",
  relationship = NULL,
  cross_by = FALSE
)
```

## Arguments

- type:

  Join type: "inner", "left", "right", "full", "semi", "anti", "cross"

- left:

  Left input AST node

- right:

  Right input AST node

- by:

  Join specification `list(left, right, op, filter)`, as returned by
  [`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md)

- keep:

  Logical, keep both key columns when names match

- suffix:

  Character vector of length 2

- na_matches:

  Character, "na" or "never"

- multiple:

  Character, one of "all"/"any"/"first"/"last" (dormant – only "all" is
  honored until Phase 7 J5)

- unmatched:

  Character, one or two of "drop"/"error" (dormant – only "drop" is
  honored until Phase 7 J5)

- relationship:

  `NULL` or one of the four cardinality strings (dormant – not enforced
  until Phase 7 J5)

- cross_by:

  `TRUE` for dplyr's deprecated `by = character()` cross spelling (`by`
  is then the empty spec; see
  [`lower_cross_by_join()`](https://bbtheo.github.io/cuplyr/reference/lower_cross_by_join.md))

## Value

An ast_join node
