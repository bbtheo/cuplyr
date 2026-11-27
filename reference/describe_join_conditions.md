# Describe a join's `by` spec as "left op right, ..." condition pairs

Shared by
`print.ast_node()`/[`ast_to_string()`](https://bbtheo.github.io/cuplyr/reference/ast_to_string.md)
(used by
[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)):
renders every condition, not just the equi ones, so a non-equi/mixed
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html) (`op`
other than `"=="`) is visible in the rendered query – e.g.
`join_by(a == b, x >= y)` shows as `"a == b, x >= y"`. A `closest()`
condition shows as `closest(x >= y)`.

## Usage

``` r
describe_join_conditions(by)
```

## Arguments

- by:

  A join spec `list(left, right, op, filter)` (see
  [`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md))

## Value

Character scalar, `""` if `by` has no conditions
