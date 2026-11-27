# Create a summarise AST node

Create a summarise AST node

## Usage

``` r
ast_summarise(
  input,
  aggregations,
  groups,
  result_groups = character(),
  on_sum_overflow = "error",
  group_order = "keys"
)
```

## Arguments

- input:

  Input AST node

- aggregations:

  List of aggregation structures

- groups:

  Character vector of group columns this node aggregates BY (drives the
  actual groupby op – always the full pre-summarise group set,
  regardless of `.groups=`)

- result_groups:

  Character vector of group columns RETAINED on the summarise's
  *output*: a subset of `groups`, computed from `.groups=` semantics by
  `resolve_summarise_groups()` (R/summarise.R) –
  [`character()`](https://rdrr.io/r/base/character.html) for
  `.groups = "drop"`/ungrouped input, all of `groups` for `"keep"`,
  all-but-the-last for the default/ `"drop_last"`. Read by
  [`propagate_groups()`](https://bbtheo.github.io/cuplyr/reference/propagate_groups.md)
  (R/execute.R).

- on_sum_overflow:

  What to do when an integer/logical
  [`sum()`](https://rdrr.io/r/base/sum.html) leaves R's integer range at
  execution. `"double"`: the column comes back FLOAT64, as in dplyr
  (eager mode reconciles the schema from the result). `"error"` (lazy
  plans, whose schema was declared INT32 statically): raise
  `cuplyr_error_sum_overflow`. A lazy
  [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
  plans only a sum of a logical this way: an INT32 sum runs at the verb
  with `"double"`.

- group_order:

  Row order of a grouped result: `"keys"` sorts by the group keys
  (ascending, `NA` last), dplyr's
  [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
  order; `"first"` (#50) orders groups by first appearance, dplyr's
  `.by =` order, and is what `summarise_by_desugar()` asks for.

## Value

An ast_summarise node
