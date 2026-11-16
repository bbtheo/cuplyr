# Optimize a single segment (no barriers)

Optimize a single segment (no barriers)

## Usage

``` r
optimize_segment(ast, required_cols = NULL, keep_mutates = FALSE)
```

## Arguments

- ast:

  Root of segment

- required_cols:

  Columns required by whatever consumes this segment's output (`NULL`
  means all of `ast`'s own output columns)

- keep_mutates:

  `TRUE` when `required_cols` is a barrier's narrower requirement (F99):
  that pruning removes columns only, so a mutate expression nothing
  reads stays and is computed (it can warn, as in dplyr), and
  [`prune_dead_columns()`](https://bbtheo.github.io/cuplyr/reference/prune_dead_columns.md)
  does not run. Its inputs stay required
  ([`push_down_projections()`](https://bbtheo.github.io/cuplyr/reference/push_down_projections.md)
  keeps them).

## Value

Optimized segment
