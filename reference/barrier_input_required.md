# What a barrier's input has to deliver

The single rule for "the columns below this node that its consumer and
the node itself read", for the barrier nodes
([`optimize_with_barriers()`](https://bbtheo.github.io/cuplyr/reference/optimize_with_barriers.md),
[`push_down_projections()`](https://bbtheo.github.io/cuplyr/reference/push_down_projections.md)
and
[`prune_dead_columns()`](https://bbtheo.github.io/cuplyr/reference/prune_dead_columns.md)
all ask it). The result may name columns the input does not have (a
window's outputs, a mutate's intermediates above): they are harmless
where a source intersects with its names, but a select must be built
from real names only.

## Usage

``` r
barrier_input_required(node, required)
```

## Arguments

- node:

  A barrier node

- required:

  Columns read from `node`'s output (`NULL`: all)

## Value

Character vector, or `NULL` for "all the input's columns" (what a node
that reads positions, the whole table, or has duplicated input names
needs)
