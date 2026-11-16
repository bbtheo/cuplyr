# What a segment holds, for skipping optimizer passes that cannot apply

One walk over the plan (both sides of a join):

- `mutate_pair`: a mutate directly over a mutate
  ([`fuse_mutates()`](https://bbtheo.github.io/cuplyr/reference/fuse_mutates.md))

- `column_nodes`: any mutate or select
  ([`prune_dead_columns()`](https://bbtheo.github.io/cuplyr/reference/prune_dead_columns.md)
  only ever changes those)

- `filter_over`: a filter directly over a mutate, select or join
  ([`push_down_filters()`](https://bbtheo.github.io/cuplyr/reference/push_down_filters.md))

- `filter_pair`: a filter directly over a filter
  ([`reorder_filters()`](https://bbtheo.github.io/cuplyr/reference/reorder_filters.md))

## Usage

``` r
segment_features(ast)
```

## Arguments

- ast:

  Root of a segment

## Value

Named logical vector
