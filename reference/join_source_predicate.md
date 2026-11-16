# Rewrite a predicate pushed below a join to the side's source names

A predicate is written against the join's OUTPUT names (`v.x`, `k.y`),
but once pushed to one side it reads that side's own table, where the
column is still called by its source name. Only IR `col` nodes are
renamed.

## Usage

``` r
join_source_predicate(pred, info, side)
```
