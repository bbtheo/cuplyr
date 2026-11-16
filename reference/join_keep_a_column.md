# Keep at least one column on each side of a keyless join

A `cross` join or a `by = character()` join has no key columns, so
projection pruning can leave a side with no needed column at all – and a
zero-column table has lost its row count, which the join still needs
(cross product size; "does y have rows" for semi/anti).

## Usage

``` r
join_keep_a_column(ast, side, needed)
```
