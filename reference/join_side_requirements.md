# Columns each side of a mutating join must keep for `required_cols`

Shared by push_down_projections() and prune_dead_columns() so the two
passes cannot drift. Besides the join keys and the source column of each
required output, pruning must not change any required column's output
NAME: a column suffixed because of a collision (`v.x`) would come out as
plain `v` if its collision partner were pruned from the other side. So
the partner is kept too, and the result is verified by recomputing the
output names on the pruned schemas (falling back to keeping both sides
whole if the simple rule is ever not enough, e.g. double suffixes).

## Usage

``` r
join_side_requirements(ast, required_cols)
```

## Details

Invariant the verification relies on: it recomputes names on each side's
columns in NATURAL (schema) order, so every pass that prunes a join side
must keep that order. The returned sets are unordered on purpose; when
the input chain is complete, an inserted `ast_select` must use
`intersect(<input names>, needed)`, never `needed` itself, because
stacked suffixes depend on column order. (A join side is always a
complete subtree.)
