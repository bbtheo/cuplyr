# Is a segment a chain no projection or pruning pass can change?

Settled when `ast` is a chain of filter and mutate nodes (each mutate
with at least one expression) ending in a source, every predicate reads
only columns of its filter's input, and `required_cols` holds every
output column of `ast` (`NULL`: they are `ast`'s own output columns, as
[`optimize_segment()`](https://bbtheo.github.io/cuplyr/reference/optimize_segment.md)
takes it). Then, by induction from the top (a node's required set always
covers its output columns: a filter's output is its input, and a
mutate's is its input plus its outputs, so taking its new outputs away
and adding what its expressions read still covers the input):

- [`push_down_projections()`](https://bbtheo.github.io/cuplyr/reference/push_down_projections.md)
  returns the same tree: the source keeps all its columns, and no filter
  has a predicate column that isn't required.

- [`prune_dead_columns()`](https://bbtheo.github.io/cuplyr/reference/prune_dead_columns.md)
  keeps every mutate expression (each output column is required), so its
  only change is `exprs[keep]` with `keep` all TRUE, which
  [`settled_prune()`](https://bbtheo.github.io/cuplyr/reference/settled_prune.md)
  repeats. Anything else (a select, a join, a barrier, an empty mutate)
  is not settled, and the passes run as usual. A segment above a barrier
  ends in a stub source and is settled like any other. Only output
  column NAMES are needed, as sets, so they are built bottom-up without
  inferring types.

## Usage

``` r
settled_chain(ast, required_cols)
```

## Arguments

- ast:

  Root of a segment

- required_cols:

  Columns required by whatever consumes the segment, or `NULL` for all
  of its output columns

## Value

`0L` not settled; `1L` settled; `2L` settled, and no mutate or filter
sits directly on a mutate or filter, so
[`segment_features()`](https://bbtheo.github.io/cuplyr/reference/segment_features.md)
would find no pair for
[`fuse_mutates()`](https://bbtheo.github.io/cuplyr/reference/fuse_mutates.md),
[`push_down_filters()`](https://bbtheo.github.io/cuplyr/reference/push_down_filters.md)
or
[`reorder_filters()`](https://bbtheo.github.io/cuplyr/reference/reorder_filters.md)
either

## Details

A narrower `required_cols` (F99: what a barrier above reads) is settled
when the requirement walked down the chain needs no select above a
filter, finds no dead mutate expression and keeps every source column.
One set does for both passes: a mutate's new outputs, the only names
projection pushdown takes back out of its requirement, are neither
source columns nor a lower node's input, so they change no answer here.
When every output is required the walk passes by the induction above.
