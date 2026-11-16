# Output column order of a summarise node

`gpu_summarise()` returns `[group keys, aggregates]`. dplyr instead lets
an aggregate named like a grouping column replace that key in place
(same position, the aggregate's type); every other aggregate follows the
keys in call order. This returns the 1-based permutation of
`c(keys, aggregates)` that yields dplyr's layout, shared by
`infer_schema.ast_summarise()` and
[`lower_summarise()`](https://bbtheo.github.io/cuplyr/reference/lower_summarise.md)
so the schema and the table cannot disagree. It is the identity when no
aggregate collides with a key. Duplicate aggregate names never reach a
node
([`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
routes them to the CPU fallback).

## Usage

``` r
summarise_output_index(groups, agg_names)
```

## Arguments

- groups:

  Character vector of group columns the node aggregates by

- agg_names:

  Character vector of aggregate output names

## Value

Integer vector indexing `c(keys, aggregates)`
