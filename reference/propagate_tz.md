# Output zones of a node, by position

The `schema$tz` of `node`'s output, from its input schema's. The
node-wise sibling of
[`propagate_factor_levels()`](https://bbtheo.github.io/cuplyr/reference/propagate_factor_levels.md):
row and column nodes carry zones with their columns;
mutate/window/summarise compute the zone of each new POSIXct column
([`ir_infer_tz()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_tz.md);
an aggregate or window op over a POSIXct column keeps that column's
zone). Joins and binds build their own schema (`build_join_schema()`,
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)).

## Usage

``` r
propagate_tz(node, input_schema, new_schema)
```

## Arguments

- node:

  The AST node being applied

- input_schema:

  The schema of the op's input (pre-op)

- new_schema:

  The node's output `names`/`types` (from
  [`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md))

## Value

A `schema$tz` value
