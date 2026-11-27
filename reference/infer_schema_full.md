# Infer a node's output schema, INCLUDING `factor_levels`

[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
(R/ast.R) computes `names`/`types` recursively via each node type's own
S3 method, but only `infer_schema.ast_join` threads `factor_levels`
through on its own – every other node type
(filter/mutate/select/arrange/summarise/distinct/slice/rename/
window/barrier) relies entirely on
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)'s
separate
[`propagate_factor_levels()`](https://bbtheo.github.io/cuplyr/reference/propagate_factor_levels.md)
step, which only ever ran at op-construction time (each verb's own call
site, one node at a time against that node's OWN immediate input
schema). That's fine for a freshly-built single-node AST, but
[`current_schema()`](https://bbtheo.github.io/cuplyr/reference/current_schema.md)/[`compute.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/compute.tbl_gpu.md)
both need to recompute the schema for an ARBITRARY-depth pending AST
from scratch (a multi-op lazy chain, or a chain built on top of another
lazy op) by calling plain
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
directly – which, for any of those non-join node types, silently drops
`factor_levels` for the whole subtree above the leaf `ast_source` (the
one place besides `ast_join` that genuinely carries `factor_levels`, via
[`input_node()`](https://bbtheo.github.io/cuplyr/reference/input_node.md)'s
`ast_source(.data$schema)`).

## Usage

``` r
infer_schema_full(node)
```

## Arguments

- node:

  An AST node (or `NULL`)

## Value

List with `names`, `types`, and (if any survive) `factor_levels`

## Details

This wraps
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
to recurse down to each node's input FIRST (recovering the leaf's real
`factor_levels` from its `ast_source` schema, or an `ast_join`'s own
self-contained computation), then reapplies
[`propagate_factor_levels()`](https://bbtheo.github.io/cuplyr/reference/propagate_factor_levels.md)
on the way back up – exactly what
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md) does
for one node, generalized to a whole AST.
