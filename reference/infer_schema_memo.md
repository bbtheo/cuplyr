# `infer_schema()` through the schema memo

What the methods, the optimizer and the lowering call to get a node's
schema. Outside a memo scope (see `schema_memo_open()`) it is plain
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md).
Inside one (an outermost
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html), an
eager chain's lowering,
[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html))
the result is remembered by node identity, so inferring every node's
input schema along a plan is linear in its depth instead of quadratic.

## Usage

``` r
infer_schema_memo(node)
```

## Arguments

- node:

  An AST node (or `NULL`)

## Value

List with `names`, `types` (plus whatever
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
gives)
