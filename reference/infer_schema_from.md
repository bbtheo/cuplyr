# Output `names`/`types` of a node from its input's schema

The body of every `infer_schema.ast_*()` method, with the input schema
passed in:
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)
calls it with the schema it already has, so recording a verb never walks
the pending plan. Joins have two inputs and go through
`infer_join_schema()`.

## Usage

``` r
infer_schema_from(node, input_schema)
```

## Arguments

- node:

  A single-input AST node

- input_schema:

  Schema of `node$input` (extra fields such as `factor_levels` are
  tolerated; pass-through nodes return it unchanged)

## Value

List with `names` and `types`
