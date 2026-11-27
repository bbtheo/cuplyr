# Bind 0-based, per-side column indices onto a non-equi join condition IR tree

The join-condition analogue of
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md).
Every `col` node in a condition IR built by
[`join_cond_ir_from_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_ir_from_spec.md)
(R/join.R) already carries an explicit `side` tag (see
[`ir_col_join()`](https://bbtheo.github.io/cuplyr/reference/ir_col_join.md)'s
docs for why sides are not resolved by lookup order: a self-join has
identical left/right schemas, so name-based resolution against
`left_schema` first would silently mis-bind a right-only column, or bind
a same-named column to the wrong side entirely). This function simply
resolves each `col` node's `$index` against `left_schema` (side `0L`) or
`right_schema` (side `1L`) directly, using that tag – never a generic
single-schema lookup.

## Usage

``` r
ir_bind_join(ir, left_schema, right_schema)
```

## Arguments

- ir:

  An IR node (or `NULL`), every `col` descendant carrying a `side` field
  ([`ir_col_join()`](https://bbtheo.github.io/cuplyr/reference/ir_col_join.md))

- left_schema, right_schema:

  List with `names`/`types` for the join's left/right inputs

## Value

The IR tree with `index` filled in on every `col` node
