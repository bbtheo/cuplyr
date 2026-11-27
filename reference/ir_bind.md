# Bind 0-based column indices onto every `col` node in an IR tree

Indices are resolved against a schema only at lowering time (never at
parse time), since projection pushdown can shift column positions in
between. Returns a new tree (does not mutate in place, matching R's
copy-on-modify semantics) with an `index` field (0-based, for C++) added
to every `col` node.

## Usage

``` r
ir_bind(ir, schema, pos = NULL)
```

## Arguments

- ir:

  An IR node (or `NULL`)

- schema:

  List with `names`/`types` to resolve column positions against

- pos:

  `NULL` (look names up in `schema$names`), or an environment mapping
  column names to their positions in `schema`, which a caller binding
  many expressions against one schema builds once

## Value

The IR tree with `index` filled in on every `col` node
