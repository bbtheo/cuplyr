# Infer the GPU output type of an IR node

Recurses; for arithmetic call nodes this delegates to the existing
[`infer_mutate_output_type()`](https://bbtheo.github.io/cuplyr/reference/infer_mutate_output_type.md)
unchanged, so D5's rules (and its pinned
STRING/DICTIONARY32/TIMESTAMP\_\* error messages) survive verbatim.
Every other op class is resolved via its
[ir_call_registry](https://bbtheo.github.io/cuplyr/reference/ir_call_registry.md)
entry's `type` function (section 1.3).

## Usage

``` r
ir_infer_type(ir, schema)
```

## Arguments

- ir:

  An IR node

- schema:

  List with `names`/`types` to resolve column types against

## Value

GPU type string
