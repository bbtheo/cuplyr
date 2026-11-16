# Create a mutate expression structure

Section 3 of `scratchpad/phase1_expression_engine.md`: the IR cutover's
mutate expression record. `ir` is the parsed expression IR (see
`R/ir.R`); `input_cols`/`output_type` are cached from
[`ir_cols()`](https://bbtheo.github.io/cuplyr/reference/ir_cols.md)/
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
since every optimizer pass (`fuse_mutates`/`try_fuse_mutate_pair`/
`prune_dead_columns`/`push_down_projections`) reads only those two
fields plus `output_col` – none of them walk the IR tree themselves.
`op`/ `scalar`/`input_types` are gone: only
[`lower_mutate()`](https://bbtheo.github.io/cuplyr/reference/lower_mutate.md)
(via `gpu_mutate_expr()`) ever needed them, and it now reads `ir`
directly.

## Usage

``` r
make_mutate_expr(output_col, ir, schema)
```

## Arguments

- output_col:

  Output column name

- ir:

  An IR node (the parsed mutate expression)

- schema:

  List with `names`/`types`, used to resolve `output_type` via
  [`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
  (which delegates to
  [`infer_mutate_output_type()`](https://bbtheo.github.io/cuplyr/reference/infer_mutate_output_type.md)
  unchanged for arithmetic ops, per section 1.3)

## Value

An expression list structure:
`list(output_col, ir, input_cols, output_type, checked_int)`;
`checked_int` (F70) is the cell of new_checked_int_cell(): read whether
the expression holds an F42 checked integer node with
record_checked_int()
