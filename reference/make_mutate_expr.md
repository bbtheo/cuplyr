# Create a mutate expression structure

The mutate expression record. `ir` is the parsed expression IR (see
`R/ir.R`); `input_cols`/`output_type` are cached from
[`ir_cols()`](https://bbtheo.github.io/cuplyr/reference/ir_cols.md)/
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
so the optimizer passes (`fuse_mutates`/
`try_fuse_mutate_pair`/`prune_dead_columns`/`push_down_projections`)
mostly need only those two fields plus `output_col`.
[`lower_mutate()`](https://bbtheo.github.io/cuplyr/reference/lower_mutate.md)
(via `gpu_mutate_expr()`) reads `ir` for execution, and some optimizer
checks look at it too (a vector literal, a checked integer node, a
rowwise aggregate flag) to keep a dot or stop a rewrite.

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
  unchanged for arithmetic ops)

## Value

An expression list structure:
`list(output_col, ir, input_cols, output_type, checked_int)`;
`checked_int` is the cell of new_checked_int_cell(): read whether the
expression holds a checked integer node with record_checked_int()
