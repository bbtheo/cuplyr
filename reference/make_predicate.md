# Create a filter predicate structure

The predicate record. `ir` is the parsed expression IR (see `R/ir.R`);
`cols` is cached from
[`ir_cols()`](https://bbtheo.github.io/cuplyr/reference/ir_cols.md)
since every optimizer filter pass needs it and none of them need to walk
the IR tree themselves.

## Usage

``` r
make_predicate(ir, schema)
```

## Arguments

- ir:

  An IR node (the parsed predicate expression)

- schema:

  List with `names`/`types` the IR is bound against; read only by
  [`ir_has_checked_int()`](https://bbtheo.github.io/cuplyr/reference/ir_has_checked_int.md)
  for `checked_int`

## Value

A predicate list structure:
`list(ir, cols, estimated_cost, is_deterministic, na_sensitive, checked_int)`;
`checked_int` is the cell of new_checked_int_cell(): read whether the
predicate holds a checked integer node with record_checked_int()
