# Create an IR literal node

Create an IR literal node

## Usage

``` r
ir_lit(value, type, na = FALSE)
```

## Arguments

- value:

  Atomic value (length 1, or length \> 1 only for a set-argument literal
  such as `%in%`'s RHS), or `NULL`

- type:

  GPU type string, or `NULL` for a bare `NA` (type is adopted from the
  sibling operand at build/lowering time; defaults to `FLOAT64` when
  inferred in isolation, see
  [`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md))

- na:

  `TRUE` iff this literal is the bare, untyped `NA` constant

## Value

A `cuplyr_ir_lit` node
