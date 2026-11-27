# Build an IR literal node from an already-evaluated R value

Length/NA rules: a bare, untyped `NA` (the logical `NA` constant)
becomes `na = TRUE` with `type = NULL` (adopts the sibling operand's
type later;
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
defaults it to `FLOAT64` when asked in isolation). Any other length-1
atomic value is typed via `gpu_type_from_r()`. Length != 1 is only legal
when `allow_vector = TRUE` (the `%in%` RHS set-argument position);
otherwise it's a diagnosable user error.

## Usage

``` r
ir_lit_from_r(value, allow_vector = FALSE)
```

## Arguments

- value:

  An R value (atomic length 1, an atomic vector, or `NULL`)

- allow_vector:

  `TRUE` iff a length != 1 vector is acceptable here (used for `%in%`'s
  RHS)

## Value

A `cuplyr_ir_lit` node

## Details

A length-0 vector is legal in the `allow_vector = TRUE` position too:
`x %in% numeric(0)` (or `character(0)`, etc.) is well-defined, ordinary
R – always `FALSE`, for every element including `NA` – not a user error,
so it must not be rejected here (the `%in%` handler in
`src/expr_eval.hpp` special-cases an empty/`NULL` RHS set to "always
FALSE").
