# Infer output type for a mutate expression

Implements D5's ordered promotion rules (see
`scratchpad/unification_design.md`). Note this deliberately mirrors R's
own arithmetic promotion, not SQL/cudf defaults: `2L^2L` is a double in
R, and `TRUE+TRUE` is an integer, and dplyr agrees with both.

## Usage

``` r
infer_mutate_output_type(op, input_types, scalar)
```

## Arguments

- op:

  The operation

- input_types:

  Types of input columns

- scalar:

  Scalar value if any

## Value

GPU type string
