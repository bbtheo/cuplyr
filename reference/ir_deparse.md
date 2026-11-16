# Deparse an IR node back to readable text

For diagnostics
([`show_query()`](https://dplyr.tidyverse.org/reference/explain.html),
error messages) – not intended to be a faithful/round-trippable R
deparser, just a readable rendering.

## Usage

``` r
ir_deparse(ir)
```

## Arguments

- ir:

  An IR node (or `NULL`)

## Value

Character scalar
