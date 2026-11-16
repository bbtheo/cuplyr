# Is an IR subtree constant (no column references)?

Is an IR subtree constant (no column references)?

## Usage

``` r
ir_is_const(ir)
```

## Arguments

- ir:

  An IR node

## Value

`TRUE` iff
[`ir_cols()`](https://bbtheo.github.io/cuplyr/reference/ir_cols.md) is
empty
