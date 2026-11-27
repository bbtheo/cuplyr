# Desugar `cumall(x)`/`cumany(x)` into a tri-state `cummin`/`cummax` encoding

Verified against dplyr 1.2.1: `cumall(c(T,T,NA,T,F,T))` is
`T,T,NA,NA, F,F`; `cumany(c(F,NA,T,F))` is `F,NA,T,T` – both are exactly
`cummin`/ `cummax` (respectively) over the tri-state encoding FALSE=0,
NA=1, TRUE=2, then decoded back to BOOL8.

## Usage

``` r
ir_parse_cumall_cumany(args, schema, scan_op)
```

## Arguments

- scan_op:

  `"cummin"` (cumall) or `"cummax"` (cumany)

## Details

[`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)'s
bare-NA value-position adoption
([`ir_resolve_value_nas()`](https://bbtheo.github.io/cuplyr/reference/ir_resolve_value_nas.md))
is applied explicitly to the decode step's `yes`/`no` pair here, exactly
as if_else()'s own registry `parse` hook would – this desugar builds
`ir_call("if_else", ...)` nodes directly rather than routing back
through `ir_parse_expr()`, so that adoption has to be done by hand.
