# Desugar `cummean(x)` into `cumsum(x) / row_number()`

Verified against dplyr 1.2.1: `cummean(c(1,2,NA,4))` is `1, 1.5, NA, NA`
– i.e. cummean inherits cumsum()'s own (sticky, once cudf's scan gets
its Phase-5 sticky-NA fix in W2 – see phase5_window_design.md section 4)
NA propagation for free, simply by dividing by row_number() (which is
never NA), with no special-casing needed here.

## Usage

``` r
ir_parse_cummean(args, schema)
```
