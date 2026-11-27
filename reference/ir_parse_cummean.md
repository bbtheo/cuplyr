# Desugar `cummean(x)` into `cumsum(x) / row_number()`

Verified against dplyr 1.2.1: `cummean(c(1,2,NA,4))` is `1, 1.5, NA, NA`
– i.e. cummean inherits cumsum()'s own (sticky) NA propagation for free,
simply by dividing by row_number() (which is never NA), with no
special-casing needed here.

## Usage

``` r
ir_parse_cummean(args, schema)
```
