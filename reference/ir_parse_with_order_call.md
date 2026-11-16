# Parse `with_order(order_by, fun, x, ...)`

`fun` is a function reference (`cummean`,
[`dplyr::lag`](https://dplyr.tidyverse.org/reference/lead-lag.html));
`x` and the further arguments are `fun`'s. Evaluated exactly as
`order_by(order_by, fun(x, ...))`: see
[`ir_parse_with_order()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_with_order.md).

## Usage

``` r
ir_parse_with_order_call(expr, env, schema)
```
