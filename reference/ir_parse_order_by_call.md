# Parse `order_by(order_by, call)`

dplyr rewrites it to
`with_order(order_by, <head of call>, <args of call>)`, so this builds
that call, lets R bind its arguments (a name
[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
would capture, such as `lag(x, order_by = y)`, makes the call an error,
i.e. `NULL` here) and parses it like
[`ir_parse_with_order()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_with_order.md).

## Usage

``` r
ir_parse_order_by_call(expr, env, schema)
```
