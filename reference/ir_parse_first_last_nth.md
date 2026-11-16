# Parse `first(x, ...)` / `last(x, ...)` / `nth(x, n, ...)`

The bare `x` (and, for
[`nth()`](https://dplyr.tidyverse.org/reference/nth.html), a
constant-integer `n`) shape, plus `order_by=` (Phase 5 W4: see
[`ir_parse_order_ref()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_order_ref.md)/
[`ir_parse_with_order()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_with_order.md)
– `first(x, order_by = y)` re-frames this spec's frame to be ordered by
`y`, matching dplyr's own "first/last/nth row of the group IN
`order_by`'s ORDER, not the group's current order" semantics, verified
empirically against dplyr 1.2.1). `default=`/`na_rm=` being present at
all – even at their own default value – still makes this return `NULL`
(not implemented this wave). Builds a window-class
`ir_call(fn_name, list(x_ir), meta = list(order_override = ))` node
([`nth()`](https://dplyr.tidyverse.org/reference/nth.html) additionally
carries `n = <int>`, the (1-based, possibly negative) element position).

## Usage

``` r
ir_parse_first_last_nth(expr, env, schema, fn_name)
```
