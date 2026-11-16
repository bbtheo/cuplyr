# Parse `lag(x, n = 1L, default = NULL, order_by = NULL)` / `lead(...)`

Builds a window-class
`ir_call(fn_name, list(x_ir), meta = list(n = , default = , order_override = ))`
node (never lowered directly – see the
[ir_call_registry](https://bbtheo.github.io/cuplyr/reference/ir_call_registry.md)
entry /
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)'s
guard). `n` must be a non-negative whole-number constant (dplyr itself
requires this); `default`, if supplied, must be a CONSTANT expression –
evaluable via
[`ir_eval_constant()`](https://bbtheo.github.io/cuplyr/reference/ir_eval_constant.md)
with no data mask, so `default = -1`, \`default = 2L

- 3L`, or any other column-free expression all work (not just a bare literal token), but `default
  = first(x)\` (a real, column-referencing dplyr default) is not
  implemented this wave.

## Usage

``` r
ir_parse_shift_call(expr, env, schema, fn_name)
```

## Details

`order_by=`, when supplied, must resolve via
[`ir_parse_order_ref()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_order_ref.md)
(a bare column, `desc(column)`, or `-column`); anything else (a general
expression) is not implemented this wave and returns `NULL` (Phase 5 W4:
`order_by=` re-frames this spec's own frame exactly like
[`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)/[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
would if wrapped around the same call – see
[`ir_parse_with_order()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_with_order.md)'s
docs – so `lag(x, order_by = y)` and `order_by(y, lag(x))` produce the
same `meta$order_override`).
