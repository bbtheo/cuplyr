# Parse dplyr's `with_order(order_by, fun, x, ...)` (what `order_by()` is)

dplyr 1.2.1: `with_order(o, f, a1, ...)` evaluates `a1` and `...` in ROW
order, permutes only `a1` into `o` order (stable, NA last), calls
`f(a1[perm], ...)` and permutes the result back. Hence:

- the windows `f` itself builds (a plain `cumsum`, or the
  [`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
  of [`cummean()`](https://dplyr.tidyverse.org/reference/cumall.html),
  the [`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html) of
  [`consecutive_id()`](https://dplyr.tidyverse.org/reference/consecutive_id.html),
  ...) run in `o` order: each gets `meta$order_override`;

- windows inside `a1` run in row order and get no override;

- an elementwise `f` whose other arguments are all scalar literals is
  the plain row-order value (permutation commutes with it);

- an elementwise `f` with another row-sized argument gives
  `f(a1, b[inverse perm])`, which has no native form: `NULL`.

Everything dplyr rejects (no `a1`, a scalar `a1`, an aggregate `f`, a
name
[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
would capture) and everything the frame cannot do exactly
(`row_number(x)`/`ntile(x)` tie-breaking by the order,
`first`/`last`/`nth`, a nested
[`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html))
returns `NULL`, so the verb's whole-call CPU fallback gives dplyr's
value or error: never an unordered native result.

## Usage

``` r
ir_parse_with_order(order_ref, head, a1, rest, env, schema)
```

## Arguments

- order_ref:

  `list(col =, desc =)` from
  [`ir_parse_order_ref()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_order_ref.md)

- head:

  The raw expression of `f` (a symbol or `pkg::sym`)

- a1:

  The raw expression of the permuted first argument

- rest:

  Named list of the raw remaining arguments (none may read a column, and
  each must evaluate to a non-NULL atomic vector of length 1)

- env:

  The environment the expressions are evaluated in

- schema:

  The input schema (`names`, `types`)

## Value

An IR node, or `NULL`

## Details

Every user sub-expression is evaluated at most once, in dplyr's order:
the first argument, then the others. The first argument is parsed once;
when it is not a plain column, `f` is parsed over a placeholder column
of the argument's type (and zone), the order is applied to that tree,
and the argument's IR then replaces the placeholder. A re-parse of the
raw argument would run a column-free call inside it
(`cumsum(x + runif(1))`) again and build the IR from the second value.
