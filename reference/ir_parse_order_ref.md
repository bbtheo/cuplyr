# Parse a window "order by" reference (`order_by=` argument, or `order_by()`/`with_order()`'s own first argument) into a schema column reference

Recognizes a bare column name, `desc(column)`/`dplyr::desc(column)`, or
`-column` (dplyr's own
[`desc()`](https://dplyr.tidyverse.org/reference/desc.html) shorthand,
also usable directly as an
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html)/[`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)
sort key) and resolves it against `schema`. Anything else – a general
expression such as `order_by(a + b, ...)` – is not implemented this wave
(mirroring `R/window.R`'s rank-family design note: only a bare,
already-existing column doubles as an order key so far, no
`..winarg..`-style hoisting for order columns yet) and returns `NULL`,
same as any other unsupported IR shape (callers fall back).

## Usage

``` r
ir_parse_order_ref(expr, schema)
```

## Arguments

- expr:

  A language object (not a quosure)

- schema:

  List with `names`/`types`

## Value

`list(col = <chr1>, desc = <lgl1>)`, or `NULL`
