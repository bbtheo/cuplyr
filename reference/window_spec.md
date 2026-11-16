# Create a window-spec structure

One record per window-function call extracted by
[`ir_extract_windows()`](https://bbtheo.github.io/cuplyr/reference/ir_extract_windows.md).
This is the exact shape section 2 of
`scratchpad/phase5_window_design.md` documents for
[`ast_window()`](https://bbtheo.github.io/cuplyr/reference/ast_window.md)'s
specs (W2), produced here so W2 can consume a finished plan's specs
directly with no reshaping.

## Usage

``` r
window_spec(
  output_col,
  fn,
  value_col = NA_character_,
  order_cols = character(),
  order_desc = logical(),
  output_type,
  n = NA_integer_,
  default = NULL,
  na_rm = FALSE,
  input_type = NA_character_
)
```

## Arguments

- output_col:

  Character scalar, the synthetic temp column name this spec's result
  will be materialized under (e.g. `"..win1.."`)

- fn:

  Character scalar, the canonical window op name (e.g. `"lag"`,
  `"cumsum"`, `"mean"`)

- value_col:

  Character scalar, the (already-resolved-to-a-plain- column) name of
  the window function's "value" argument, or `NA_character_` for a call
  with no value argument
  ([`n()`](https://dplyr.tidyverse.org/reference/context.html), bare
  [`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html))

- order_cols:

  Character vector of order-by column names for this spec's frame:
  rank-family calls with a value argument default this to `value_col`
  itself (ranking a column inherently orders by that same column),
  UNLESS an explicit
  `order_by=`/[`order_by()`](https://dplyr.tidyverse.org/reference/order_by.html)/[`with_order()`](https://dplyr.tidyverse.org/reference/with_order.html)
  override (Phase 5 W4, `R/ir.R`'s
  [`ir_parse_with_order()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_with_order.md))
  is present, in which case that wins for ANY window kind
  (shift/rank/agg)

- order_desc:

  Logical vector, parallel to `order_cols`, `TRUE` for descending

- output_type:

  GPU type string for this spec's result column

- n:

  Integer scalar, an op-specific count argument (`lag`/`lead`'s shift
  amount, `nth`'s element position), or `NA_integer_` if not applicable

- default:

  An IR literal node (the op's constant fill/default value), or `NULL`
  if not applicable/not supplied

- na_rm:

  Logical scalar, `FALSE` unless a future wave adds an `na_rm=`-aware
  window op

- input_type:

  (F80) GPU type string of an integer
  [`sum()`](https://rdrr.io/r/base/sum.html)'s argument (`"INT32"` or
  `"BOOL8"`), `NA_character_` for every other call; a lazy
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) runs a
  call holding an INT32 sum at the verb (see `window_spec_int_sum()`)

## Value

A list with exactly these named fields
