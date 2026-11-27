# Parse `sum(x)` / `sum(x, na.rm = <literal>)`

[`sum()`](https://rdrr.io/r/base/sum.html)'s ordinary registry entry
(`arity = 1L`, agg-family window kind) only accepts the bare
single-argument form – the overwhelmingly common shape, and the one this
window-agg family's `na.rm = FALSE` default (see `src/ops_window.cpp`'s
"sticky-NA"-style whole-group-NA-propagation rule, matching R's own
[`sum()`](https://rdrr.io/r/base/sum.html) default) is built for. But
real, unmodified `dplyr::add_tally(wt = )` (a plain, non-generic dplyr
function we don't control – `dplyr:::tally_n()`) UNCONDITIONALLY emits
`base::sum(<wt>, na.rm = TRUE)` as its internal
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) dot,
regardless of whether the weight column actually has any NAs – so this
2-arg shape must parse too, or `add_tally(wt = )` hard-errors.

## Usage

``` r
ir_parse_sum_call(expr, env, schema)
```

## Details

`na.rm = TRUE` desugars `x` to `coalesce(x, 0)` before the window `sum`
call: replacing every `NA` with the additive identity is exactly
equivalent to excluding it from the sum (matching what R's own
`sum(x, na.rm = TRUE)` computes), and since the desugared column has no
nulls left at all, the window kernel's whole-group-NA rule never fires –
so no C++ change is needed. `na.rm = FALSE` (if named explicitly) is a
pure no-op, since propagating NA is already this family's own default.

A second argument with any other name (or a non-literal-logical value)
doesn't match this shape – returns `NULL`, same as any other unsupported
call shape
([`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)
hard-errors on it;
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) falls
back to CPU evaluation).
