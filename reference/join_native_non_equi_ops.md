# The non-equi operators the native `mixed_join`/`conditional_join` lowering understands

[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)'s own
infix parser only ever produces one of these five for a condition
(verified empirically: `==`, `>=`, `>`, `<=`, `<`; `!=` isn't reachable
via any of
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)'s own
helpers –
[`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/
`overlaps()` all pre-expand to `>=`/`<=` pairs, `closest()` wraps one of
the five – but is included here defensively since `ast_op_from_name()`
(`src/expr_eval.hpp`) already has an entry for it and nothing about this
set is dplyr-specific).

## Usage

``` r
join_native_non_equi_ops
```
