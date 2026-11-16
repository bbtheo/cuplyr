# Desugar single-column `consecutive_id(x)` into a two-stage `lag`/ `row_number`/`cumsum` formula

Multi-column `consecutive_id(a, b, ...)` is not implemented this wave
(returns `NULL`, i.e. falls back) – this is enforced for free by the
registry entry's `arity = 1L` (a 2+-arg call fails the generic arity
check in `ir_parse_expr()` before `parse` is ever invoked), so this
function itself only ever sees exactly one argument.

## Usage

``` r
ir_parse_consecutive_id(args, schema)
```

## Details

Verified against dplyr 1.2.1, INCLUDING NA-adjacency (the tricky part:
two consecutive `NA`s are treated as "the same value", i.e. NOT a new
group, which the `is.na(x) & is.na(lx)` disjunct captures):
`consecutive_id(c(1,NA,NA,2))` is `1,2,2,3`;
`consecutive_id(c(NA,NA,1, 1,NA))` is `1,1,2,2,3`;
`consecutive_id(c(1,2,2,NA,NA,2))` is `1,2,2,3,3,4`.

Builds `cumsum(if_else(rn==1L, 1L, if_else(same_as_prev, 0L, 1L)))`
where `rn = row_number()` and
`same_as_prev = coalesce(x == lag(x), FALSE) | (is.na(x) & is.na(lag(x)))`;
for a double `x` the "both missing" clause is split into "both NaN" and
"both NA but not NaN" (vctrs keeps NaN and NA apart).
`lag(x)`/[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
are themselves window calls nested inside
[`cumsum()`](https://rdrr.io/r/base/cumsum.html)'s argument – exactly
the "nested window calls needing 2 decomposition passes" shape
`R/ window.R`'s
[`ir_extract_windows()`](https://bbtheo.github.io/cuplyr/reference/ir_extract_windows.md)/[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)
handles generically (see test-window-plan.R), so this desugar doesn't
need to know anything about staging itself.
