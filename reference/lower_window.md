# Lower window node

Phase 5, task W2 (`scratchpad/phase5_window_design.md` section 2.1):
binds every spec's `value_col`/`order_cols` (and the node's own
`group_cols`) to 0-based indices against `ast$input`'s schema, groups
the specs into "frames" keyed by `(order_cols, order_desc)`, and hands
ONE `gpu_window()` call (`src/ops_window.cpp`) to the C++ side: the
six-step stable-sort / gather / groupby-scan-or-shift / scatter-back
algorithm (section 2.2).

## Usage

``` r
lower_window(ast, source_ptr)
```

## Details

`out_pos` (each spec's 0-based position among the node's own appended
columns) is simply the spec's position in `ast$specs` – the same order
`infer_schema.ast_window()` (`R/ast.R`) appends spec output columns in,
so the two always agree on where a given spec's result lands in the
final schema.

As of Phase 5 task W4, every window kind is lowerable: the "shift"
family
([`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)/[`lead()`](https://dplyr.tidyverse.org/reference/lead-lag.html)),
the "scan" family
([`cumsum()`](https://rdrr.io/r/base/cumsum.html)/[`cummax()`](https://rdrr.io/r/base/cumsum.html)/
[`cummin()`](https://rdrr.io/r/base/cumsum.html)/[`cumprod()`](https://rdrr.io/r/base/cumsum.html),
plus bare
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)'s
no-order COUNT_ALL form), the "rank" family
(`row_number(x)`/[`min_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/[`dense_rank()`](https://dplyr.tidyverse.org/reference/row_number.html)/
[`percent_rank()`](https://dplyr.tidyverse.org/reference/percent_rank.html)/[`cume_dist()`](https://dplyr.tidyverse.org/reference/percent_rank.html)),
and the "agg" family
([`mean()`](https://rdrr.io/r/base/mean.html)/[`sum()`](https://rdrr.io/r/base/sum.html)/[`min()`](https://rdrr.io/r/base/Extremes.html)/[`max()`](https://rdrr.io/r/base/Extremes.html)/[`n()`](https://dplyr.tidyverse.org/reference/context.html)/[`sd()`](https://rdrr.io/r/stats/sd.html)/[`var()`](https://rdrr.io/r/stats/cor.html)/[`first()`](https://dplyr.tidyverse.org/reference/nth.html)/
[`last()`](https://dplyr.tidyverse.org/reference/nth.html)/[`nth()`](https://dplyr.tidyverse.org/reference/nth.html))
– see
[`validate_window_spec()`](https://bbtheo.github.io/cuplyr/reference/validate_window_spec.md).
A later task (the rolling-window benchmark, scratchpad/todo.md) adds a
sixth, "rolling"
([`roll_mean()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_sum()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_min()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_max()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`roll_sd()`](https://bbtheo.github.io/cuplyr/reference/roll.md),
but NOT
[`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md) –
see
[`window_spec_lowerable()`](https://bbtheo.github.io/cuplyr/reference/window_spec_lowerable.md)'s
own docs). A spec whose `fn` isn't recognized AT ALL (no registry entry,
or a registry entry with a `NULL` `window` field) hits a hard internal
error here (never a user-facing message: no verb decomposes an
expression into an unrecognized spec, so reaching this function with one
is a cuplyr bug, not a user mistake).
