# Reattach factor levels after `bind_rows()`

[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
(R/bind.R) already computes and attaches the unioned
`schema$factor_levels` on its own result – since `x_aligned`/`y_aligned`
are already aligned to the exact same `target_schema`,
[`bind_rows_gpu()`](https://bbtheo.github.io/cuplyr/reference/bind_rows_gpu.md)'s
own internal
[`compute_unified_schema()`](https://bbtheo.github.io/cuplyr/reference/compute_unified_schema.md)
call recomputes the identical schema
[`check_set_op_compatible()`](https://bbtheo.github.io/cuplyr/reference/check_set_op_compatible.md)
already computed here, so this is a belt-and-braces re-assignment (kept
for clarity rather than trusting that invariant to hold silently at a
distance).
[`intersect()`](https://generics.r-lib.org/reference/setops.html)/[`setdiff()`](https://generics.r-lib.org/reference/setops.html)
(built from
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html) +
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html),
which already preserve `factor_levels` via
[`propagate_factor_levels()`](https://bbtheo.github.io/cuplyr/reference/propagate_factor_levels.md),
R/execute.R) don't need this at all.

## Usage

``` r
restore_set_op_factor_levels(result, target_schema)
```
