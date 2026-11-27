# Get all column names from a GPU table (dplyr's `tbl_vars()` generic)

[`dplyr::tbl_vars()`](https://dplyr.tidyverse.org/reference/tbl_vars.html)
is a thin generic: its top-level function body always returns
`new_sel_vars(tbl_vars_dispatch(x), group_vars(x))`, where
`tbl_vars_dispatch()` re-dispatches to `tbl_vars.<class>()` if one
exists (falling back to `tbl_vars.default <- function(x) names(x)`
otherwise). So a `tbl_vars.tbl_gpu` method plugs straight into that
machinery – it only needs to return every column name (grouped or not);
[`group_vars()`](https://dplyr.tidyverse.org/reference/group_data.html)
(already implemented, see above) supplies the grouping half.

## Usage

``` r
# S3 method for class 'tbl_gpu'
tbl_vars(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

A character vector of every column name (schema-aware: reflects pending
lazy ops via
[`current_schema()`](https://bbtheo.github.io/cuplyr/reference/current_schema.md),
not just the base `$schema`).

## Details

Added to support
[`dplyr::add_tally()`](https://dplyr.tidyverse.org/reference/count.html),
which is a plain (non-generic) function in dplyr 1.2 that calls
`tbl_vars(x)` directly for its name- collision check
(`check_n_name(name, tbl_vars(x))`, i.e. it avoids colliding with ANY
existing column, not just the grouping columns – see
[`tally()`](https://dplyr.tidyverse.org/reference/count.html)'s own
narrower `check_n_name(name, group_vars(x))` check for the contrasting
case, R/count.R). Before this method existed,
`dplyr::add_tally(gpu_table)` errored inside `tbl_vars_dispatch()`
(`UseMethod("tbl_vars")` found no applicable method for a plain-list
`tbl_gpu` object, and the generic's own `tbl_vars.default` doesn't apply
since `tbl_gpu` isn't a data.frame subclass).

[`dplyr::tbl_nongroup_vars()`](https://dplyr.tidyverse.org/reference/tbl_vars.html)
is a plain, non-generic function defined purely in terms of
`setdiff(tbl_vars(x), group_vars(x))` – it works transparently for
`tbl_gpu` via this method plus
[`group_vars.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/group_vars.tbl_gpu.md),
with no `tbl_gpu`-specific code of its own.
