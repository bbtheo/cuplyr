# Get the schema a new op should be built against

The table's stored `$schema`. For a lazy table with pending operations
it is the output schema of `$lazy_ops`, factor levels and POSIXct zones
included: every writer of `lazy_ops`
([`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md),
[`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md),
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
[`collapse()`](https://dplyr.tidyverse.org/reference/compute.html)'s
barrier, the callers that copy both fields) stores the schema it
implies, so the invariant `schema == infer_schema_full(lazy_ops)` holds
and reading it is O(1), whatever the plan's depth. Set
`options(cuplyr.check_schema_cache = TRUE)` or the environment variable
`CUPLYR_CHECK_SCHEMA_CACHE=1` to verify the invariant on every call
(internal debug switch for test runs; aborts with
`cuplyr_error_internal`).

## Usage

``` r
current_schema(.data)
```

## Arguments

- .data:

  A `tbl_gpu` object

## Value

List with `names` and `types` (and possibly `factor_levels`, `tz`)
