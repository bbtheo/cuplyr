# Resolve `y` for a set operation: must be a `tbl_gpu` or a `data.frame`

Unlike the join verbs, dplyr's own set-op generics have no `copy=`
argument at all (verified via `args(dplyr::union)`: `x, y, ...`) – a
plain `data.frame` `y` is uploaded automatically (mirrors
`collect_other_side()`'s inverse, R/fallback.R), and anything else (e.g.
a bare vector) gets dplyr's own exact error text (verified empirically:
`union(tibble(a = 1), 1:3)` errors with "`y` must be a data frame.", no
bullet marker).

## Usage

``` r
resolve_set_op_other_side(y)
```
