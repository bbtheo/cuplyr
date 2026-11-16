# `semi_join()`/`anti_join()`: native via `cudf::filtered_join`

Native GPU implementation via `cudf::filtered_join` (Phase 7 J2). Unlike
the mutating joins, dplyr's own
[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)/[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
signature has no `suffix`/`keep`/`multiple`/`unmatched`/`relationship`
arguments at all (verified via `args(dplyr:::semi_join.data.frame)`:
`x, y, by, copy, ..., na_matches`) – there's nothing to merge, so no
column-collision handling is needed. The result is x's own schema
verbatim (a row subset of x), and x's
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
grouping is preserved (as for every join type, see
[`push_join()`](https://bbtheo.github.io/cuplyr/reference/push_join.md),
`R/execute.R`).

## Usage

``` r
# S3 method for class 'tbl_gpu'
semi_join(x, y, by = NULL, copy = FALSE, ..., na_matches = "na")

# S3 method for class 'tbl_gpu'
anti_join(x, y, by = NULL, copy = FALSE, ..., na_matches = "na")
```

## Arguments

- x, y:

  `tbl_gpu` objects to join. If `y` is not a `tbl_gpu`, set
  `copy = TRUE` to upload it automatically.

- by:

  Join specification – see "Details" above and `?mutating-joins`.

- copy:

  Whether to upload a non-`tbl_gpu` `y` via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Not used; dplyr itself requires these to be empty.

- na_matches:

  `"na"` (default, `NA` keys match each other) or `"never"` (`NA` keys
  never match anything).

## Value

A `tbl_gpu` with `x`'s own schema, filtered to the rows that
([`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html))
or don't
([`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html))
have a match in `y`. `x`'s
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
grouping is preserved; on a `.drop = FALSE` table every group of `x` is
kept, including the ones the join empties, like dplyr.

## Details

`by` accepts the same forms as the mutating joins (see
`?mutating-joins`): `NULL`, a character vector, or a
[`dplyr::join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
spec, including non-equi conditions (`>=`, `>`, `<=`, `<`,
[`between()`](https://dplyr.tidyverse.org/reference/between.html)/[`within()`](https://rdrr.io/r/base/with.html)/
`overlaps()`) via `cudf::conditional_join`/`mixed_join` (Phase 7 J7). A
single `closest()` condition is native (it asks whether an x row has a
closest match), as are non-equi conditions over two `STRING` columns;
shapes outside that fall back to CPU evaluation transparently (with a
`cuplyr.fallback` notification), same as the mutating joins.
