# Filtering joins for `tbl_gpu` (`semi_join()`, `anti_join()`)

[`semi_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
keeps the rows of `x` that have a match in `y`, and
[`anti_join()`](https://dplyr.tidyverse.org/reference/filter-joins.html)
the rows that have none, as in dplyr. Both run on the GPU. As in dplyr,
they take no `suffix`/`keep`/`multiple`/`unmatched`/ `relationship`
arguments: nothing from `y` is added, so the result has `x`'s columns
and a subset of its rows (each kept row once, duplicates of `x`
included), and keeps `x`'s
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
grouping.

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
`overlaps()`). A single `closest()` condition runs on the GPU (it asks
whether an x row has a closest match), as do non-equi conditions over
two string columns; other shapes run in dplyr on the CPU, as for the
mutating joins (set `options(cuplyr.fallback = "warn")` to be told when;
see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)).
