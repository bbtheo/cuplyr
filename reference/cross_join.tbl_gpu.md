# Cross join for `tbl_gpu`

[`cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
pairs every row of `x` with every row of `y`, on the GPU, as
[`dplyr::cross_join()`](https://dplyr.tidyverse.org/reference/cross_join.html)
does. As in dplyr, there are no keys, so it takes no
`by`/`keep`/`na_matches`/`multiple`/`unmatched`/`relationship`
arguments. Also as in dplyr:

- All of `x`'s columns come first (in their original order), then all of
  `y`'s.

- Every column name the two tables share gets both suffixes: two tables
  that both have `id` and `val` give `id.x`, `val.x`, `id.y`, `val.y`,
  never a bare shared name.

- `x`'s grouping columns keep grouping the result, except a grouping
  column whose name gets a suffix (`"g"` -\> `"g.x"`), which drops out.
  `y`'s grouping is ignored.

- A 0-row `x` or `y` gives a 0-row result with the merged columns.

- `copy = TRUE` with a plain `data.frame` `y` works like every other
  join.

- A side with rows but no columns repeats each row of the other side
  once per row it has, and two such tables give a table of
  `nrow(x) * nrow(y)` rows and no columns.

## Usage

``` r
# S3 method for class 'tbl_gpu'
cross_join(x, y, ..., copy = FALSE, suffix = c(".x", ".y"))
```

## Arguments

- x, y:

  `tbl_gpu` objects. If `y` is not a `tbl_gpu`, set `copy = TRUE` to
  upload it automatically.

- ...:

  Not used; dplyr itself requires these to be empty.

- copy:

  Whether to upload a non-`tbl_gpu` `y` via
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- suffix:

  Length-2 character vector disambiguating every common column name
  (every shared name gets both suffixes, since there are no join keys to
  coalesce). A `suffix = c("", ...)` that still leaves two columns with
  the same name runs in dplyr on the CPU, as for the mutating joins.

## Value

A `tbl_gpu` with every row of `x` paired with every row of `y`
(`nrow(x) * nrow(y)` rows), columns from `x` first then `y`. `x`'s
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
grouping is preserved for any group column whose name survives suffixing
unchanged.
