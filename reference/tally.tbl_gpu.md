# Count the rows in each group (no additional grouping)

Like
[`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md),
but without any `...`: counts the rows of the groups `x` already has (or
of the whole table, if it is ungrouped). It is one
[`summarise.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)
call and runs on the GPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
tally(x, wt = NULL, sort = FALSE, name = NULL)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- wt:

  Optional weight column/expression, as
  [`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md).

- sort:

  If `TRUE`, sorts the result in descending order of the count column.

- name:

  The name of the count column. If `NULL` (default), uses `"n"`, or
  `"nn"`/`"nnn"`/... if `"n"` already names one of `x`'s grouping
  columns. As in dplyr, only the grouping columns are checked, where
  [`add_count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/add_count.tbl_gpu.md)
  and
  [`dplyr::add_tally()`](https://dplyr.tidyverse.org/reference/count.html)
  check every column.

## Value

A `tbl_gpu` with one row per group: the grouping columns and the count
column. As after
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
the result drops the last grouping column and keeps the others; this
differs from
[`count.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md),
which keeps `x`'s grouping in full.

## See also

[`count.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/count.tbl_gpu.md)

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  gpu_mtcars |> group_by(cyl) |> tally() |> collect()
}
#> # A tibble: 3 × 2
#>     cyl     n
#>   <dbl> <int>
#> 1     4    11
#> 2     6     7
#> 3     8    14
```
