# Create a new GPU table with only the specified columns

Computes new columns (or replaces existing ones) from expressions, like
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md),
but keeps only the columns named or created by `...` (plus grouping
columns), as
[`dplyr::transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)
does. dplyr supersedes
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html) in
favor of `mutate(.keep = "none")`; it still works here, and its column
order differs from `mutate(.keep = "none")`'s (see "Details" below).

## Usage

``` r
# S3 method for class 'tbl_gpu'
transmute(.data, ...)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Name-value pairs of expressions, exactly as for
  [`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md).
  A bare column name (`transmute(df, x)`) keeps that column.

## Value

A `tbl_gpu` object with only the columns `...` names/creates, plus any
group columns not otherwise mentioned.

## Details

As in dplyr, and unlike
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) (which
keeps existing columns where they are and appends new ones at the end,
see
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)),
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)'s
output order is the order of the dots, whether a dot is a bare existing
column or a new expression. Grouping columns not mentioned in `...` come
first (in
[`group_vars()`](https://dplyr.tidyverse.org/reference/group_data.html)
order); a grouping column that is mentioned in `...` takes its dot's
position instead.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  result <- gpu_mtcars |>
    transmute(kpl = mpg * 0.425, cyl) |>
    collect()
}
```
