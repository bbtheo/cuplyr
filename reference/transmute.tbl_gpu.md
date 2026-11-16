# Create a new GPU table with only the specified columns

Computes new columns (or replaces existing ones) from expressions, like
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md),
but keeps ONLY the columns named/created by `...` (plus group columns),
similar to
[`dplyr::transmute()`](https://dplyr.tidyverse.org/reference/transmute.html).
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html) is
superseded in dplyr (in favor of `mutate(.keep = "none")`) but still
fully supported, and its column-ordering rule is subtly different from
`mutate(.keep = "none")`'s – see "Details" below.

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

Verified empirically against dplyr 1.2.1: unlike
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) (whose
output order keeps existing columns in their original relative position
and appends new ones at the end, see
[`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)'s
docs),
[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)'s
output order is simply the literal `...` dot order – whether a dot is a
bare existing column or a brand-new expression. Group columns NOT
explicitly mentioned in `...` are prepended at the very front (in
[`group_vars()`](https://dplyr.tidyverse.org/reference/group_data.html)
order); a group column that IS explicitly mentioned in `...` takes its
dot position instead of being prepended.

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
