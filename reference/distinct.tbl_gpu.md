# Keep distinct rows

[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)
keeps the first row of each distinct combination of values, in their
original order, as
[`dplyr::distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)
does. It runs on the GPU.

## Usage

``` r
# S3 method for class 'tbl_gpu'
distinct(.data, ..., .keep_all = FALSE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Optional variables to use when determining uniqueness. When every dot
  is an unnamed column reference (`distinct(df, x, .data$y)`) the keys
  are those columns, looked up by name without evaluating anything. When
  any dot is named or computed (`distinct(df, z = x + 1)`), the computed
  keys are added as in
  [`mutate.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
  before the duplicates are dropped; a key expression the GPU can't
  compute runs the whole call in dplyr on the CPU. If omitted,
  uniqueness is determined using all columns.

- .keep_all:

  If `TRUE`, keeps all columns (in their original order, with any newly
  computed key columns appended at the end, as
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) adds
  them). If `FALSE` (the default), only the columns used for the
  uniqueness check are kept, in the order given in `...`.

## Value

A `tbl_gpu` object with duplicate rows removed.

## Details

### Column selection and order

With no `...`, every column is the key and every column is kept
(`.keep_all` makes no difference then). With `...`, the output's columns
(when `.keep_all = FALSE`) are exactly the `...` expressions, in the
order given: `distinct(df, y, x)` returns columns `y, x`, not `x, y`.

### Grouped tables

If `.data` is grouped, the grouping columns are always part of the key
and always appear in the result, even if not named in `...`: as in
dplyr, grouping columns missing from `...` come first, ahead of the
`...` columns (a grouping column already in `...` keeps its position).
The result keeps the grouping of `.data`.

### NA and NaN

Two `NA` key values are equal (one `NA` row survives), and so are two
`NaN` key values, as in
[`dplyr::distinct()`](https://dplyr.tidyverse.org/reference/distinct.html).

## See also

[`mutate.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/mutate.tbl_gpu.md)
for creating new columns,
[`group_by.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md)
for grouping,
[`collect.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)
for retrieving results

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # All columns as key, original row order preserved
  result <- gpu_mtcars |> distinct() |> collect()

  # Key on specific columns, only those columns in the result
  result <- gpu_mtcars |> distinct(cyl, gear) |> collect()

  # Key on specific columns, but keep every column (first row per key)
  result <- gpu_mtcars |> distinct(cyl, .keep_all = TRUE) |> collect()
}
```
