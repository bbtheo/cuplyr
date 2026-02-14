# Group a GPU table by one or more columns

Marks columns to group by for subsequent operations like
[`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).
The grouping is stored as metadata and does not perform any computation
until an aggregation is requested.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_by(.data, ..., .add = FALSE, .drop = TRUE)
```

## Arguments

- .data:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Column names to group by. Can be unquoted column names or tidyselect
  expressions.

- .add:

  If `FALSE` (default), will override existing groups. If `TRUE`, will
  add to existing groups.

- .drop:

  Ignored. Included for compatibility with dplyr generic.

## Value

A grouped `tbl_gpu` object. The object has the same data but with
grouping columns recorded for use by
[`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html).

## Details

Unlike operations like
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html)
or
[`dplyr::mutate()`](https://dplyr.tidyverse.org/reference/mutate.html),
`group_by()` does not perform any GPU computation. It simply records
which columns should be used for grouping in subsequent aggregation
operations.

The actual groupby computation happens when you call
[`dplyr::summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
on the grouped table. This lazy approach allows you to chain multiple
operations before executing the expensive groupby operation.

## See also

[`summarise.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/summarise.tbl_gpu.md)
for aggregating grouped data,
[`ungroup.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/ungroup.tbl_gpu.md)
for removing grouping

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Group by a single column
  by_cyl <- gpu_mtcars |>
    group_by(cyl)

  # Group by multiple columns
  by_cyl_gear <- gpu_mtcars |>
    group_by(cyl, gear)

  # Use with summarise for aggregation
  result <- gpu_mtcars |>
    group_by(cyl) |>
    summarise(mean_mpg = mean(mpg)) |>
    collect()
}
#> Error in group_by(gpu_mtcars, cyl): could not find function "group_by"
```
