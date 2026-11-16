# Get grouping variables from a GPU table

Returns the names of columns used for grouping.

## Usage

``` r
# S3 method for class 'tbl_gpu'
group_vars(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

A character vector of grouping column names.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  grouped <- gpu_mtcars |>
    group_by(cyl, gear)

  group_vars(grouped)  # c("cyl", "gear")
}
#> [1] "cyl"  "gear"
```
