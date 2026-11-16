# Get grouping information from a GPU table

Returns a list of symbols representing the grouping columns.

## Usage

``` r
# S3 method for class 'tbl_gpu'
groups(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

A list of symbols for the grouping columns.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  grouped <- gpu_mtcars |>
    group_by(cyl)

  groups(grouped)  # list(as.symbol("cyl"))
}
#> [[1]]
#> cyl
#> 
```
