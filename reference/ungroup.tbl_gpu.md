# Remove grouping from a GPU table

Removes grouping from a grouped `tbl_gpu` object: all of it, or only the
group columns selected in `...`.

## Usage

``` r
# S3 method for class 'tbl_gpu'
ungroup(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Columns to remove from the grouping (tidyselect; renaming is an
  error). Empty removes all groups. Must be empty for an ungrouped
  table.

## Value

An ungrouped `tbl_gpu` object.

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  grouped <- gpu_mtcars |>
    group_by(cyl)

  # Remove grouping
  ungrouped <- grouped |>
    ungroup()

  # Verify groups are removed
  length(group_vars(ungrouped))  # 0
}
#> [1] 0
```
