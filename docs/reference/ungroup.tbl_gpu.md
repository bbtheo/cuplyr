# Remove grouping from a GPU table

Removes all grouping information from a grouped `tbl_gpu` object.

## Usage

``` r
# S3 method for class 'tbl_gpu'
ungroup(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Ignored. Included for compatibility with dplyr generic.

## Value

An ungrouped `tbl_gpu` object.

## Examples

``` r
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
#> Error in group_by(gpu_mtcars, cyl): could not find function "group_by"
```
