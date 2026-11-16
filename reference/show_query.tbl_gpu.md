# Show pending operations for a lazy tbl_gpu

`tbl_gpu` method for dplyr's
[`dplyr::show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
generic. Prints the pending AST (nothing is executed).

## Usage

``` r
# S3 method for class 'tbl_gpu'
show_query(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Not used.

## Value

`x`, invisibly.
