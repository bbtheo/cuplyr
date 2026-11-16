# Explain a tbl_gpu query

`tbl_gpu` method for dplyr's
[`dplyr::explain()`](https://dplyr.tidyverse.org/reference/explain.html)
generic. Prints
[`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)'s
output and, when operations are pending, the optimized plan
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) would
run. Nothing is executed.

## Usage

``` r
# S3 method for class 'tbl_gpu'
explain(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Not used.

## Value

`x`, invisibly.
