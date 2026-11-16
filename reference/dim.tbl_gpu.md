# Get dimensions of a GPU table

Returns the number of rows and columns in a GPU table. For a lazy table
with pending operations the row count is that of the RESULT, so the plan
is run on the GPU
([`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
nothing is copied to R and `x` itself is not changed).
[`ncol()`](https://rdrr.io/r/base/nrow.html)/[`NCOL()`](https://rdrr.io/r/base/nrow.html)
only need the column count, so they are answered from the schema without
running anything.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dim(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

An integer vector of length 2: c(nrow, ncol). The row count is `NA` for
a table that has no GPU data yet, for
[`ncol()`](https://rdrr.io/r/base/nrow.html)/[`NCOL()`](https://rdrr.io/r/base/nrow.html)
of a lazy table with pending operations (which ignore it), and for a
zero-column table whose row count was never recorded.

## Examples

``` r
if (has_gpu()) {
  dim(tbl_gpu(mtcars))
}
#> [1] 32 11
```
