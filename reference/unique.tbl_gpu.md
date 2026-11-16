# Unique rows of a GPU table

`unique(x)` is `distinct(x)`: duplicate rows over all columns are
dropped.

## Usage

``` r
# S3 method for class 'tbl_gpu'
unique(x, incomparables = FALSE, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- incomparables:

  Only `FALSE`, as in
  [`unique.data.frame()`](https://rdrr.io/r/base/unique.html).

- ...:

  Only `fromLast = FALSE` is supported (a GPU
  [`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)
  keeps the first occurrence); `fromLast = TRUE` and other arguments are
  errors rather than silently ignored.

## Value

A `tbl_gpu` with duplicate rows removed (groups are kept).
