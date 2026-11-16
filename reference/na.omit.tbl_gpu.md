# Drop rows with missing values

[`stats::na.omit()`](https://rdrr.io/r/stats/na.fail.html) for a
`tbl_gpu`: keeps the rows where no column is `NA` (`NaN` counts as
missing, as in [`is.na()`](https://rdrr.io/r/base/NA.html); an
[`addNA()`](https://rdrr.io/r/base/factor.html) factor level is a value,
not a missing one). It is an ordinary native
[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html),
so the table keeps its execution mode, grouping and attributes, and lazy
stays lazy.

## Usage

``` r
# S3 method for class 'tbl_gpu'
na.omit(object, ...)
```

## Arguments

- object:

  A `tbl_gpu`.

- ...:

  Ignored (as in `na.omit.data.frame()`).

## Value

A `tbl_gpu` without the incomplete rows.

## Details

Unlike `na.omit.data.frame()`, the result has no `na.action` attribute:
that attribute holds the row numbers dropped, which would need an extra
pass over the table and a transfer to the host just to be computed.
