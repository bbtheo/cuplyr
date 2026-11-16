# Coerce a GPU table to a base data frame, tibble, list or matrix

These collect the table (running any pending lazy operations) and coerce
the result as the corresponding method for a tibble does: groups are
dropped (as for a `grouped_df`), and
[`as.list()`](https://rdrr.io/r/base/list.html) gives a plain named list
of the columns, so
[`lapply()`](https://rdrr.io/r/base/lapply.html)/[`sapply()`](https://rdrr.io/r/base/lapply.html)
over a `tbl_gpu` iterate the columns.

## Usage

``` r
# S3 method for class 'tbl_gpu'
as.data.frame(x, row.names = NULL, optional = FALSE, ...)

# S3 method for class 'tbl_gpu'
as_tibble(x, ...)

# S3 method for class 'tbl_gpu'
as.list(x, ...)

# S3 method for class 'tbl_gpu'
as.matrix(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- row.names, optional:

  Passed to
  [`base::as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html).

- ...:

  Passed on to the tibble/data frame method.

## Value

A `data.frame`, `tibble`, `list` or `matrix`.
