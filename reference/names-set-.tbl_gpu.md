# Set column names of a GPU table

Set column names of a GPU table

## Usage

``` r
# S3 method for class 'tbl_gpu'
names(x) <- value
```

## Arguments

- x:

  A `tbl_gpu` object.

- value:

  A character vector of new column names. Must have the same length as
  the number of columns. `NA` and `""` are allowed, as in a data frame
  (data-masking verbs then refuse the table, like dplyr).

## Value

The modified `tbl_gpu` object.
