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
  the number of columns, contain no `NA` values, and no empty strings.

## Value

The modified `tbl_gpu` object.
