# Dimnames of a GPU table

Rows are unnamed; the column names are the table's current names (no
plan is run), so [`colnames()`](https://rdrr.io/r/base/colnames.html)
works.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dimnames(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

`list(NULL, <column names>)`.
