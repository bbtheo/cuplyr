# Set the dimnames of a GPU table

Only the column names can be set (`colnames(x) <- value`); rows are
unnamed, so `value[[1]]` must be `NULL`.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dimnames(x) <- value
```

## Arguments

- x:

  A `tbl_gpu` object.

- value:

  `list(NULL, <column names>)`.

## Value

The modified `tbl_gpu` object.
