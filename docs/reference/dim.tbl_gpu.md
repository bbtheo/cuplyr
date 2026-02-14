# Get dimensions of a GPU table

Returns the number of rows and columns in a GPU table.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dim(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

An integer vector of length 2: c(nrow, ncol). Returns \`c(NA, ncol)

## Examples

``` r
if (has_gpu()) {



}
#> NULL
```
