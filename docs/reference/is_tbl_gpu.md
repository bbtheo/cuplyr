# Test if an object is a GPU table

Checks whether an object inherits from the `tbl_gpu` class.

## Usage

``` r
is_tbl_gpu(x)
```

## Arguments

- x:

  An R object to test.

## Value

`TRUE` if `x` is a `tbl_gpu` object, `FALSE` otherwise.

## Examples

``` r
if (has_gpu()) {
  gpu_df <- tbl_gpu(mtcars)
  is_tbl_gpu(gpu_df)
  is_tbl_gpu(mtcars)
}
#> [1] FALSE
```
