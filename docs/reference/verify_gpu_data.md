# Verify that data resides on GPU

Performs checks to confirm that a `tbl_gpu` object has its data stored
on the GPU, not in R memory. This is useful for debugging and ensuring
GPU operations are working correctly.

## Usage

``` r
verify_gpu_data(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

`TRUE` if all checks pass and data is verified to be on GPU, `FALSE`
otherwise.

## Details

This function performs multiple verification steps:

1.  Object has the `tbl_gpu` class

2.  Object has a valid external pointer

3.  GPU operations (dim, types) work on the pointer

4.  R object is small (no data copy in R memory)

## See also

[`gpu_object_info`](gpu_object_info.md) for detailed object information,
[`tbl_gpu`](tbl_gpu.md) for creating GPU tables

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Should return TRUE
  verify_gpu_data(gpu_mtcars)

  # Regular data frames return FALSE
  verify_gpu_data(mtcars)
}
#> [1] FALSE
```
