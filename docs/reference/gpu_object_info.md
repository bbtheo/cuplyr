# Get detailed information about a GPU table object

Returns comprehensive information about a `tbl_gpu` object including its
dimensions, column types, estimated memory usage, and verification that
data resides on the GPU.

## Usage

``` r
gpu_object_info(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

A list with the following components:

- valid:

  Logical: TRUE if the object is a valid tbl_gpu with GPU data

- nrow:

  Number of rows

- ncol:

  Number of columns

- column_names:

  Character vector of column names

- column_types:

  Character vector of GPU column types

- estimated_gpu_bytes:

  Estimated GPU memory usage in bytes

- estimated_gpu_mb:

  Estimated GPU memory usage in megabytes

- r_object_bytes:

  Size of the R object (should be small)

- data_on_gpu:

  Logical: TRUE if data is verified to be on GPU

- pointer_valid:

  Logical: TRUE if the external pointer is valid

## See also

[`gpu_memory_usage`](gpu_memory_usage.md) for just the memory estimate,
[`verify_gpu_data`](verify_gpu_data.md) to check if data is on GPU

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  info <- gpu_object_info(gpu_mtcars)
  cat("Rows:", info$nrow, "\n")
  cat("GPU memory:", round(info$estimated_gpu_mb, 2), "MB\n")
  cat("Data on GPU:", info$data_on_gpu, "\n")
}
#> Rows: 32 
#> GPU memory: 0 MB
#> Data on GPU: TRUE 
```
