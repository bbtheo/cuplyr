# Compare R object size vs GPU data size

Computes the ratio of GPU memory usage to R object size for a `tbl_gpu`
object. A high ratio confirms that data is stored on GPU, not in R.

## Usage

``` r
gpu_size_comparison(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

A list with:

- r_bytes:

  Size of the R object in bytes

- gpu_bytes:

  Estimated GPU memory in bytes

- ratio:

  GPU size divided by R size (should be \> 1 if data is on GPU)

## See also

[`verify_gpu_data`](https://bbtheo.github.io/cuplyr/reference/verify_gpu_data.md)
for boolean verification,
[`gpu_object_info`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
for detailed information

## Examples

``` r
if (has_gpu()) {
  # Create a larger dataset
  df <- data.frame(matrix(runif(10000), ncol = 10))
  gpu_df <- tbl_gpu(df)

  comparison <- gpu_size_comparison(gpu_df)
  cat("R object:", round(comparison$r_bytes / 1024, 1), "KB\n")
  cat("GPU data:", round(comparison$gpu_bytes / 1024, 1), "KB\n")
  cat("Ratio:", round(comparison$ratio, 1), "x\n")
}
#> R object: 3.2 KB
#> GPU data: 79.3 KB
#> Ratio: 24.8 x
```
