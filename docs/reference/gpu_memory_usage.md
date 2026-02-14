# Estimate GPU memory usage of a tbl_gpu object

Calculates the estimated GPU memory footprint of a GPU table based on
its dimensions and column types. This is useful for understanding memory
requirements before working with large datasets.

## Usage

``` r
gpu_memory_usage(x)
```

## Arguments

- x:

  A `tbl_gpu` object.

## Value

The estimated memory usage in bytes (numeric), or `NA` if the object is
not a valid `tbl_gpu` or has no data on GPU.

## Details

The estimate includes:

- Column data (varies by type: 8 bytes for FLOAT64, 4 bytes for INT32,
  etc.)

- Validity bitmasks for NA handling (1 bit per row per column)

String columns use an average estimate of 32 bytes per element, which
may vary significantly based on actual string lengths.

This is an estimate and actual GPU memory usage may be higher due to:

- Memory alignment requirements

- RMM memory pool overhead

- Temporary allocations

## See also

[`gpu_details`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
for overall GPU memory info,
[`gpu_object_info`](https://bbtheo.github.io/cuplyr/reference/gpu_object_info.md)
for detailed object information

## Examples

``` r
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)
  size <- gpu_memory_usage(gpu_mtcars)
  cat("Estimated GPU memory:", round(size / 1024, 1), "KB\n")
}
#> Estimated GPU memory: 2.8 KB
```
