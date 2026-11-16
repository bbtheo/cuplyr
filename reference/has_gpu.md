# Check GPU availability

Tests whether a CUDA-capable GPU is available and accessible. This
function is useful for conditional code that should only run when GPU
acceleration is possible.

## Usage

``` r
has_gpu()
```

## Value

`TRUE` if a GPU is available and CUDA is properly configured, `FALSE`
otherwise. Returns `FALSE` (not an error) if CUDA libraries are not
found.

## Details

This function checks:

1.  CUDA driver is loaded

2.  At least one CUDA device is present

3.  Device is accessible (not in exclusive mode by another process)

## See also

[`gpu_details`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
for detailed GPU information,
[`show_gpu`](https://bbtheo.github.io/cuplyr/reference/show_gpu.md) for
formatted GPU info display

## Examples

``` r
if (has_gpu()) {
  message("GPU acceleration available!")
  gpu_df <- tbl_gpu(mtcars)
} else {
  message("No GPU found, using CPU")
}
#> GPU acceleration available!
```
