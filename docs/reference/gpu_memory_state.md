# Get GPU memory snapshot

Returns the current GPU memory state including total, free, and used
memory. Useful for monitoring GPU memory usage during operations.

## Usage

``` r
gpu_memory_state()
```

## Value

A list with:

- available:

  Logical: TRUE if GPU is available

- total_bytes:

  Total GPU memory in bytes

- free_bytes:

  Free GPU memory in bytes

- used_bytes:

  Used GPU memory in bytes

- total_gb:

  Total GPU memory in gigabytes

- free_gb:

  Free GPU memory in gigabytes

- used_gb:

  Used GPU memory in gigabytes

## See also

[`gpu_details`](gpu_details.md) for device information,
[`gpu_memory_usage`](gpu_memory_usage.md) for per-object memory
estimates

## Examples

``` r
if (has_gpu()) {
  # Check memory before allocation
  before <- gpu_memory_state()

  # Allocate some GPU data
  gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))

  # Check memory after allocation
  after <- gpu_memory_state()

  cat("Memory used:", after$used_gb - before$used_gb, "GB\n")
}
#> Memory used: 0.008388608 GB
```
