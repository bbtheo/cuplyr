# Get detailed GPU information

Retrieves comprehensive information about the available GPU including
device name, compute capability, memory capacity, and multiprocessor
count.

## Usage

``` r
gpu_details()
```

## Value

A named list with GPU details:

- available:

  Logical: TRUE if GPU is available

- device_count:

  Number of CUDA devices

- device_id:

  ID of the current device (0-indexed)

- name:

  GPU model name (e.g., "NVIDIA GeForce RTX 4090")

- compute_capability:

  CUDA compute capability (e.g., "8.9")

- total_memory:

  Total GPU memory in bytes

- free_memory:

  Currently available GPU memory in bytes

- multiprocessors:

  Number of streaming multiprocessors (SMs)

If no GPU is available, returns
`list(available = FALSE, device_count = 0)`.

## Details

### Compute capability

The compute capability indicates the GPU architecture and supported
features:

- 7.x - Volta/Turing (V100, RTX 20 series)

- 8.x - Ampere (A100, RTX 30 series)

- 8.9 - Ada Lovelace (RTX 40 series)

- 9.x - Hopper (H100)

- 10.x+ - Blackwell and newer

### Memory

The `free_memory` value reflects memory available at the time of the
call. Other applications or CUDA contexts may be using GPU memory.

## See also

[`has_gpu`](https://bbtheo.github.io/cuplyr/reference/has_gpu.md) for
simple availability check,
[`show_gpu`](https://bbtheo.github.io/cuplyr/reference/show_gpu.md) for
formatted display

## Examples

``` r
info <- gpu_details()
if (info$available) {
  cat("GPU:", info$name, "\n")
  cat("Memory:", round(info$total_memory / 1e9, 1), "GB\n")
  cat("Compute:", info$compute_capability, "\n")
}
#> GPU: NVIDIA GeForce RTX 5070 
#> Memory: 12.3 GB
#> Compute: 12.0 
```
