# Display GPU information

Prints formatted information about the available GPU to the console.
Useful for verifying GPU setup and checking available resources.

## Usage

``` r
show_gpu()
```

## Value

Invisibly returns the GPU info list (same as
[`gpu_details()`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)).

## Details

Output includes:

- Device name and compute capability

- Total, free, and used memory (device level: includes the memory
  cuplyr's memory resource holds, see
  [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md))

- Number of streaming multiprocessors

## See also

[`has_gpu`](https://bbtheo.github.io/cuplyr/reference/has_gpu.md) for
availability check,
[`gpu_details`](https://bbtheo.github.io/cuplyr/reference/gpu_details.md)
for programmatic access

## Examples

``` r
show_gpu()
#> GPU Information
#> ---------------------------------------- 
#> Device:       NVIDIA GeForce RTX 5070 
#> Compute:      12.0 
#> Memory:       12.4 GB total
#>               11.6 GB free
#>               0.8 GB used
#> SMs:          1 
```
