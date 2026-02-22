# Display GPU information

Prints formatted information about the available GPU to the console.
Useful for verifying GPU setup and checking available resources.

## Usage

``` r
show_gpu()
```

## Value

Invisibly returns the GPU info list (same as
[`gpu_details()`](gpu_details.md)).

## Details

Output includes:

- Device name and compute capability

- Total, free, and used memory

- Number of streaming multiprocessors

## See also

[`has_gpu`](has_gpu.md) for availability check,
[`gpu_details`](gpu_details.md) for programmatic access

## Examples

``` r
show_gpu()
#> GPU Information
#> ---------------------------------------- 
#> Device:       NVIDIA GeForce RTX 5070 
#> Compute:      12.0 
#> Memory:       12.3 GB total
#>               10.9 GB free
#>               1.4 GB used
#> SMs:          1 
```
