# Configure LD_LIBRARY_PATH for cloud environments

Sets LD_LIBRARY_PATH with conda lib FIRST (for newer libstdc++), then
driver path (for libcuda.so.1), then CUDA runtime.

## Usage

``` r
configure_cloud_library_paths(driver_lib, conda_prefix)
```

## Arguments

- driver_lib:

  Path to directory with real libcuda.so.1.

- conda_prefix:

  RAPIDS/conda prefix.
