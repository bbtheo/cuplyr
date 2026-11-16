# Validate cloud runtime loader resolution after install

Confirms cuplyr.so RUNPATH includes conda lib and that ldd resolves
libstdc++.so.6 from the conda prefix (not system path).

## Usage

``` r
validate_cloud_loader_resolution(conda_prefix, verbose = FALSE)
```

## Arguments

- conda_prefix:

  RAPIDS/conda prefix.

- verbose:

  Logical; print pass details.
