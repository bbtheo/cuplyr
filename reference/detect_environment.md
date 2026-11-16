# Detect the runtime environment

Identifies whether we're running on Colab, in a container, on a cloud
GPU instance, or locally. Used by
[`install_cuplyr()`](https://bbtheo.github.io/cuplyr/reference/install_cuplyr.md)
to apply environment-specific fixes (e.g. CUDA stub disabling on Colab).

## Usage

``` r
detect_environment(override = Sys.getenv("CUPLYR_ENV"))
```

## Arguments

- override:

  Explicit override, e.g. from `CUPLYR_ENV` env var.

## Value

One of `"local"`, `"colab"`, `"container"`, `"cloud_gpu"`.
