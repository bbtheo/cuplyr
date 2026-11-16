# System diagnostics for cuplyr

Collects comprehensive system information for debugging installation
issues or filing bug reports. Includes cuplyr version, R/Rcpp versions,
CUDA/cuDF paths, GPU info, and environment variables.

## Usage

``` r
diagnostics(redact = TRUE)
```

## Arguments

- redact:

  If `TRUE` (default), redact potentially sensitive paths (home
  directory, usernames). Set to `FALSE` for local debugging.

## Value

Invisibly returns a list with all diagnostic information. Also prints
formatted output to the console.

## Details

The output includes:

- **Package versions**: cuplyr, R, Rcpp, dplyr

- **CUDA info**: toolkit version, driver version, CUDA_HOME path

- **libcudf info**: CONDA_PREFIX, CUDF_HOME, detected paths

- **GPU info**: device name, compute capability, memory

- **Library paths**: LD_LIBRARY_PATH, RPATH of cuplyr.so

- **Environment**: OS, distro, key environment variables

## See also

[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
for pre-install checks,
[`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
for post-install verification

## Examples

``` r
# Full diagnostics
diagnostics()
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.1.1.9000 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  ~/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     ~/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.9 GiB free
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues

# Redacted for sharing in bug reports
diagnostics(redact = TRUE)
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.1.1.9000 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  ~/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     ~/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.9 GiB free
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues

# Unredacted for local debugging
diagnostics(redact = FALSE)
#> cuplyr diagnostics
#> ================================================== 
#> 
#> Package Versions
#>   cuplyr:     0.1.1.9000 
#>   R:         4.6.1 (x86_64-pc-linux-gnu)
#>   Rcpp:       1.1.2 
#>   dplyr:      1.2.1 
#> 
#> System
#>   OS:         Linux 
#>   Distro:     Omarchy 
#> 
#> CUDA
#>   CUDA_HOME:  /opt/cuda 
#>   CUDA ver:   13.4 
#>   Driver:     615.78.08 
#> 
#> RAPIDS / libcudf
#>   CONDA_PREFIX:  /home/theo/cuplr/.pixi/envs/default 
#>   CUDF_HOME:     /home/theo/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf 
#> 
#> GPU
#>   Available:  YES
#>   Name:       NVIDIA GeForce RTX 5070 
#>   Compute:    12.0 
#>   Memory:     11.6 GiB total,  9.9 GiB free
#> 
#> Library Paths
#>   LD_LIBRARY_PATH:  /usr/lib64/R/lib:/usr/lib/jvm/java-26-openjdk/lib/server 
#>   cuplyr.so:        not found 
#> 
#> ================================================== 
#> Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues
```
