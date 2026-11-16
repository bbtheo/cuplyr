# Check cuplyr dependencies

Performs a pre-flight check of all dependencies required to install and
run cuplyr. Reports status of NVIDIA driver, CUDA toolkit, libcudf, R
version, and required R packages.

## Usage

``` r
check_deps(format = c("text", "json"), verbose = FALSE)
```

## Arguments

- format:

  Output format: `"text"` for human-readable console output, `"json"`
  for machine-readable JSON string.

- verbose:

  If `TRUE`, print additional details about each check.

## Value

Invisibly returns a list with:

- ok:

  Logical: `TRUE` if all required checks pass

- checks:

  Named list of individual check results, each with `ok`, `value`, and
  `message` fields

## Details

The following checks are performed:

1.  **NVIDIA Driver**: Verifies `nvidia-smi` is available and reports
    driver version

2.  **CUDA Toolkit**: Checks for `nvcc`, validates CUDA \>= 12.2, and
    reports CUDA version

3.  **libcudf**: Looks for cudf headers in standard paths

4.  **R Version**: Checks R \>= 4.3.0

5.  **R Packages**: Verifies required R packages are installed

6.  **GPU Access**: Tests if a GPU device is accessible (if cuplyr is
    already installed)

## See also

[`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
for post-install verification,
[`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
for full system diagnostics

## Examples

``` r
# Human-readable output
check_deps()
#> cuplyr dependency check
#> -------------------------------------------------- 
#> ✔ NVIDIA Driver: nvidia-smi found (version unknown)
#> ✔ CUDA Toolkit: CUDA 13.4 at /opt/cuda
#> ✔ libcudf: Found headers and libcudf.so at /home/theo/cuplr/.pixi/envs/default
#> ✔ R Version: R 4.6.1
#> ✔ R Packages: All 7 required packages installed
#> ✔ GPU Access: GPU accessible: NVIDIA GeForce RTX 5070
#> -------------------------------------------------- 
#> All checks passed. 
#> Next step: ./configure && R CMD INSTALL .

# Machine-readable JSON
check_deps(format = "json")
#> {
#>   "ok": true,
#>   "checks": {
#>     "nvidia_driver": {"ok": true, "value": null, "message": "nvidia-smi found (version unknown)"},
#>     "cuda_toolkit": {"ok": true, "value": "13.4", "message": "CUDA 13.4 at /opt/cuda"},
#>     "libcudf": {"ok": true, "value": "/home/theo/cuplr/.pixi/envs/default", "message": "Found headers and libcudf.so at /home/theo/cuplr/.pixi/envs/default"},
#>     "r_version": {"ok": true, "value": "4.6.1", "message": "R 4.6.1"},
#>     "r_packages": {"ok": true, "value": "Rcpp, dplyr, rlang, vctrs, cli, tidyselect, tibble", "message": "All 7 required packages installed"},
#>     "gpu_access": {"ok": true, "value": "NVIDIA GeForce RTX 5070", "message": "GPU accessible: NVIDIA GeForce RTX 5070"}
#>   }
#> } 

# Programmatic use
result <- check_deps()
#> cuplyr dependency check
#> -------------------------------------------------- 
#> ✔ NVIDIA Driver: nvidia-smi found (version unknown)
#> ✔ CUDA Toolkit: CUDA 13.4 at /opt/cuda
#> ✔ libcudf: Found headers and libcudf.so at /home/theo/cuplr/.pixi/envs/default
#> ✔ R Version: R 4.6.1
#> ✔ R Packages: All 7 required packages installed
#> ✔ GPU Access: GPU accessible: NVIDIA GeForce RTX 5070
#> -------------------------------------------------- 
#> All checks passed. 
#> Next step: ./configure && R CMD INSTALL .
if (!result$ok) {
  message("Some checks failed. See result$checks for details.")
}
```
