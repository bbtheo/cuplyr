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
    the driver version, read with
    `nvidia-smi --query-gpu=driver_version` and, failing that, from the
    `Driver Version:` or `KMD Version:` header

2.  **CUDA Toolkit**: Looks for `nvcc` in `CUDA_HOME`, `CONDA_PREFIX`,
    the environment of the libcudf that cuplyr has loaded, the pixi
    environment (`PIXI_PROJECT_ROOT`, or `.pixi` in the working
    directory), the system CUDA directories and `PATH`, in that order.
    Validates CUDA \>= 12.2 and reports the CUDA version

3.  **libcudf**: Reports the libcudf that cuplyr has loaded, if any.
    Otherwise looks for cudf headers and `libcudf.so` under `CUDF_HOME`,
    `CONDA_PREFIX`, the pixi environment (`PIXI_PROJECT_ROOT`, or
    `.pixi` in the working directory) and standard paths. A `CUDF_HOME`
    or `CONDA_PREFIX` that points to a missing directory is named in the
    message but does not fail the check by itself. The check reports the
    libcudf version it finds (from `cudf/version_config.hpp`) and fails
    for a libcudf older than 26.02; a version it can't read is not a
    failure.

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
#> ✔ NVIDIA Driver: Driver version 615.78.08
#> ✔ CUDA Toolkit: CUDA 13.4 at /opt/cuda
#> ✔ libcudf: libcudf loaded by cuplyr from /home/theo/cuplr/.pixi/envs/default/lib/libcudf.so (libcudf 26.10.2). Note: CUDF_HOME is set to '/home/theo/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf', which does not exist.
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
#>     "nvidia_driver": {"ok": true, "value": "615.78.08", "message": "Driver version 615.78.08"},
#>     "cuda_toolkit": {"ok": true, "value": "13.4", "message": "CUDA 13.4 at /opt/cuda"},
#>     "libcudf": {"ok": true, "value": "/home/theo/cuplr/.pixi/envs/default", "message": "libcudf loaded by cuplyr from /home/theo/cuplr/.pixi/envs/default/lib/libcudf.so (libcudf 26.10.2). Note: CUDF_HOME is set to '/home/theo/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf', which does not exist."},
#>     "r_version": {"ok": true, "value": "4.6.1", "message": "R 4.6.1"},
#>     "r_packages": {"ok": true, "value": "Rcpp, dplyr, rlang, vctrs, cli, tidyselect, tibble", "message": "All 7 required packages installed"},
#>     "gpu_access": {"ok": true, "value": "NVIDIA GeForce RTX 5070", "message": "GPU accessible: NVIDIA GeForce RTX 5070"}
#>   }
#> } 

# Programmatic use
result <- check_deps()
#> cuplyr dependency check
#> -------------------------------------------------- 
#> ✔ NVIDIA Driver: Driver version 615.78.08
#> ✔ CUDA Toolkit: CUDA 13.4 at /opt/cuda
#> ✔ libcudf: libcudf loaded by cuplyr from /home/theo/cuplr/.pixi/envs/default/lib/libcudf.so (libcudf 26.10.2). Note: CUDF_HOME is set to '/home/theo/cuplr/rapids-dev/lib/python3.12/site-packages/libcudf', which does not exist.
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
