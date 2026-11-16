# Install cuplyr from source

Installs cuplyr by setting up build dependencies (if needed) and running
`./configure && R CMD INSTALL .`. Supports multiple methods for
providing CUDA/cuDF dependencies.

## Usage

``` r
install_cuplyr(
  method = c("auto", "pixi", "conda", "system"),
  repo = NULL,
  ref = "main",
  conda_prefix = NULL,
  configure_args = character(),
  dry_run = FALSE,
  verbose = FALSE
)
```

## Arguments

- method:

  How to obtain build dependencies:

  `"auto"`

  :   Try methods in order: pixi, conda, system

  `"pixi"`

  :   Use pixi to manage a reproducible environment

  `"conda"`

  :   Use conda/mamba to install RAPIDS into a prefix

  `"system"`

  :   Assume CUDA and cuDF are already installed on the system

- repo:

  GitHub repository URL or local path to cuplyr source. If `NULL`,
  assumes current directory is the cuplyr source tree.

- ref:

  Git branch or tag to install. Only used when `repo` is a URL.

- conda_prefix:

  Path for the conda environment when `method = "conda"`. Defaults to
  the active conda environment (unless it is `base`), then
  `"/opt/rapids"` on Colab and cloud GPUs, then
  `~/.local/share/cuplyr/rapids` (or under `XDG_DATA_HOME`). An existing
  environment is added to, never replaced.

- configure_args:

  Extra arguments passed to `./configure`.

- dry_run:

  If `TRUE`, print what would be done without executing.

- verbose:

  If `TRUE`, print detailed progress.

## Value

Invisibly returns `TRUE` on success.

## Details

### Method details

**auto** (default): Tries methods in order of preference:

1.  If `pixi` is on PATH and the source tree has `pixi.toml`, use pixi

2.  If `mamba` or `conda` is on PATH, use conda

3.  If CUDA and cuDF are already available, use system

**pixi**: Runs `pixi run install` which handles everything. Requires
[pixi](https://pixi.sh) to be installed.

**conda**: Installs RAPIDS 25.12 for CUDA 12, with the CUDA compiler and
headers, into a conda environment, then configures and builds cuplyr
against it. Requires `mamba` or `conda`.

**system**: Assumes `CUDA_HOME` and `CONDA_PREFIX`/`CUDF_HOME` are
already set. Runs `./configure && R CMD INSTALL .` directly.

### Pre-flight

Before building, `install_cuplyr()` runs
[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
and reports any missing dependencies. With `method = "conda"`, missing
RAPIDS packages are installed automatically.

## See also

[`check_deps()`](https://bbtheo.github.io/cuplyr/reference/check_deps.md)
for dependency checking,
[`verify_installation()`](https://bbtheo.github.io/cuplyr/reference/verify_installation.md)
for post-install verification,
[`diagnostics()`](https://bbtheo.github.io/cuplyr/reference/diagnostics.md)
for troubleshooting

## Examples

``` r
if (FALSE) { # \dontrun{
# Auto-detect best method
install_cuplyr()

# Use conda explicitly
install_cuplyr(method = "conda")

# System install (CUDA/cuDF already available)
install_cuplyr(method = "system")

# Preview what would happen
install_cuplyr(dry_run = TRUE)

# Install from GitHub
install_cuplyr(repo = "https://github.com/bbtheo/cuplyr.git")
} # }
```
