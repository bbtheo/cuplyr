# Building the r-cuplyr conda package

This directory holds the recipe for `r-cuplyr`, the conda package of cuplyr.
It is for maintainers: the package is not published yet (the channel is still
to be decided). Users who want conda today follow the Installation section of
the main README.

| File | Purpose |
|---|---|
| `recipe.yaml` | The recipe for rattler-build (preferred). |
| `meta.yaml` | The same recipe for conda-build. Keep the two in step. |
| `variants.yaml` | The RAPIDS series (`rapids_version`), the CUDA major (`cuda_version`), and the C standard library and compiler versions to build with. |
| `build.sh` | Build script shared by both recipes. Uses no network and ignores the builder's R startup files. |
| `test-package.R` | The package test: loads cuplyr and checks that libcudf and librmm resolve. Needs no GPU. |

## What the package contains

A binary built against one RAPIDS series, for Linux. RAPIDS has no ABI across
series (libcudf's own package pins itself to its series, and `libcudf.so` has
no versioned soname), so the package requires exactly the libcudf, librmm and
libkvikio series it was built against. `variants.yaml` lists the series to
build, 26.10 by default. cuplyr itself supports 26.02 and later, so to also
build 26.02, add `"26.02"` to `rapids_version` (quoted: unquoted, 26.10 is the
number 26.1) or pass `--variant rapids_version=26.02`. `configure` rejects an
older series.

The version comes from the recipe (`context.version` in `recipe.yaml`,
`{% set version %}` in `meta.yaml`); a test keeps both equal to `Version` in
`DESCRIPTION`, so change them together when the version changes.

### Glibc floor

`variants.yaml` pins the C standard library to `sysroot` 2.28 and the
compilers to version 14, as libcudf is built. The pin matters: rattler-build
has no default for `stdlib("c")`, and an unpinned sysroot follows the newest
one on the build machine, which would raise the package's glibc requirement
above libcudf's and make it uninstallable on systems such as Debian 12, RHEL 9
and Colab.

### How libcuda is linked

The package needs no NVIDIA driver to build or to load, and finds the user's
driver at run time.

- The build links libcuda from the **stub** in the toolkit
  (`cuda-driver-dev`), even if the build machine has a driver. `build.sh` sets
  `CUPLYR_FORCE_CUDA_STUBS=1` for `configure`, which also fails clearly if no
  stub is found.
- libcuda is linked `--as-needed`, and no driver or stub directory goes on the
  library's RUNPATH. cuplyr calls only the CUDA runtime API, so `libcuda.so.1`
  is not listed as needed in `cuplyr.so`, and the package loads on a machine
  without a GPU (the package test relies on this).
- At run time the CUDA runtime (`libcudart`) loads the user's `libcuda.so.1`
  through the system loader, as libcudf and every conda CUDA package do. The
  stub is never on the run-time search path, so it can't mask the driver.

If cuplyr ever calls the driver API directly, `--as-needed` keeps libcuda as a
needed library; the GPU-less package test would then fail, which is the signal
to revisit this.

## Build

From a fresh `git clone`, not a development checkout. conda-build copies the
whole directory, including a checkout's `.pixi/` environment (gigabytes) and
any compiled objects; rattler-build honours `.gitignore`, but a clone is still
the safest way to build exactly what is committed. `build.sh` removes
`src/*.o`, `src/*.so` and `src/Makevars` from the copy, and the `inst/` paths
that `.Rbuildignore` excludes (such as the vendored dplyr tests), so the binary
holds what the source package holds. The channels are the ones `pixi.toml`
uses.

With rattler-build:

```bash
rattler-build build --recipe conda/recipe.yaml -m conda/variants.yaml \
  -c rapidsai -c conda-forge -c nvidia --output-dir <dir>
```

With conda-build (it does not read `variants.yaml` on its own, so `-m` is
needed):

```bash
conda build conda/ -m conda/variants.yaml \
  -c rapidsai -c conda-forge -c nvidia --output-folder <dir>
```

Neither needs a GPU. The package test runs at the end of the build.

`build.sh` sets `R_ENVIRON_USER`, `R_PROFILE_USER` and `R_MAKEVARS_USER` to
`/dev/null`, so R reads no `~/.Renviron`, `~/.Rprofile` or `~/.R/Makevars`.
conda-build passes `HOME` on (rattler-build does outside strict isolation), and
a `CUDA_HOME` or `CUDF_HOME` in `~/.Renviron` would send `configure` to a
system toolkit, while a personal `LDFLAGS` would replace the toolchain's and
search the system libraries first. It runs `configure` once itself and calls
`R CMD INSTALL --no-configure`, so that run is the one the compile uses.

## Install into a fresh environment

```bash
mamba create -n cuplyr-test --no-channel-priority \
  -c file://<dir> -c rapidsai -c conda-forge -c nvidia r-cuplyr
mamba activate cuplyr-test
```

`<dir>` is the output directory of the build, as an absolute path
(`file://` needs one). `rattler-build` writes the channel there; with
conda-build, run `conda index <dir>` first if it has not been indexed.

`--no-channel-priority` is for configurations that set strict channel priority,
where the solve would otherwise fail (as `install.sh` does).

## Smoke test (needs a GPU)

On a machine with an NVIDIA driver, in the new environment:

```r
library(cuplyr)
verify_installation()

tbl_gpu(mtcars) |>
  dplyr::filter(cyl > 4) |>
  dplyr::group_by(cyl) |>
  dplyr::summarise(mpg = mean(mpg)) |>
  dplyr::collect()
```

`verify_installation()` should report no failures, and the pipeline should
return one row per remaining `cyl`.

## Publishing

Where the package is published is not decided yet. The plan is the project's
own channel first, conda-forge later. The recipe assumes no private channel.
For conda-forge it would additionally need:

- libcudf, librmm and libkvikio from conda-forge itself (today they come from
  the `rapidsai` channel);
- a feedstock and its review;
- the conda-forge global pinning for `r-base` and CUDA in place of the
  `rapids_version` and `cuda_version` variants.
