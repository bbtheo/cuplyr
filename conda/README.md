# Building the r-cuplyr conda package

This directory holds the recipe for `r-cuplyr`, the conda package of cuplyr,
which is published on the [`bbtheo`](https://anaconda.org/bbtheo) channel on
anaconda.org. It is for maintainers: users install the package as the
Installation section of the main README shows.

Maintainers build and upload the package with
[rattler-build](https://rattler.build), the only build tool the recipe
supports. Install it once with pixi (it is on conda-forge):

```bash
pixi global install rattler-build
```

| File | Purpose |
|---|---|
| `recipe.yaml` | The recipe (rattler-build, recipe schema v1). |
| `variants.yaml` | The RAPIDS series (`rapids_version`), the CUDA major (`cuda_version`), and the C standard library and compiler versions to build with. |
| `build.sh` | The recipe's build script. Uses no network and ignores the builder's R startup files. |
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

The version comes from `context.version` in `recipe.yaml`; a test keeps it
equal to `Version` in `DESCRIPTION`, so change them together when the version
changes.

### CUDA floor

`cuda_version: "13"` builds against the newest CUDA 13.x the channels have.
`cuda-cudart-dev` exports a run requirement on the CUDA runtime of at least
the version it was built with, so the CUDA 13 minor of the build becomes the
users' floor: an environment can't install the package with an older 13.x.

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

From a fresh `git clone`, not a development checkout. rattler-build honours
`.gitignore`, but a clone is still the safest way to build exactly what is
committed. `build.sh` removes `src/*.o`, `src/*.so` and `src/Makevars` from
the copy, and the `inst/` paths that `.Rbuildignore` excludes (such as the
vendored dplyr tests), so the binary leaves out what the source package leaves
out. The channels are the ones `pixi.toml` uses.

```bash
rattler-build build --recipe conda/recipe.yaml -m conda/variants.yaml \
  -c rapidsai -c conda-forge -c nvidia --output-dir <dir>
```

No GPU is needed. The package test runs at the end of the build, and the
package is written to `<dir>/linux-64/`.

`build.sh` sets `R_ENVIRON_USER`, `R_PROFILE_USER` and `R_MAKEVARS_USER` to
`/dev/null`, so R reads no `~/.Renviron`, `~/.Rprofile` or `~/.R/Makevars`.
rattler-build passes `HOME` on outside strict isolation, and a `CUDA_HOME` or
`CUDF_HOME` in `~/.Renviron` would send `configure` to a system toolkit, while
a personal `LDFLAGS` would replace the toolchain's and search the system
libraries first. It runs `configure` once itself and calls
`R CMD INSTALL --no-configure`, so that run is the one the compile uses.

After `configure`, `build.sh` checks every absolute path it wrote into
`src/Makevars`. If the host environment lacks part of the toolkit (`nvcc`, for
one), `configure` would fall back to a toolkit elsewhere on the build machine,
such as `/usr/local/cuda`; `build.sh` then stops and lists the paths outside
the host environment, with the variable that holds each. It also stops if the
CUDA `crt/` headers, which `cuda_runtime.h` includes and
`cuda-crt-dev_<platform>` ships, are missing. `build.sh` refuses to run
without `SRC_DIR` and `PREFIX`, which rattler-build sets, so running it by
hand in a checkout deletes nothing.

## Install into a fresh environment

```bash
mamba create -n cuplyr-test --no-channel-priority \
  -c file://<dir> -c rapidsai -c conda-forge -c nvidia r-cuplyr
mamba activate cuplyr-test
```

`<dir>` is the output directory of the build, as an absolute path
(`file://` needs one); rattler-build writes the channel index there.

`--no-channel-priority` is for configurations that set strict channel priority,
where the solve would otherwise fail (as `install.sh` does).

## Smoke test (needs a GPU)

The package needs an NVIDIA GPU with compute capability 7.5 (Turing) or newer and a driver that supports CUDA 13.

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

## Upload

Once the smoke test passes, a maintainer uploads the package to the `bbtheo`
channel with rattler-build itself (no anaconda-client is needed).

Log in once, which keeps the token out of the shell history:

```bash
read -rs T && rattler-build auth login api.anaconda.org --conda-token "$T"; unset T
```

`read -rs` reads the anaconda.org token (an API token allowed to upload to
`bbtheo`) without echoing it, and rattler-build stores it in its keychain/auth
file. The host is `api.anaconda.org` because that is where
`rattler-build upload anaconda` looks the token up. `rattler-build auth status`
shows which host a stored token is under.

After that, the upload reads the token from the stored login:

```bash
rattler-build upload anaconda --owner bbtheo \
  <dir>/linux-64/r-cuplyr-<version>-<build>.conda
```

For a single command instead of the login, set `ANACONDA_API_KEY` (or pass
`--api-key`) for that command.

Upload nothing but `r-cuplyr` to `bbtheo`. pixi uses strict channel priority
(and conda, unless it is disabled, also takes a package from the first channel
that has it), and `bbtheo` comes first in the install commands, so any other
package on it would shadow the conda-forge package of the same name.

anaconda.org refuses a file that is already on the channel; add `--force` to
replace it, for example after rebuilding with the same build number. Raise
`build.number` in `recipe.yaml` instead when the new build should reach users
who already have the old one.

## conda-forge

The package is not on conda-forge, because the libraries it links are not:
conda-forge has no libcudf or libkvikio, and its librmm stops at 24.04. The
recipe assumes no private channel beyond the channel list in the build
command, so conda-forge can follow if the RAPIDS libraries arrive there. It
would then also need a feedstock and its review, and the conda-forge global
pinning for `r-base` and CUDA in place of the `rapids_version` and
`cuda_version` variants.
