#!/usr/bin/env bash
# Build script shared by conda/recipe.yaml (rattler-build) and conda/meta.yaml
# (conda-build). Both run it from the source directory with PREFIX set to the
# host environment and R set to the host environment's R.
#
# Nothing is downloaded here: every dependency comes from the host environment.
set -euo pipefail

cd "${SRC_DIR:-.}"

# The source is a copy of the checkout. Drop objects and a Makevars left by a
# development build, which would point at another machine's paths.
rm -f src/*.o src/*.so src/Makevars

# The install below copies the checkout, not a tarball from R CMD build, so
# .Rbuildignore has not been applied. Remove from inst/ what it excludes
# (inst/dplyr-conformance, the vendored dplyr tests), so the binary matches
# the source package. Only paths inside this source copy are touched.
if [ -d inst ] && [ -f .Rbuildignore ]; then
    patterns="$(mktemp)"
    grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' .Rbuildignore > "$patterns" || true
    if [ -s "$patterns" ]; then
        find inst -mindepth 1 | { grep -E -i -f "$patterns" || true; } | while IFS= read -r path; do
            rm -rf -- "$path"
        done
    fi
    rm -f "$patterns"
fi

# Keep the builder's own R setup out of the build. conda-build passes HOME on
# (so does rattler-build outside strict isolation), and R reads
# ~/.Renviron, ~/.Rprofile and ~/.R/Makevars from it: a CUDA_HOME or CUDF_HOME
# set there would redirect configure to a system toolkit, and a personal
# LDFLAGS would replace the toolchain's and search the system libraries first.
# R_LIBS_USER keeps a user library out of the package's load test.
export R_ENVIRON_USER=/dev/null
export R_PROFILE_USER=/dev/null
export R_MAKEVARS_USER=/dev/null
export R_LIBS_USER="$PREFIX/lib/R/library"

# Configure against the host environment only. A leftover CUDA_HOME or
# CUDF_HOME from the caller's shell would win over CONDA_PREFIX in configure,
# and an active environment's CONDA_PREFIX would be searched first.
unset CUDA_HOME CUDF_HOME
export CONDA_PREFIX="$PREFIX"
export CUDF_HOME="$PREFIX"

# Link libcuda from the toolkit's stub, even if the build machine has a
# driver, and only as needed. cuplyr calls just the CUDA runtime API, so the
# package then loads without a driver, and libcudart finds the user's driver
# at run time. configure puts no driver or stub directory on RUNPATH.
export CUPLYR_FORCE_CUDA_STUBS=1

./configure

# R CMD INSTALL loads the package once after linking; with libcuda linked
# --as-needed that works on a machine without a GPU. --no-configure makes the
# configure run above the one the compile uses, instead of a second run from
# R's own process.
"${R:-$PREFIX/bin/R}" CMD INSTALL --no-configure --library="$PREFIX/lib/R/library" .
