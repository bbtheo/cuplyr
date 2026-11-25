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
# --as-needed that works on a machine without a GPU.
"${R:-$PREFIX/bin/R}" CMD INSTALL --library="$PREFIX/lib/R/library" .
