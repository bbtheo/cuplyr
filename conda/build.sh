#!/usr/bin/env bash
# Build script of conda/recipe.yaml. rattler-build runs it from the source
# directory with SRC_DIR set to that directory, PREFIX set to the host
# environment and R set to the host environment's R.
#
# Nothing is downloaded here: every dependency comes from the host environment.
set -euo pipefail

# Stop before anything is touched unless the build tool's variables are set.
# Run by hand in a checkout, the cleanup below would delete tracked files.
: "${SRC_DIR:?is not set: build.sh is run by the recipe build tool, not by hand}"
: "${PREFIX:?is not set: build.sh is run by the recipe build tool, not by hand}"

cd "$SRC_DIR"

# The source is a copy of the checkout. Drop objects and a Makevars left by a
# development build, which would point at another machine's paths.
rm -f src/*.o src/*.so src/Makevars

# The install below copies the checkout, not a tarball from R CMD build, so
# .Rbuildignore has not been applied. Remove from inst/ what it excludes
# (inst/dplyr-conformance, the vendored dplyr tests), so the binary leaves out
# what the source package leaves out. Only paths inside this source copy are
# touched.
#
# R CMD build matches the patterns as Perl regular expressions, ignoring case;
# grep -E -i agrees with it on the plain patterns .Rbuildignore holds, so keep
# PCRE-only syntax (\d, (?:...)) out of it.
if [ -d inst ] && [ -f .Rbuildignore ]; then
    patterns="$(mktemp)"
    trap 'rm -f "$patterns"' EXIT
    # grep exits 1 when nothing is selected, which is fine; 2 is an error.
    grep -v -e '^[[:space:]]*$' -e '^[[:space:]]*#' .Rbuildignore > "$patterns" || [ $? -eq 1 ]
    if [ -s "$patterns" ]; then
        # List every match before deleting any: removing a directory while
        # find is still walking it makes find fail.
        matches="$(find inst -mindepth 1 | { grep -E -i -f "$patterns" || [ $? -eq 1 ]; })" || {
            echo "ERROR: could not match inst/ against .Rbuildignore (see the message above)." >&2
            exit 1
        }
        # find lists a directory before its contents, so once a directory is
        # removed the matches under it that follow can be skipped.
        removed=""
        while IFS= read -r path; do
            [ -n "$path" ] || continue
            if [ -n "$removed" ] && [[ "$path" == "$removed"/* ]]; then
                continue
            fi
            rm -rf -- "$path"
            removed="$path"
        done <<< "$matches"
    fi
fi

# Keep the builder's own R setup out of the build. rattler-build passes HOME
# on outside strict isolation, and R reads ~/.Renviron, ~/.Rprofile and
# ~/.R/Makevars from it: a CUDA_HOME or CUDF_HOME set there would redirect
# configure to a system toolkit, and a personal LDFLAGS would replace the
# toolchain's and search the system libraries first.
# R_LIBS_USER, and no R_LIBS, keep the caller's libraries out of the package's
# load test.
export R_ENVIRON_USER=/dev/null
export R_PROFILE_USER=/dev/null
export R_MAKEVARS_USER=/dev/null
export R_LIBS_USER="$PREFIX/lib/R/library"
unset R_LIBS

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

# configure falls back to a toolkit elsewhere on the machine (/usr/local/cuda,
# /opt/cuda) when the host environment lacks one, for example without nvcc.
# The binary must be built from the host environment alone, so check every
# absolute path in src/Makevars before compiling anything, whichever variable
# holds it.

# The value of one variable in src/Makevars ("" when absent or empty).
makevar() {
    awk -v key="$1" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' src/Makevars
}

# The host prefix as given and with symlinks resolved; paths under either count.
host_prefix="${PREFIX%/}"
host_prefix_real="$(cd "$PREFIX" && pwd -P)"
in_host_prefix() {
    case "$1" in
        "$host_prefix" | "$host_prefix"/* | "$host_prefix_real" | "$host_prefix_real"/*) return 0 ;;
    esac
    return 1
}

problems=()
# The paths the compile and link can't do without. CCCL_INCLUDE may be empty:
# it is when libcudf's prefix has no CCCL copy (the toolkit's is used).
for var in CUDA_HOME CUDA_INCLUDE CUDA_DRIVER_LIB CUDA_RUNTIME_LIB CUDF_INCLUDE CUDF_LIB RMM_INCLUDE RAPIDS_RPATHS; do
    [ -n "$(makevar "$var")" ] || problems+=("$var is empty")
done

# Every absolute path on a NAME=value line. Values are split on spaces,
# colons (RAPIDS_RPATHS is a search path) and commas (-Wl,-rpath,<dir>), and
# a -L or -I prefix is dropped. References such as $(CUDF_LIB) in PKG_LIBS
# are not paths: the variable they name is checked on its own line.
assignment='^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*[:+?]?=(.*)$'
while IFS= read -r line; do
    [[ "$line" =~ $assignment ]] || continue
    name="${BASH_REMATCH[1]}"
    IFS=$' \t:,' read -r -a words <<< "${BASH_REMATCH[2]}"
    for word in ${words[@]+"${words[@]}"}; do
        case "$word" in
            -L* | -I*) word="${word#-?}" ;;
            -rpath=* | --rpath=*) word="${word#*=}" ;;
        esac
        case "$word" in
            /*) in_host_prefix "$word" || problems+=("$name: $word") ;;
        esac
    done
done < src/Makevars

# forced-stub mode records no driver directory in the library
driver_rpath="$(makevar CUDA_DRIVER_RPATH_FLAG)"
[ -z "$driver_rpath" ] || problems+=("CUDA_DRIVER_RPATH_FLAG=$driver_rpath")

if [ "${#problems[@]}" -gt 0 ]; then
    {
        echo "ERROR: configure took CUDA or libcudf paths from outside the host environment"
        echo "       $host_prefix:"
        printf '         %s\n' "${problems[@]}"
        echo "       The host requirements must provide the whole CUDA toolkit (nvcc, the"
        echo "       headers and the libcuda stub) and RAPIDS; check the recipe's host list."
    } >&2
    exit 1
fi

# cuda_runtime.h includes the crt/ headers, which cuda-crt-dev_<platform> ships
# (cuda-nvcc-tools and cuda-cudart-dev don't bring them). Without them the
# compile fails at the first source file; say so by name here instead.
cuda_include="$(makevar CUDA_INCLUDE)"
if [ ! -f "$cuda_include/crt/host_config.h" ]; then
    {
        echo "ERROR: $cuda_include/crt/host_config.h is missing, and cuda_runtime.h"
        echo "       includes it. Add cuda-crt-dev_<target_platform> to the host requirements."
    } >&2
    exit 1
fi

# R CMD INSTALL loads the package once after linking; with libcuda linked
# --as-needed that works on a machine without a GPU. --no-configure makes the
# configure run above the one the compile uses, instead of a second run from
# R's own process.
"${R:-$PREFIX/bin/R}" CMD INSTALL --no-configure --library="$PREFIX/lib/R/library" .
