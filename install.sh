#!/usr/bin/env bash
set -euo pipefail

# install.sh - Install cuplyr from source
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/bbtheo/cuplyr/master/install.sh | bash
#   # or locally:
#   ./install.sh [--method=auto|pixi|conda|system] [--dry-run] [--verbose]

usage() {
  cat <<'USAGE'
Install cuplyr (GPU-accelerated dplyr backend)

Usage:
  install.sh [OPTIONS]

Options:
  --method=METHOD    Install method: auto, pixi, conda, system (default: auto)
  --conda-prefix=DIR RAPIDS install location (default: /opt/rapids or tmpdir)
  --ref=REF          Git branch/tag to install (default: master)
  --dry-run          Print what would be done without executing
  --verbose          Print detailed progress
  -h, --help         Show this help

Methods:
  auto     Try pixi, then conda, then system (default)
  pixi     Use pixi to manage environment (recommended for contributors)
  conda    Use mamba/conda to install RAPIDS, then build cuplyr
  system   Assume CUDA + cuDF already installed

Examples:
  ./install.sh                          # Auto-detect method
  ./install.sh --method=conda           # Use conda/mamba
  ./install.sh --method=system          # CUDA/cuDF already on system
  ./install.sh --dry-run --verbose      # Preview what would happen
USAGE
}

# --- Defaults ---
METHOD="auto"
CONDA_PREFIX_ARG=""
REF="master"
DRY_RUN=false
VERBOSE=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method=*) METHOD="${1#*=}"; shift ;;
    --conda-prefix=*) CONDA_PREFIX_ARG="${1#*=}"; shift ;;
    --ref=*) REF="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --verbose) VERBOSE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# --- Helpers ---
log() { echo "==> $*"; }
vlog() { if $VERBOSE; then echo "    $*"; fi; }
run() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    if $VERBOSE; then
      "$@"
    else
      # Suppress stdout but preserve stderr for errors
      "$@" > /dev/null
    fi
  fi
}

has_cmd() { command -v "$1" > /dev/null 2>&1; }

detect_distro() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "$ID"
  else
    echo "unknown"
  fi
}

# --- Method detection ---
detect_method() {
  if has_cmd pixi && [ -f "pixi.toml" ]; then
    echo "pixi"
  elif has_cmd mamba || has_cmd conda; then
    echo "conda"
  elif [ -f "${CUDA_HOME:-/usr/local/cuda}/include/cuda.h" ]; then
    echo "system"
  else
    echo ""
  fi
}

# --- Clone source if needed ---
SRC_DIR=""
ensure_source() {
  if [ -f "DESCRIPTION" ] && [ -f "configure" ]; then
    SRC_DIR="$(pwd)"
    return
  fi

  log "Cloning cuplyr (ref: $REF)..."
  SRC_DIR="$(mktemp -d)/cuplyr"
  run git clone --depth 1 -b "$REF" https://github.com/bbtheo/cuplyr.git "$SRC_DIR"
}

# --- Install via pixi ---
install_pixi() {
  log "Installing via pixi..."
  if $DRY_RUN; then
    echo "[dry-run] (cd \"$SRC_DIR\" && pixi run install)"
    return
  fi
  cd "$SRC_DIR"
  run pixi run install
}

# --- Detect environment ---
detect_env() {
  # Colab detection (need 2+ signals)
  local colab_signals=0
  [ -d "/content" ] && colab_signals=$((colab_signals + 1))
  [ -n "${COLAB_RELEASE_TAG:-}" ] && colab_signals=$((colab_signals + 1))
  [ -n "${COLAB_GPU:-}" ] && colab_signals=$((colab_signals + 1))
  [ -f "/usr/local/share/jupyter/kernels/ir" ] && colab_signals=$((colab_signals + 1))
  if [ "$colab_signals" -ge 2 ]; then
    echo "colab"
    return
  fi

  # Container
  if [ -f "/.dockerenv" ] || [ -n "${KUBERNETES_SERVICE_HOST:-}" ]; then
    echo "container"
    return
  fi

  # Cloud GPU (non-standard driver path)
  if has_cmd nvidia-smi; then
    if [ -f "/usr/lib64-nvidia/libcuda.so.1" ] || [ -f "/usr/lib/nvidia/libcuda.so.1" ]; then
      if [ ! -f "/usr/lib/x86_64-linux-gnu/libcuda.so.1" ]; then
        echo "cloud_gpu"
        return
      fi
    fi
  fi

  echo "local"
}

# --- Find real NVIDIA driver ---
find_real_driver() {
  for p in /usr/lib64-nvidia /usr/lib/x86_64-linux-gnu; do
    if [ -f "$p/libcuda.so.1" ]; then
      local size
      size=$(stat -c %s "$p/libcuda.so.1" 2>/dev/null || echo 0)
      # Real driver is 30MB+, stubs are <1MB
      if [ "$size" -ge 1000000 ]; then
        echo "$p"
        return
      fi
    fi
  done
  echo ""
}

# --- Disable CUDA stubs ---
disable_stubs() {
  local lib_dir="$1"
  for stub_name in libcuda.so libcuda.so.1; do
    for loc in "$lib_dir/stubs" "$lib_dir"; do
      local stub_path="$loc/$stub_name"
      if [ -f "$stub_path" ]; then
        local size
        size=$(stat -c %s "$stub_path" 2>/dev/null || echo 0)
        # Only disable if it's a stub (small file)
        if [ "$size" -lt 1000000 ]; then
          vlog "Disabling stub: $stub_path"
          mv "$stub_path" "$stub_path.disabled" 2>/dev/null || true
        fi
      fi
    done
  done
}

# --- Install via conda ---
install_conda() {
  log "Installing via conda..."

  local env
  env=$(detect_env)
  vlog "Environment: $env"

  local conda_cmd
  if has_cmd mamba; then conda_cmd="mamba"; else conda_cmd="conda"; fi

  local prefix="${CONDA_PREFIX_ARG:-${CONDA_PREFIX:-}}"
  if [ -z "$prefix" ]; then
    if [ "$env" = "colab" ] || [ "$env" = "cloud_gpu" ]; then
      prefix="/opt/rapids"
    else
      prefix="$(mktemp -d)/cuplyr-rapids"
    fi
  fi

  # Cloud: disable stubs before anything
  if [ "$env" = "colab" ] || [ "$env" = "cloud_gpu" ]; then
    if [ -d "$prefix/lib" ]; then
      log "Disabling CUDA stubs..."
      disable_stubs "$prefix/lib"
    fi
  fi

  # Check if RAPIDS already installed
  if [ ! -f "$prefix/include/cudf/types.hpp" ]; then
    log "Installing RAPIDS packages into $prefix ..."

    # Try pinned versions first, fall back to unpinned
    local solved=0
    for attempt in 1 2; do
      if [ "$attempt" -eq 1 ]; then
        vlog "Trying pinned RAPIDS 25.12..."
        run "$conda_cmd" create -y -p "$prefix" \
          -c rapidsai -c conda-forge -c nvidia \
          "libcudf=25.12" "librmm=25.12" "libkvikio=25.12" spdlog fmt && solved=1
      else
        log "Retrying with unpinned RAPIDS versions..."
        run "$conda_cmd" create -y -p "$prefix" \
          -c rapidsai -c conda-forge -c nvidia \
          libcudf librmm libkvikio spdlog fmt && solved=1
      fi

      if [ "$solved" -eq 1 ]; then
        # Check for dev packages if headers missing
        if [ ! -f "$prefix/include/cudf/types.hpp" ]; then
          vlog "Installing -dev packages..."
          run "$conda_cmd" install -y -p "$prefix" \
            -c rapidsai -c conda-forge -c nvidia \
            libcudf-dev librmm-dev libkvikio-dev || true
        fi
        [ -f "$prefix/include/cudf/types.hpp" ] && break
        solved=0
      fi
    done

    if [ "$solved" -eq 0 ] || [ ! -f "$prefix/include/cudf/types.hpp" ]; then
      echo "ERROR: Failed to install RAPIDS packages with usable headers/libraries" >&2
      exit 1
    fi

    # Disable stubs again (conda may recreate them)
    if [ "$env" = "colab" ] || [ "$env" = "cloud_gpu" ]; then
      disable_stubs "$prefix/lib"
    fi
  else
    log "RAPIDS packages found at $prefix"
  fi

  export CONDA_PREFIX="$prefix"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

  # Cloud: configure library paths
  local driver_lib=""
  if [ "$env" = "colab" ] || [ "$env" = "cloud_gpu" ]; then
    driver_lib=$(find_real_driver)
    if [ -z "$driver_lib" ]; then
      echo "ERROR: Could not find real NVIDIA driver" >&2
      exit 1
    fi
    log "Real driver found at: $driver_lib"
    export LD_LIBRARY_PATH="$driver_lib:$CUDA_HOME/lib64:$prefix/lib:${LD_LIBRARY_PATH:-}"
  fi

  log "Configuring cuplyr..."
  if $DRY_RUN; then
    echo "[dry-run] (cd \"$SRC_DIR\" && ./configure)"
    echo "[dry-run] (cd \"$SRC_DIR\" && R CMD INSTALL .)"
    return
  fi
  cd "$SRC_DIR"
  run ./configure

  # Cloud: patch Makevars to prepend driver RUNPATH
  if [ -n "$driver_lib" ] && [ -f "src/Makevars" ]; then
    log "Patching Makevars for cloud driver path..."
    # Prepend -Wl,--enable-new-dtags -Wl,-rpath,$driver_lib to PKG_LIBS
    sed -i.bak "s|^PKG_LIBS=|PKG_LIBS=-Wl,--enable-new-dtags -Wl,-rpath,$driver_lib |" src/Makevars
  fi

  log "Building cuplyr..."
  run R CMD INSTALL .
}

# --- Install via system ---
install_system() {
  log "Installing from system dependencies..."

  # Quick checks
  if [ ! -f "${CUDA_HOME:-/usr/local/cuda}/include/cuda.h" ]; then
    echo "ERROR: CUDA toolkit not found." >&2
    echo "Set CUDA_HOME or install CUDA, then retry." >&2
    echo "Or use: ./install.sh --method=conda" >&2
    exit 1
  fi

  if $DRY_RUN; then
    echo "[dry-run] (cd \"$SRC_DIR\" && ./configure)"
    echo "[dry-run] (cd \"$SRC_DIR\" && R CMD INSTALL .)"
    return
  fi

  cd "$SRC_DIR"

  log "Configuring cuplyr..."
  run ./configure

  log "Building cuplyr..."
  run R CMD INSTALL .
}

# --- Main ---
main() {
  log "cuplyr installer"

  # Check R is available
  if ! has_cmd R; then
    echo "ERROR: R not found. Install R >= 4.3 first." >&2
    echo "  https://cran.r-project.org/" >&2
    exit 1
  fi

  # Check R packages
  log "Checking R dependencies..."
  if ! $DRY_RUN; then
    R --no-save --no-restore -e '
      needed <- c("Rcpp", "dplyr", "rlang", "vctrs", "pillar", "glue", "cli", "tidyselect", "tibble")
      missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
      if (length(missing) > 0) {
        message("Installing missing R packages: ", paste(missing, collapse = ", "))
        install.packages(missing, repos = "https://cloud.r-project.org")
      }
    '
  fi

  # Detect method
  if [ "$METHOD" = "auto" ]; then
    ensure_source
    METHOD=$(detect_method)
    if [ -z "$METHOD" ]; then
      echo "ERROR: Could not auto-detect install method." >&2
      echo "" >&2
      echo "Options:" >&2
      echo "  1. Install pixi: curl -fsSL https://pixi.sh/install.sh | bash" >&2
      echo "  2. Install mamba: https://mamba.readthedocs.io/" >&2
      echo "  3. Install CUDA + cuDF, then: ./install.sh --method=system" >&2
      exit 1
    fi
    log "Auto-detected method: $METHOD"
  else
    ensure_source
  fi

  case "$METHOD" in
    pixi) install_pixi ;;
    conda) install_conda ;;
    system) install_system ;;
    *) echo "Unknown method: $METHOD" >&2; exit 1 ;;
  esac

  log "Done!"
  echo ""
  echo "Verify installation:"
  echo '  R -e "library(cuplyr); verify_installation()"'
}

main
