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
      "$@" > /dev/null 2>&1
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
  cd "$SRC_DIR"
  run pixi run install
}

# --- Install via conda ---
install_conda() {
  log "Installing via conda..."

  local conda_cmd
  if has_cmd mamba; then conda_cmd="mamba"; else conda_cmd="conda"; fi

  local prefix="${CONDA_PREFIX_ARG:-${CONDA_PREFIX:-}}"
  if [ -z "$prefix" ]; then
    prefix="/opt/rapids"
    # Fall back to temp if /opt is not writable
    if ! mkdir -p "$prefix" 2>/dev/null; then
      prefix="$(mktemp -d)/cuplyr-rapids"
    fi
  fi

  # Check if RAPIDS already installed
  if [ ! -f "$prefix/include/cudf/types.hpp" ]; then
    log "Installing RAPIDS packages into $prefix ..."
    run "$conda_cmd" create -y -p "$prefix" \
      -c rapidsai -c conda-forge -c nvidia \
      "libcudf>=25.12" "librmm>=25.12" "libkvikio>=25.12" \
      spdlog fmt "cuda-toolkit>=12.0,<13"
  else
    log "RAPIDS packages found at $prefix"
  fi

  export CONDA_PREFIX="$prefix"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

  log "Configuring cuplyr..."
  cd "$SRC_DIR"
  run ./configure

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
    ' || true
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
