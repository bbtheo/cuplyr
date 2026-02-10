#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build a package tarball from a staged copy (avoids local disk quota issues).

Usage:
  scripts/build-tarball.sh [--out-dir DIR] [--stage-dir DIR] [--keep-stage]

Options:
  --out-dir DIR    Directory for final tarball (default: <repo>, same as R CMD build .)
  --stage-dir DIR  Staging parent directory (default: /tmp)
  --keep-stage     Do not delete staged copy after build
  -h, --help       Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${REPO_ROOT}"
STAGE_PARENT="/tmp"
KEEP_STAGE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --stage-dir)
      STAGE_PARENT="$2"
      shift 2
      ;;
    --keep-stage)
      KEEP_STAGE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ ! -f DESCRIPTION ]]; then
  echo "DESCRIPTION not found in ${REPO_ROOT}" >&2
  exit 1
fi

PKG_NAME="$(Rscript -e 'd <- read.dcf("DESCRIPTION"); cat(d[1, "Package"])')"
PKG_VERSION="$(Rscript -e 'd <- read.dcf("DESCRIPTION"); cat(d[1, "Version"])')"
REPO_BASENAME="$(basename "${REPO_ROOT}")"

STAGE_ROOT="${STAGE_PARENT%/}/${REPO_BASENAME}_build_stage"
STAGE_PKG_DIR="${STAGE_ROOT}/${REPO_BASENAME}"
TARBALL_NAME="${PKG_NAME}_${PKG_VERSION}.tar.gz"

mkdir -p "${OUT_DIR}"
rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_PKG_DIR}"

rsync -a --delete \
  --exclude '.git' \
  --exclude '.pixi' \
  --exclude 'benchmark_data' \
  --exclude 'scratchpad' \
  "${REPO_ROOT}/" "${STAGE_PKG_DIR}/"

(
  cd "${STAGE_ROOT}"
  R CMD build --no-build-vignettes --no-manual "${REPO_BASENAME}"
)

cp -f "${STAGE_ROOT}/${TARBALL_NAME}" "${OUT_DIR}/"
echo "Built tarball: ${OUT_DIR}/${TARBALL_NAME}"

if [[ "${KEEP_STAGE}" != "true" ]]; then
  rm -rf "${STAGE_ROOT}"
fi
