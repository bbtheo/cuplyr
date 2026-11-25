# Shared setup for the tools/ scripts. Source it; don't execute it.
#
# Every worktree (feature, integration, main checkout) shares one pixi env and
# one set of GPU locks, both anchored at the MAIN checkout. Worktrees never run
# `pixi` themselves: the env's metadata pins the main pixi.toml path, so a
# `pixi run` from a worktree risks re-installing into the shared env.

set -euo pipefail

# The worktree is the one this script lives in, NOT the caller's cwd: an agent
# calling /path/to/cuplr-wt/<id>/tools/test-file always tests <id>, wherever
# its shell happens to be.
CUPLYR_WT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_common="$(git -C "$CUPLYR_WT_ROOT" rev-parse --path-format=absolute --git-common-dir)"
CUPLYR_MAIN="$(dirname "$_common")"
CUPLYR_LOCK_DIR="$_common/cuplyr-locks"
CUPLYR_PARITY_DIR="$CUPLYR_MAIN/scratchpad/parity"
CUPLYR_WT_PARENT="$(dirname "$CUPLYR_MAIN")/cuplr-wt"
# The one long-lived branch. Feature branches start from origin's copy and
# land on it through a pull request (tools/merge-queue).
CUPLYR_TRUNK="main"
mkdir -p "$CUPLYR_LOCK_DIR"

# Which pixi environment the tools run in. Default: the main checkout's
# pixi.toml, default environment (what every worktree and the merge gate use).
#   CUPLYR_PIXI_MANIFEST  path to a pixi.toml       (default: $CUPLYR_MAIN/pixi.toml)
#   CUPLYR_PIXI_ENV       environment name in it    (default: default)
# Each of the two is taken from the first of these that sets it:
#   1. the variable itself, in the caller's environment;
#   2. the environment pixi activated, when the tools run under `pixi run -e X`
#      or `pixi shell -e X` (PIXI_PROJECT_MANIFEST and PIXI_ENVIRONMENT_NAME,
#      only when BOTH are set: half a pair is not something pixi activated), so
#      `pixi run -e rapids2602 conformance` runs on 26.02;
#   3. a worktree's untracked .cuplyr-env file at its root (KEY=VALUE lines),
#      which pins a worktree to an environment when nothing above says otherwise;
#   4. the default above.
_file_manifest=""
_file_env=""
if [[ -f "$CUPLYR_WT_ROOT/.cuplyr-env" ]]; then
  while IFS='=' read -r _k _v || [[ -n "$_k" ]]; do
    case "$_k" in
      CUPLYR_PIXI_MANIFEST) _file_manifest="$_v" ;;
      CUPLYR_PIXI_ENV) _file_env="$_v" ;;
    esac
  done < "$CUPLYR_WT_ROOT/.cuplyr-env"
fi
_pixi_manifest=""
_pixi_env=""
if [[ -n "${PIXI_PROJECT_MANIFEST:-}" && -n "${PIXI_ENVIRONMENT_NAME:-}" ]]; then
  _pixi_manifest="$PIXI_PROJECT_MANIFEST"
  _pixi_env="$PIXI_ENVIRONMENT_NAME"
fi
CUPLYR_PIXI_MANIFEST="${CUPLYR_PIXI_MANIFEST:-${_pixi_manifest:-${_file_manifest:-$CUPLYR_MAIN/pixi.toml}}}"
CUPLYR_PIXI_ENV="${CUPLYR_PIXI_ENV:-${_pixi_env:-${_file_env:-default}}}"
export CUPLYR_PIXI_MANIFEST CUPLYR_PIXI_ENV

cuplyr_activate_env() {
  if [[ -z "${CUPLYR_ENV_ACTIVE:-}" ]]; then
    # conda activation scripts reference unset variables; relax -u for them.
    set +u
    eval "$(pixi shell-hook --manifest-path "$CUPLYR_PIXI_MANIFEST" -e "$CUPLYR_PIXI_ENV")"
    set -u
    export CUPLYR_ENV_ACTIVE=1
  fi
}

# Compiles are CPU-only and happen outside the GPU lock where possible
# (tools/build); parallelize them.
export MAKEFLAGS="${MAKEFLAGS:--j8}"

# pkgbuild/pkgload default to debug builds (-O0), 2.5-5x slower than the -O2
# build perf baselines are recorded with. Turn the extra debug flags off for
# every compile path that goes through a tool.
export PKG_BUILD_EXTRA_FLAGS=false
