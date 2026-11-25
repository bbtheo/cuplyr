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
# A worktree can pin both in an untracked .cuplyr-env file at its root
# (KEY=VALUE lines); variables set in the caller's environment win over it.
if [[ -f "$CUPLYR_WT_ROOT/.cuplyr-env" ]]; then
  while IFS='=' read -r _k _v; do
    case "$_k" in
      CUPLYR_PIXI_MANIFEST|CUPLYR_PIXI_ENV)
        [[ -n "${!_k:-}" ]] || export "$_k=$_v" ;;
    esac
  done < "$CUPLYR_WT_ROOT/.cuplyr-env"
fi
CUPLYR_PIXI_MANIFEST="${CUPLYR_PIXI_MANIFEST:-$CUPLYR_MAIN/pixi.toml}"
CUPLYR_PIXI_ENV="${CUPLYR_PIXI_ENV:-default}"
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
