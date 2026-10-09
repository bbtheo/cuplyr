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
mkdir -p "$CUPLYR_LOCK_DIR"

cuplyr_activate_env() {
  if [[ -z "${CUPLYR_ENV_ACTIVE:-}" ]]; then
    # conda activation scripts reference unset variables; relax -u for them.
    set +u
    eval "$(pixi shell-hook --manifest-path "$CUPLYR_MAIN/pixi.toml")"
    set -u
    export CUPLYR_ENV_ACTIVE=1
  fi
}

# Compiles are CPU-only and happen outside the GPU lock where possible
# (tools/build); parallelize them.
export MAKEFLAGS="${MAKEFLAGS:--j8}"
