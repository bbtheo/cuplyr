#!/usr/bin/env bash
# PreToolUse(Bash) guard for every Claude session in this repo.
# Exit 2 blocks the call and shows stderr to the agent.
cmd="$(jq -r '.tool_input.command // ""')"
deny() { echo "BLOCKED by .claude/hooks/guard-bash.sh: $1" >&2; exit 2; }

# A command word in shell-command position (start, or after ; & | ( $( ),
# optionally behind VAR=val assignments or `env`.
pos='(^|[;&|(]|\$\()[[:space:]]*(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'

if grep -Eq 'R CMD INSTALL|devtools::install|remotes::install|pak::|install_cuplyr' <<<"$cmd"; then
  deny "cuplyr is never installed. Tests run via pkgload::load_all() through tools/test, tools/test-file, tools/r."
fi
if grep -Eq "${pos}pixi([[:space:]]|$)" <<<"$cmd"; then
  deny "don't call pixi from agents (it can re-install the shared env from a worktree). Use tools/test, tools/test-file, tools/perf, tools/conformance, tools/r, tools/document, tools/build."
fi
if grep -Eq "${pos}(Rscript|R)([[:space:]]|$)" <<<"$cmd"; then
  deny "raw R/Rscript bypasses the GPU lock. Use tools/r '<expr>' for ad-hoc R, or tools/test-file / tools/test / tools/document."
fi
if grep -q 'CUPLYR_PERF_RECORD' <<<"$cmd"; then
  deny "re-recording the perf baseline is Theo's decision. Report the perf result instead."
fi
if grep -Eq 'perf_baseline\.json|conformance_baseline\.tsv' <<<"$cmd" &&
   grep -Eq '>|sed[[:space:]]+-i|(^|[[:space:]])(cp|mv|rm|tee|truncate|dd)[[:space:]]' <<<"$cmd"; then
  deny "the perf and conformance baselines are not writable by agents (tools/merge-queue ratchets the conformance one)."
fi
if grep -Eq "${pos}git[[:space:]]+push" <<<"$cmd"; then
  deny "agents never push. Theo pushes."
fi
# Checking dev out (outside the integration worktree) would let commits land
# on it; --detach only moves HEAD, so it's allowed.
if grep -Eq "${pos}git[[:space:]]+(checkout|switch)[[:space:]]+(-[^[:space:]]+[[:space:]]+)*dev([[:space:]]|$)" <<<"$cmd" &&
   ! grep -q -- '--detach' <<<"$cmd"; then
  deny "dev only moves via tools/merge-queue (git switch --detach dev is fine)."
fi
if grep -Eq "git[[:space:]]+branch[[:space:]]+(-f|-D|-d|-M|-m)[[:space:]].*\bdev\b|update-ref[[:space:]]+refs/heads/dev" <<<"$cmd"; then
  deny "dev only moves via tools/merge-queue."
fi
exit 0
