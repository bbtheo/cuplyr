#!/usr/bin/env bash
# Coder-only PreToolUse guard (wired in .claude/agents/coder.md).
# Blocks edits to PM-owned acceptance tests, the parity state files, and
# worktree/merge tooling.
#
# Protected lists live in the main checkout at
# scratchpad/parity/features/<worktree-name>.protected
# (line 1 = SHA of the PM's acceptance-test commit, then repo-relative paths).
input="$(cat)"
deny() { echo "BLOCKED by .claude/hooks/coder-guard.sh: $1" >&2; exit 2; }

main="$(dirname "$(git -C "$CLAUDE_PROJECT_DIR" rev-parse --path-format=absolute --git-common-dir)")"
feat="$main/scratchpad/parity/features"
f="$(jq -r '.tool_input.file_path // ""' <<<"$input")"
cmd="$(jq -r '.tool_input.command // ""' <<<"$input")"

if grep -Eq 'tools/(merge-queue|merge-gate|parity-init|wt-new)' <<<"$cmd"; then
  deny "worktrees and merging belong to the PM and orchestrator; the roadmap board to the orchestrator."
fi

if grep -Eq 'git[[:space:]]+(-C[[:space:]]+[^[:space:]]+[[:space:]]+)*push|gh[[:space:]]+(pr|issue|release|label|repo)[[:space:]]+(create|merge|close|comment|edit|delete|reopen|ready|review)|gh[[:space:]]+api[[:space:]].*(-X|--method|-f|-F)|gh[[:space:]]+project[[:space:]]+(item-add|item-edit|item-delete|item-archive|edit|create|delete|copy|close|field-create|field-delete|link|unlink|mark-template)|tools/board[[:space:]]+(add|status)' <<<"$cmd"; then
  deny "pushing and writing to GitHub (pull requests, issues) belong to the PM and orchestrator."
fi
if grep -Eq 'tools/(test|perf|conformance)([[:space:]]|$)' <<<"$cmd"; then
  deny "full-suite, perf and conformance runs belong to the tester. Use tools/test-file for targeted runs."
fi

if [[ -n "$f" ]]; then
  case "$f" in
    "$main"/scratchpad/parity/*) deny "scratchpad/parity/ (briefs, run state, history) belongs to the PM and orchestrator." ;;
  esac
  # The file's own worktree decides which protected list applies.
  root="$(git -C "$(dirname "$f")" rev-parse --show-toplevel 2>/dev/null)" || exit 0
  prot="$feat/$(basename "$root").protected"
  [[ -f "$prot" ]] || exit 0
  while IFS= read -r p; do
    [[ -n "$p" && "$f" == "$root/$p" ]] &&
      deny "$p is a PM-owned acceptance test. Put new tests in another file; if you think this test is wrong, say so in your report."
  done < <(tail -n +2 "$prot")
  exit 0
fi

# Bash: the shell's cwd is unreliable, so check against every feature's list.
# A protected path is at risk when it is a redirect target, or when the command
# also runs something that writes files. A redirect elsewhere (2>&1, > log)
# next to a read-only run of the protected test is fine.
if [[ -n "$cmd" ]]; then
  targets="$(grep -Eo '[0-9]*>>?[[:space:]]*[^[:space:]&|;<>()]+' <<<"$cmd" | grep -Ev '>&?[0-9]+$' || true)"
  writer=0
  grep -Eq 'sed[[:space:]]+-i|(^|[[:space:];&|(])(cp|mv|rm|tee|truncate|dd|install|patch|git[[:space:]]+(-C[[:space:]]+[^[:space:]]+[[:space:]]+)*(checkout|restore|rm|mv|apply|stash))([[:space:]]|$)' <<<"$cmd" && writer=1
  for prot in "$feat"/*.protected; do
    [[ -f "$prot" ]] || continue
    while IFS= read -r p; do
      [[ -n "$p" ]] || continue
      grep -qF "$p" <<<"$cmd" || continue
      if grep -qF "$p" <<<"$targets" || [[ $writer == 1 ]]; then
        deny "$p is a PM-owned acceptance test and this command could modify it."
      fi
    done < <(tail -n +2 "$prot")
  done
fi
exit 0
