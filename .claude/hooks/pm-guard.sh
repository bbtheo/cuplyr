#!/usr/bin/env bash
# PM-only PreToolUse(Bash) guard (wired in .claude/agents/pm.md).
# Merging and integration setup belong to the orchestrator.
cmd="$(jq -r '.tool_input.command // ""')"
if grep -Eq 'tools/(merge-queue|merge-gate|parity-init)' <<<"$cmd"; then
  echo "BLOCKED by .claude/hooks/pm-guard.sh: merging is the orchestrator's job. Report READY_TO_MERGE instead." >&2
  exit 2
fi
exit 0
