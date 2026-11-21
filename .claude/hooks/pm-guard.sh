#!/usr/bin/env bash
# PM-only PreToolUse(Bash) guard (wired in .claude/agents/pm.md).
# Merging, integration setup and the roadmap board belong to the orchestrator.
cmd="$(jq -r '.tool_input.command // ""')"
if grep -Eq 'tools/(merge-queue|merge-gate|parity-init)' <<<"$cmd"; then
  echo "BLOCKED by .claude/hooks/pm-guard.sh: merging is the orchestrator's job. Report READY_TO_MERGE instead." >&2
  exit 2
fi
if grep -Eq 'gh[[:space:]]+project[[:space:]]+(item-add|item-edit|item-delete|item-archive|edit|create|delete|copy|close|field-create|field-delete|link|unlink|mark-template)|tools/board[[:space:]]+(add|status)' <<<"$cmd"; then
  echo "BLOCKED by .claude/hooks/pm-guard.sh: the roadmap board is the orchestrator's. Report your status; the orchestrator moves the issue." >&2
  exit 2
fi
exit 0
