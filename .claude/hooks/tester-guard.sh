#!/usr/bin/env bash
# Tester-only PreToolUse(Bash) guard (wired in .claude/agents/tester.md).
# Testers run the wrapped test commands and read/write logs. Nothing else.
cmd="$(jq -r '.tool_input.command // ""')"
deny() { echo "BLOCKED by .claude/hooks/tester-guard.sh: $1" >&2; exit 2; }
if grep -Eq 'git[[:space:]]+(commit|checkout|switch|reset|restore|stash|merge|rebase|cherry-pick|apply|clean|rm|mv|add)' <<<"$cmd"; then
  deny "testers never change the repo."
fi
if grep -Eq 'git[[:space:]]+(-C[[:space:]]+[^[:space:]]+[[:space:]]+)*push|gh[[:space:]]+(pr|issue|release|label|repo)[[:space:]]+(create|merge|close|comment|edit|delete|reopen|ready|review)|gh[[:space:]]+api[[:space:]].*(-X|--method|-f|-F)|gh[[:space:]]+project[[:space:]]+(item-add|item-edit|item-delete|item-archive|edit|create|delete|copy|close|field-create|field-delete|link|unlink|mark-template)|tools/board[[:space:]]+(add|status)' <<<"$cmd"; then
  deny "pushing and writing to GitHub (pull requests, issues) belong to the PM and orchestrator; the roadmap board to the orchestrator."
fi
if grep -Eq '(^|[;&|[:space:]])(sed[[:space:]]+-i|rm|mv|cp)[[:space:]]' <<<"$cmd"; then
  deny "testers only run tests and write logs/reports under scratchpad/parity/."
fi
exit 0
