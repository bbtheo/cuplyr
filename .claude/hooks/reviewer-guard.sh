#!/usr/bin/env bash
# Reviewer-only PreToolUse(Bash) guard (wired in .claude/agents/reviewer.md).
# The reviewer reads diffs and logs; it never runs code or changes the repo.
# To get something run, it asks a tester subagent.
cmd="$(jq -r '.tool_input.command // ""')"
deny() { echo "BLOCKED by .claude/hooks/reviewer-guard.sh: $1" >&2; exit 2; }
if grep -Eq 'tools/(gpu-run|test|test-file|perf|conformance|r|build|document|merge-queue|merge-gate|wt-new)' <<<"$cmd"; then
  deny "reviewers don't run code. Spawn a tester subagent with the exact command you want run."
fi
if grep -Eq 'git[[:space:]]+(commit|checkout|switch|reset|restore|stash|merge|rebase|cherry-pick|am|apply|clean|rm|mv|add|tag|branch[[:space:]]+-)' <<<"$cmd" ||
   grep -Eq '>[^&]|sed[[:space:]]+-i|(^|[;&|[:space:]])(rm|mv|cp|tee|truncate)[[:space:]]' <<<"$cmd"; then
  deny "reviewers are read-only."
fi
exit 0
