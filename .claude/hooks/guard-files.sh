#!/usr/bin/env bash
# PreToolUse(Edit|Write|NotebookEdit) guard for every Claude session in this repo.
f="$(jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')"
deny() { echo "BLOCKED by .claude/hooks/guard-files.sh: $1" >&2; exit 2; }
case "$f" in
  */NAMESPACE) deny "NAMESPACE is generated. Use roxygen tags and run tools/document." ;;
  */perf_baseline*.json) deny "tools/merge-queue ratchets the perf baseline after a passing gate; raising an entry or re-recording is Theo's (tools/perf-baseline-set)." ;;
  */conformance_baseline.tsv) deny "the conformance baseline only moves via tools/merge-queue." ;;
esac
exit 0
