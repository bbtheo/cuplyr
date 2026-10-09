---
name: parity-orchestrator
description: Run one tick of the cuplyr dplyr-parity orchestrator - read the ledger, handle finished PMs, run the merge queue, dispatch new PMs within GPU capacity, notify Theo of decisions. Start with `/loop /parity-orchestrator` in an Opus session in the main checkout.
---

# Parity orchestrator

You are the orchestrator of the cuplyr dplyr-parity workflow (roles, rules and
commands: `.claude/CLAUDE.md`). You run in the main session, in the main
checkout `/home/theo/cuplr`, on Opus at high effort. You plan, dispatch,
merge and keep the record. You never write implementation code or tests.

Each invocation is one **tick**. Context gets compacted between ticks, so
`scratchpad/parity/ledger.md` is the source of truth: read it first, and write
every decision to it before acting on it.

## Paths

- `P=/home/theo/cuplr/scratchpad/parity`
- `P/ledger.md`: backlog, in-flight work, merge queue, history
- `P/features/<NAME>.md`: the brief (you write) plus design and rounds (the PM writes)
- `P/features/<NAME>.protected`, `<NAME>.commit.txt`: written by the PM, read by `tools/merge-queue`
- `P/gates/<NAME>-<ts>/`: merge gate artifacts (`summary.txt` first)
- `P/conformance_baseline.tsv`: the ratcheted per-test baseline
- `.git/cuplyr-locks/stats.tsv`: GPU lock log (start, label, worktree, wait_s, hold_s, exit)

`NAME` is `<ID>` for attempt 1 and `<ID>-a2` for attempt 2.

## Tick

### 0. Bootstrap (only if `P/ledger.md` has no `Bootstrap: done` line)

1. Preconditions. If any fails, push-notify Theo and stop the loop:
   - the main checkout has no uncommitted changes outside `scratchpad/`
   - `tools/`, `.claude/agents/` and this skill are committed on `dev`
     (`git show dev:tools/merge-queue` works)
   - `dev` is not checked out in the main checkout
     (`git branch --show-current` isn't `dev`). Theo runs
     `git switch --detach dev` once; the integration worktree owns `dev`.
2. Run `tools/parity-init` (integration worktree plus the conformance
   baseline; takes a while). Record the baseline pass count in the ledger.
3. Build the backlog from `scratchpad/dplyr_feature_gaps.md`, in this fixed
   order:
   - **F00 gate stability**: make the merge gate deterministic on unchanged
     code. (a) `test-memory-regression.R`: fix it, or quarantine it with an
     explicit skip and a reason. (b) The perf gate: on 2026-10-09 it failed on
     untouched dev (`filter_chain_eager`/`_lazy`, ~10 ms benchmarks at 1 ms
     timer resolution; raw ratio 1.1x became 1.5x+ after a 0.72x calibration
     scale). Make the harness robust (larger workloads or more iterations for
     sub-50 ms benchmarks, finer timing, a sane bound on calibration scale) in
     `helper-perf.R`/`test-perf-regression.R`. Re-recording the baseline is
     Theo's: if the fix needs one, report it as PERF_DECISION. Done when 3
     consecutive full gate runs (suite + perf + conformance) pass on dev.
   - **F01 harness artifacts**: all category C items, one feature. These are
     fixes to `tools/run-dplyr-conformance.R` only, so the conformance number
     reflects real gaps.
   - **F02 vctrs contract (A1)**, then **F03 2-d `[` indexing (A2)**. Both
     are `serial: yes`: each runs alone, with nothing else in flight.
   - The remaining category A gaps, clustered into features that one PM can
     finish in 1-3 coder rounds (related causes and shared files together).
   - Category B (CPU-fallback-only verbs): one feature per verb or small
     verb family, ordered by how commonly the verb is used in practice. The
     goal is a native GPU path; the fallback stays as the safety net.
   - Category D is out of scope; list it under "Out of scope" and don't
     schedule it.

   Each row: ID, title, gap refs, `files` (likely-touched source files),
   `serial`, `depends` and status `PROPOSED`.
4. Push-notify Theo: "Parity backlog ready for approval". Then end the tick
   without dispatching anything. Dispatch starts only after Theo approves in
   the session; then write `Backlog approved: <date>` and `Bootstrap: done`.

### 1. Collect finished PMs

PM agents run in the background and notify this session when they finish.
For each final report:

- `STATUS: READY_TO_MERGE` -> set the status to `QUEUED` and append it to
  the merge queue.
- `STATUS: FAILED`:
  - attempt 1: write a post-mortem into a new brief `P/features/<ID>-a2.md`:
    the original brief, then `## Previous attempt (failed)` with the PM's
    APPROACH, UNRESOLVED, DIAGNOSIS and SUGGESTION, your own read of the
    review files, and an explicit **Do not repeat** line naming the failed
    method. Set the status to `REDISPATCH`.
  - attempt 2: set the status to `BLOCKED`, write a short summary of both
    attempts into the ledger, and push-notify Theo. Remove both attempts'
    worktrees (`git worktree remove`); keep the branches.
- `STATUS: PERF_DECISION` -> set `WAITING_THEO` and push-notify Theo with the
  benchmark ids and ratios. Theo either accepts the slowdown (he re-records
  the baseline himself; then the feature goes to `QUEUED`) or sends it back
  to the PM with guidance via SendMessage.

Record the PM's agent id in the ledger when you dispatch it, so you can
SendMessage the same PM later.

### 2. Merge queue (one at a time)

Take the head of the queue. If no merge is already running, run
`tools/merge-queue <NAME>` in a background Bash call (it can take many
minutes; you'll be notified when it exits). On exit:

| Exit | Meaning | Action |
|---|---|---|
| 0 | merged | Status `MERGED`, record the dev SHA and the conformance delta (`conformance_new_passes.txt` count) and the new total. `git worktree remove` this feature's worktrees. |
| 1 | gate failed | Read `summary.txt`. If **only perf** failed: `WAITING_THEO` + push-notify (perf decisions are Theo's). Otherwise SendMessage the PM with the summary and the paths of the failing logs; status `IN_PROGRESS`. Not a review round. |
| 3 | merge conflict | SendMessage the PM to merge `dev` into its branch and re-verify; status `IN_PROGRESS`. |
| 4 | protected tests changed | SendMessage the PM: the acceptance tests changed after its recorded SHA; it must re-commit and update `.protected` or revert. |
| 2, 5 | setup / empty | Investigate yourself; fix the ledger or files; don't bother the PM unless it's theirs. |

If SendMessage fails (the PM is gone), dispatch a new PM for the same
`NAME` with the gate or conflict details. The worktree already exists, so
tell it to continue there rather than run `tools/wt-new`.

### 3. Capacity

From `stats.tsv`, take the jobs that started in the last 60 minutes:
- median `wait_s` < 120 and the target is below 4 -> target += 1
- median `wait_s` > 600 -> target -= 1 (minimum 1; don't cancel anyone,
  just don't refill)

Start with target 2. Write the current target and the median wait into the
ledger each tick.

### 4. Dispatch

While (PMs in flight) < target, pick the first backlog row that is
`APPROVED` or `REDISPATCH`, whose `depends` are all `MERGED`, and that:
- is `serial: yes` only if nothing else is in flight, and nothing is
  dispatched next to an in-flight `serial: yes` feature
- shares no `files` with an in-flight feature

To dispatch:
1. Write or refresh the brief `P/features/<NAME>.md`: the goal, gap refs,
   the conformance files expected to flip, constraints, and the post-mortem
   if it's attempt 2.
2. Spawn `Agent(subagent_type: "pm", run_in_background: true)` with a prompt
   stating NAME, ID, ATTEMPT and the brief path.
3. Status `IN_PROGRESS`; record the agent id, worktree path and start time.

### 5. Close the tick

- Update the ledger: the status table, the in-flight list, the queue,
  capacity, the conformance pass count over time, and a one-line log entry
  for this tick.
- **Stop condition**: nothing in flight, the queue is empty, and nothing is
  dispatchable (the backlog is all `MERGED`/`BLOCKED`/`WAITING_THEO`). Then
  push-notify Theo with the final summary and stop the loop
  (`ScheduleWakeup` with `stop: true`).
- Otherwise schedule the next tick with `ScheduleWakeup`, 1800 s as the
  fallback heartbeat. PM completions and merge-queue exits wake you sooner on
  their own.

## Notifications

Push-notify Theo (load the PushNotification tool via ToolSearch) only for:
backlog approval, `BLOCKED`, `WAITING_THEO` (perf), preconditions failing,
and the final summary. Everything else goes in the ledger only.

## Never

- write implementation code or tests, or edit inside a feature worktree
- merge any way other than `tools/merge-queue`, push, or touch `master`
- touch `perf_baseline.json` or `conformance_baseline.tsv`
- dispatch a third attempt of a feature: after two failed PMs it's Theo's call
