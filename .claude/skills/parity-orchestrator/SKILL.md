---
name: parity-orchestrator
description: Run one tick of the cuplyr dplyr-parity orchestrator - read the ledger, handle finished PMs, run the merge queue, dispatch new PMs within GPU capacity, notify Theo of decisions. Start with `/loop /parity-orchestrator` in an Opus session in the main checkout.
---

# Parity orchestrator

You are the orchestrator of the cuplyr dplyr-parity workflow (roles, rules and
commands: `.claude/CLAUDE.md`). You run in the main session, in the main
checkout `/home/theo/cuplr`, on Opus at high effort. You plan, dispatch,
merge and keep the record. You never write implementation code or tests.

The backlog is public: every row of the ledger is a GitHub issue, and every
feature lands on `main` through a pull request that `tools/merge-queue`
gates and squash-merges. The ledger holds what GitHub doesn't: dispatch
state, agent ids, worktrees, the merge queue and the tick log.

Each invocation is one **tick**. Context gets compacted between ticks, so
`scratchpad/parity/ledger.md` is the source of truth: read it first, and write
every decision to it before acting on it.

**Every turn ends with `ScheduleWakeup`** (prompt `/loop /parity-orchestrator`),
unless the stop condition in step 6 holds. That includes turns where you only
answer Theo or record his decision. A turn that ends without one ends the
loop: the pending wakeup is dropped. On 2026-10-09 this left the orchestrator
dead for three hours after a reply turn.

## Paths

- `P=/home/theo/cuplr/scratchpad/parity`
- `P/ledger.md`: backlog rows with their issue numbers, in-flight work, merge queue, history
- `P/features/<NAME>.md`: the brief (you write) plus design and rounds (the PM writes)
- `P/features/<NAME>.protected`, `<NAME>.commit.txt`: written by the PM, read by `tools/merge-queue`
- `P/gates/<NAME>-<ts>/`: merge gate artifacts (`summary.txt` first)
- `P/conformance_baseline.tsv`: the ratcheted per-test baseline
- `/home/theo/cuplr/scratchpad/perf_baseline.json` (the main checkout's `scratchpad/`, not `P/`): the perf baseline; `tools/merge-queue` ratchets it after a passing gate (gate dir: `perf_observations.tsv`, `perf_ratchet.txt`)
- `.git/cuplyr-locks/stats.tsv`: GPU lock log (start, label, worktree, wait_s, hold_s, exit)

`NAME` is `<ID>` for attempt 1 and `<ID>-a2` for attempt 2. Rows created
before 2026-11-12 keep their `F<n>` id; their issue numbers are in the
ledger's `## Issues` table. A new row's `ID` is its issue number.

## GitHub

- **New backlog rows start as issues.** When you find follow-up work (a PM's
  report, a review, a gate log), open an issue first with
  `gh issue create --title ... --label ... --body-file ...`: `bug` or
  `feature` or `upkeep`, plus an area label (`performance`, `dplyr parity`,
  `io`, `fallback`). Two or three sentences on the problem, the proposed
  approach, and a "Done when" line. Then add the row as `PROPOSED` with the
  issue number. Theo approves rows in the session, as before.
- **Issues opened by other people** are not backlog rows until Theo says so.
  List new ones in the tick log (`gh issue list --state open`) and tell Theo
  in the next notification; don't dispatch them and don't reply on them.
- **Labels mirror the ledger.** Add `wip` when a row goes `IN_PROGRESS` and
  remove it when the row is `MERGED`, `BLOCKED` or back to `APPROVED`. Add
  `needs decision` with a short comment stating the question when a row goes
  `WAITING_THEO` or `BLOCKED`; remove it when Theo decides.
  (`gh issue edit <n> --add-label wip`, `gh issue comment <n> --body-file ...`.)
- **Public text.** Everything you write to GitHub is public and permanent:
  no local paths, agent ids, ledger statuses or round logs. Say what the
  feature does and what is being asked.
- **Never** `gh pr merge`, close or reopen pull requests and issues by hand,
  edit other people's comments, or push anything. A merged pull request
  closes its issue through `Fixes #n`.

## Tick

### 0. Bootstrap (only if `P/ledger.md` has no `Bootstrap: done` line)

1. Preconditions. If any fails, push-notify Theo and stop the loop:
   - the main checkout has no uncommitted changes outside `scratchpad/`
   - `tools/`, `.claude/agents/` and this skill are committed on `main`
     (`git show origin/main:tools/merge-queue` works)
   - `main` is not checked out in the main checkout
     (`git branch --show-current` isn't `main`). Theo runs
     `git switch --detach origin/main` once; the integration worktree owns
     `main`.
   - `gh auth status` succeeds
2. Run `tools/parity-init` (integration worktree plus the conformance
   baseline; takes a while). Record the baseline pass count in the ledger.
3. Build the backlog from `scratchpad/dplyr_feature_gaps.md`, in this fixed
   order:
   - **F00 gate stability**: make the merge gate deterministic on unchanged
     code. (a) `test-memory-regression.R`: fix it, or quarantine it with an
     explicit skip and a reason. (b) The perf gate: on 2026-10-09 it failed on
     untouched main (`filter_chain_eager`/`_lazy`, ~10 ms benchmarks at 1 ms
     timer resolution; raw ratio 1.1x became 1.5x+ after a 0.72x calibration
     scale). Make the harness robust (larger workloads or more iterations for
     sub-50 ms benchmarks, finer timing, a sane bound on calibration scale) in
     `helper-perf.R`/`test-perf-regression.R`. Raising the baseline is
     Theo's: if the fix needs one, report it as PERF_DECISION. Done when 3
     consecutive full gate runs (suite + perf + conformance) pass on main.
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

### 1. Recover PMs lost with an earlier session

PM agents (and their coders, testers and reviewers) live inside this
session's process. If the session ends while one is running (crash, reboot,
closed app), the PM is gone and no completion notification will ever come.
So every tick, before collecting:

1. Call `ListAgents`. Every ledger row that is `IN_PROGRESS` and whose PM
   agent id isn't listed is lost. (`QUEUED` and `WAITING_THEO` rows have no
   running PM, so skip them.)
2. Check whether the lost PM finished before it died: read the last
   assistant message in
   `~/.claude/projects/-home-theo-cuplr/*/subagents/agent-<id>.jsonl`. If it
   is a final report with a `STATUS:` line, handle it in step 2 as if it had
   just arrived.
3. Otherwise redispatch it in place. Spawn
   `Agent(subagent_type: "pm", run_in_background: true)` with the same NAME,
   ID, ATTEMPT and ISSUE (a lost session is not a failed attempt), the brief path,
   and `RESUME: <worktree path>`. Tell it which round it was in according to
   the brief's `## Rounds` log and `P/features/<NAME>/runs/` and `reviews/`,
   and what was in progress when it died.
4. Record the new agent id in the ledger, keep the original start time, and
   add a tick-log line: `<NAME> PM lost with session; redispatched (resume
   round <N>)`.

A merge started in an earlier session is dead too: treat it as not running.
Re-running `tools/merge-queue` exits 2 if the interrupted merge left
`_integration` dirty or off `main`; then follow the exit-2 row in step 3.
If the interrupted run had already merged the pull request on GitHub
(`gh pr view <n> --json state`), the row is `MERGED`: record it, and note
in the ledger that the baselines were not ratcheted for it.
Don't reset `_integration` yourself without checking what's in it.

### 2. Collect finished PMs

PM agents run in the background and notify this session when they finish.
For each final report:

- `STATUS: READY_TO_MERGE` -> record the pull request number, set the status
  to `QUEUED` and append it to the merge queue.
- `STATUS: FAILED`:
  - attempt 1: write a post-mortem into a new brief `P/features/<ID>-a2.md`:
    the original brief, then `## Previous attempt (failed)` with the PM's
    APPROACH, UNRESOLVED, DIAGNOSIS and SUGGESTION, your own read of the
    review files, and an explicit **Do not repeat** line naming the failed
    method. Set the status to `REDISPATCH`.
  - attempt 2: set the status to `BLOCKED`, write a short summary of both
    attempts into the ledger, and push-notify Theo. Remove both attempts'
    worktrees (`git worktree remove`); keep the branches. Label the issue
    `needs decision` and comment what was tried and what is still open.
- `STATUS: PERF_DECISION` -> set `WAITING_THEO`, label the issue
  `needs decision`, and push-notify Theo with the benchmark ids and ratios. Theo either accepts the slowdown (he raises the
  entries with `tools/perf-baseline-set` or re-records; then the feature goes
  to `QUEUED`) or sends it back to the PM with guidance via SendMessage.

Record the PM's agent id in the ledger when you dispatch it, so you can
SendMessage the same PM later.

### 3. Merge queue (one at a time)

Take the head of the queue. If no merge is already running, run
`tools/merge-queue <NAME>` in a background Bash call (it can take many
minutes; you'll be notified when it exits). It pushes the branch if GitHub's
copy is behind, runs the gate on `main` + the squashed branch, posts the
summary on the pull request with a `gpu-gate` commit status, and on a pass
squash-merges the pull request and fast-forwards the local `main`. On exit:

| Exit | Meaning | Action |
|---|---|---|
| 0 | merged | Status `MERGED`, record the pull request number, the main SHA and the conformance delta (`conformance_new_passes.txt` count) and the new total, plus the `perf baseline:` lines from `summary.txt` (what the ratchet lowered or added). `git worktree remove` this feature's worktrees. Remove the issue's `wip` label; the merge closed the issue. |
| 1 | gate failed | Read `summary.txt`. If **only perf** failed: `WAITING_THEO` + push-notify (perf decisions are Theo's). Otherwise SendMessage the PM with the summary and the paths of the failing logs; status `IN_PROGRESS`. Not a review round. |
| 3 | merge conflict | SendMessage the PM to merge `origin/main` into its branch, re-verify and push; status `IN_PROGRESS`. |
| 4 | protected tests changed | SendMessage the PM: the acceptance tests changed after its recorded SHA; it must re-commit and update `.protected` or revert. |
| 6 | merged, perf ratchet failed | Main is merged and the perf baseline is unchanged. Treat the feature as `MERGED` (as in exit 0), read the error in the merge output, and push-notify Theo. Don't re-run the merge. |
| 7 | gate passed, GitHub refused the merge | Main is unchanged. Read the error (a ruleset, a conflict that appeared on GitHub, the head moved). If the head moved or there is a conflict, treat it as exit 3. Otherwise push-notify Theo. Don't retry in a loop. |
| 8 | merged, tree differs from the gated candidate | The pull request is merged, but `main` got something the gate didn't test (another merge landed in between, or the local `main` couldn't fast-forward). Treat the feature as `MERGED`, don't ratchet anything by hand, and push-notify Theo with both SHAs. |
| 2, 5 | setup / empty | Investigate yourself; fix the ledger or files; don't bother the PM unless it's theirs. `no open pull request` means the PM skipped its last step: SendMessage it. |

If SendMessage fails (the PM is gone), dispatch a new PM for the same
`NAME` with `RESUME: <worktree path>` and the gate or conflict details.

### 4. Capacity

From `stats.tsv`, take the jobs that started in the last 60 minutes:
- median `wait_s` < 120 and the target is below 4 -> target += 1
- median `wait_s` > 600 -> target -= 1 (minimum 1; don't cancel anyone,
  just don't refill)

Start with target 2. Write the current target and the median wait into the
ledger each tick.

### 5. Dispatch

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
   stating NAME, ID, ATTEMPT, ISSUE and the brief path.
3. Status `IN_PROGRESS`; record the agent id, worktree path and start time.
   Add the `wip` label to the issue.

### 6. Close the tick

- Update the ledger: the status table, the in-flight list, the queue,
  capacity, the conformance pass count over time, and a one-line log entry
  for this tick.
- **Stop condition**: nothing in flight, the queue is empty, and nothing is
  dispatchable (the backlog is all `MERGED`/`BLOCKED`/`WAITING_THEO`). Then
  push-notify Theo with the final summary and stop the loop
  (`ScheduleWakeup` with `stop: true`).
- Otherwise schedule the next tick with `ScheduleWakeup`, 1800 s as the
  fallback heartbeat. PM completions and merge-queue exits wake you sooner on
  their own. Theo's messages wake you too. When you answer one, still end
  that turn with `ScheduleWakeup`, or the loop stops.

## Notifications

Push-notify Theo (load the PushNotification tool via ToolSearch) only for:
backlog approval, `BLOCKED`, `WAITING_THEO` (perf), preconditions failing,
perf ratchet failed (merge-queue exit 6), and the final summary. Everything else goes in the ledger only.

## Never

- write implementation code or tests, or edit inside a feature worktree
- merge any way other than `tools/merge-queue`, run `gh pr merge`, push, or
  touch `main`
- touch `perf_baseline.json` or `conformance_baseline.tsv`, or run
  `tools/perf-baseline-set` (the merge queue ratchets them; raising a perf entry is Theo's)
- dispatch a third attempt of a feature: after two failed PMs it's Theo's call
