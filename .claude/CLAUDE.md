# cuplyr – Notes for Coding Agents

GPU-backed dplyr API in R, with C++/Rcpp bindings to libcudf (cuDF 25.12).
The current goal is **full dplyr feature parity on the GPU**, built by a
multi-agent workflow (see "Parity workflow" below). Work is tracked in GitHub
issues and lands on `main` through pull requests; the contributor-facing
description is `.github/CONTRIBUTING.md`.

For how the code works (source map, AST/optimizer, joins, factor handling,
cuDF API gotchas, test helpers), read `.claude/reference/architecture.md`
before changing code. This file only covers what every agent needs.

## Core facts

```r
# tbl_gpu (R/tbl-gpu.R)
list(
  ptr = <externalptr>,      # XPtr to cudf::table
  schema = list(names = c("a", "b"), types = c("FLOAT64", "INT32")),
  groups = c("a"),          # group_by() columns, metadata only
  exec_mode = "eager",      # or "lazy"
  lazy_ops = NULL           # pending AST when lazy
  # rowwise = TRUE          # present ONLY on a rowwise table (F49): `groups` then
  #                         # holds the rowwise vars, which are NOT groups; every verb
  #                         # is either rowwise-aware or runs whole on the CPU
)
```

| R type | GPU type | Notes |
|---|---|---|
| logical | BOOL8 | |
| integer | INT32 | |
| double | FLOAT64 | |
| character | STRING | |
| Date | TIMESTAMP_DAYS | |
| POSIXct | TIMESTAMP_MICROSECONDS | |
| factor | DICTIONARY32 (logical) / INT32 (physical) | 1-based codes on GPU, labels in `schema$factor_levels`; see the reference for which call sites use which view |
| integer64 | FLOAT64 | loses precision above 2^53, warns |

Every verb builds one AST node and hands it to `push_op()` (`R/execute.R`);
`R/lower.R::lower_and_execute()` is the single execution path for eager and
lazy. The optimizer runs only from `compute()`.

## Hard rules

These are enforced by hooks in `.claude/hooks/` where possible. Don't try to
work around a block; if a rule stops legitimate work, say so in your report.

1. **Nothing is ever installed.** No `R CMD INSTALL`, `pixi run install`,
   `devtools::install()`. Code is loaded with `pkgload::load_all()` from the
   worktree being tested.
2. **All R goes through the GPU lock.** One RTX 5070 (12 GB) serves every
   worktree, so every R process uses a `tools/` wrapper (table below). No raw
   `Rscript`/`R -e`, and no `pixi` calls from agents.
3. **Call tools by absolute path in the worktree you mean.** The scripts test
   the worktree they live in, not your shell's cwd:
   `/home/theo/cuplr-wt/<id>/tools/test-file tests/testthat/test-x.R`.
4. **Never edit `NAMESPACE`** or the Rcpp export files. Use roxygen tags
   (`@export`, `@importFrom`) and run `tools/document`.
5. **Test-first bugfixes.** Add a failing test that reproduces the bug before
   the fix. If it passes unexpectedly, the bug isn't understood yet.
6. **Baselines move only through the merge queue or Theo.** Agents never
   write `scratchpad/perf_baseline.json` or `conformance_baseline.tsv`, or
   set `CUPLYR_PERF_RECORD`. After a fully passing gate, `tools/merge-queue`
   ratchets both: the conformance baseline takes new passes, and the perf
   baseline is lowered to the slower of an entry's last two observations
   (new ids are added). Raising a perf entry (`tools/perf-baseline-set`) and a
   full re-record are Theo's. A manual re-record takes no `merge.lock`:
   don't run one while `tools/merge-queue` is running.
7. **Git and GitHub**: `main` is the only long-lived branch. Every change is
   a pull request from a `feat/<name>` branch, linked to an issue with
   `Fixes #n`, and `main` only moves when `tools/merge-queue` squash-merges
   that pull request after a passing gate. Agents push feature branches and
   nothing else: `git push origin feat/<name>`, never forced, never `main`,
   tags or other refs. Only the PM and the orchestrator push or write to
   GitHub (pull requests, issue comments, labels); coders, testers and
   reviewers don't. Nobody runs `gh pr merge` by hand. Commits keep Theo as
   author and end with the trailer
   `Co-Authored-By: Claude <noreply@anthropic.com>`.
8. **Non-publishable writing** (plans, post-mortems, reviews, logs) goes in
   `scratchpad/` (gitignored), never in the package.

## Commands

| Command | Lock | Use |
|---|---|---|
| `tools/build` | no | Compile this worktree's C++ (runs `./configure` first if needed). Do this after C++ edits so locked runs don't compile. |
| `tools/test-file <file>... [--out f.tsv]` | yes | Run specific test files. |
| `tools/r '<R expr>'` | yes | Ad-hoc R with the package loaded and dplyr attached. |
| `tools/document` | yes | Regenerate NAMESPACE, man/, Rcpp exports. |
| `tools/test [--out f.tsv]` | yes | Full testthat suite. Tester agents only. |
| `tools/perf [--out f.tsv]` | yes | Perf gate (`CUPLYR_PERF=1`, max ratio 1.5 vs the main checkout's baseline). Tester agents only. |
| `tools/perf-baseline-set [--baseline f] <id>=<s>...` | merge.lock | Raise (or lower) perf baseline entries by hand. Theo only; `tools/merge-queue` runs `tools/perf-ratchet` itself. |
| `tools/conformance [files] [--ids f.tsv] [--report f.md]` | yes | dplyr's own vendored test suite run against `tbl_gpu`. Tester agents only. |
| `tools/site` | yes | Build the pkgdown site into `docs/` (vignettes and examples run on the GPU). |
| `tools/site-publish [--no-push]` | yes | Build the site from `origin/main` and push it to `gh-pages`. Theo only. |
| `tools/wt-new <id> [attempt]` | no | Create feature worktree `../cuplr-wt/<id>` on branch `feat/<id>` off `origin/main`. PM only. |
| `tools/merge-queue [--no-merge] <name>` | gate | Run the full gate on `main` + the squashed `feat/<name>`, post the result on its pull request (comment and `gpu-gate` commit status), then squash-merge the pull request. `--no-merge` stops after the report. Orchestrator only. |

`tools/gpu-run` is the lock itself; the wrappers above call it. A gate run
blocks new GPU jobs from starting until it finishes. Lock waits are logged to
`.git/cuplyr-locks/stats.tsv`.

`pixi run test|perf|conformance|document|build|pkgdown|site-publish` are aliases for Theo's
terminal in the main checkout.

## Testing conventions

- Every GPU test starts with `skip_if_no_gpu()`.
- Parity tests compare against real dplyr with `expect_same_as_dplyr(df,
  pipeline, ...)` / `expect_same_as_dplyr_lazy()` (`tests/testthat/helper-oracle.R`).
  Use `arrange_by =` only where dplyr itself doesn't define row order.
- Compare eager and lazy results through `tibble::as_tibble()`.
- `test-memory-regression.R` measures GPU memory deltas and is sensitive to
  other GPU activity; that is one reason everything runs under the lock.

## Parity workflow

Roles (definitions in `.claude/agents/`, orchestrator procedure in
`.claude/skills/parity-orchestrator/SKILL.md`):

| Role | Model | Does |
|---|---|---|
| Orchestrator | Opus (main session) | Owns the backlog and ledger, dispatches PMs, writes post-mortems, runs the merge queue. |
| `pm` | Opus | One feature in its own worktree. Writes the acceptance tests, then drives coder -> tester -> reviewer rounds. |
| `coder` | Sonnet (rounds 1-2), Opus (rounds 3-4) | Writes the code. Can't touch the PM's acceptance tests. Runs only targeted test files. |
| `tester` | Haiku | Runs the full suite, perf and conformance; writes logs; reports facts only. |
| `reviewer` | Opus | Adversarial review of diff and logs. Never runs code; asks a tester instead. |

How one feature moves:

1. **Issue.** Every backlog item is a GitHub issue. It holds the problem, the
   agreed approach and any discussion with outside contributors. The
   orchestrator opens an issue for a new idea before it enters the backlog,
   and Theo approves it.
2. **Branch.** The PM works in a worktree on `feat/<name>`, cut from
   `origin/main`.
3. **Pull request.** When the reviewer approves, the PM merges `origin/main`
   into the branch, pushes it and opens a pull request with `Fixes #n`.
4. **Gate and merge.** The orchestrator runs `tools/merge-queue <name>`. The
   gate result is posted on the pull request; on a pass the pull request is
   squash-merged, which closes the issue.

Labels follow the tidyverse set (`bug`, `feature`, `upkeep`, `documentation`,
`breaking change ☠️`, `wip`, `reprex`) plus area labels (`performance`,
`dplyr parity`, `io`, `fallback`). `wip` marks an issue a PM is working on;
`needs decision` marks one that waits for Theo.

Text that goes to GitHub is public and permanent. Write it for an outside
reader: no local paths, agent ids or ledger jargon, and no secrets. Internal
detail (round logs, reviews, post-mortems) stays in `scratchpad/`.

State lives in `scratchpad/parity/` in the main checkout:
`ledger.md` (dispatch state per backlog row with its issue number, merge
queue, tick log), `features/<name>.md` (brief, design, round history),
`features/<name>/` (test runs, reviews),
`features/<name>.protected` and `.commit.txt` (read by `tools/merge-queue`),
`gates/` (merge gate artifacts), `conformance_baseline.tsv`.

The gap analysis the backlog is built from is
`scratchpad/dplyr_feature_gaps.md`. `scratchpad/workflow_state.md` is the
frozen history of phases 0-13.
