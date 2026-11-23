# cuplyr – Notes for Coding Agents

GPU-backed dplyr API in R, with C++/Rcpp bindings to libcudf (cuDF 25.12).
The current goal is **full dplyr feature parity on the GPU**, built by a
multi-agent workflow (see "Parity workflow" below). Work is tracked in GitHub
issues, ordered on the [roadmap board](https://github.com/users/bbtheo/projects/1),
and lands on `main` through pull requests; the contributor-facing
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
  # origin = list(...)      # present ONLY on a lazy result of a CPU step (#10):
  #                         # plain-text history that explain() prints
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
lazy. Only `compute()` (and `collect()` through it) runs an optimized plan;
`explain()`/`show_query()` call the optimizer only to print the plan.

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
| `tools/board list\|next\|show\|add\|status ...` | no | Read the roadmap board and set an issue's status. Orchestrator only; moving an item to Approved or reordering is a maintainer's. |

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
| Orchestrator | Opus (main session) | Dispatches PMs in board order, keeps the board status and the local run state, opens follow-up issues, writes post-mortems, runs the merge queue. |
| `pm` | Opus | One feature in its own worktree. Writes the acceptance tests, then drives coder -> tester -> reviewer rounds. |
| `coder` | Sonnet (rounds 1-2), Opus (rounds 3-4) | Writes the code. Can't touch the PM's acceptance tests. Runs only targeted test files. |
| `tester` | Haiku | Runs the full suite, perf and conformance; writes logs; reports facts only. |
| `reviewer` | Opus | Adversarial review of diff and logs. Never runs code; asks a tester instead. |

How one feature moves:

1. **Issue.** Every backlog item is a GitHub issue on the roadmap board. It
   holds the problem, the agreed approach and any discussion with outside
   contributors. New issues start as Proposed; an item is agreed when a
   maintainer moves it to Approved. The board order is the priority order.
   Open questions carry the `tbd` label and a comment stating the question.
2. **Branch and draft.** The PM works in a worktree on `feat/<name>`, cut
   from `origin/main`. Once the acceptance tests are committed, it pushes
   the branch and opens a draft pull request with `Fixes #n`; after each
   round it pushes again and comments on where the work stands.
3. **Ready for review.** When the reviewer approves, the PM merges
   `origin/main` into the branch, pushes it, completes the description and
   marks the pull request ready. `tools/merge-queue` refuses drafts.
4. **Gate and merge.** The orchestrator moves the issue to In review and
   runs `tools/merge-queue <name>`. The gate result is posted on the pull
   request; on a pass the pull request is squash-merged, which closes the
   issue (Done).

Labels follow the tidyverse set (`bug`, `feature`, `upkeep`, `documentation`,
`breaking change ☠️`, `reprex`, `good first issue ❤️`, `help wanted ❤️`) plus
area labels (`performance`, `dplyr parity`, `io`, `fallback`) and `tbd` (an
open question). Who is working on an issue is its board status, not a label.

Text that goes to GitHub is public and permanent. Write it for an outside
reader: no local paths, agent ids, internal ids (`F124`, `N6`) or round
logs, and no secrets. Decisions belong to "maintainers", not to a named
person. Every issue is self-contained: what is wrong and why it matters, a
reproducible example where there is one, links to related issues by number,
and a **Done when** line. Internal detail (round logs, reviews,
post-mortems) stays in `scratchpad/`.

Local state lives in `scratchpad/parity/` in the main checkout:
`state.md` (runs in flight, merge queue, capacity), `log.md` (tick log),
`ledger.md` (frozen history up to 2026-11-21), `features/<name>.md` (brief,
design, round history),
`features/<name>/` (test runs, reviews),
`features/<name>.protected` and `.commit.txt` (read by `tools/merge-queue`),
`gates/` (merge gate artifacts), `conformance_baseline.tsv`.

The gap analysis the backlog is built from is
`scratchpad/dplyr_feature_gaps.md`. `scratchpad/workflow_state.md` is the
frozen history of phases 0-13.
