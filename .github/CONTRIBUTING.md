# Contributing to cuplyr

cuplyr runs dplyr pipelines on an NVIDIA GPU through libcudf. The aim is that
any dplyr pipeline gives the same answer on a GPU table as on a data frame.
Bug reports, feature ideas, documentation fixes and code are all welcome.

## How work moves

1. Every bug and feature starts as a [GitHub issue](https://github.com/bbtheo/cuplyr/issues).
   The issue is where the approach gets discussed and agreed.
2. Every change lands through a pull request that links its issue with
   `Fixes #123`. The maintainer and coding agents work this way too.
3. A maintainer runs [the merge gate](#the-merge-gate) on the pull request
   and posts the result as a comment.
4. A passing pull request is squash-merged, so the default branch gets one
   commit per pull request.

Typos and small documentation fixes can skip the issue. Open the pull
request directly.

## Scope

In scope:

- dplyr verbs, helpers and arguments, with dplyr's semantics. A result that
  differs from dplyr is a bug.
- GPU code for verbs that use the CPU fallback today. The fallback stays as
  the safety net.
- Reading and writing data on the GPU, memory management, and the lazy
  optimizer.
- Installation and packaging.

Out of scope for now:

- General lubridate parity. lubridate functions run through the CPU fallback,
  apart from fixed-unit `floor_date()`, `ceiling_date()` and `round_date()`.
- Custom and S4 column classes (`ts`, `difftime`, `hms`, `IDate` and others).
  Upload raises `cuplyr_error_unsupported_class`.
- Multi-GPU execution.

Something outside this list can still come in. Open an issue and make the
case. Changing the scope means a pull request against this section.

### Known differences from dplyr

These differences are deliberate, not bugs:

- A `Date` with a fractional day is floored at upload, with a
  `cuplyr_warning_fractional_date` warning.
- Duplicated column names are allowed only on ungrouped tables.
- `integer64` columns are stored as doubles and lose precision above 2^53,
  with a warning.

## Reporting a bug

Open an issue with the template. A good report has:

- a small example that builds its own data, ideally made with
  `reprex::reprex()`
- what you expected, which is usually what dplyr returns for a data frame
- the output of `cuplyr::diagnostics()`

If you know, say whether the problem shows on eager tables, lazy tables or
both.

## Proposing a feature

Open an issue and show the code you want to work. Wait until a maintainer
agrees on the approach before you write a large pull request.

Issues labelled `help wanted ❤️` are agreed and open to anyone. Comment on the
issue before you start so two people don't do the same work.

## Pull requests

- Fork the repository and branch from the default branch. Name the branch
  after the issue, for example `142-slice-sample-weights`.
- Keep one topic per pull request. A feature plus an unrelated refactor is
  two pull requests.
- Put `Fixes #123` in the description.
- Add a bullet to `NEWS.md` under "development version" for anything a user
  can see. Start with the function name and write what the user can now do.
- Don't change the version in `DESCRIPTION`. Only release pull requests do.
- Give the pull request a conventional-commit title (`feat: ...`, `fix: ...`,
  `perf: ...`, `docs: ...`, `test: ...`, `refactor: ...`). The title becomes
  the squashed commit's subject.
- Fix review comments with new commits. Don't force-push once a review has
  started.

### The merge gate

Every test needs a CUDA GPU, and GitHub's free runners have none. A
maintainer runs the gate on a workstation GPU instead, against your branch
merged into the current default branch. All three parts must pass:

| Part | What it checks | Run it yourself |
|---|---|---|
| Test suite | Every testthat file, no failures or errors | `pixi run test` |
| Performance | No benchmark is more than 1.5x slower than the recorded baseline | `pixi run perf` |
| dplyr conformance | dplyr's own test suite, run against GPU tables. A test that passes today must still pass | `pixi run conformance` |

The performance baseline comes from the maintainer's GPU, so your own
`pixi run perf` numbers are only a rough guide. If the gate fails, the
maintainer posts the failing tests or benchmarks on the pull request.

You can contribute without a GPU. Say so in the pull request and a maintainer
runs the tests for you. This takes longer.

## Developer setup

You need Linux, an NVIDIA GPU (Volta or newer), CUDA 12.2 or newer, and
[pixi](https://pixi.sh). pixi installs R, RAPIDS libcudf and every R package
into the project directory.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # once
git clone https://github.com/<you>/cuplyr.git
cd cuplyr
pixi run build-dll                              # compile the C++ in place
pixi run check-deps                             # confirm the GPU and libraries are found
```

Daily commands:

| Task | Command |
|---|---|
| Compile after a C++ change | `pixi run build-dll` |
| Run one or more test files | `pixi run test-file tests/testthat/test-filter.R` |
| Run the full suite | `pixi run test` |
| Regenerate `NAMESPACE`, `man/` and the Rcpp exports | `pixi run document` |
| Start R with the package environment | `pixi run r`, then `pkgload::load_all()` |
| Build the pkgdown site into `docs/` | `pixi run pkgdown` |

The test commands load the package from the source tree with
`pkgload::load_all()`, so you don't need to install it. They also take a file
lock so two runs never share the GPU. The memory tests need that to stay
stable.

`pixi run install` installs the package into the pixi environment. Use it to
try cuplyr as a user would.

`.claude/reference/architecture.md` explains how the code works: the AST and
optimizer, joins, factors and the libcudf calls to be careful with. Read it
before a larger change.

## Code guidelines

### Exports

Never edit `NAMESPACE` or the Rcpp export files by hand. Add roxygen tags
(`@export`, `@importFrom`) and run `pixi run document`.
Commit the regenerated `src/RcppExports.cpp` and `R/RcppExports.R` with your
C++ change. A source install from a clone uses them as they are.

### Tests

- Tests use testthat, edition 3. Every GPU test starts with `skip_if_no_gpu()`.
- Compare against real dplyr with `expect_same_as_dplyr()` and
  `expect_same_as_dplyr_lazy()` from `tests/testthat/helper-oracle.R`. Cover
  eager and lazy tables.
- Pass `arrange_by =` only where dplyr itself leaves the row order undefined.
- A bug fix starts with a test that fails. If the test passes before the fix,
  the bug isn't understood yet.

```r
test_that("filter() keeps rows where the condition is TRUE", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, NA, 3), g = c("a", "b", "a"))
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, x > 1))
})
```

### Style

Follow the [tidyverse style guide](https://style.tidyverse.org) for R, and
match the surrounding code for C++. Signal errors with `cli::cli_abort()` and
a class that starts with `cuplyr_error_`.

## AI-assisted contributions

Claude Code agents write most of cuplyr's code. The maintainer directs and
reviews them, and their commits carry a `Co-Authored-By` trailer. You may use
AI tools too, on these terms:

- Say so in the pull request, using the line in the template.
- Read every changed line and be able to explain and debug it. If you can't
  answer review questions, the pull request gets closed.
- Tests still come first. Generated code without oracle tests against dplyr
  won't be reviewed.
- Don't post generated text as issue comments or reviews without reading it.

`AGENTS.md` holds the rules that agents working in this repository follow.

## Releases

The default branch always carries a development version ending in `.9000`.
A release is a pull request that sets the version in `DESCRIPTION` and tidies
`NEWS.md`. After the merge, the maintainer:

- tags the commit
- publishes a GitHub release from the NEWS section
- publishes the [website](#website)
- returns the version to `.9000`

Release notes name the RAPIDS version the release was built and tested
against.

## Website

The site at <https://bbtheo.github.io/cuplyr/> is built with pkgdown and
served by GitHub Pages from the `gh-pages` branch. The vignettes and
reference examples run on the GPU, so GitHub's runners can't build it. The
maintainer builds it locally and pushes the result.

- `pixi run pkgdown` builds the site into `docs/` from the current checkout.
  Open `docs/index.html` to check it. `docs/` is not committed.
- `pixi run site-publish` builds the site from `main` and replaces the
  contents of `gh-pages` with it. It refuses to run on a dirty checkout or on
  any commit other than `origin/main`.

The maintainer publishes the site at every release, and after any merge
that changes user-facing docs such as the vignettes, the README or the
documentation of exported functions.

## Code of conduct

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md).

## Project structure

```
cuplyr/
├── R/               # R code (dplyr verbs, AST, optimizer)
├── src/             # C++ code (Rcpp bindings to libcudf)
├── tests/testthat/  # Test suite, including the oracle helpers
├── tools/           # Wrappers for builds, tests, the perf and conformance gates
├── vignettes/       # Long-form documentation
├── man/             # Generated docs (don't edit directly)
├── configure        # Detects CUDA and libcudf, writes src/Makevars
├── pixi.toml        # Dependencies and tasks
└── DESCRIPTION      # R package metadata
```
