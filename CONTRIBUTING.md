# Contributing to cuplyr

## Developer Setup

cuplyr uses [pixi](https://pixi.sh) to manage CUDA and RAPIDS dependencies reproducibly.

### First time setup

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and build
git clone https://github.com/bbtheo/cuplyr.git
cd cuplyr
pixi run install
```

### Daily workflow

**Option A: `pixi shell`** (recommended — run normal R commands)

```bash
pixi shell          # Activates the environment
R CMD INSTALL .     # Build after C++ changes
R -e 'devtools::load_all()'  # Quick reload for R-only changes
R -e 'devtools::test()'      # Run tests
R                   # Interactive R session
exit                # Leave pixi shell
```

**Option B: `pixi run`** (no shell activation)

```bash
pixi run install    # Configure + build
pixi run load-dev   # Quick reload (R only)
pixi run test       # Run tests
pixi run r          # Start R
```

### When to use which command

| What changed | Command |
|-------------|---------|
| R code only | `pixi run load-dev` or `devtools::load_all()` |
| C++ code | `pixi run install` |
| CUDA/cudf paths changed | `pixi run dev` (clean rebuild) |
| Added new exported function | `devtools::document()` then `pixi run install` |
| Added new C++ function | `Rcpp::compileAttributes()` then `pixi run install` |

### Available pixi tasks

| Task | Description |
|------|-------------|
| `pixi run install` | Configure + install (the standard build) |
| `pixi run dev` | Clean rebuild from scratch |
| `pixi run test` | Run test suite |
| `pixi run check-deps` | Check system dependencies |
| `pixi run load-dev` | Quick R-only reload |
| `pixi run configure` | Regenerate `src/Makevars` |
| `pixi run clean` | Remove build artifacts |
| `pixi run build` | Build source tarball |
| `pixi run pkgdown` | Build documentation site |

## Code Guidelines

### Exports and NAMESPACE

**Always use roxygen2 for exports.** Never edit `NAMESPACE` by hand.

```r
#' My new function
#'
#' @param x Input
#' @return Output
#' @export
my_function <- function(x) { ... }
```

Then run `devtools::document()` to regenerate `NAMESPACE`.

### Adding a new dplyr verb

See the "Implementing a New dplyr Verb" section in `.claude/CLAUDE.md` for the full pattern (C++ implementation, R wrapper, NAMESPACE exports, tests).

### Testing

- All tests use `testthat` (edition 3)
- GPU tests must start with `skip_if_no_gpu()`
- Run tests: `pixi run test`
- Test helpers are in `tests/testthat/helper-gpu.R`

```r
test_that("my feature works", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5)
  result <- tbl_gpu(df) |> my_verb() |> collect()
  expect_equal(result$x, expected)
})
```

### Bug fixes

When fixing a bug, always write a failing test first, then implement the fix.

## Project Structure

```
cuplyr/
├── R/               # R code (dplyr verbs, AST, optimizer)
├── src/             # C++ code (Rcpp bindings to libcudf)
├── tests/testthat/  # Test suite
├── man/             # Generated docs (don't edit directly)
├── configure        # Build configuration script
├── pixi.toml        # Pixi dependency manager config
└── DESCRIPTION      # R package metadata
```
