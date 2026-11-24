# Acceptance tests for #20, harness side: the five vendored dplyr tests the
# issue lists, each run through the conformance harness exactly as
# tools/conformance runs it (the test_that() block is copied verbatim from
# inst/dplyr-conformance/tests/ into a staged copy of that directory, with its
# helpers and _snaps).
#
# Two of them (the indirect many-to-many join warning and the colwise quosure
# deprecation) failed for a harness reason: dplyr's own test run sets
# TESTTHAT_PKG = "dplyr" (testthat::test_check("dplyr")), so code written in
# a test file counts as a *direct* caller for dplyr's is_direct() and for
# lifecycle. The harness ran testthat::test_file() without it, so every test
# body was an indirect caller: the many-to-many warning of a direct call was
# not raised, and lifecycle's `always = TRUE` deprecations warned only once
# per session.
#
# The other two harness causes: the shim's class() turned every collected
# result into a tibble (as_tibble()), although a tbl_gpu uploaded from a
# data.frame collects to a data.frame; and harness fixtures are lazy, so a
# warning that depends on the data (many-to-many) is raised when the result
# is computed, which expect_warning() never did. Inside expect_warning() and
# expect_no_warning() the harness now computes a lazy tbl_gpu result.

p20_tools_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
}

p20_vendored_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "inst", "dplyr-conformance", "tests"),
                mustWork = FALSE)
}

p20_lib_cache <- new.env(parent = emptyenv())

p20_lib <- function() {
  tools_dir <- p20_tools_dir()
  if (!dir.exists(tools_dir) || !dir.exists(p20_vendored_dir())) {
    testthat::skip("tools/ or the vendored dplyr tests are not present; harness tests need a checkout")
  }
  if (is.null(p20_lib_cache$env)) {
    env <- new.env(parent = globalenv())
    sys.source(file.path(tools_dir, "conformance-lib.R"), envir = env)
    p20_lib_cache$env <- env
  }
  p20_lib_cache$env
}

# The test_that() call named `test` in the vendored file `file`, as text.
p20_vendored_block <- function(file, test) {
  exprs <- parse(file.path(p20_vendored_dir(), file), keep.source = FALSE)
  hits <- Filter(function(e) {
    is.call(e) && identical(e[[1]], as.name("test_that")) && identical(e[[2]], test)
  }, as.list(exprs))
  if (length(hits) != 1L) {
    stop(sprintf("expected one test '%s' in %s, found %d", test, file, length(hits)))
  }
  deparse(hits[[1]], width.cutoff = 500L)
}

# Run `code` (lines) as tests/<file> in a staged copy of the vendored tests
# directory (helpers and _snaps included), through the harness.
p20_run <- function(file, code, upload = TRUE,
                    exclusions = NULL) {
  lib <- p20_lib()
  # tools/run-dplyr-conformance.R attaches these, as dplyr's own test run has
  # them on the search path (a function moved into rlang's namespace, as the
  # join test does, finds join_by() there)
  for (pkg in c("dplyr", "tibble", "withr", "purrr")) withr::local_package(pkg)
  dir <- tempfile("p20-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  src <- p20_vendored_dir()
  entries <- list.files(src, all.files = TRUE, no.. = TRUE)
  entries <- entries[!grepl("^test-.*\\.[Rr]$", entries)]
  file.copy(file.path(src, entries), dir, recursive = TRUE)
  writeLines(code, file.path(dir, file))
  lib$conformance_run_file(file.path(dir, file), exclusions = exclusions, upload = upload)
}

p20_row <- function(res, test) {
  ids <- res$ids
  row <- ids[ids$test == test, , drop = FALSE]
  if (nrow(row) != 1L) {
    stop(sprintf("expected one result row for test '%s', found %d", test, nrow(row)))
  }
  row
}

p20_run_vendored <- function(file, test, upload = TRUE, exclusions = NULL) {
  res <- p20_run(file, p20_vendored_block(file, test), upload = upload, exclusions = exclusions)
  p20_row(res, test)
}

p20_join_test <- "mutating joins don't trigger many-to-many warning when called indirectly"
p20_colwise_test <- "colwise verbs deprecate quosures (#4330)"
p20_count_test <- "name must be string"
p20_arrange_test <- "arrange ignores NULLs (#6193)"
p20_bind_test <- "bind_cols() infers classes from first result (#1692)"

test_that("the harness runs test files with TESTTHAT_PKG = \"dplyr\", like dplyr's own test run", {
  skip_if_no_gpu()
  before <- Sys.getenv("TESTTHAT_PKG", unset = NA)
  res <- p20_run("test-fake.R", c(
    'test_that("pkg", {',
    '  expect_equal(Sys.getenv("TESTTHAT_PKG"), "dplyr")',
    '})'
  ))
  expect_equal(p20_row(res, "pkg")$raw_status, "pass")
  # ... and leaves the caller's environment as it was
  expect_identical(Sys.getenv("TESTTHAT_PKG", unset = NA), before)
})

test_that("code in a harness test body is a direct caller for dplyr and lifecycle", {
  skip_if_no_gpu()
  res <- p20_run("test-fake.R", c(
    'test_that("direct", {',
    '  expect_true(dplyr:::is_direct(environment()))',
    '  f <- function() lifecycle::deprecate_warn("1.0.0", "p20_old()", always = TRUE, id = "p20-harness-always")',
    '  expect_warning(f(), class = "lifecycle_warning_deprecated")',
    '  expect_warning(f(), class = "lifecycle_warning_deprecated")',
    '})'
  ))
  expect_equal(p20_row(res, "direct")$raw_status, "pass")
})

test_that("vendored: mutating joins don't trigger many-to-many warning when called indirectly", {
  skip_if_no_gpu()
  row <- p20_run_vendored("test-join.R", p20_join_test)
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
  expect_gt(as.integer(row$uploads), 0L)
})

test_that("vendored: colwise verbs deprecate quosures (#4330), also when run again", {
  skip_if_no_gpu()
  # twice in one session: lifecycle's `always = TRUE` warns every time only
  # for a direct caller (an indirect one is muted after the first warning)
  for (i in 1:2) {
    row <- p20_run_vendored("test-colwise-mutate.R", p20_colwise_test)
    expect_equal(row$raw_status, "pass", info = paste("run", i, row$message))
  }
})

test_that("vendored: count() name must be string", {
  skip_if_no_gpu()
  row <- p20_run_vendored("test-count-tally.R", p20_count_test)
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
  # both statements credited against dplyr's reference, no infrastructure error
  expect_equal(as.integer(row$snapshots_relaxed), 2L)
})

test_that("vendored: arrange ignores NULLs (#6193)", {
  skip_if_no_gpu()
  row <- p20_run_vendored("test-arrange.R", p20_arrange_test)
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
})

test_that("vendored: bind_cols() infers classes from first result (#1692)", {
  skip_if_no_gpu()
  row <- p20_run_vendored("test-bind-cols.R", p20_bind_test)
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
  expect_true(is.na(row$item))
})

test_that("class() of a result reports the class collect() gives it", {
  skip_if_no_gpu()
  res <- p20_run("test-fake.R", c(
    'test_that("df result", {',
    '  df <- data.frame(x = 1:3)',
    '  expect_equal(class(filter(df, x > 1)), "data.frame")',
    '  expect_equal(class(group_by(df, x)), c("grouped_df", "tbl_df", "tbl", "data.frame"))',
    '})',
    'test_that("tibble result", {',
    '  tb <- tibble(x = 1:3)',
    '  expect_equal(class(filter(tb, x > 1)), c("tbl_df", "tbl", "data.frame"))',
    '})',
    'test_that("df result is not a tibble", {',
    '  df <- data.frame(x = 1:3)',
    '  expect_equal(class(filter(df, x > 1)), c("tbl_df", "tbl", "data.frame"))',
    '})'
  ))
  expect_equal(p20_row(res, "df result")$raw_status, "pass", info = p20_row(res, "df result")$message)
  expect_equal(p20_row(res, "tibble result")$raw_status, "pass", info = p20_row(res, "tibble result")$message)
  expect_equal(p20_row(res, "df result is not a tibble")$raw_status, "fail")
})

test_that("a warning a lazy result raises when it is computed counts for expect_warning()", {
  skip_if_no_gpu()
  res <- p20_run("test-fake.R", c(
    'test_that("lazy warns", {',
    '  df <- tibble(x = c(1, 1))',
    '  expect_warning(left_join(df, df, join_by(x)), class = "dplyr_warning_join_relationship_many_to_many")',
    '})',
    'test_that("lazy silenced", {',
    '  df <- tibble(x = c(1, 1))',
    '  expect_no_warning(left_join(df, df, join_by(x), relationship = "many-to-many"))',
    '  expect_no_warning(out <- left_join(df, df, join_by(x), relationship = "many-to-many"))',
    '  expect_equal(nrow(collect(out)), 4L)',
    '})',
    'test_that("no warning is still no warning", {',
    '  df <- tibble(x = c(1, 2))',
    '  expect_warning(left_join(df, df, join_by(x)))',
    '})',
    'test_that("a deferred warning still fails expect_no_warning", {',
    '  df <- tibble(x = c(1, 1))',
    '  expect_no_warning(left_join(df, df, join_by(x)))',
    '})'
  ))
  for (t in c("lazy warns", "lazy silenced")) {
    expect_equal(p20_row(res, t)$raw_status, "pass", info = paste(t, p20_row(res, t)$message))
    expect_true(p20_row(res, t)$exercised, info = t)
  }
  expect_equal(p20_row(res, "no warning is still no warning")$raw_status, "fail")
  expect_equal(p20_row(res, "a deferred warning still fails expect_no_warning")$raw_status, "fail")
})
