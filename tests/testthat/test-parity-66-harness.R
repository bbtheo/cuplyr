# Acceptance tests for #66, harness side: vendored dplyr tests that hand a
# tbl_gpu to dplyr internals, vctrs or tibble, or hand-build a subclass of a
# converted table.
#
# - Six of them test dplyr's own helper code (the non-exported compute_by()
#   and join_rows()) or build their expected value / lookup table with vctrs
#   or tibble functions that need a data frame. They are excluded in
#   tools/conformance-exclusions.tsv with a reason; test-parity-66-native.R
#   checks that the same operations through cuplyr's verbs match dplyr.
# - Two of them prepend a class with `class(x) <- c("cls", class(x))`. The
#   harness's class() reports the class collect() gives a tbl_gpu, so the
#   assignment wrote dplyr's classes onto the tbl_gpu object and corrupted it.
#   The harness's `class<-` is now the inverse of its class(): the prepended
#   classes land in front of the tbl_gpu's own class. They are not excluded:
#   "strips off subclass" measures cuplyr and passes; the slice-helper dispatch
#   test (#6084) runs on a real subclassed tbl_gpu.
# - "colwise verbs deprecate quosures (#4330)" passes since #121 and is
#   covered by test-parity-20-harness.R.
#
# Each vendored test_that() block runs verbatim in a staged copy of the
# vendored tests directory (helpers and _snaps included), as tools/conformance
# runs it.

p66_tools_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
}

p66_vendored_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "inst", "dplyr-conformance", "tests"),
                mustWork = FALSE)
}

p66_lib_cache <- new.env(parent = emptyenv())

p66_lib <- function() {
  tools_dir <- p66_tools_dir()
  if (!dir.exists(tools_dir) || !dir.exists(p66_vendored_dir())) {
    testthat::skip("tools/ or the vendored dplyr tests are not present; harness tests need a checkout")
  }
  if (is.null(p66_lib_cache$env)) {
    env <- new.env(parent = globalenv())
    sys.source(file.path(tools_dir, "conformance-lib.R"), envir = env)
    p66_lib_cache$env <- env
  }
  p66_lib_cache$env
}

p66_manifest <- function() {
  p66_lib()$conformance_load_exclusions(file.path(p66_tools_dir(), "conformance-exclusions.tsv"))
}

# The test_that() call named `test` in the vendored file `file`, as text.
p66_vendored_block <- function(file, test) {
  exprs <- parse(file.path(p66_vendored_dir(), file), keep.source = FALSE)
  hits <- Filter(function(e) {
    is.call(e) && identical(e[[1]], as.name("test_that")) && identical(e[[2]], test)
  }, as.list(exprs))
  if (length(hits) != 1L) {
    stop(sprintf("expected one test '%s' in %s, found %d", test, file, length(hits)))
  }
  deparse(hits[[1]], width.cutoff = 500L)
}

# Run `code` (lines) as tests/<file> in a staged copy of the vendored tests
# directory, through the harness.
p66_run <- function(file, code, upload = TRUE, exclusions = NULL) {
  lib <- p66_lib()
  # tools/run-dplyr-conformance.R attaches these, as dplyr's own test run has
  # them on the search path
  for (pkg in c("dplyr", "tibble", "withr", "purrr")) withr::local_package(pkg)
  dir <- tempfile("p66-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  src <- p66_vendored_dir()
  entries <- list.files(src, all.files = TRUE, no.. = TRUE)
  entries <- entries[!grepl("^test-.*\\.[Rr]$", entries)]
  file.copy(file.path(src, entries), dir, recursive = TRUE)
  writeLines(code, file.path(dir, file))
  lib$conformance_run_file(file.path(dir, file), exclusions = exclusions, upload = upload)
}

p66_row <- function(res, test) {
  ids <- res$ids
  row <- ids[ids$test == test, , drop = FALSE]
  if (nrow(row) != 1L) {
    stop(sprintf("expected one result row for test '%s', found %d", test, nrow(row)))
  }
  row
}

p66_run_vendored <- function(file, test, upload = TRUE, exclusions = NULL) {
  res <- p66_run(file, p66_vendored_block(file, test), upload = upload, exclusions = exclusions)
  p66_row(res, test)
}

# (file, test, item) of the tests that are excluded
p66_excluded <- data.frame(
  file = c(
    "test-by.R", "test-by.R", "test-by.R",
    "test-by.R", "test-by.R", "test-by.R",
    "test-join-rows.R",
    "test-join.R",
    "test-recode-values.R"
  ),
  test = c(
    "computes group data when `by` is set",
    "computes `by` group data in order of appearance",
    "extracts existing data when `by = NULL`",
    # these three pass in the run only because a relaxed `error = TRUE`
    # snapshot credits any error: dplyr's grouped/rowwise check in
    # compute_by() is an inherits() test a grouped tbl_gpu doesn't meet, so
    # compute_by() goes on and fails on the `[` of the tbl_gpu instead
    "can't set `.by` with a grouped-df",
    "can't set `.by` with a rowwise-df",
    "can tweak the error args",
    "missing values only match with `==`, `>=`, and `<=` conditions",
    "joins using `overlaps(bounds =)` work as expected (#6488)",
    "`NA` is considered unmatched unless handled explicitly"
  ),
  item = c("C3", "C3", "C3", "C3", "C3", "C3", "C3", "C1", "C1"),
  stringsAsFactors = FALSE
)

p66_subclass_summarise <- "strips off subclass"
p66_subclass_slice <- "slice_helpers do call slice() and benefit from dispatch (#6084)"
p66_colwise <- "colwise verbs deprecate quosures (#4330)"

# ---------------------------------------------------------------------------
# The exclusion manifest

test_that("the tests that call dplyr internals or need a data frame helper input are excluded with a reason", {
  m <- p66_manifest()
  for (i in seq_len(nrow(p66_excluded))) {
    hit <- m[m$file == p66_excluded$file[[i]] & m$test == p66_excluded$test[[i]], , drop = FALSE]
    info <- paste(p66_excluded$file[[i]], p66_excluded$test[[i]], sep = " :: ")
    expect_equal(nrow(hit), 1L, info = info)
    if (nrow(hit) == 1L) {
      expect_equal(hit$item, p66_excluded$item[[i]], info = info)
      expect_gt(nchar(trimws(hit$reason)), 40L)
    }
  }
})

test_that("the hand-built subclass tests and the colwise test are not excluded", {
  m <- p66_manifest()
  key <- paste(m$file, m$test, sep = " :: ")
  expect_false(paste("test-summarise.R", p66_subclass_summarise, sep = " :: ") %in% key)
  expect_false(paste("test-slice.R", p66_subclass_slice, sep = " :: ") %in% key)
  expect_false(paste("test-colwise-mutate.R", p66_colwise, sep = " :: ") %in% key)
})

# ---------------------------------------------------------------------------
# The excluded tests, run as tools/conformance runs them

test_that("vendored: each excluded test reports `excluded`, and passes on plain dplyr", {
  skip_if_no_gpu()
  m <- p66_manifest()
  for (i in seq_len(nrow(p66_excluded))) {
    f <- p66_excluded$file[[i]]
    t <- p66_excluded$test[[i]]
    row <- p66_run_vendored(f, t, exclusions = m)
    expect_equal(row$status, "excluded", info = paste(f, t, row$message))
    expect_equal(row$item, p66_excluded$item[[i]], info = paste(f, t))
    # the test itself is sound: without uploads it passes on dplyr/tibble
    control <- p66_run_vendored(f, t, upload = FALSE)
    expect_equal(control$raw_status, "pass", info = paste("control:", f, t, control$message))
  }
})

# ---------------------------------------------------------------------------
# `class<-` in the shim env is the inverse of its class()

test_that("prepending a class to an uploaded fixture keeps it a tbl_gpu", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("fixture subclass", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66cls", class(df))',
    '  expect_identical(base::class(df), c("p66cls", "tbl_gpu", "tbl"))',
    '  expect_identical(class(df), c("p66cls", "tbl_df", "tbl", "data.frame"))',
    '  expect_true(inherits(df, "p66cls"))',
    '  expect_equal(nrow(collect(filter(df, x > 1))), 2L)',
    '})',
    'test_that("data.frame fixture subclass", {',
    '  df <- data.frame(x = 1:3)',
    '  class(df) <- c("p66cls", class(df))',
    '  expect_identical(base::class(df), c("p66cls", "tbl_gpu", "tbl"))',
    '  expect_identical(class(df), c("p66cls", "data.frame"))',
    '  expect_s3_class(df, c("p66cls", "data.frame"), exact = TRUE)',
    '  expect_s3_class(df, class(df), exact = TRUE)',
    '})',
    'test_that("data.frame fixture subclass is not a bare data.frame", {',
    '  df <- data.frame(x = 1:3)',
    '  class(df) <- c("p66cls", class(df))',
    '  expect_s3_class(df, "data.frame", exact = TRUE)',
    '})'
  ))
  # as on plain dplyr: a subclassed data.frame is not exactly a data.frame
  expect_equal(p66_row(res, "data.frame fixture subclass is not a bare data.frame")$raw_status, "fail")
  for (t in c("fixture subclass", "data.frame fixture subclass")) {
    row <- p66_row(res, t)
    expect_equal(row$raw_status, "pass", info = paste(t, row$message))
    expect_true(row$exercised, info = t)
  }
})

test_that("several classes, a second prepend and a no-op assignment round-trip", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("several", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66a", "p66b", class(df))',
    '  expect_identical(base::class(df), c("p66a", "p66b", "tbl_gpu", "tbl"))',
    '  expect_identical(class(df), c("p66a", "p66b", "tbl_df", "tbl", "data.frame"))',
    '})',
    'test_that("second prepend", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66a", class(df))',
    '  class(df) <- c("p66b", class(df))',
    '  expect_identical(base::class(df), c("p66b", "p66a", "tbl_gpu", "tbl"))',
    '  expect_identical(class(df), c("p66b", "p66a", "tbl_df", "tbl", "data.frame"))',
    '})',
    'test_that("no-op", {',
    '  gdf <- tibble(x = 1:4, g = c(1, 1, 2, 2)) |> group_by(g)',
    '  class(gdf) <- class(gdf)',
    '  expect_identical(base::class(gdf), c("tbl_gpu", "tbl"))',
    '  class(gdf) <- c("p66a", class(gdf))',
    '  class(gdf) <- class(gdf)',
    '  expect_identical(base::class(gdf), c("p66a", "tbl_gpu", "tbl"))',
    '})'
  ))
  for (t in c("several", "second prepend", "no-op")) {
    row <- p66_row(res, t)
    expect_equal(row$raw_status, "pass", info = paste(t, row$message))
    expect_true(row$exercised, info = t)
  }
})

test_that("expect_s3_class() sees a prepended class, as class() does", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("with subclass", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66a", class(df))',
    '  expect_s3_class(df, c("p66a", "tbl_df", "tbl", "data.frame"), exact = TRUE)',
    '  expect_s3_class(df, "p66a")',
    '})',
    'test_that("subclass is not ignored", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66a", class(df))',
    '  expect_s3_class(df, c("tbl_df", "tbl", "data.frame"), exact = TRUE)',
    '})',
    'test_that("a verb result without the subclass", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66a", class(df))',
    '  out <- summarise(df, n = n())',
    '  expect_s3_class(out, c("tbl_df", "tbl", "data.frame"), exact = TRUE)',
    '})'
  ))
  expect_equal(p66_row(res, "with subclass")$raw_status, "pass", info = p66_row(res, "with subclass")$message)
  expect_equal(p66_row(res, "subclass is not ignored")$raw_status, "fail")
  expect_equal(p66_row(res, "a verb result without the subclass")$raw_status, "pass",
               info = p66_row(res, "a verb result without the subclass")$message)
})

test_that("prepending a class to a grouped result keeps it a grouped tbl_gpu", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("grouped subclass", {',
    '  gdf <- tibble(x = 1:4, g = c(1, 1, 2, 2)) |> group_by(g)',
    '  class(gdf) <- c("p66cls", class(gdf))',
    '  expect_identical(base::class(gdf), c("p66cls", "tbl_gpu", "tbl"))',
    '  expect_identical(class(gdf), c("p66cls", "grouped_df", "tbl_df", "tbl", "data.frame"))',
    '  expect_identical(group_vars(gdf), "g")',
    '  out <- summarise(gdf, n = n(), .groups = "keep")',
    '  expect_s3_class(out, c("grouped_df", "tbl_df", "tbl", "data.frame"), exact = TRUE)',
    '})'
  ))
  row <- p66_row(res, "grouped subclass")
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
})

test_that("a subclass slice() method is dispatched to on a converted table", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("dispatch", {',
    '  local_methods(',
    '    slice.p66noisy = function(.data, ..., .preserve = FALSE) {',
    '      warning("noisy")',
    '      NextMethod()',
    '    }',
    '  )',
    '  nf <- tibble(x = 1:10, g = rep(1:2, each = 5)) |> group_by(g)',
    '  class(nf) <- c("p66noisy", class(nf))',
    '  expect_warning(out <- slice(nf, 1:2), "noisy")',
    '  expect_equal(nrow(collect(out)), 4L)',
    '})'
  ))
  row <- p66_row(res, "dispatch")
  expect_equal(row$raw_status, "pass", info = row$message)
  expect_true(row$exercised)
})

test_that("other class assignments behave as in base R", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("plain objects", {',
    '  l <- list(a = 1)',
    '  class(l) <- "p66foo"',
    '  expect_identical(base::class(l), "p66foo")',
    '  class(l) <- NULL',
    '  expect_identical(base::class(l), "list")',
    '})',
    'test_that("a class that does not extend the reported one", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- "p66only"',
    '  expect_identical(base::class(df), "p66only")',
    '})'
  ))
  for (t in c("plain objects", "a class that does not extend the reported one")) {
    row <- p66_row(res, t)
    expect_equal(row$raw_status, "pass", info = paste(t, row$message))
  }
})

test_that("without uploads, prepending a class is base R's", {
  skip_if_no_gpu()
  res <- p66_run("test-fake.R", c(
    'test_that("control", {',
    '  df <- tibble(x = 1:3)',
    '  class(df) <- c("p66cls", class(df))',
    '  expect_identical(base::class(df), c("p66cls", "tbl_df", "tbl", "data.frame"))',
    '})'
  ), upload = FALSE)
  row <- p66_row(res, "control")
  expect_equal(row$raw_status, "pass", info = row$message)
})

# ---------------------------------------------------------------------------
# The two hand-built subclass tests, verbatim

test_that("vendored: summarise() strips off subclass", {
  skip_if_no_gpu()
  row <- p66_run_vendored("test-summarise.R", p66_subclass_summarise, exclusions = p66_manifest())
  expect_equal(row$status, "pass", info = row$message)
  expect_true(row$exercised)
  expect_gt(as.integer(row$uploads), 0L)
})

test_that("vendored: the slice-helper dispatch test (#6084) runs on a subclassed tbl_gpu", {
  skip_if_no_gpu()
  row <- p66_run_vendored("test-slice.R", p66_subclass_slice, exclusions = p66_manifest())
  # It no longer errors on a corrupt object. (cuplyr's slice helpers are
  # native GPU methods that don't route through slice(), so the expectations
  # after the first one can fail: that is a measured difference, not an
  # artefact, and it is not excluded.)
  expect_true(row$status %in% c("pass", "fail"), info = paste(row$status, row$message))
  expect_false(grepl("valid <grouped_df>|groups. attribute", row$message %||% ""), info = row$message)
  expect_true(row$exercised)
  expect_true(is.na(row$item))
})
