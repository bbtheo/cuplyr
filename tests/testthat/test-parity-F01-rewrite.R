# F01 (coder-owned) extra tests for the conformance harness: the source
# rewriter that decides which constructor calls are uploaded, the class()
# mask, and the missing-feature guard on relaxed error snapshots.
# Skipped when tools/ is absent (build-ignored).

f01x_lib <- local({
  cache <- NULL
  function() {
    tools_dir <- normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
    if (!dir.exists(tools_dir)) {
      testthat::skip("tools/ is not present (build-ignored); harness tests need a checkout")
    }
    if (is.null(cache)) {
      cache <<- new.env(parent = globalenv())
      sys.source(file.path(tools_dir, "conformance-lib.R"), envir = cache)
    }
    cache
  }
})

f01x_rewrite <- function(code) {
  lib <- f01x_lib()
  lib$conformance_rewrite_source(code)
}

f01x_wrapped <- function(code) {
  out <- f01x_rewrite(code)
  lengths(regmatches(out, gregexpr("cuplyr.conformance.upload", out, fixed = TRUE)))
}

test_that("F01 rewrite: statement-level assignments and verb table arguments are wrapped", {
  expect_equal(f01x_wrapped("df <- tibble(x = 1)"), 1L)
  expect_equal(f01x_wrapped("test_that('a', { df <- data.frame(x = 1); y <- 2 })"), 1L)
  expect_equal(f01x_wrapped("a <- b <- tibble(x = 1)"), 1L)
  expect_equal(f01x_wrapped("filter(tibble(x = 1), x > 0)"), 1L)
  expect_equal(f01x_wrapped("dplyr::filter(.data = tibble(x = 1), x > 0)"), 1L)
  expect_equal(f01x_wrapped("tibble(x = 1) |> filter(x > 0)"), 1L)
  expect_equal(f01x_wrapped("tibble(x = 1) %>% filter(x > 0)"), 1L)
  expect_equal(f01x_wrapped("left_join(tibble(k = 1), tibble(k = 1), by = 'k')"), 2L)
  expect_equal(f01x_wrapped("tibble(k = 1) %>% left_join(tibble(k = 1), by = 'k')"), 2L)
  expect_equal(f01x_wrapped("bind_rows(tibble(x = 1), tibble(x = 2), .id = 'i')"), 2L)
  expect_equal(f01x_wrapped("expect_equal(arrange(tibble(x = 2:1), x), tibble(x = 1:2))"), 1L)
})

test_that("F01 rewrite: other constructor calls stay plain", {
  expect_equal(f01x_wrapped("make <- function() tibble(x = 1)"), 0L)
  expect_equal(f01x_wrapped("lapply(1:2, \\(i) data.frame(i = i))"), 0L)
  expect_equal(f01x_wrapped("df <- tibble(z = data.frame(a = 1))"), 1L)
  expect_equal(f01x_wrapped("mutate(df, data.frame(z = x))"), 0L)
  expect_equal(f01x_wrapped("df %>% mutate(data.frame(z = x))"), 0L)
  expect_equal(f01x_wrapped("mutate(df, y = { z <- tibble(a = 1); z })"), 0L)
  expect_equal(f01x_wrapped("helper(data.frame(x = 1))"), 0L)
  expect_equal(f01x_wrapped("x$y <- tibble(a = 1)"), 0L)
  expect_equal(f01x_wrapped("df <- tibble::tibble(x = 1)"), 0L)
  expect_equal(f01x_wrapped("q <- quote(tibble(x = 1))"), 0L)
  expect_equal(f01x_wrapped("expect_equal(out, tibble(x = 1))"), 0L)
  expect_equal(f01x_wrapped("df %>% mutate(y = 1) %>% helper(tibble(a = 1))"), 0L)
})

test_that("F01 rewrite: the wrapped code is the original code plus the upload calls", {
  code <- c(
    "test_that('x', {",
    "  df <- tibble(x = c(1, NA), y = 'a')  # comment",
    "  expect_equal(filter(df, !is.na(x))$y, 'a')",
    "  f <- function(a = 1L, ...) if (a > 0) a else -a",
    "})"
  )
  out <- f01x_rewrite(code)
  unwrap <- function(e) {
    if (!is.call(e)) {
      return(e)
    }
    if (is.call(e[[1]]) && identical(deparse(e[[1]]), "(base::getOption(\"cuplyr.conformance.upload\"))")) {
      return(unwrap(e[[2]]))
    }
    for (i in seq_along(e)) {
      if (!identical(e[[i]], quote(expr = )) && is.call(e[[i]])) e[[i]] <- unwrap(e[[i]])
    }
    e
  }
  expect_match(out, "cuplyr.conformance.upload", fixed = TRUE)
  expect_identical(
    unwrap(parse(text = out, keep.source = FALSE)[[1]]),
    parse(text = code, keep.source = FALSE)[[1]]
  )
})

test_that("F01 rewrite: every vendored file round-trips through deparse unchanged", {
  lib <- f01x_lib()
  vdir <- normalizePath(testthat::test_path("..", "..", "inst", "dplyr-conformance", "tests"), mustWork = FALSE)
  skip_if_not(dir.exists(vdir), "vendored dplyr tests not present")
  files <- list.files(vdir, pattern = "^(test|helper)-.*[.]R$", full.names = TRUE)
  expect_gt(length(files), 50L)
  for (f in files) {
    lines <- readLines(f, warn = FALSE, encoding = "UTF-8")
    before <- parse(text = lines, keep.source = FALSE)
    after <- parse(text = lib$conformance_rewrite_source(lines, wrap = FALSE), keep.source = FALSE)
    expect_equal(length(after), length(before), info = basename(f))
    for (i in seq_along(before)) {
      # Complex literals (2i) deparse as (0+2i): same value, different parse tree.
      if (grepl("[0-9]i\\b", paste(deparse(before[[i]]), collapse = " "))) next
      expect_identical(after[[i]], before[[i]], info = paste(basename(f), i))
    }
  }
})

test_that("F01: class() of a tbl_gpu fixture reports the fixture's class, results report tibble classes", {
  skip_if_no_gpu()
  lib <- f01x_lib()
  dir <- withr::local_tempdir("f01x-")
  writeLines(c(
    'test_that("class of fixtures", {',
    '  tb <- tibble(g = c(1, 1, 2), x = 1:3)',
    '  df <- data.frame(g = c(1, 1, 2), x = 1:3)',
    '  expect_equal(class(tb), c("tbl_df", "tbl", "data.frame"))',
    '  expect_equal(class(df), "data.frame")',
    '  expect_equal(class(mutate(tb, y = x)), c("tbl_df", "tbl", "data.frame"))',
    '  expect_equal(class(group_by(tb, g)), c("grouped_df", "tbl_df", "tbl", "data.frame"))',
    '  expect_s3_class(mutate(tb, y = x), class(tb), exact = TRUE)',
    '})',
    'test_that("bare data.frame fixtures compare as bare data.frames", {',
    '  df <- data.frame(x = 1:3)',
    '  expect_s3_class(mutate(df, y = x), class(df), exact = TRUE)',
    '  expect_equal(mutate(df, y = x), data.frame(x = 1:3, y = 1:3))',
    '})',
    'test_that("a grouped result is not a bare data.frame", {',
    '  df <- data.frame(x = 1:3)',
    '  expect_s3_class(group_by(df, x), class(df), exact = TRUE)',
    '})'
  ), file.path(dir, "test-cls.R"))
  res <- lib$conformance_run_file(file.path(dir, "test-cls.R"))
  expect_equal(res$ids$status, c("pass", "pass", "fail"))
})

test_that("F01: fixtures evaluated in the global environment are still uploaded", {
  skip_if_no_gpu()
  lib <- f01x_lib()
  dir <- withr::local_tempdir("f01x-")
  writeLines(c(
    'test_that("global env", {',
    '  out <- rlang::eval_bare(rlang::expr(dplyr::filter(data.frame(x = 1:3), x > 1)), rlang::env(rlang::global_env()))',
    '  expect_true(inherits(out, "tbl_gpu"))',
    '})'
  ), file.path(dir, "test-glob.R"))
  res <- lib$conformance_run_file(file.path(dir, "test-glob.R"))
  expect_equal(res$ids$status, "pass")
  expect_equal(res$ids$uploads, 1L)
})

# Run a fake vendored file with a _snaps reference; returns the file result.
f01x_run_snap <- function(code, snaps) {
  lib <- f01x_lib()
  dir <- withr::local_tempdir("f01x-", .local_envir = parent.frame())
  writeLines(code, file.path(dir, "test-s.R"))
  dir.create(file.path(dir, "_snaps"))
  writeLines(snaps, file.path(dir, "_snaps", "s.md"))
  lib$conformance_run_file(file.path(dir, "test-s.R"))
}

test_that("F01 C4: an error snapshot whose only error is a missing method is a failure", {
  skip_if_no_gpu()
  res <- f01x_run_snap(c(
    'test_that("missing method", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, filter_out(df, x > 1))',
    '})',
    'test_that("real error", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, filter(df, y > 1))',
    '})'
  ), c(
    "# missing method", "", "    Code", "      filter_out(df, x > 1)", "    Condition", "      Error:", "      ! nope", "",
    "# real error", "", "    Code", "      filter(df, y > 1)", "    Condition", "      Error in `filter()`:", "      ! Problem", ""
  ))
  expect_equal(res$ids$status, c("fail", "pass"))
  expect_equal(res$ids$snapshots_relaxed, c(0L, 1L))
  expect_equal(nrow(res$relaxed_snapshots), 1L)
  expect_match(res$relaxed_snapshots$reference, "Problem")
})

test_that("F01 C4 (B1): an error snapshot is credited statement by statement against the reference", {
  skip_if_no_gpu()
  res <- f01x_run_snap(c(
    'test_that("all agree", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, {',
    '    stop("a")',
    '    nrow(df)',
    '    stop("b")',
    '  })',
    '})',
    'test_that("cuplyr rejects what dplyr accepts", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, {',
    '    stop("a")',
    '    stop("b")',
    '  })',
    '})',
    'test_that("cuplyr accepts what dplyr rejects", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, {',
    '    stop("a")',
    '    nrow(df)',
    '  })',
    '})',
    'test_that("no reference block is not credited", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("a"))',
    '})',
    'test_that("second snapshot uses the second block", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(nrow(df))',
    '  expect_snapshot(error = TRUE, stop("a"))',
    '})',
    'test_that("infrastructure errors are not credited", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("GPU operation \'inner_join\' failed: CUDF failure at x"))',
    '})'
  ), c(
    "# all agree", "", "    Code", "      stop(\"a\")", "    Condition", "      Error:", "      ! a",
    "    Code", "      nrow(df)", "    Output", "      [1] 3",
    "    Code", "      stop(\"b\")", "    Condition", "      Error:", "      ! b", "",
    "# cuplyr rejects what dplyr accepts", "", "    Code", "      stop(\"a\")", "    Condition", "      Error:", "      ! a",
    "    Code", "      nrow(df)", "    Output", "      [1] 3", "",
    "# cuplyr accepts what dplyr rejects", "", "    Code", "      stop(\"a\")", "    Condition", "      Error:", "      ! a",
    "    Code", "      stop(\"b\")", "    Condition", "      Error:", "      ! b", "",
    "# second snapshot uses the second block", "", "    Code", "      nrow(df)", "    Output", "      [1] 3", "",
    "---", "", "    Code", "      stop(\"a\")", "    Condition", "      Error:", "      ! a", "",
    "# infrastructure errors are not credited", "", "    Code", "      stop(\"x\")", "    Condition", "      Error:", "      ! x", ""
  ))
  expect_equal(res$ids$status, c("pass", "fail", "fail", "skip", "pass", "fail"))
  expect_equal(res$ids$snapshots_relaxed, c(1L, 0L, 0L, 0L, 1L, 0L))
  # All messages of a credited snapshot are kept for audit.
  expect_equal(res$relaxed_snapshots$cuplyr_message[[1]], "a ; b")
  expect_match(res$ids$message[[6]], "infrastructure")
})

test_that("F01 rewrite (B2): only the verb's own first formal is the table", {
  # A dot argument called `x` is not the table ...
  expect_equal(f01x_wrapped("summarise(tibble(g = 1), x = x + 1)"), 1L)
  expect_equal(f01x_wrapped("tibble(g1 = 1, g2 = 1, x = 1) |> group_by(g1, g2) |> summarise(x = x + 1)"), 1L)
  expect_equal(f01x_wrapped("mutate(df, x = tibble(a = 1))"), 0L)
  # ... the real first argument is, named or not.
  expect_equal(f01x_wrapped("count(x = tibble(a = 1))"), 1L)
  expect_equal(f01x_wrapped("mutate(.data = tibble(a = 1), x = 2)"), 1L)
  expect_equal(f01x_wrapped("left_join(x = tibble(k = 1), y = tibble(k = 1), by = 'k')"), 2L)
  expect_equal(f01x_wrapped("left_join(tibble(k = 1), x = tibble(k = 1))"), 2L)
})

test_that("F01: a control run (upload = FALSE) uploads nothing", {
  skip_if_no_gpu()
  lib <- f01x_lib()
  dir <- withr::local_tempdir("f01x-")
  writeLines(c(
    'test_that("plain", {',
    '  df <- tibble(x = 1:3)',
    '  expect_false(inherits(df, "tbl_gpu"))',
    '})'
  ), file.path(dir, "test-ctl.R"))
  res <- lib$conformance_run_file(file.path(dir, "test-ctl.R"), upload = FALSE)
  expect_equal(res$ids$raw_status, "pass")
  expect_equal(res$ids$uploads, 0L)
  expect_equal(res$ids$status, "not_exercised")
})

test_that("F01 C4 (B3/B4): an infrastructure-looking error counts only when the reference of that statement lacks it", {
  skip_if_no_gpu()
  ref <- function(test, ...) c(paste("#", test), "", ..., "")
  err <- function(code, ...) {
    l <- c(...)
    c("    Code", paste0("      ", code), "    Condition", "      Error:", paste0("      ", c(paste0("! ", l[[1]]), l[-1])))
  }
  res <- f01x_run_snap(c(
    'test_that("dplyr own unused argument", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("unused argument (1)"))',
    '})',
    'test_that("GPU failure still fails", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("GPU operation \'filter\' failed: CUDF failure at x"))',
    '})',
    'test_that("text in another statement does not exempt", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, {',
    '    stop("a")',
    '    stop("unused argument (1)")',
    '  })',
    '})',
    'test_that("pick outside a mask is not the reference error", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("Must only be used inside data-masking verbs like `mutate()`."))',
    '})',
    'test_that("pick outside a mask is the reference error", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("Must only be used inside data-masking verbs like `mutate()`, `filter()`, and `group_by()`."))',
    '})',
    'test_that("unvalidated name", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("The LHS of `:=` must be a string, not the number 1."))',
    '})',
    'test_that("wrapped infrastructure error", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, rlang::abort("Problem", parent = simpleError("CUDF failure at y")))',
    '})'
  ), c(
    ref("dplyr own unused argument", err("closest(a >= b, 1)", "unused argument (1)")),
    ref("GPU failure still fails", err("filter(df, x)", "nope")),
    ref("text in another statement does not exempt",
      err("f()", "unused argument (1)"), err("g()", "something else")),
    ref("pick outside a mask is not the reference error", err("pick()", "Must supply at least one input to `pick()`.")),
    ref("pick outside a mask is the reference error",
      err("pick()", "Must only be used inside data-masking verbs like `mutate()`, `filter()`, and", "  `group_by()`.")),
    ref("unvalidated name", err("count(df, x, name = 1)", "`name` must be a single string, not the number 1.")),
    ref("wrapped infrastructure error", err("f()", "Problem"))
  ))
  expect_equal(res$ids$status, c("pass", "fail", "fail", "fail", "pass", "fail", "fail"))
  expect_equal(res$ids$snapshots_relaxed, c(1L, 0L, 0L, 0L, 1L, 0L, 0L))
  expect_true(all(grepl("infrastructure", res$ids$message[c(2, 3, 4, 6, 7)])))
  # Same innermost line (the reference's wrapped line is joined): not flagged.
  expect_equal(res$relaxed_snapshots$final_line_differs, c(FALSE, FALSE))
})

test_that("F01 C4 (N1): an uncredited snapshot does not end the test", {
  skip_if_no_gpu()
  res <- f01x_run_snap(c(
    'test_that("later failure still runs", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("a"))',
    '  expect_equal(1, 2)',
    '})',
    'test_that("later pass still runs", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("a"))',
    '  expect_equal(nrow(df), 3L)',
    '})',
    'test_that("only an uncredited snapshot", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("a"))',
    '})'
  ), c("# unrelated", "", "    Code", "      1", "    Output", "      [1] 1", ""))
  expect_equal(res$ids$status, c("fail", "pass", "skip"))
  expect_equal(res$ids$snapshots_uncredited, c(1L, 1L, 1L))
  expect_equal(res$ids$snapshots_relaxed, c(0L, 0L, 0L))
})

test_that("F01 C4 (N3): a credited snapshot whose innermost error differs is flagged for audit", {
  skip_if_no_gpu()
  res <- f01x_run_snap(c(
    'test_that("differs", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, stop("cuplyr wording"))',
    '})',
    'test_that("same", {',
    '  df <- tibble(x = 1:3)',
    '  expect_snapshot(error = TRUE, rlang::abort("Problem", parent = simpleError("Column `y` not found.")))',
    '})'
  ), c(
    "# differs", "", "    Code", "      f()", "    Condition", "      Error:", "      ! dplyr wording", "",
    "# same", "", "    Code", "      f()", "    Condition", "      Error in `f()`:", "      i In argument: `y`.",
    "      Caused by error:", "      ! Column `y` not found.", ""
  ))
  expect_equal(res$ids$status, c("pass", "pass"))
  expect_equal(res$relaxed_snapshots$final_line_differs, c(TRUE, FALSE))
  lib <- f01x_lib()
  rep <- lib$conformance_report(list(res))
  expect_true(any(grepl("AUDIT (differs)", rep, fixed = TRUE)))
  expect_true(any(grepl("^1 test\\(s\\) have a check whose final", rep)))
})
