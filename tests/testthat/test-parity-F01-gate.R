# F01 (coder-owned) tests for the merge gate's conformance decision:
# conformance_gate_check(), conformance_gate_feature_name() and
# conformance_load_known_failures() in tools/conformance-lib.R. Pure R on
# small data frames; no GPU. Skipped when tools/ is absent (build-ignored).

f01g_lib <- local({
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

f01g_ids <- function(...) {
  x <- c(...)
  data.frame(file = "test-a.R", test = names(x), status = unname(x), stringsAsFactors = FALSE)
}

f01g_known <- function(tests = character(), owner = "F07") {
  data.frame(file = rep("test-a.R", length(tests)), test = tests,
    owner = rep(owner, length.out = length(tests)), reason = rep("r", length(tests)),
    stringsAsFactors = FALSE)
}

f01g_check <- function(baseline, new, known_candidate = f01g_known(), known_base = f01g_known(),
                       feature = "F99", harness_changed = TRUE) {
  f01g_lib()$conformance_gate_check(baseline, new, known_candidate, known_base, feature,
    harness_changed)
}

test_that("F01 gate: pass -> excluded / not_exercised are reclassified, not regressions", {
  base <- f01g_ids(t1 = "pass", t2 = "pass", t3 = "pass")
  new <- f01g_ids(t1 = "pass", t2 = "excluded", t3 = "not_exercised")
  r <- f01g_check(base, new, harness_changed = TRUE)
  expect_true(r$ok)
  expect_equal(nrow(r$regressions), 0)
  expect_setequal(r$reclassified$test, c("t2", "t3"))
  s <- f01g_lib()$conformance_gate_summary(r)
  expect_length(s, 1)
  expect_match(s, "^conformance: PASS \\(1 passing, \\+0 new, 0 regressions, 2 reclassified")
})

test_that("F01 gate: pass -> not_exercised is a regression when the harness is unchanged", {
  base <- f01g_ids(t2 = "pass", t3 = "pass")
  new <- f01g_ids(t2 = "excluded", t3 = "not_exercised")
  r <- f01g_check(base, new, harness_changed = FALSE)
  expect_false(r$ok)
  expect_equal(r$regressions$test, "t3")
  expect_equal(r$reclassified$test, "t2")
  expect_match(f01g_lib()$conformance_gate_summary(r)[1], "^conformance: FAIL")
})

test_that("F01 gate: listed pass -> fail/error is an accepted known failure", {
  base <- f01g_ids(t1 = "pass", t2 = "pass", t3 = "pass")
  new <- f01g_ids(t1 = "fail", t2 = "error", t3 = "pass")
  # one row listed only in the candidate's copy, one only in dev's
  r <- f01g_check(base, new, known_candidate = f01g_known("t1"), known_base = f01g_known("t2"))
  expect_true(r$ok)
  expect_setequal(r$accepted$test, c("t1", "t2"))
  expect_equal(nrow(r$regressions), 0)
  expect_match(f01g_lib()$conformance_gate_summary(r)[1], "2 accepted known failures\\)$")
})

test_that("F01 gate: skip and missing are never accepted, unlisted fail is a regression", {
  base <- f01g_ids(t1 = "pass", t2 = "pass", t3 = "pass")
  new <- f01g_ids(t1 = "skip", t3 = "fail")
  r <- f01g_check(base, new, known_candidate = f01g_known(c("t1", "t2")))
  expect_false(r$ok)
  expect_equal(r$regressions$test, c("t1", "t2", "t3"))
  expect_equal(r$regressions$new_status, c("skip", "missing", "fail"))
  expect_equal(nrow(r$accepted), 0)
})

test_that("F01 gate: the owner is blocked until its rows pass and are deleted", {
  base <- f01g_ids(t1 = "pass", t2 = "fail")
  # owner merges, row still failing, row still listed
  r <- f01g_check(base, f01g_ids(t1 = "fail", t2 = "fail"),
    known_candidate = f01g_known("t1"), known_base = f01g_known("t1"), feature = "F07")
  expect_false(r$ok)
  expect_length(r$owner_failures, 2)
  expect_match(r$owner_failures[1], "t1, which is fail")
  expect_match(r$owner_failures[2], "must delete its row")
  # row only in dev's copy (deleted on the branch) but still failing
  r <- f01g_check(base, f01g_ids(t1 = "fail", t2 = "fail"),
    known_base = f01g_known("t1"), feature = "F07")
  expect_false(r$ok)
  expect_length(r$owner_failures, 1)
  # a row the owner added on its own branch is checked too
  r <- f01g_check(base, f01g_ids(t1 = "pass", t2 = "fail"),
    known_candidate = f01g_known("t2"), feature = "F07")
  expect_false(r$ok)
  expect_match(r$owner_failures[1], "t2, which is fail")
  # fixed: passes and is no longer listed by the candidate
  r <- f01g_check(base, f01g_ids(t1 = "pass", t2 = "pass"),
    known_base = f01g_known(c("t1", "t2")), feature = "F07")
  expect_true(r$ok)
  expect_length(r$owner_failures, 0)
  expect_length(r$warnings, 0)
  # fixed but the row was not deleted
  r <- f01g_check(base, f01g_ids(t1 = "pass", t2 = "pass"),
    known_candidate = f01g_known("t1"), known_base = f01g_known("t1"), feature = "F07")
  expect_false(r$ok)
  expect_match(r$owner_failures, "must delete its row test-a.R :: t1")
})

test_that("F01 gate: a retry branch (-a2) is the same owner", {
  base <- f01g_ids(t1 = "pass")
  r <- f01g_check(base, f01g_ids(t1 = "fail"), known_base = f01g_known("t1"), feature = "F07-a2")
  expect_false(r$ok)
  expect_length(r$owner_failures, 1)
  r <- f01g_check(base, f01g_ids(t1 = "pass"), known_base = f01g_known("t1"), feature = "F07-a2")
  expect_true(r$ok)
  # F070 is not F07
  r <- f01g_check(base, f01g_ids(t1 = "fail"), known_base = f01g_known("t1"), feature = "F070")
  expect_true(r$ok)
  expect_equal(r$accepted$test, "t1")
})

test_that("F01 gate: a listed test passing on a non-owner's merge only warns", {
  base <- f01g_ids(t1 = "fail")
  r <- f01g_check(base, f01g_ids(t1 = "pass"), known_candidate = f01g_known("t1"),
    known_base = f01g_known("t1"), feature = "F05")
  expect_true(r$ok)
  expect_length(r$warnings, 1)
  s <- f01g_lib()$conformance_gate_summary(r)
  expect_match(s[1], "^conformance: PASS \\(1 passing, \\+1 new")
  expect_match(s[2], "^conformance: WARNING: known failure test-a.R :: t1 \\(owner F07\\)")
})

test_that("F01 gate: feature name comes from the gate dir, bad names are errors", {
  fn <- f01g_lib()$conformance_gate_feature_name
  expect_equal(fn("/x/scratchpad/parity/gates/F00-20261009T210542"), "F00")
  expect_equal(fn("/x/gates/F07-a2-20261009T210542/"), "F07-a2")
  expect_error(fn("/x/gates/F07"), "does not end in")
  expect_error(fn("/x/gates/F07-2026100T210542"), "does not end in")
  expect_error(fn("/x/gates/-20261009T210542"), "does not end in")
})

test_that("F01 gate: known-failures file is validated", {
  load <- f01g_lib()$conformance_load_known_failures
  write <- function(lines) {
    p <- withr::local_tempfile(fileext = ".tsv", .local_envir = parent.frame())
    writeLines(lines, p)
    p
  }
  expect_equal(nrow(load(NULL)), 0)
  expect_equal(nrow(load(file.path(tempdir(), "no-such-file.tsv"))), 0)
  ok <- load(write(c("file\ttest\towner\treason", "test-a.R\tt 1\tF07\twhy")))
  expect_equal(ok$owner, "F07")
  expect_error(load(write(c("file\ttest\treason", "test-a.R\tt1\twhy"))), "lacks column")
  expect_error(load(write(c("file\ttest\towner\treason", "test-a.R\tt1\t\twhy"))),
    "`owner` must be non-empty")
  expect_error(load(write(c("file\ttest\towner\treason", "test-a.R\tt1\tF07\t "))),
    "`reason` must be non-empty")
  expect_error(load(write(c("file\ttest\towner\treason", "test-a.R\tt1\tF07\twhy",
    "test-a.R\tt1\tF13\twhy2"))), "duplicate")
})

test_that("F01 gate: the committed known-failures file is valid", {
  f01g_lib()
  path <- normalizePath(testthat::test_path("..", "..", "tools", "conformance-known-failures.tsv"),
    mustWork = FALSE)
  kf <- f01g_lib()$conformance_load_known_failures(path)
  expect_true(file.exists(path))
  # Owners delete their rows when they merge; only additions need a test edit.
  seed <- c(
    "test-slice.R\tslice handles zero-row and zero-column inputs (#1219, #2490)",
    "test-summarise.R\tsummarise(.groups=) in global environment",
    "test-group-by.R\tcan partially `ungroup()` (#6606)",
    "test-join.R\tfiltering joins finalize unspecified columns (#6804)")
  expect_true(all(paste(kf$file, kf$test, sep = "\t") %in% seed))
  expect_true(all(grepl("^F[0-9]{2}$", kf$owner)))
})

test_that("F01 gate: owner tokens are validated", {
  load <- f01g_lib()$conformance_load_known_failures
  p <- withr::local_tempfile(fileext = ".tsv")
  for (o in c("F7", "f07", "F07-a", "F007", "owner")) {
    writeLines(c("file\ttest\towner\treason", paste0("test-a.R\tt1\t", o, "\twhy")), p)
    expect_error(load(p), "`owner` must look like", info = o)
  }
  writeLines(c("file\ttest\towner\treason", "test-a.R\tt1\tF07-a12\twhy"), p)
  expect_equal(load(p)$owner, "F07-a12")
})

test_that("F01 gate: input TSVs must have file/test/status columns", {
  read <- f01g_lib()$conformance_read_gate_ids
  p <- withr::local_tempfile(fileext = ".tsv")
  writeLines(c("file\ttest", "test-a.R\tt1"), p)
  expect_error(read(p), "lacks column\\(s\\): status")
  writeLines(c("file\ttest\tstatus", "test-a.R\tt1\tpass"), p)
  expect_equal(read(p)$status, "pass")
})

test_that("F01 gate: rows the candidate adds relative to dev are reported", {
  added <- f01g_lib()$conformance_gate_added
  kc <- f01g_known(c("t1", "t2"), owner = "F07")
  kb <- f01g_known("t1", owner = "F07")
  ec <- data.frame(file = "test-a.R", test = c("e1", "e2"), item = "C2", stringsAsFactors = FALSE)
  eb <- ec[1, ]
  expect_equal(added(kc, kb, ec, eb), c(
    "conformance: NOTE: candidate adds known-failure row test-a.R :: t2 (owner F07)",
    "conformance: NOTE: candidate adds exclusion test-a.R :: e2 (C2)"))
  # dev has no exclusions file; deletions add nothing
  expect_length(added(kb, kc, NULL, NULL), 0)
  expect_length(added(kc[0, ], kc, ec, NULL), 2)
})
