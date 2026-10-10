# F01 acceptance tests: the dplyr conformance harness's own logic.
#
# The harness lives in tools/ (build-ignored), so these tests skip when
# tools/ is absent (R CMD check). When tools/ is present the harness library
# tools/conformance-lib.R must exist and expose:
#
#   conformance_run_file(path, exclusions = NULL)
#     Runs one vendored-style test file under the shim. Returns a list with
#     `ids` (data.frame; columns include file, test, status, raw_status,
#     exercised, uploads, item) and `unmatched_exclusions` (data.frame with
#     at least a `test` column: manifest rows for this file that matched no
#     test).
#   conformance_load_exclusions(path)
#     Reads and validates the exclusion manifest (file, test, item, reason).
#   conformance_write_ids(ids, path)
#     Writes the per-test TSV read by the merge gate (file, test, status
#     first).
#   conformance_compare(old_ids, new_ids)
#     Per-file pass counts and the pass -> non-pass transitions.
#
# Status vocabulary: pass, fail, error, skip, not_exercised, excluded.

f01_tools_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
}

f01_lib_cache <- new.env(parent = emptyenv())

f01_lib <- function() {
  tools_dir <- f01_tools_dir()
  if (!dir.exists(tools_dir)) {
    testthat::skip("tools/ is not present (build-ignored); harness tests need a checkout")
  }
  if (is.null(f01_lib_cache$env)) {
    lib_path <- file.path(tools_dir, "conformance-lib.R")
    if (!file.exists(lib_path)) {
      stop("tools/conformance-lib.R does not exist")
    }
    env <- new.env(parent = globalenv())
    sys.source(lib_path, envir = env)
    f01_lib_cache$env <- env
  }
  f01_lib_cache$env
}

# Writes `code` as tests/<name> in a fresh temp dir (plus an optional
# _snaps/<stem>.md) and runs it through the harness.
f01_run_fake <- function(code, name = "test-fake.R", snaps = NULL,
                         exclusions = NULL, dir = NULL) {
  lib <- f01_lib()
  if (is.null(dir)) {
    dir <- tempfile("f01-fake-")
  }
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(code, file.path(dir, name))
  if (!is.null(snaps)) {
    dir.create(file.path(dir, "_snaps"), showWarnings = FALSE)
    stem <- sub("^test-", "", sub("\\.R$", "", name))
    writeLines(snaps, file.path(dir, "_snaps", paste0(stem, ".md")))
  }
  lib$conformance_run_file(file.path(dir, name), exclusions = exclusions)
}

f01_row <- function(res, test) {
  ids <- res$ids
  row <- ids[ids$test == test, , drop = FALSE]
  if (nrow(row) != 1) {
    stop(sprintf("expected exactly one result row for test '%s', found %d", test, nrow(row)))
  }
  row
}

f01_status <- function(res, test) f01_row(res, test)$status
f01_raw <- function(res, test) f01_row(res, test)$raw_status
f01_uploads <- function(res, test) as.integer(f01_row(res, test)$uploads)

# ---------------------------------------------------------------------------
# Fake file A: shim scoping (C1), normalization (C7, grouping), exercised /
# not_exercised classification (C5), stable ids.

f01_fake_a <- r"---(
df_file <- data.frame(x = 1:3)

test_that("U1 tibble assignment uploads", {
  df <- tibble(x = 1:3)
  expect_true(inherits(df, "tbl_gpu"))
})

test_that("U1 data.frame assignment uploads", {
  df <- data.frame(x = 1:3)
  expect_true(inherits(df, "tbl_gpu"))
})

test_that("U1 tribble assignment uploads", {
  df <- tribble(~x, ~y, 1, "a", 2, "b")
  expect_true(inherits(df, "tbl_gpu"))
})

test_that("U1 assignment inside braces uploads", {
  {
    df <- tibble(x = 1:3)
  }
  expect_true(inherits(df, "tbl_gpu"))
})

test_that("U2 first verb argument uploads", {
  out <- filter(tibble(x = 1:3), x > 1)
  expect_true(inherits(out, "tbl_gpu"))
})

test_that("U2 magrittr pipe head uploads", {
  out <- tibble(x = 1:3) %>% filter(x > 1)
  expect_true(inherits(out, "tbl_gpu"))
})

test_that("U2 native pipe head uploads", {
  out <- tibble(x = 1:3) |> filter(x > 1)
  expect_true(inherits(out, "tbl_gpu"))
})

test_that("U2 group_by on a constructor uploads", {
  gf <- group_by(tibble(g = c(1, 1, 2), x = 1:3), g)
  expect_true(inherits(gf, "tbl_gpu"))
})

test_that("U2 verb inside an expectation uploads", {
  expect_true(inherits(arrange(tibble(x = 3:1), x), "tbl_gpu"))
})

test_that("U2 second table of a join uploads", {
  a <- tibble(k = 1:2, v = 1:2)
  out <- left_join(a, tibble(k = 1:2, w = 3:4), by = "k")
  expect_true(inherits(out, "tbl_gpu"))
})

test_that("U2 bind_rows arguments upload", {
  out <- bind_rows(tibble(x = 1:2), tibble(x = 3:4))
  expect_true(inherits(out, "tbl_gpu"))
})

test_that("U3 file-level fixture uploads", {
  expect_true(inherits(df_file, "tbl_gpu"))
})

test_that("P1 constructor inside a closure stays plain", {
  df <- tibble(y = 1)
  make <- function(x) tibble(a = x, b = x + 1)
  expect_false(inherits(make(1), "tbl_gpu"))
  res <- lapply(1:2, function(i) data.frame(i = i))
  expect_false(any(vapply(res, inherits, logical(1), "tbl_gpu")))
})

test_that("P2 nested column constructor stays plain", {
  t <- tibble(z = data.frame(A = 1:2, B = 3:4))
  expect_false(inherits(t$z, "tbl_gpu"))
  expect_true(is.data.frame(t$z))
})

test_that("P3 constructor in a verb's dots stays plain", {
  df <- tibble(x = 1:3, y = c(2, 4, 6))
  try(mutate(df, data.frame(z = x / y)), silent = TRUE)
  try(reframe(df, data.frame(m = x)), silent = TRUE)
  expect_true(inherits(df, "tbl_gpu"))
})

test_that("P4 constructor passed to a non-verb helper stays plain", {
  df <- tibble(y = 1)
  wrap <- function(d) structure(d, class = c("wrapped", class(d)))
  w <- wrap(data.frame(x = 1:3))
  expect_false(inherits(w, "tbl_gpu"))
  expect_s3_class(w, "wrapped")
})

test_that("P5 expected values are compared as plain data", {
  df <- tibble(x = 1:3)
  expect_equal(filter(df, x > 1), tibble(x = 2:3))
  expect_identical(filter(data.frame(x = 1:3), x > 1), data.frame(x = 2:3))
})

test_that("S1 different data still fails", {
  df <- tibble(x = 1:3)
  expect_equal(filter(df, x > 1), tibble(x = 3:4))
})

test_that("S2 grouping is part of the comparison", {
  gf <- group_by(tibble(g = c(1, 1, 2), x = 1:3), g)
  expect_equal(gf, tibble::tibble(g = c(1, 1, 2), x = 1:3))
})

test_that("S3 grouped results compare equal to a grouped_df", {
  gf <- group_by(tibble(g = c(1, 1, 2), x = 1:3), g)
  expect_equal(gf, dplyr::group_by(tibble::tibble(g = c(1, 1, 2), x = 1:3), g))
  expect_s3_class(gf, "grouped_df")
})

test_that("S4 lists of tbl_gpu are normalized element-wise", {
  df <- tibble(x = 1:3)
  expect_equal(list(df, df), list(tibble::tibble(x = 1:3), tibble::tibble(x = 1:3)))
  expect_identical(list(a = df), list(a = tibble::tibble(x = 1:3)))
})

test_that("S5 lists of tbl_gpu still compare data", {
  df <- tibble(x = 1:3)
  expect_equal(list(df, df), list(tibble::tibble(x = 1:3), tibble::tibble(x = 4:6)))
})

test_that("S6 list-columns of tbl_gpu are normalized", {
  df <- tibble(x = 1:3)
  nested <- tibble::tibble(g = 1:2, data = list(df, df))
  expect_equal(
    nested,
    tibble::tibble(g = 1:2, data = list(tibble::tibble(x = 1:3), tibble::tibble(x = 1:3)))
  )
})

test_that("S7 list-columns of tbl_gpu still compare data", {
  df <- tibble(x = 1:3)
  nested <- tibble::tibble(g = 1:2, data = list(df, df))
  expect_equal(
    nested,
    tibble::tibble(g = 1:2, data = list(tibble::tibble(x = 1:3), tibble::tibble(x = 0L)))
  )
})

test_that("N1 failed upload that passes is not exercised", {
  df <- tibble(x = list(1, 2))
  expect_equal(length(df$x), 2L)
})

test_that("N2 closure-built fixture is not exercised", {
  make <- function() tibble(x = 1:3)
  expect_equal(nrow(filter(make(), x > 1)), 2L)
})

test_that("N3 file-level fixture through a verb is exercised", {
  expect_equal(filter(df_file, x > 2)$x, 3L)
})

test_that("N4 vector-only test stays pass", {
  expect_equal(1 + 1, 2)
})

test_that("N5 failing test without a tbl_gpu stays fail", {
  df <- tibble(x = list(1, 2))
  expect_equal(length(df$x), 3L)
})

test_that("N6 skipped test stays skip", {
  skip("not today")
})

test_that("dup", {
  expect_true(TRUE)
})

test_that("dup", {
  expect_true(TRUE)
})
)---"

f01_res_a_cache <- new.env(parent = emptyenv())
f01_res_a <- function() {
  if (is.null(f01_res_a_cache$res)) {
    f01_res_a_cache$res <- f01_run_fake(f01_fake_a)
  }
  f01_res_a_cache$res
}

test_that("F01: harness runs a test file and reports one row per test", {
  skip_if_no_gpu()
  res <- f01_res_a()
  ids <- res$ids
  expect_true(all(c("file", "test", "status", "raw_status", "exercised", "uploads", "item") %in% names(ids)))
  expect_equal(unique(ids$file), "test-fake.R")
  expect_equal(nrow(ids), 32L)
  expect_true(all(ids$status %in% c("pass", "fail", "error", "skip", "not_exercised", "excluded")))
  expect_true(all(ids$raw_status %in% c("pass", "fail", "error", "skip")))
  # Repeated test names keep the stable " #k" suffix the merge gate relies on.
  expect_equal(sum(ids$test == "dup"), 1L)
  expect_equal(sum(ids$test == "dup #2"), 1L)
})

test_that("F01 C1: fixtures assigned at test-body level are uploaded", {
  skip_if_no_gpu()
  res <- f01_res_a()
  for (t in c(
    "U1 tibble assignment uploads", "U1 data.frame assignment uploads",
    "U1 tribble assignment uploads", "U1 assignment inside braces uploads"
  )) {
    expect_equal(f01_raw(res, t), "pass", info = t)
    expect_equal(f01_status(res, t), "pass", info = t)
    expect_equal(f01_uploads(res, t), 1L, info = t)
  }
})

test_that("F01 C1: constructors that are a verb's table input are uploaded", {
  skip_if_no_gpu()
  res <- f01_res_a()
  for (t in c(
    "U2 first verb argument uploads", "U2 magrittr pipe head uploads",
    "U2 native pipe head uploads", "U2 group_by on a constructor uploads",
    "U2 verb inside an expectation uploads"
  )) {
    expect_equal(f01_raw(res, t), "pass", info = t)
    expect_equal(f01_uploads(res, t), 1L, info = t)
  }
  for (t in c("U2 second table of a join uploads", "U2 bind_rows arguments upload")) {
    expect_equal(f01_raw(res, t), "pass", info = t)
    expect_equal(f01_uploads(res, t), 2L, info = t)
  }
})

test_that("F01 C1: file-level fixtures are uploaded", {
  skip_if_no_gpu()
  res <- f01_res_a()
  expect_equal(f01_raw(res, "U3 file-level fixture uploads"), "pass")
})

test_that("F01 C1: constructors in closures, nested columns, verb dots and helper arguments stay plain", {
  skip_if_no_gpu()
  res <- f01_res_a()
  expect_equal(f01_raw(res, "P1 constructor inside a closure stays plain"), "pass")
  expect_equal(f01_uploads(res, "P1 constructor inside a closure stays plain"), 1L)
  expect_equal(f01_raw(res, "P2 nested column constructor stays plain"), "pass")
  expect_equal(f01_raw(res, "P3 constructor in a verb's dots stays plain"), "pass")
  expect_equal(f01_uploads(res, "P3 constructor in a verb's dots stays plain"), 1L)
  expect_equal(f01_raw(res, "P4 constructor passed to a non-verb helper stays plain"), "pass")
  expect_equal(f01_uploads(res, "P4 constructor passed to a non-verb helper stays plain"), 1L)
})

test_that("F01 C1: expected values are not uploaded and compare as plain data", {
  skip_if_no_gpu()
  res <- f01_res_a()
  expect_equal(f01_raw(res, "P5 expected values are compared as plain data"), "pass")
  # Only the two pipeline inputs are uploaded, never the expected values.
  expect_equal(f01_uploads(res, "P5 expected values are compared as plain data"), 2L)
})

test_that("F01: comparisons still compare real data and grouping", {
  skip_if_no_gpu()
  res <- f01_res_a()
  expect_equal(f01_raw(res, "S1 different data still fails"), "fail")
  expect_equal(f01_raw(res, "S2 grouping is part of the comparison"), "fail")
  expect_equal(f01_raw(res, "S3 grouped results compare equal to a grouped_df"), "pass")
})

test_that("F01 C7: lists and list-columns of tbl_gpu are normalized recursively", {
  skip_if_no_gpu()
  res <- f01_res_a()
  expect_equal(f01_raw(res, "S4 lists of tbl_gpu are normalized element-wise"), "pass")
  expect_equal(f01_raw(res, "S5 lists of tbl_gpu still compare data"), "fail")
  expect_equal(f01_raw(res, "S6 list-columns of tbl_gpu are normalized"), "pass")
  expect_equal(f01_raw(res, "S7 list-columns of tbl_gpu still compare data"), "fail")
})

test_that("F01 C5: a passing test that never touched a tbl_gpu is not_exercised", {
  skip_if_no_gpu()
  res <- f01_res_a()
  n1 <- f01_row(res, "N1 failed upload that passes is not exercised")
  expect_equal(n1$raw_status, "pass")
  expect_equal(n1$status, "not_exercised")
  expect_false(as.logical(n1$exercised))
  expect_equal(as.integer(n1$uploads), 0L)

  n2 <- f01_row(res, "N2 closure-built fixture is not exercised")
  expect_equal(n2$raw_status, "pass")
  expect_equal(n2$status, "not_exercised")
  expect_false(as.logical(n2$exercised))
})

test_that("F01 C5: exercised, vector-only, failing and skipped tests keep their status", {
  skip_if_no_gpu()
  res <- f01_res_a()
  n3 <- f01_row(res, "N3 file-level fixture through a verb is exercised")
  expect_equal(n3$status, "pass")
  expect_true(as.logical(n3$exercised))

  n4 <- f01_row(res, "N4 vector-only test stays pass")
  expect_equal(n4$status, "pass")
  expect_false(as.logical(n4$exercised))

  expect_equal(f01_status(res, "N5 failing test without a tbl_gpu stays fail"), "fail")
  expect_equal(f01_status(res, "N6 skipped test stays skip"), "skip")

  u1 <- f01_row(res, "U1 tibble assignment uploads")
  expect_true(as.logical(u1$exercised))
})

test_that("F01: running a file leaves no instrumentation or state behind", {
  skip_if_no_gpu()
  ns <- asNamespace("cuplyr")
  before_fn <- get("new_tbl_gpu", envir = ns)
  before_tbl_gpu <- get("tbl_gpu", envir = ns)
  withr::local_envvar(NOT_CRAN = "sentinel")
  f01_run_fake(c(
    'test_that("one", { df <- tibble(x = 1:3); expect_true(inherits(df, "tbl_gpu")) })'
  ))
  expect_identical(get("new_tbl_gpu", envir = ns), before_fn)
  expect_identical(get("tbl_gpu", envir = ns), before_tbl_gpu)
  expect_equal(Sys.getenv("NOT_CRAN"), "sentinel")
})

# ---------------------------------------------------------------------------
# Snapshots (C4): wording is never compared; whether the code errors is.

f01_fake_snap <- r"---(
test_that("snap error with different wording passes", {
  df <- tibble(x = 1:3)
  expect_snapshot(stop("cuplyr wording"), error = TRUE)
})

test_that("snap error that does not happen fails", {
  df <- tibble(x = 1:3)
  expect_snapshot(nrow(df), error = TRUE)
})

test_that("snap unexpected error is an error", {
  df <- tibble(x = 1:3)
  expect_snapshot(stop("boom"))
})

test_that("snap output only is not counted as a pass", {
  df <- tibble(x = 1:3)
  expect_snapshot(nrow(df))
})
)---"

f01_snaps_ref <- c(
  "# snap error with different wording passes",
  "",
  "    Code",
  "      stop(\"dplyr wording\")",
  "    Condition",
  "      Error:",
  "      ! dplyr wording",
  "",
  "# snap output only is not counted as a pass",
  "",
  "    Code",
  "      nrow(df)",
  "    Output",
  "      [1] 99",
  ""
)

test_that("F01 C4: error snapshots check that an error happens, not its wording", {
  skip_if_no_gpu()
  dir <- withr::local_tempdir("f01-snap-")
  # Even with snapshot comparison switched on in the caller's environment,
  # the harness must not compare wording and must not write .new.md files
  # into the vendored tests directory.
  withr::local_envvar(NOT_CRAN = "true")
  res <- f01_run_fake(f01_fake_snap, name = "test-snap.R", snaps = f01_snaps_ref, dir = dir)
  expect_equal(f01_status(res, "snap error with different wording passes"), "pass")
  expect_equal(f01_status(res, "snap error that does not happen fails"), "fail")
  expect_equal(f01_status(res, "snap unexpected error is an error"), "error")
  expect_equal(f01_status(res, "snap output only is not counted as a pass"), "skip")
  expect_setequal(
    list.files(dir, recursive = TRUE),
    c("test-snap.R", file.path("_snaps", "snap.md"))
  )
})

# ---------------------------------------------------------------------------
# Exclusion manifest (C2-C8 bookkeeping)

f01_write_manifest <- function(rows, path) {
  lines <- c(
    "file\ttest\titem\treason",
    vapply(rows, function(r) paste(r, collapse = "\t"), character(1))
  )
  writeLines(lines, path)
  path
}

test_that("F01: exclusion manifest loads and validates", {
  lib <- f01_lib()
  dir <- withr::local_tempdir("f01-manifest-")

  ok <- f01_write_manifest(list(
    c("test-a.R", "first test", "C2", "uses count_regroups(), dplyr-internal instrumentation"),
    c("test-a.R", "second test", "C3", "calls dplyr:::eval_relocate() directly")
  ), file.path(dir, "ok.tsv"))
  m <- lib$conformance_load_exclusions(ok)
  expect_true(is.data.frame(m))
  expect_true(all(c("file", "test", "item", "reason") %in% names(m)))
  expect_equal(nrow(m), 2L)
  expect_equal(m$item, c("C2", "C3"))

  empty_reason <- f01_write_manifest(list(
    c("test-a.R", "first test", "C2", "")
  ), file.path(dir, "empty.tsv"))
  expect_error(lib$conformance_load_exclusions(empty_reason))

  bad_item <- f01_write_manifest(list(
    c("test-a.R", "first test", "C9", "not a harness item")
  ), file.path(dir, "bad-item.tsv"))
  expect_error(lib$conformance_load_exclusions(bad_item))

  dup <- f01_write_manifest(list(
    c("test-a.R", "first test", "C2", "reason one"),
    c("test-a.R", "first test", "C3", "reason two")
  ), file.path(dir, "dup.tsv"))
  expect_error(lib$conformance_load_exclusions(dup))

  no_reason_col <- file.path(dir, "no-col.tsv")
  writeLines(c("file\ttest\titem", "test-a.R\tfirst test\tC2"), no_reason_col)
  expect_error(lib$conformance_load_exclusions(no_reason_col))
})

test_that("F01: excluded tests get status excluded, keep their raw outcome, and stale entries are reported", {
  skip_if_no_gpu()
  lib <- f01_lib()
  dir <- withr::local_tempdir("f01-excl-")
  manifest <- f01_write_manifest(list(
    c("test-excl.R", "excluded but passing", "C2", "harness artifact one"),
    c("test-excl.R", "excluded and failing", "C3", "harness artifact two"),
    c("test-excl.R", "no such test", "C8", "stale entry"),
    c("test-other.R", "excluded and failing", "C2", "belongs to another file")
  ), file.path(dir, "manifest.tsv"))
  excl <- lib$conformance_load_exclusions(manifest)

  res <- f01_run_fake(c(
    'test_that("excluded but passing", { df <- tibble(x = 1:3); expect_true(inherits(df, "tbl_gpu")) })',
    'test_that("excluded and failing", { df <- tibble(x = 1:3); expect_true(FALSE) })',
    'test_that("not excluded", { df <- tibble(x = 1:3); expect_true(inherits(df, "tbl_gpu")) })'
  ), name = "test-excl.R", exclusions = excl, dir = file.path(dir, "tests"))

  a <- f01_row(res, "excluded but passing")
  expect_equal(a$status, "excluded")
  expect_equal(a$raw_status, "pass")
  expect_equal(a$item, "C2")

  b <- f01_row(res, "excluded and failing")
  expect_equal(b$status, "excluded")
  expect_equal(b$raw_status, "fail")
  expect_equal(b$item, "C3")

  nx <- f01_row(res, "not excluded")
  expect_equal(nx$status, "pass")
  expect_true(is.na(nx$item))

  expect_equal(res$unmatched_exclusions$test, "no such test")
})

# ---------------------------------------------------------------------------
# The real manifest (tools/conformance-exclusions.tsv)

f01_vendored_dir <- function() {
  normalizePath(testthat::test_path("..", "..", "inst", "dplyr-conformance", "tests"), mustWork = FALSE)
}

# Literal test_that() names per vendored file, with the harness's " #k"
# suffix for repeats, plus the set of function names each body calls.
f01_vendored_tests <- function(path) {
  exprs <- parse(path, keep.source = FALSE)
  out <- list()
  for (e in exprs) {
    if (is.call(e) && identical(e[[1]], as.name("test_that")) && is.character(e[[2]])) {
      out[[length(out) + 1L]] <- list(name = e[[2]], calls = all.names(e[[3]]))
    }
  }
  nms <- vapply(out, `[[`, character(1), "name")
  occ <- stats::ave(seq_along(nms), nms, FUN = seq_along)
  ids <- ifelse(occ > 1, paste0(nms, " #", occ), nms)
  ids <- gsub("[\t\n\r]+", " ", ids)
  for (i in seq_along(out)) out[[i]]$id <- ids[[i]]
  out
}

test_that("F01: the real exclusion manifest is valid and every entry names a real test", {
  lib <- f01_lib()
  path <- file.path(f01_tools_dir(), "conformance-exclusions.tsv")
  expect_true(file.exists(path))
  m <- lib$conformance_load_exclusions(path)
  expect_gt(nrow(m), 0L)
  expect_true(all(nzchar(trimws(m$reason))))
  expect_true(all(m$item %in% paste0("C", 1:8)))

  vdir <- f01_vendored_dir()
  for (f in unique(m$file)) {
    expect_true(file.exists(file.path(vdir, f)), info = f)
    if (!file.exists(file.path(vdir, f))) next
    ids <- vapply(f01_vendored_tests(file.path(vdir, f)), `[[`, character(1), "id")
    missing <- setdiff(m$test[m$file == f], ids)
    expect_equal(missing, character(0), info = f)
  }
})

test_that("F01 C2/C3/C6: tests of dplyr internals and the unruly-class tests are excluded", {
  lib <- f01_lib()
  m <- lib$conformance_load_exclusions(file.path(f01_tools_dir(), "conformance-exclusions.tsv"))
  key <- paste(m$file, m$test, sep = "::")
  item_of <- stats::setNames(m$item, key)

  vdir <- f01_vendored_dir()
  manifest_env <- new.env()
  sys.source(file.path(vdir, "..", "manifest.R"), envir = manifest_env)
  runnable <- manifest_env$runnable_files(vdir)

  expected <- character(0)
  for (f in runnable) {
    for (t in f01_vendored_tests(file.path(vdir, f))) {
      k <- paste(f, t$id, sep = "::")
      if ("count_regroups" %in% t$calls) expected[[k]] <- "C2"
      if (any(c("eval_relocate", "rows_check_unique") %in% t$calls)) expected[[k]] <- "C3"
    }
  }
  expected[["test-filter.R::can filter() with unruly class"]] <- "C6"
  expected[["test-arrange.R::can arrange() with unruly class"]] <- "C6"
  expect_gt(length(expected), 10L)

  for (k in names(expected)) {
    expect_true(k %in% key, info = k)
    if (k %in% key) expect_equal(item_of[[k]], expected[[k]], info = k)
  }
})

# ---------------------------------------------------------------------------
# Output for the merge gate and the before/after comparison

test_that("F01: per-test TSV keeps file, test, status as its first columns", {
  lib <- f01_lib()
  ids <- data.frame(
    file = c("test-a.R", "test-a.R"),
    test = c("plain name", "name with\ttab and\nnewline"),
    status = c("pass", "not_exercised"),
    raw_status = c("pass", "pass"),
    exercised = c(TRUE, FALSE),
    uploads = c(1L, 0L),
    item = c(NA_character_, NA_character_),
    stringsAsFactors = FALSE
  )
  path <- withr::local_tempfile(fileext = ".tsv")
  lib$conformance_write_ids(ids, path)
  back <- utils::read.delim(path, quote = "", comment.char = "", stringsAsFactors = FALSE,
    colClasses = "character")
  expect_equal(names(back)[1:3], c("file", "test", "status"))
  expect_equal(nrow(back), 2L)
  expect_equal(back$status, c("pass", "not_exercised"))
  expect_false(any(grepl("[\t\n\r]", back$test)))
  expect_equal(length(readLines(path)), 3L)
})

test_that("F01: compare reports per-file pass counts and lost passes", {
  lib <- f01_lib()
  old <- data.frame(
    file = c("test-a.R", "test-a.R", "test-a.R", "test-b.R"),
    test = c("t1", "t2", "t3", "u1"),
    status = c("pass", "pass", "fail", "pass"),
    stringsAsFactors = FALSE
  )
  new <- data.frame(
    file = c("test-a.R", "test-a.R", "test-a.R"),
    test = c("t1", "t2", "t3"),
    status = c("pass", "not_exercised", "pass"),
    stringsAsFactors = FALSE
  )
  cmp <- lib$conformance_compare(old, new)

  lost <- cmp$lost_passes[order(cmp$lost_passes$file, cmp$lost_passes$test), ]
  expect_equal(lost$file, c("test-a.R", "test-b.R"))
  expect_equal(lost$test, c("t2", "u1"))
  expect_equal(lost$new_status, c("not_exercised", NA))

  expect_equal(cmp$new_passes$test, "t3")

  pf <- cmp$per_file[order(cmp$per_file$file), ]
  expect_equal(pf$file, c("test-a.R", "test-b.R"))
  expect_equal(as.integer(pf$pass_before), c(2L, 1L))
  expect_equal(as.integer(pf$pass_after), c(2L, 0L))
})
