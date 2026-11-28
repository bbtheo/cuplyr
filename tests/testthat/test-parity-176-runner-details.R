# #176, details of tools/run-tests.R that the acceptance file does not pin:
# strict option parsing, shard planning with zero durations, serial files in
# processes of their own, progress lines while the shards run, and durations
# recorded from a run with a dead shard. Fixture test directories are written
# to a temp dir; no run rebuilds the package library (see the acceptance file).

rd176_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

rd176_skip <- function() {
  skip_if_no_gpu()
  root <- rd176_root()
  skip_if_not(
    file.exists(file.path(root, "tools", "run-tests.R")) &&
      file.exists(file.path(root, "DESCRIPTION")) &&
      dir.exists(file.path(root, "src")),
    "source tree not available in this test context"
  )
}

rd176_lib <- function() {
  e <- new.env(parent = globalenv())
  sys.source(file.path(rd176_root(), "tools", "run-tests-lib.R"), envir = e)
  e
}

rd176_run <- function(args, env = character()) {
  gpu_free_for_child()
  vars <- c(CUPLYR_MEMORY_POOL = "0")
  vars[names(env)] <- env
  out <- withr::with_dir(rd176_root(), withr::with_envvar(
    vars,
    suppressWarnings(system2(file.path(R.home("bin"), "Rscript"),
                             c("tools/run-tests.R", args), stdout = TRUE, stderr = TRUE))
  ))
  list(status = as.integer(attr(out, "status") %||% 0L), output = as.character(out))
}

rd176_read_tsv <- function(path) {
  utils::read.delim(path, sep = "\t", quote = "", stringsAsFactors = FALSE,
                    colClasses = "character", na.strings = character())
}

test_that("#176: a shard count is a positive whole number in digits", {
  rd176_skip()
  lib <- rd176_lib()
  expect_identical(lib$check_shard_count("3", "--shards"), 3L)
  expect_identical(lib$check_shard_count(" 12 ", "--shards"), 12L)
  for (bad in c("2.5", "1e1", "0", "-1", "", "abc", "3 4", "99999999999")) {
    expect_error(lib$check_shard_count(bad, "--shards"), "--shards must be a positive whole number",
                 info = bad)
  }
  expect_error(withr::with_envvar(c(CUPLYR_TEST_SHARDS = "2.5"), lib$default_shards()),
               "CUPLYR_TEST_SHARDS must be a positive whole number")
})

test_that("#176: the default shard count is 3, or a third of the cores when that is fewer", {
  rd176_skip()
  lib <- rd176_lib()
  cores <- parallel::detectCores()
  want <- min(3L, max(1L, cores %/% 3L))
  expect_identical(withr::with_envvar(c(CUPLYR_TEST_SHARDS = NA), lib$default_shards()), want)
  # an explicit count is used as given, even above a third of the cores
  expect_identical(withr::with_envvar(c(CUPLYR_TEST_SHARDS = "40"), lib$default_shards()), 40L)
})

test_that("#176: suite_plan() spreads files whose recorded durations are all zero", {
  rd176_skip()
  lib <- rd176_lib()
  files <- sprintf("test-%s.R", letters[1:7])
  zero <- stats::setNames(numeric(length(files)), files)
  plan <- lib$suite_plan(files, zero, serial = character(), shards = 3L)
  expect_length(plan$shards, 3L)
  expect_true(all(lengths(plan$shards) > 0L))
  expect_identical(sort(unlist(plan$shards)), files)
  # one file with time and the rest zero: the zero files still fill the other shards
  one <- zero
  one[["test-a.R"]] <- 50
  plan1 <- lib$suite_plan(files, one, serial = character(), shards = 3L)
  expect_true(all(lengths(plan1$shards) > 0L))
  expect_identical(plan1$shards[[1]], "test-a.R")
})

test_that("#176: an unknown option or a malformed --shards fails before anything runs", {
  rd176_skip()
  r <- rd176_run(c("suite", "--shard", "2"))
  expect_false(identical(r$status, 0L))
  expect_true(any(grepl("unknown option --shard", r$output, fixed = TRUE)), info = paste(r$output, collapse = "\n"))

  r <- rd176_run(c("file", "tests/testthat/test-none.R", "--bogus"))
  expect_false(identical(r$status, 0L))
  expect_true(any(grepl("unknown option --bogus", r$output, fixed = TRUE)), info = paste(r$output, collapse = "\n"))

  for (bad in c("2.5", "1e1", "0", "two")) {
    r <- rd176_run(c("suite", "--shards", bad))
    expect_false(identical(r$status, 0L), info = bad)
    expect_true(any(grepl("--shards must be a positive whole number", r$output, fixed = TRUE)),
                info = paste(bad, paste(r$output, collapse = "\n")))
  }
})

# Two serial files record their process ids; p and q are parallel files that
# take a moment, so the progress lines of both shards are in the output.
rd176_fixture <- function(dir) {
  w <- function(name, lines) writeLines(lines, file.path(dir, paste0("test-", name, ".R")))
  pidline <- function(x) sprintf('cat(Sys.getpid(), file = file.path(Sys.getenv("P176_STATE"), "pid-%s"))', x)
  w("p", c(pidline("p"), 'test_that("p", { Sys.sleep(1); expect_true(TRUE) })'))
  w("q", c(pidline("q"), 'test_that("q", { Sys.sleep(1); expect_true(TRUE) })'))
  w("s1", c(pidline("s1"), 'test_that("s1", { expect_true(TRUE) })'))
  w("s2", c(pidline("s2"), 'test_that("s2", { expect_true(TRUE) })'))
  writeLines(c("test-s1.R", "test-s2.R"), file.path(dir, "serial.txt"))
  dir
}

test_that("#176: each serial file runs in a process of its own, and progress lines stream", {
  rd176_skip()
  base <- withr::local_tempdir("cuplyr176d-")
  fix <- file.path(base, "tests")
  state <- file.path(base, "state")
  dir.create(fix)
  dir.create(state)
  rd176_fixture(fix)
  out <- file.path(base, "r.tsv")
  r <- rd176_run(c("suite", "--dir", fix, "--shards", "2",
                   "--serial-list", file.path(fix, "serial.txt"),
                   "--durations", file.path(base, "none.tsv"), "--out", out),
                 env = c(P176_STATE = state))
  log <- paste(r$output, collapse = "\n")
  expect_identical(r$status, 0L, info = log)
  pid <- function(x) readLines(file.path(state, paste0("pid-", x)))
  expect_false(pid("s1") == pid("s2"))
  expect_false(pid("p") == pid("q"))
  expect_false(pid("s1") %in% c(pid("p"), pid("q")))

  expect_true(any(grepl("^\\[1\\] started test-p\\.R$", r$output)), info = log)
  expect_true(any(grepl("^\\[2\\] started test-q\\.R$", r$output)), info = log)
  expect_true(any(grepl("^\\[serial\\] started test-s1\\.R$", r$output)), info = log)
  expect_true(any(grepl("^\\[serial\\] finished test-s2\\.R", r$output)), info = log)
  # the progress lines come before the first child's full log
  first_log <- grep("^===== ", r$output)[[1]]
  expect_lt(grep("^\\[1\\] started test-p\\.R$", r$output)[[1]], first_log)

  res <- rd176_read_tsv(out)
  expect_identical(res$shard[res$file %in% c("test-s1.R", "test-s2.R")], c("serial", "serial"))

  # one PEAK line with separate figures per phase
  peak <- grep("^PEAK:", r$output, value = TRUE)
  expect_length(peak, 1L)
  expect_match(peak, "before the first child", fixed = TRUE)
  expect_match(peak, "parallel phase", fixed = TRUE)
  expect_match(peak, "serial phase", fixed = TRUE)
})

test_that("#176: --record-durations leaves out files a dead shard never finished", {
  rd176_skip()
  base <- withr::local_tempdir("cuplyr176r-")
  fix <- file.path(base, "tests")
  state <- file.path(base, "state")
  dir.create(fix)
  dir.create(state)
  w <- function(name, lines) writeLines(lines, file.path(fix, paste0("test-", name, ".R")))
  w("a", 'test_that("a", { expect_true(TRUE) })')
  w("b", c('test_that("b dies", { tools::pskill(Sys.getpid(), tools::SIGKILL); Sys.sleep(30) })'))
  w("c", 'test_that("c", { expect_true(TRUE) })')
  # both on one shard, b then c: a finishes on the other shard
  utils::write.table(data.frame(file = c("test-a.R", "test-b.R", "test-c.R"), seconds = c(10, 4, 3)),
                     file.path(base, "durations.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  rec <- file.path(base, "rec.tsv")
  r <- rd176_run(c("suite", "--dir", fix, "--shards", "2",
                   "--serial-list", file.path(base, "none.txt"),
                   "--durations", file.path(base, "durations.tsv"),
                   "--record-durations", rec, "--out", file.path(base, "r.tsv")),
                 env = c(P176_STATE = state))
  expect_identical(r$status, 1L, info = paste(r$output, collapse = "\n"))
  recorded <- rd176_lib()$read_durations(rec)
  expect_identical(names(recorded), "test-a.R")
})
