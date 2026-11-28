# #176: the suite records how long each file takes and runs in parallel shards.
#
# tools/run-tests.R is driven here on small fixture test directories (written
# to a temp dir), so the runner's contract is checked without running the real
# suite: per-test and per-file durations, the slowest-files list, deterministic
# shard assignment from recorded durations, a serial phase that runs after the
# shards, merged results identical to a one-process run, and a shard that dies
# failing the run instead of reporting zero failures. The planning functions are
# in tools/run-tests-lib.R, which has no side effects when sourced.
#
# The child runs use CUPLYR_MEMORY_POOL=0 so each fixture process holds only a
# CUDA context, not a memory pool.

p176_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p176_skip <- function() {
  skip_if_no_gpu()
  root <- p176_root()
  skip_if_not(
    file.exists(file.path(root, "tools", "run-tests.R")) &&
      file.exists(file.path(root, "DESCRIPTION")) &&
      dir.exists(file.path(root, "src")),
    "source tree not available in this test context"
  )
}

p176_lib <- function() {
  path <- file.path(p176_root(), "tools", "run-tests-lib.R")
  if (!file.exists(path)) stop("tools/run-tests-lib.R does not exist", call. = FALSE)
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

# Run `Rscript tools/run-tests.R <args>` from the worktree root.
p176_run <- function(args, env = character()) {
  root <- p176_root()
  # A runner without the sharding library would ignore --dir and run the real
  # suite; fail fast instead.
  p176_lib()
  gpu_free_for_child()
  rscript <- file.path(R.home("bin"), "Rscript")
  out <- withr::with_dir(root, withr::with_envvar(
    c(CUPLYR_MEMORY_POOL = "0", env),
    suppressWarnings(system2(rscript, c("tools/run-tests.R", args),
                             stdout = TRUE, stderr = TRUE))
  ))
  list(status = as.integer(attr(out, "status") %||% 0L), output = as.character(out))
}

p176_read_tsv <- function(path) {
  utils::read.delim(path, sep = "\t", quote = "", stringsAsFactors = FALSE,
                    colClasses = "character", na.strings = character())
}

p176_result_line <- function(output) {
  grep("^RESULT: [0-9]+ pass, [0-9]+ fail, [0-9]+ error, [0-9]+ skip$", output, value = TRUE)
}

p176_result_counts <- function(output) {
  line <- p176_result_line(output)
  if (length(line) != 1L) return(NULL)
  n <- as.integer(regmatches(line, gregexpr("[0-9]+", line))[[1]])
  stats::setNames(n, c("pass", "fail", "error", "skip"))
}

p176_write_durations <- function(path, secs) {
  utils::write.table(data.frame(file = names(secs), seconds = unname(secs)), path,
                     sep = "\t", quote = FALSE, row.names = FALSE)
}

# --- fixtures ----------------------------------------------------------------

# Every fixture file appends its name to $P176_STATE/order when it is sourced,
# and a pid-<x> file with its process id; parallel files leave a done-<x>
# marker when they finish. The serial file records which done markers existed
# when it started.
p176_header <- function(x) {
  c(
    sprintf('cat("%s\\n", file = file.path(Sys.getenv("P176_STATE"), "order"), append = TRUE)', x),
    sprintf('cat(Sys.getpid(), file = file.path(Sys.getenv("P176_STATE"), "pid-%s"))', x)
  )
}
p176_done <- function(x) {
  sprintf('test_that("%s done", { file.create(file.path(Sys.getenv("P176_STATE"), "done-%s")); expect_true(TRUE) })', x, x)
}

# Main fixture: a (2 passes), b (slow pass + skip), c (a failure and an error),
# d (no test_that() blocks at all), e and f (passes), ab-serial (serial list).
p176_main_fixture <- function(dir) {
  w <- function(name, lines) writeLines(lines, file.path(dir, paste0("test-", name, ".R")))
  w("a", c(p176_header("a"),
           'test_that("a one", { expect_true(TRUE) })',
           'test_that("a two", { expect_equal(1, 1) })',
           p176_done("a")))
  w("b", c(p176_header("b"),
           'test_that("b slow", { Sys.sleep(2); expect_true(TRUE) })',
           'test_that("b skipped", { skip("fixture skip") })',
           p176_done("b")))
  w("c", c(p176_header("c"),
           'test_that("c fails", { expect_equal(1, 2) })',
           'test_that("c errors", { stop("fixture boom") })',
           'test_that("c passes", { expect_true(TRUE) })',
           p176_done("c")))
  w("d", c(p176_header("d"),
           '# no test_that() blocks in this file',
           'file.create(file.path(Sys.getenv("P176_STATE"), "done-d"))'))
  w("e", c(p176_header("e"), 'test_that("e one", { expect_true(TRUE) })', p176_done("e")))
  w("f", c(p176_header("f"), 'test_that("f one", { expect_true(TRUE) })', p176_done("f")))
  w("ab-serial", c(
    p176_header("ab-serial"),
    'test_that("ab-serial records what finished before it", {',
    '  st <- Sys.getenv("P176_STATE")',
    '  seen <- sub("^done-", "", list.files(st, pattern = "^done-"))',
    '  writeLines(seen, file.path(st, "serial-saw"))',
    '  expect_true(TRUE)',
    '})'))
  writeLines(c("# fixture serial list", "", "test-ab-serial.R   # runs alone, after the shards"),
             file.path(dir, "serial.txt"))
  # LPT over 3 shards: b (30) -> 1, a (20) -> 2, c (10) -> 3, e (10) -> 3,
  # f (5) -> 2 (tie with 3, lowest index), d (1) -> 3.
  p176_write_durations(file.path(dir, "durations.tsv"),
                       c("test-a.R" = 20, "test-b.R" = 30, "test-c.R" = 10,
                         "test-d.R" = 1, "test-e.R" = 10, "test-f.R" = 5,
                         "test-ab-serial.R" = 50))
  dir
}

p176_main_files <- c("test-a.R", "test-ab-serial.R", "test-b.R", "test-c.R",
                     "test-d.R", "test-e.R", "test-f.R")

# Crash fixture: a (pass), b (pass), c (dies mid-file, how is $P176_DEATH),
# d (pass). LPT over 2 shards: a (10) -> 1, c (4) -> 2, b (3) -> 2, d (2) -> 2,
# so shard 2 runs b, c, d in that order and d never starts.
p176_crash_fixture <- function(dir) {
  w <- function(name, lines) writeLines(lines, file.path(dir, paste0("test-", name, ".R")))
  w("a", c(p176_header("a"), 'test_that("a one", { expect_true(TRUE) })'))
  w("b", c(p176_header("b"), 'test_that("b one", { expect_true(TRUE) })'))
  w("c", c(p176_header("c"),
           'test_that("c dies", {',
           '  cat("P176-MARKER-BEFORE-DEATH\\n", file = stderr())',
           '  if (identical(Sys.getenv("P176_DEATH"), "quit0")) quit(save = "no", status = 0L)',
           '  tools::pskill(Sys.getpid(), tools::SIGKILL)',
           '  Sys.sleep(30)',
           '})',
           'test_that("c after death", { expect_true(TRUE) })'))
  w("d", c(p176_header("d"), 'test_that("d one", { expect_true(TRUE) })'))
  writeLines(character(), file.path(dir, "serial.txt"))
  p176_write_durations(file.path(dir, "durations.tsv"),
                       c("test-a.R" = 10, "test-b.R" = 3, "test-c.R" = 4, "test-d.R" = 2))
  dir
}

# One run of the main fixture per mode, shared by the tests below.
p176_cache <- new.env(parent = emptyenv())

p176_main_run <- function(shards) {
  key <- paste0("main-", shards)
  if (!is.null(p176_cache[[key]])) return(p176_cache[[key]])
  base <- tempfile("cuplyr176-")
  dir.create(base)
  fix <- file.path(base, "tests")
  state <- file.path(base, "state")
  dir.create(fix)
  dir.create(state)
  p176_main_fixture(fix)
  out <- file.path(base, "results.tsv")
  rec <- file.path(base, "recorded.tsv")
  r <- p176_run(c("suite", "--dir", fix, "--shards", shards,
                  "--serial-list", file.path(fix, "serial.txt"),
                  "--durations", file.path(fix, "durations.tsv"),
                  "--record-durations", rec, "--out", out),
                env = c(P176_STATE = state))
  r$out <- out
  r$rec <- rec
  r$state <- state
  p176_cache[[key]] <- r
  r
}

p176_crash_run <- function(death) {
  key <- paste0("crash-", death)
  if (!is.null(p176_cache[[key]])) return(p176_cache[[key]])
  base <- tempfile("cuplyr176c-")
  dir.create(base)
  fix <- file.path(base, "tests")
  state <- file.path(base, "state")
  dir.create(fix)
  dir.create(state)
  p176_crash_fixture(fix)
  out <- file.path(base, "results.tsv")
  r <- p176_run(c("suite", "--dir", fix, "--shards", "2",
                  "--serial-list", file.path(fix, "serial.txt"),
                  "--durations", file.path(fix, "durations.tsv"), "--out", out),
                env = c(P176_STATE = state, P176_DEATH = death))
  r$out <- out
  r$state <- state
  p176_cache[[key]] <- r
  r
}

p176_pid <- function(state, x) {
  f <- file.path(state, paste0("pid-", x))
  if (!file.exists(f)) return(NA_character_)
  readLines(f, warn = FALSE)[[1]]
}

# --- planning (tools/run-tests-lib.R) ------------------------------------------

test_that("#176: suite_plan() assigns files longest first to the least loaded shard", {
  p176_skip()
  lib <- p176_lib()
  files <- c("test-a.R", "test-b.R", "test-c.R", "test-d.R", "test-e.R", "test-f.R",
             "test-ab-serial.R")
  durations <- c("test-a.R" = 20, "test-b.R" = 30, "test-c.R" = 10, "test-d.R" = 1,
                 "test-e.R" = 10, "test-f.R" = 5, "test-ab-serial.R" = 50)
  plan <- lib$suite_plan(files, durations, serial = "test-ab-serial.R", shards = 3L)
  expect_identical(plan$serial, "test-ab-serial.R")
  expect_identical(plan$shards, list(
    "test-b.R",
    c("test-a.R", "test-f.R"),
    c("test-c.R", "test-d.R", "test-e.R")
  ))

  plan2 <- lib$suite_plan(c("test-a.R", "test-b.R", "test-c.R", "test-d.R"),
                          c("test-a.R" = 10, "test-b.R" = 3, "test-c.R" = 4, "test-d.R" = 2),
                          serial = character(), shards = 2L)
  expect_identical(plan2$serial, character())
  expect_identical(plan2$shards, list("test-a.R", c("test-b.R", "test-c.R", "test-d.R")))
})

test_that("#176: suite_plan() is deterministic and covers every file exactly once", {
  p176_skip()
  lib <- p176_lib()
  files <- sprintf("test-%02d.R", 1:23)
  # ties on purpose, and some files without a recorded duration
  durations <- stats::setNames(rep(c(5, 5, 2, 9), length.out = 15), files[1:15])
  serial <- c("test-04.R", "test-20.R")
  p1 <- lib$suite_plan(files, durations, serial = serial, shards = 4L)
  p2 <- lib$suite_plan(rev(files), durations[rev(names(durations))], serial = rev(serial),
                       shards = 4L)
  expect_identical(p1, p2)
  expect_identical(p1$serial, sort(serial, method = "radix"))
  expect_length(p1$shards, 4L)
  expect_true(all(lengths(p1$shards) > 0L))
  ran <- c(unlist(p1$shards), p1$serial)
  expect_identical(sort(ran), sort(files))
  expect_false(any(duplicated(ran)))
  # each shard runs its files in name order (byte order, whatever the locale)
  for (s in p1$shards) expect_identical(s, sort(s, method = "radix"))
})

test_that("#176: suite_plan() never makes empty shards", {
  p176_skip()
  lib <- p176_lib()
  plan <- lib$suite_plan(c("test-x.R", "test-y.R", "test-z.R"), numeric(),
                         serial = "test-z.R", shards = 6L)
  expect_length(plan$shards, 2L)
  expect_identical(sort(unlist(plan$shards)), c("test-x.R", "test-y.R"))
  expect_identical(plan$serial, "test-z.R")

  only_serial <- lib$suite_plan("test-z.R", numeric(), serial = "test-z.R", shards = 3L)
  expect_length(only_serial$shards, 0L)
  expect_identical(only_serial$serial, "test-z.R")
})

test_that("#176: the serial list and durations files are read with comments and blanks ignored", {
  p176_skip()
  lib <- p176_lib()
  f <- withr::local_tempfile(fileext = ".txt")
  writeLines(c("# memory and timing files", "", "  test-x.R  ", "test-y.R # why", "#test-z.R"), f)
  expect_identical(lib$read_serial_list(f), c("test-x.R", "test-y.R"))
  expect_identical(lib$read_serial_list(tempfile()), character())

  d <- withr::local_tempfile(fileext = ".tsv")
  p176_write_durations(d, c("test-x.R" = 1.5, "test-y.R" = 20))
  expect_equal(lib$read_durations(d), c("test-x.R" = 1.5, "test-y.R" = 20))
  expect_identical(lib$read_durations(tempfile()), stats::setNames(numeric(), character()))
})

test_that("#176: default_shards() reads CUPLYR_TEST_SHARDS and stays within a third of the cores", {
  p176_skip()
  lib <- p176_lib()
  cap <- max(1L, parallel::detectCores() %/% 3L)
  n <- withr::with_envvar(c(CUPLYR_TEST_SHARDS = NA), lib$default_shards())
  expect_true(is.integer(n) && length(n) == 1L)
  expect_gte(n, 1L)
  expect_lte(n, cap)
  expect_identical(withr::with_envvar(c(CUPLYR_TEST_SHARDS = "2"), lib$default_shards()), 2L)
  expect_error(withr::with_envvar(c(CUPLYR_TEST_SHARDS = "abc"), lib$default_shards()),
               "CUPLYR_TEST_SHARDS")
  expect_error(withr::with_envvar(c(CUPLYR_TEST_SHARDS = "0"), lib$default_shards()),
               "CUPLYR_TEST_SHARDS")
})

test_that("#176: the real suite's serial list exists and covers the memory files", {
  p176_skip()
  lib <- p176_lib()
  root <- p176_root()
  serial <- lib$read_serial_list(file.path(root, "tools", "test-serial.txt"))
  tests <- list.files(file.path(root, "tests", "testthat"), pattern = "^test-.*\\.R$")
  expect_true(length(serial) > 0L)
  expect_identical(setdiff(serial, tests), character())
  expect_true(all(c("test-memory-regression.R", "test-gpu-memory.R") %in% serial))

  durations <- lib$read_durations(file.path(root, "tools", "test-durations.tsv"))
  expect_true(length(durations) > 0L)
  expect_true(all(is.finite(durations) & durations >= 0))

  plan <- lib$suite_plan(tests, durations, serial = serial, shards = lib$default_shards())
  ran <- c(unlist(plan$shards), plan$serial)
  expect_identical(sort(ran), sort(tests))
  expect_false(any(duplicated(ran)))
  expect_identical(plan, lib$suite_plan(rev(tests), durations, serial = serial,
                                        shards = lib$default_shards()))
})

# --- a sharded run of the main fixture -----------------------------------------

test_that("#176: a sharded run keeps the results format, the RESULT line and the exit status", {
  p176_skip()
  r <- p176_main_run(3L)
  expect_identical(r$status, 1L, info = paste(r$output, collapse = "\n"))
  expect_identical(p176_result_counts(r$output), c(pass = 12L, fail = 1L, error = 1L, skip = 1L),
                   info = paste(r$output, collapse = "\n"))
  expect_true(any(grepl("^FAILING:", r$output)))
  expect_true(any(grepl("[fail] test-c.R :: c fails", r$output, fixed = TRUE)))
  expect_true(any(grepl("[error] test-c.R :: c errors", r$output, fixed = TRUE)))

  res <- p176_read_tsv(r$out)
  expect_identical(names(res)[1:3], c("file", "test", "status"))
  expect_true(all(c("seconds", "file_seconds", "shard") %in% names(res)))
  expect_identical(sort(unique(res$file), method = "radix"), setdiff(p176_main_files, "test-d.R"))
})

test_that("#176: merged sharded results equal a one-process run, row for row", {
  p176_skip()
  serial <- p176_main_run(1L)
  sharded <- p176_main_run(3L)
  a <- p176_read_tsv(serial$out)[, c("file", "test", "status")]
  b <- p176_read_tsv(sharded$out)[, c("file", "test", "status")]
  expect_identical(b, a)
  expect_identical(p176_result_line(sharded$output), p176_result_line(serial$output))
  expect_identical(sharded$status, serial$status)
})

test_that("#176: --shards 1 runs every file in one process, in name order", {
  p176_skip()
  r <- p176_main_run(1L)
  order <- readLines(file.path(r$state, "order"))
  expect_identical(order, sub("^test-(.*)\\.R$", "\\1", p176_main_files))
  pids <- vapply(c("a", "ab-serial", "b", "c", "d", "e", "f"),
                 function(x) p176_pid(r$state, x), character(1))
  expect_false(anyNA(pids))
  expect_length(unique(pids), 1L)
})

test_that("#176: each shard runs the files its plan gives it, in separate processes", {
  p176_skip()
  r <- p176_main_run(3L)
  pid <- function(x) p176_pid(r$state, x)
  expect_false(anyNA(c(pid("a"), pid("b"), pid("c"), pid("e"), pid("f"))))
  expect_identical(pid("a"), pid("f"))
  expect_identical(pid("c"), pid("e"))
  expect_identical(pid("c"), pid("d"))
  expect_length(unique(c(pid("a"), pid("b"), pid("c"))), 3L)

  res <- p176_read_tsv(r$out)
  shard_of <- function(f) unique(res$shard[res$file == f])
  expect_length(shard_of("test-a.R"), 1L)
  expect_identical(shard_of("test-a.R"), shard_of("test-f.R"))
  expect_identical(shard_of("test-c.R"), shard_of("test-e.R"))
  expect_false(identical(shard_of("test-a.R"), shard_of("test-b.R")))
  expect_identical(shard_of("test-ab-serial.R"), "serial")
})

test_that("#176: the serial phase starts only after every shard has finished", {
  p176_skip()
  r <- p176_main_run(3L)
  saw <- readLines(file.path(r$state, "serial-saw"))
  expect_setequal(saw, c("a", "b", "c", "d", "e", "f"))
  expect_false(p176_pid(r$state, "ab-serial") %in%
                 vapply(c("a", "b", "c"), function(x) p176_pid(r$state, x), character(1)))
})

test_that("#176: a run records per-test and per-file durations and lists the slowest files", {
  p176_skip()
  r <- p176_main_run(3L)
  res <- p176_read_tsv(r$out)
  secs <- suppressWarnings(as.numeric(res$seconds))
  fsecs <- suppressWarnings(as.numeric(res$file_seconds))
  expect_false(anyNA(secs))
  expect_false(anyNA(fsecs))
  expect_true(all(secs >= 0 & fsecs >= 0))
  expect_gte(secs[res$file == "test-b.R" & res$test == "b slow"], 1.5)
  expect_gte(unique(fsecs[res$file == "test-b.R"]), 1.5)
  # one file duration per file, at least as long as its tests together
  for (f in unique(res$file)) {
    expect_length(unique(fsecs[res$file == f]), 1L)
    expect_gte(unique(fsecs[res$file == f]) + 0.01, sum(secs[res$file == f]))
  }

  i <- grep("^SLOWEST", r$output)
  expect_length(i, 1L)
  listed <- grep("test-[A-Za-z0-9-]+\\.R", r$output[seq.int(i + 1L, length(r$output))],
                 value = TRUE)
  expect_true(length(listed) > 0L)
  expect_match(listed[[1]], "test-b.R", fixed = TRUE)
})

test_that("#176: --record-durations writes every file run, including one without tests", {
  p176_skip()
  lib <- p176_lib()
  r <- p176_main_run(3L)
  rec <- lib$read_durations(r$rec)
  expect_setequal(names(rec), p176_main_files)
  expect_gte(rec[["test-b.R"]], 1.5)
  expect_true(all(is.finite(rec) & rec >= 0))
})

test_that("#176: a suite run reports peak GPU and host memory", {
  p176_skip()
  skip_if_not(nzchar(Sys.which("nvidia-smi")), "nvidia-smi not available")
  r <- p176_main_run(3L)
  peak <- grep("^PEAK:", r$output, value = TRUE)
  expect_length(peak, 1L)
  expect_match(peak, "gpu", ignore.case = TRUE)
  expect_match(peak, "host", ignore.case = TRUE)
})

# --- a shard that dies -----------------------------------------------------------

p176_expect_crash_reported <- function(r) {
  log <- paste(r$output, collapse = "\n")
  expect_identical(r$status, 1L, info = log)
  counts <- p176_result_counts(r$output)
  expect_false(is.null(counts), info = log)
  expect_gte(counts[["error"]], 1L)
  expect_true(any(grepl("P176-MARKER-BEFORE-DEATH", r$output, fixed = TRUE)), info = log)

  expect_true(file.exists(r$out), info = log)
  res <- p176_read_tsv(r$out)
  expect_true(all(c("test-a.R", "test-b.R", "test-c.R", "test-d.R") %in% res$file), info = log)
  expect_true(any(res$file == "test-c.R" & res$status == "error"), info = log)
  expect_true(all(res$status[res$file == "test-a.R"] == "pass"))
  # the file the shard finished before it died keeps its result
  expect_true(all(res$status[res$file == "test-b.R"] == "pass"))
  expect_true(any(grepl("[error] test-c.R", r$output, fixed = TRUE)), info = log)
}

test_that("#176: a shard killed mid-file fails the run and keeps its output", {
  p176_skip()
  p176_expect_crash_reported(p176_crash_run("kill"))
})

test_that("#176: a shard that exits 0 before finishing its files fails the run", {
  p176_skip()
  p176_expect_crash_reported(p176_crash_run("quit0"))
})

# --- file mode -------------------------------------------------------------------

test_that("#176: tools/test-file mode records durations in the same columns", {
  p176_skip()
  base <- withr::local_tempdir("cuplyr176f-")
  dir.create(file.path(base, "tests"))
  dir.create(file.path(base, "state"))
  p176_main_fixture(file.path(base, "tests"))
  out <- file.path(base, "file.tsv")
  r <- p176_run(c("file", file.path(base, "tests", "test-a.R"), "--out", out),
                env = c(P176_STATE = file.path(base, "state")))
  expect_identical(r$status, 0L, info = paste(r$output, collapse = "\n"))
  expect_identical(p176_result_counts(r$output), c(pass = 3L, fail = 0L, error = 0L, skip = 0L))
  res <- p176_read_tsv(out)
  expect_identical(names(res)[1:3], c("file", "test", "status"))
  expect_true(all(c("seconds", "file_seconds") %in% names(res)))
  expect_identical(res$test, c("a one", "a two", "a done"))
})
