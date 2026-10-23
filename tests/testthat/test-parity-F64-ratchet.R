# F64 acceptance tests (PM-owned): the adaptive perf baseline.
#
# tools/merge-queue ratchets scratchpad/perf_baseline.json down after a fully
# passing gate. The rules live in tools/perf-ratchet-lib.R; merge-queue calls
# them through the tools/perf-ratchet wrapper. Everything here works on temp
# files only; the real baseline is never read or written. No GPU.
#
# Rule under test (see the F64 design): an observation is the gate's
# calibrated per-benchmark median (observed_s). Each entry keeps
# last_observed_s from the previous passing gate; the confirmed value is
# max(last_observed_s, observed_s) and median_s <- min(median_s, confirmed).
# Entries without last_observed_s are only seeded. New ids enter with their
# first observation. Calibration entries are never ratcheted.

f64_tools_dir <- function() {
  d <- normalizePath(testthat::test_path("..", "..", "tools"), mustWork = FALSE)
  if (!dir.exists(d)) {
    testthat::skip("tools/ is not present (build-ignored); ratchet tests need a checkout")
  }
  d
}

f64_lib <- local({
  cache <- NULL
  function() {
    tools_dir <- f64_tools_dir()
    if (is.null(cache)) {
      env <- new.env(parent = globalenv())
      sys.source(file.path(tools_dir, "perf-ratchet-lib.R"), envir = env)
      cache <<- env
    }
    cache
  }
})

f64_entry <- function(median_s, last = NULL, commit = "377eda7",
                      recorded_at = "2026-07-30T20:15:59+0000", iterations = 5) {
  e <- list(median_s = median_s, recorded_at = recorded_at, commit = commit, iterations = iterations)
  if (!is.null(last)) e$last_observed_s <- last
  e
}

f64_obs <- function(..., iterations = 5L) {
  x <- c(...)
  data.frame(
    id = names(x), calibration = "kernel", current_s = unname(x), raw_scale = 1, scale = 1,
    observed_s = unname(x), iterations = as.integer(iterations), reps = 3L,
    stringsAsFactors = FALSE
  )
}

f64_apply <- function(baseline, obs) {
  f64_lib()$perf_ratchet_apply(baseline, obs, commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000")
}

f64_passing_summary <- c(
  "suite: PASS",
  "RESULT: 4038 pass, 0 fail, 0 error, 25 skip",
  "perf: PASS",
  "conformance: PASS (1067 passing, +0 new, 0 regressions, 0 reclassified, 0 accepted known failures)",
  "conformance: feature F64, harness unchanged vs dev"
)

f64_write_json <- function(path, data) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  txt <- jsonlite::toJSON(data, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
  writeLines(as.character(txt), path)
  invisible(path)
}

f64_write_obs <- function(path, obs) {
  utils::write.table(obs, path, sep = "\t", quote = FALSE, row.names = FALSE)
  invisible(path)
}

f64_md5 <- function(path) unname(tools::md5sum(path))

# A temp workspace: <root>/base/baseline.json, <root>/gate/{obs,summary}.
f64_workspace <- function(baseline, obs, summary = f64_passing_summary, env = parent.frame()) {
  root <- withr::local_tempdir(.local_envir = env)
  ws <- list(
    baseline = file.path(root, "base", "baseline.json"),
    obs = file.path(root, "gate", "perf_observations.tsv"),
    summary = file.path(root, "gate", "summary.txt"),
    out = file.path(root, "gate")
  )
  dir.create(ws$out, recursive = TRUE)
  f64_write_json(ws$baseline, baseline)
  if (!is.null(obs)) f64_write_obs(ws$obs, obs)
  writeLines(summary, ws$summary)
  ws
}

# ---------------------------------------------------------------------------
# perf_ratchet_apply(): the rules
# ---------------------------------------------------------------------------

test_that("F64 ratchet: a value confirmed by two consecutive gates lowers the entry", {
  b <- list(filter_chain_eager = f64_entry(0.010, last = 0.009))
  r <- f64_apply(b, f64_obs(filter_chain_eager = 0.0085))
  e <- r$baseline$filter_chain_eager
  expect_equal(e$median_s, 0.009)
  expect_equal(e$last_observed_s, 0.0085)
  expect_identical(e$commit, "abc1234")
  expect_identical(e$recorded_at, "2026-10-23T12:00:00+0000")
  expect_equal(nrow(r$changes), 1)
  expect_identical(r$changes$id, "filter_chain_eager")
  expect_equal(r$changes$old_s, 0.010)
  expect_equal(r$changes$new_s, 0.009)
  expect_identical(r$changes$kind, "lowered")
})

test_that("F64 ratchet: a single lucky gate does not drag the baseline down", {
  b <- list(filter_chain_eager = f64_entry(0.010, last = 0.0095))
  r1 <- f64_apply(b, f64_obs(filter_chain_eager = 0.0070))
  expect_equal(r1$baseline$filter_chain_eager$median_s, 0.0095)
  expect_equal(r1$baseline$filter_chain_eager$last_observed_s, 0.0070)
  # Next gate reads normal again: the confirmed value is max(0.0070, 0.0093).
  r2 <- f64_apply(r1$baseline, f64_obs(filter_chain_eager = 0.0093))
  expect_equal(r2$baseline$filter_chain_eager$median_s, 0.0093)
  expect_equal(r2$baseline$filter_chain_eager$last_observed_s, 0.0093)
  expect_identical(r2$changes$kind, "lowered")
})

test_that("F64 ratchet: an entry without last_observed_s is only seeded", {
  b <- list(filter_chain_eager = f64_entry(0.010))
  r <- f64_apply(b, f64_obs(filter_chain_eager = 0.005))
  e <- r$baseline$filter_chain_eager
  expect_equal(e$median_s, 0.010)
  expect_equal(e$last_observed_s, 0.005)
  expect_identical(e$commit, "377eda7")
  expect_identical(e$recorded_at, "2026-07-30T20:15:59+0000")
  expect_equal(nrow(r$changes), 0)
  # ... and lowers at the next passing gate.
  r2 <- f64_apply(r$baseline, f64_obs(filter_chain_eager = 0.006))
  expect_equal(r2$baseline$filter_chain_eager$median_s, 0.006)
})

test_that("F64 ratchet: slower or equal observations never raise or rewrite an entry", {
  b <- list(
    slower = f64_entry(0.010, last = 0.012),
    equal = f64_entry(0.010, last = 0.010)
  )
  r <- f64_apply(b, f64_obs(slower = 0.011, equal = 0.010))
  expect_equal(r$baseline$slower$median_s, 0.010)
  expect_equal(r$baseline$slower$last_observed_s, 0.011)
  expect_identical(r$baseline$slower$commit, "377eda7")
  expect_equal(r$baseline$equal$median_s, 0.010)
  expect_identical(r$baseline$equal$recorded_at, "2026-07-30T20:15:59+0000")
  expect_equal(nrow(r$changes), 0)
})

test_that("F64 ratchet: ids missing from the baseline enter with their first observation", {
  b <- list(filter_chain_eager = f64_entry(0.010, last = 0.010))
  r <- f64_apply(b, f64_obs(filter_chain_eager = 0.011, int_arith_eager = 0.0123, iterations = 7L))
  expect_identical(names(r$baseline), c("filter_chain_eager", "int_arith_eager"))
  e <- r$baseline$int_arith_eager
  expect_equal(e$median_s, 0.0123)
  expect_equal(e$last_observed_s, 0.0123)
  expect_identical(e$commit, "abc1234")
  expect_identical(e$recorded_at, "2026-10-23T12:00:00+0000")
  expect_equal(as.numeric(e$iterations), 7)
  expect_equal(nrow(r$changes), 1)
  expect_identical(r$changes$id, "int_arith_eager")
  expect_true(is.na(r$changes$old_s))
  expect_equal(r$changes$new_s, 0.0123)
  expect_identical(r$changes$kind, "new")
})

test_that("F64 ratchet: unobserved entries and calibration entries are left exactly as they were", {
  b <- list(
    `_calibration` = f64_entry(0.065),
    `_calibration_transfer` = f64_entry(0.019),
    untouched = f64_entry(0.5, last = 0.49),
    a = f64_entry(0.010, last = 0.009)
  )
  obs <- f64_obs(`_calibration` = 0.040, `_calibration_transfer` = 0.010, a = 0.009)
  r <- f64_apply(b, obs)
  expect_identical(r$baseline[["_calibration"]], b[["_calibration"]])
  expect_identical(r$baseline[["_calibration_transfer"]], b[["_calibration_transfer"]])
  expect_identical(r$baseline$untouched, b$untouched)
  expect_identical(names(r$baseline), names(b))
  expect_false(any(startsWith(r$changes$id, "_calibration")))
  expect_equal(r$baseline$a$median_s, 0.009)
})

test_that("F64 ratchet: missing, infinite or non-positive observations are ignored", {
  b <- list(a = f64_entry(0.010, last = 0.009), b = f64_entry(0.010, last = 0.009))
  obs <- f64_obs(a = NA_real_, b = Inf, c = 0, d = -1)
  r <- f64_apply(b, obs)
  expect_identical(r$baseline, b)
  expect_equal(nrow(r$changes), 0)
})

test_that("F64 ratchet: duplicate ids in the observations are an error", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  expect_error(f64_apply(b, f64_obs(a = 0.008, a = 0.007)), "duplicate", ignore.case = TRUE)
})

# ---------------------------------------------------------------------------
# perf_ratchet_gate_passed()
# ---------------------------------------------------------------------------

test_that("F64 ratchet: only a fully passing gate summary counts as passed", {
  passed <- f64_lib()$perf_ratchet_gate_passed
  expect_true(passed(f64_passing_summary))
  expect_true(passed(c(f64_passing_summary,
    "conformance: NOTE: candidate adds known-failure row test-a.R :: t1 (owner F64)")))

  swap <- function(from, to) sub(from, to, f64_passing_summary, fixed = TRUE)
  expect_false(passed(swap("suite: PASS", "suite: FAIL")))
  expect_false(passed(swap("perf: PASS", "perf: FAIL")))
  expect_false(passed(swap("conformance: PASS (", "conformance: FAIL (")))
  expect_false(passed(c(f64_passing_summary, "conformance: OWNER FAIL: test-a.R :: t1")))
  expect_false(passed(f64_passing_summary[!startsWith(f64_passing_summary, "perf:")]))
  expect_false(passed(f64_passing_summary[!startsWith(f64_passing_summary, "suite:")]))
  expect_false(passed(f64_passing_summary[!startsWith(f64_passing_summary, "conformance:")]))
  expect_false(passed(character()))
})

# ---------------------------------------------------------------------------
# perf_ratchet_run(): the merge-queue step on temp files
# ---------------------------------------------------------------------------

test_that("F64 ratchet: a passing gate writes the baseline, perf_ratchet.txt and the summary lines", {
  b <- list(
    `_calibration` = f64_entry(0.065),
    filter_chain_eager = f64_entry(0.009, last = 0.0088),
    join_left_eager = f64_entry(0.32, last = 0.33)
  )
  obs <- f64_obs(filter_chain_eager = 0.0087, join_left_eager = 0.35, int_arith_eager = 0.0123456789012345)
  ws <- f64_workspace(b, obs)
  changes <- f64_lib()$perf_ratchet_run(ws$baseline, ws$obs, ws$summary, ws$out,
    commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000")
  expect_s3_class(changes, "data.frame")
  expect_setequal(changes$id, c("filter_chain_eager", "int_arith_eager"))

  # Readable by the perf harness itself, at full precision.
  got <- perf_read_baseline(ws$baseline)
  expect_equal(as.numeric(got$filter_chain_eager$median_s), 0.0088)
  expect_identical(got$filter_chain_eager$commit, "abc1234")
  expect_equal(as.numeric(got$int_arith_eager$median_s), 0.0123456789012345, tolerance = 1e-14)
  expect_equal(as.numeric(got$join_left_eager$median_s), 0.32)
  expect_equal(as.numeric(got$join_left_eager$last_observed_s), 0.35)
  expect_identical(got[["_calibration"]]$commit, "377eda7")
  expect_equal(as.numeric(got[["_calibration"]]$median_s), 0.065)

  summ <- readLines(ws$summary)
  expect_identical(summ[seq_along(f64_passing_summary)], f64_passing_summary)
  expect_true("perf baseline: filter_chain_eager 0.009 -> 0.0088" %in% summ)
  expect_true("perf baseline: int_arith_eager new -> 0.0123457" %in% summ)
  expect_false(any(grepl("join_left_eager", summ)))

  log <- readLines(file.path(ws$out, "perf_ratchet.txt"))
  expect_true("filter_chain_eager 0.009 -> 0.0088" %in% log)
  expect_true("int_arith_eager new -> 0.0123457" %in% log)

  # Atomic write left nothing behind next to the baseline.
  expect_identical(list.files(dirname(ws$baseline), all.files = TRUE, no.. = TRUE), "baseline.json")
})

test_that("F64 ratchet: a passing gate with nothing to lower says so", {
  b <- list(a = f64_entry(0.010, last = 0.012))
  ws <- f64_workspace(b, f64_obs(a = 0.011))
  f64_lib()$perf_ratchet_run(ws$baseline, ws$obs, ws$summary, ws$out,
    commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000")
  expect_true("perf baseline: unchanged" %in% readLines(ws$summary))
  expect_equal(as.numeric(perf_read_baseline(ws$baseline)$a$median_s), 0.010)
})

test_that("F64 ratchet: a failed gate never touches the baseline", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  for (bad in c("suite: FAIL", "perf: FAIL")) {
    summary <- sub(paste0(sub(":.*", "", bad), ": PASS"), bad, f64_passing_summary, fixed = TRUE)
    ws <- f64_workspace(b, f64_obs(a = 0.005), summary = summary)
    before <- f64_md5(ws$baseline)
    mtime <- file.mtime(ws$baseline)
    res <- f64_lib()$perf_ratchet_run(ws$baseline, ws$obs, ws$summary, ws$out,
      commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000")
    expect_null(res)
    expect_identical(f64_md5(ws$baseline), before)
    expect_identical(file.mtime(ws$baseline), mtime)
  }
})

test_that("F64 ratchet: a passing gate without an observations file is an error and changes nothing", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  ws <- f64_workspace(b, obs = NULL)
  before <- f64_md5(ws$baseline)
  expect_error(f64_lib()$perf_ratchet_run(ws$baseline, ws$obs, ws$summary, ws$out,
    commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000"))
  expect_identical(f64_md5(ws$baseline), before)
})

test_that("F64 ratchet: only the part status lines decide; FAIL inside a test name does not", {
  passed <- f64_lib()$perf_ratchet_gate_passed
  expect_true(passed(c(f64_passing_summary,
    "conformance: NOTE: candidate adds exclusion test-a.R :: FAILS on empty input (r)",
    "conformance: WARNING: test-a.R :: errors FAIL loudly passes on F64's merge")))
  expect_false(passed(c(f64_passing_summary, "conformance: OWNER FAIL: test-a.R :: t1")))
})

test_that("F64 ratchet: an error while logging leaves the baseline untouched", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  ws <- f64_workspace(b, f64_obs(a = 0.0085))
  before <- f64_md5(ws$baseline)
  missing_out <- file.path(ws$out, "no-such-dir")
  expect_error(f64_lib()$perf_ratchet_run(ws$baseline, ws$obs, ws$summary, missing_out,
    commit = "abc1234", recorded_at = "2026-10-23T12:00:00+0000"))
  expect_identical(f64_md5(ws$baseline), before)
})

# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

test_that("F64 ratchet: the baseline write is atomic and round-trips through the perf harness", {
  dir <- withr::local_tempdir()
  path <- file.path(dir, "baseline.json")
  data <- list(
    `_calibration` = f64_entry(0.065),
    summarise_grouped_eager = f64_entry(0.0016789123456789, last = 0.0017)
  )
  f64_lib()$perf_ratchet_write_baseline(path, data)
  expect_identical(list.files(dir, all.files = TRUE, no.. = TRUE), "baseline.json")
  got <- perf_read_baseline(path)
  expect_equal(as.numeric(got$summarise_grouped_eager$median_s), 0.0016789123456789, tolerance = 1e-14)
  expect_equal(f64_lib()$perf_ratchet_read_baseline(path), got)

  # A directory we cannot write into: the write fails and the old file survives.
  before <- f64_md5(path)
  Sys.chmod(dir, "0555")
  withr::defer(Sys.chmod(dir, "0755"))
  skip_if(file.access(dir, 2) == 0, "running as a user who can write a read-only directory")
  expect_error(f64_lib()$perf_ratchet_write_baseline(path, list(a = f64_entry(1))))
  expect_identical(f64_md5(path), before)
  expect_identical(list.files(dir, all.files = TRUE, no.. = TRUE), "baseline.json")
})

# ---------------------------------------------------------------------------
# The wrapper merge-queue calls, end to end on temp files
# ---------------------------------------------------------------------------

f64_run_tool <- function(tool, args) {
  out <- suppressWarnings(system2("bash", c(shQuote(file.path(f64_tools_dir(), tool)), args),
    stdout = TRUE, stderr = TRUE))
  list(status = attr(out, "status") %||% 0L, output = out)
}

test_that("F64 ratchet: tools/perf-ratchet applies a passing gate to the baseline it is given", {
  b <- list(a = f64_entry(0.010, last = 0.009), `_calibration` = f64_entry(0.065))
  ws <- f64_workspace(b, f64_obs(a = 0.0085, new_bench = 0.02))
  withr::local_envvar(CUPLYR_PERF_BASELINE = NA)
  res <- f64_run_tool("perf-ratchet", c(
    "--baseline", shQuote(ws$baseline), "--observations", shQuote(ws$obs),
    "--summary", shQuote(ws$summary), "--out", shQuote(ws$out), "--commit", "abc1234"
  ))
  expect_equal(res$status, 0L, info = paste(res$output, collapse = "\n"))
  got <- perf_read_baseline(ws$baseline)
  expect_equal(as.numeric(got$a$median_s), 0.009)
  expect_identical(got$a$commit, "abc1234")
  expect_equal(as.numeric(got$new_bench$median_s), 0.02)
  expect_true("perf baseline: a 0.01 -> 0.009" %in% readLines(ws$summary))
  expect_true(file.exists(file.path(ws$out, "perf_ratchet.txt")))
})

test_that("F64 ratchet: tools/perf-ratchet honours CUPLYR_PERF_BASELINE", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  ws <- f64_workspace(b, f64_obs(a = 0.0085))
  withr::local_envvar(CUPLYR_PERF_BASELINE = ws$baseline)
  res <- f64_run_tool("perf-ratchet", c(
    "--observations", shQuote(ws$obs), "--summary", shQuote(ws$summary),
    "--out", shQuote(ws$out), "--commit", "abc1234"
  ))
  expect_equal(res$status, 0L, info = paste(res$output, collapse = "\n"))
  expect_equal(as.numeric(perf_read_baseline(ws$baseline)$a$median_s), 0.009)
})

test_that("F64 ratchet: tools/perf-ratchet is a no-op on a failed gate and fails loudly on bad input", {
  b <- list(a = f64_entry(0.010, last = 0.009))
  failed <- sub("perf: PASS", "perf: FAIL", f64_passing_summary, fixed = TRUE)
  ws <- f64_workspace(b, f64_obs(a = 0.005), summary = failed)
  withr::local_envvar(CUPLYR_PERF_BASELINE = NA)
  before <- f64_md5(ws$baseline)
  res <- f64_run_tool("perf-ratchet", c(
    "--baseline", shQuote(ws$baseline), "--observations", shQuote(ws$obs),
    "--summary", shQuote(ws$summary), "--out", shQuote(ws$out), "--commit", "abc1234"
  ))
  expect_equal(res$status, 0L, info = paste(res$output, collapse = "\n"))
  expect_identical(f64_md5(ws$baseline), before)

  # Passing gate, but the observations file is missing: non-zero, untouched.
  ws2 <- f64_workspace(b, obs = NULL)
  before2 <- f64_md5(ws2$baseline)
  res2 <- f64_run_tool("perf-ratchet", c(
    "--baseline", shQuote(ws2$baseline), "--observations", shQuote(ws2$obs),
    "--summary", shQuote(ws2$summary), "--out", shQuote(ws2$out), "--commit", "abc1234"
  ))
  expect_false(identical(res2$status, 0L))
  expect_identical(f64_md5(ws2$baseline), before2)

  # No baseline given at all (no --baseline, no CUPLYR_PERF_BASELINE): an
  # error, never a default. The observations path doesn't exist either, so a
  # wrong default could not write anything.
  res3 <- f64_run_tool("perf-ratchet", c(
    "--observations", shQuote(ws2$obs), "--summary", shQuote(ws2$summary),
    "--out", shQuote(ws2$out), "--commit", "abc1234"
  ))
  expect_false(identical(res3$status, 0L))
})

# ---------------------------------------------------------------------------
# Theo's deliberate set (raise) of an entry
# ---------------------------------------------------------------------------

test_that("F64 ratchet: perf_baseline_set raises or adds entries and refuses bad requests", {
  dir <- withr::local_tempdir()
  path <- file.path(dir, "baseline.json")
  f64_write_json(path, list(
    `_calibration` = f64_entry(0.065),
    filter_chain_eager = f64_entry(0.009, last = 0.0088)
  ))
  set <- f64_lib()$perf_baseline_set

  set(path, c(filter_chain_eager = 0.012, new_bench = 0.05), recorded_at = "2026-10-23T12:00:00+0000")
  got <- perf_read_baseline(path)
  expect_equal(as.numeric(got$filter_chain_eager$median_s), 0.012)
  expect_identical(got$filter_chain_eager$commit, "manual")
  expect_identical(got$filter_chain_eager$recorded_at, "2026-10-23T12:00:00+0000")
  expect_null(got$filter_chain_eager$last_observed_s)
  expect_equal(as.numeric(got$new_bench$median_s), 0.05)
  expect_equal(as.numeric(got[["_calibration"]]$median_s), 0.065)

  before <- f64_md5(path)
  expect_error(set(path, c(`_calibration` = 0.1)), "reserved", ignore.case = TRUE)
  expect_error(set(path, c(`_calibration_transfer` = 0.1)), "reserved", ignore.case = TRUE)
  expect_error(set(path, c(filter_chain_eager = 0)))
  expect_error(set(path, c(filter_chain_eager = -1)))
  expect_error(set(path, c(filter_chain_eager = NA_real_)))
  expect_error(set(path, c(filter_chain_eager = Inf)))
  expect_identical(f64_md5(path), before)
  expect_identical(list.files(dir, all.files = TRUE, no.. = TRUE), "baseline.json")
})

test_that("F64 ratchet: tools/perf-baseline-set sets entries in the baseline it is given", {
  dir <- withr::local_tempdir()
  path <- file.path(dir, "baseline.json")
  decoy <- file.path(dir, "env-baseline.json")
  start <- list(`_calibration` = f64_entry(0.065), filter_chain_eager = f64_entry(0.009, last = 0.0088))
  f64_write_json(path, start)
  f64_write_json(decoy, start)
  # Always point the env var at a temp file too, so the wrapper can never fall
  # through to the real baseline; --baseline must win over it.
  withr::local_envvar(CUPLYR_PERF_BASELINE = decoy)
  decoy_md5 <- f64_md5(decoy)

  res <- f64_run_tool("perf-baseline-set", c("--baseline", shQuote(path), "filter_chain_eager=0.02", "new_bench=0.05"))
  if (any(grepl("merge is running", res$output))) testthat::skip("a merge holds merge.lock")
  expect_equal(res$status, 0L, info = paste(res$output, collapse = "\n"))
  got <- perf_read_baseline(path)
  expect_equal(as.numeric(got$filter_chain_eager$median_s), 0.02)
  expect_identical(got$filter_chain_eager$commit, "manual")
  expect_null(got$filter_chain_eager$last_observed_s)
  expect_equal(as.numeric(got$new_bench$median_s), 0.05)
  expect_identical(f64_md5(decoy), decoy_md5)

  # Without --baseline it uses CUPLYR_PERF_BASELINE.
  res <- f64_run_tool("perf-baseline-set", "filter_chain_eager=0.03")
  if (any(grepl("merge is running", res$output))) testthat::skip("a merge holds merge.lock")
  expect_equal(res$status, 0L, info = paste(res$output, collapse = "\n"))
  expect_equal(as.numeric(perf_read_baseline(decoy)$filter_chain_eager$median_s), 0.03)

  # Refusals leave the file alone.
  before <- f64_md5(path)
  res <- f64_run_tool("perf-baseline-set", c("--baseline", shQuote(path), "_calibration=0.1"))
  expect_false(identical(res$status, 0L))
  res <- f64_run_tool("perf-baseline-set", c("--baseline", shQuote(path), "filter_chain_eager=-1"))
  expect_false(identical(res$status, 0L))
  expect_identical(f64_md5(path), before)
})

test_that("F64 ratchet: tools/perf-ratchet never takes merge.lock (merge-queue holds it)", {
  for (f in c("perf-ratchet", "perf-ratchet.R", "perf-ratchet-lib.R")) {
    txt <- readLines(file.path(f64_tools_dir(), f))
    code <- txt[!grepl("^\\s*#", txt)]
    expect_false(any(grepl("flock|merge\\.lock", code)), info = f)
  }
})

test_that("F64 ratchet: tools/perf records observations like the gate does", {
  perf <- paste(readLines(file.path(f64_tools_dir(), "perf")), collapse = "\n")
  expect_match(perf, "CUPLYR_PERF_OBSERVATIONS", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# Call sites: merge-queue runs the ratchet only after a passing gate and its
# ff-merge; merge-gate records the observations
# ---------------------------------------------------------------------------

test_that("F64 ratchet: merge-queue calls tools/perf-ratchet in its success branch only", {
  mq <- readLines(file.path(f64_tools_dir(), "merge-queue"))
  ff <- grep("merge -q --ff-only", mq, fixed = TRUE)
  code <- !grepl("^\\s*#", mq)
  ratchet <- which(grepl("tools/perf-ratchet", mq, fixed = TRUE) & code)
  els <- grep("^else\\b", trimws(mq))
  expect_length(ff, 1)
  expect_length(ratchet, 1)
  expect_true(length(els) >= 1)
  els <- els[els > ff][1]
  expect_true(ff < ratchet && ratchet < els)

  # The whole call, joining backslash continuations.
  block <- paste(mq[ratchet:(els - 1)], collapse = " ")
  expect_match(block, "--baseline \"$CUPLYR_MAIN/scratchpad/perf_baseline.json\"", fixed = TRUE)
  expect_match(block, "--observations \"$out/perf_observations.tsv\"", fixed = TRUE)
  expect_match(block, "--summary \"$out/summary.txt\"", fixed = TRUE)
  expect_match(block, "--out \"$out\"", fixed = TRUE)
  expect_match(block, "exit 6", fixed = TRUE)
  expect_true(any(grepl("6 .*perf", mq[1:20])))
})

test_that("F64 ratchet: merge-gate records the perf observations into the gate directory", {
  mg <- paste(readLines(file.path(f64_tools_dir(), "merge-gate")), collapse = "\n")
  expect_match(mg, "CUPLYR_PERF_OBSERVATIONS=\"$out/perf_observations.tsv\"", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# Guard hooks
# ---------------------------------------------------------------------------

f64_hook <- function(hook, input) {
  path <- normalizePath(testthat::test_path("..", "..", ".claude", "hooks", hook), mustWork = FALSE)
  if (!file.exists(path)) testthat::skip(paste(hook, "is not present"))
  if (!nzchar(Sys.which("jq"))) testthat::skip("jq is not available")
  json <- as.character(jsonlite::toJSON(list(tool_input = input), auto_unbox = TRUE))
  out <- suppressWarnings(system2("bash", shQuote(path), input = json, stdout = TRUE, stderr = TRUE))
  list(status = attr(out, "status") %||% 0L, output = paste(out, collapse = "\n"))
}

test_that("F64 ratchet: guard-bash keeps agents off the perf baseline and explains the ratchet", {
  real <- "/home/theo/cuplr/scratchpad/perf_baseline.json"

  r <- f64_hook("guard-bash.sh", list(command = "/home/theo/cuplr/tools/perf-baseline-set filter_chain_eager=0.02"))
  expect_equal(r$status, 2L)

  r <- f64_hook("guard-bash.sh", list(command = paste(
    "/home/theo/cuplr-wt/F64/tools/perf-ratchet --baseline", real,
    "--observations /tmp/o.tsv --summary /tmp/s.txt --out /tmp")))
  expect_equal(r$status, 2L)

  # The same wrapper on a temp copy is fine.
  r <- f64_hook("guard-bash.sh", list(command = paste(
    "/home/theo/cuplr-wt/F64/tools/perf-ratchet --baseline /tmp/RtmpX/file1.json",
    "--observations /tmp/o.tsv --summary /tmp/s.txt --out /tmp")))
  expect_equal(r$status, 0L)

  r <- f64_hook("guard-bash.sh", list(command = paste("cp /tmp/x.json", real)))
  expect_equal(r$status, 2L)
  expect_match(r$output, "ratchet")
  expect_match(r$output, "Theo")

  r <- f64_hook("guard-bash.sh", list(command = paste0("CUPLYR_PERF_", "RECORD=1 tools/perf")))
  expect_equal(r$status, 2L)
  expect_match(r$output, "Theo")
})

test_that("F64 ratchet: guard-files explains who moves the perf baseline", {
  r <- f64_hook("guard-files.sh", list(file_path = "/home/theo/cuplr/scratchpad/perf_baseline.json"))
  expect_equal(r$status, 2L)
  expect_match(r$output, "merge-queue")
  expect_match(r$output, "Theo")
})
