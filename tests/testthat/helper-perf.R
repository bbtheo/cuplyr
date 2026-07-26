# Performance-regression harness
#
# This complements the correctness oracles in helper-oracle.R: those check
# that GPU results match dplyr-on-tibble; this checks that GPU results
# don't get *slower* over time as core-execution code is rewritten.
#
# Three-piece API:
#
#   perf_time(fn, iterations, warmup)        - median wall time of fn()
#   perf_baseline_path()                     - where the baseline JSON lives
#   expect_no_perf_regression(id, fn, ...)   - the test-facing entry point
#
# Design notes:
#
# 1. Opt-in: perf checks are expensive (they build ~1e6-row tables and run
#    multiple iterations per benchmark) and machine-dependent (a baseline
#    recorded on one machine/GPU is meaningless on another). They are
#    therefore skipped unless CUPLYR_PERF=1 is set, matching skip_if_no_gpu()
#    style opt-outs elsewhere in the suite.
#
# 2. Two modes, both gated behind CUPLYR_PERF=1:
#      - CUPLYR_PERF_RECORD=1: time `fn`, write the result into the baseline
#        file as the new reference for `id`, then skip (recording is not a
#        pass/fail check).
#      - otherwise: time `fn`, compare against the stored baseline for `id`
#        with expect_lt(current, baseline * max_ratio). No baseline for
#        `id` yet -> skip with a clear message rather than fail, since a
#        fresh checkout has no baseline file at all.
#
# 3. Baseline storage: a single JSON file (default
#    scratchpad/perf_baseline.json, overridable via CUPLYR_PERF_BASELINE)
#    keyed by benchmark `id`. scratchpad/ is gitignored, so baselines are
#    never committed -- they are machine/GPU-specific and would be
#    meaningless (or actively misleading) shared across machines.
#
# 4. JSON backend: jsonlite is used when available (it is present in the
#    dev/pixi environment though not a formal package dependency); a small
#    hand-rolled reader/writer is used otherwise. The hand-rolled version
#    only needs to round-trip the flat structure this file itself writes
#    (id -> list(median_s, recorded_at, commit, iterations)), not arbitrary
#    JSON, so it stays intentionally minimal.

# =============================================================================
# Timing
# =============================================================================

#' Time a zero-arg closure, returning the median of several timed runs
#'
#' @param fn A zero-argument function. For lazy pipelines it must call
#'   collect() (or compute()) internally so the pipeline actually executes;
#'   otherwise only AST construction would be timed.
#' @param iterations Number of timed runs (median is reported).
#' @param warmup Number of untimed runs executed first (JIT/caching/allocator
#'   warmup), discarded.
#' @return Median elapsed time in seconds (numeric scalar).
perf_time <- function(fn, iterations = 5, warmup = 2) {
  stopifnot(is.function(fn))
  stopifnot(is.numeric(iterations), length(iterations) == 1, iterations >= 1)
  stopifnot(is.numeric(warmup), length(warmup) == 1, warmup >= 0)

  for (i in seq_len(warmup)) {
    fn()
  }
  if (warmup > 0) gc_gpu()

  times <- numeric(iterations)
  for (i in seq_len(iterations)) {
    gc_gpu()
    # proc.time()["elapsed"] is a monotonic wall-clock reading (unaffected by
    # system clock adjustments), unlike Sys.time().
    t0 <- proc.time()[["elapsed"]]
    fn()
    times[i] <- proc.time()[["elapsed"]] - t0
  }

  stats::median(times)
}

# =============================================================================
# Baseline file location
# =============================================================================

#' Look like an absolute (or home-relative) path
#' @keywords internal
perf_path_is_absolute <- function(path) {
  grepl("^(/|~|[A-Za-z]:[\\\\/])", path)
}

#' Resolve the perf baseline JSON path
#'
#' Defaults to scratchpad/perf_baseline.json relative to the package root
#' (tests run with a working directory of tests/testthat/, so the default is
#' resolved via testthat::test_path("../.."), the same pattern used in
#' test-install-script.R for locating install.sh). CUPLYR_PERF_BASELINE can
#' override the location entirely; if it is itself absolute (or "~..."), it
#' is used as-is rather than re-anchored to the package root.
#' @keywords internal
perf_baseline_path <- function() {
  configured <- Sys.getenv("CUPLYR_PERF_BASELINE", "scratchpad/perf_baseline.json")

  if (perf_path_is_absolute(configured)) {
    return(configured)
  }

  pkg_root <- tryCatch(
    testthat::test_path("..", ".."),
    error = function(e) ".."
  )

  file.path(pkg_root, configured)
}

# =============================================================================
# Baseline JSON read/write
# =============================================================================

#' Read the baseline file into a named list (id -> record list)
#'
#' Robust to a missing or corrupt file: both are treated as "no baselines
#' recorded yet" (empty list) rather than an error, since the perf harness
#' should never fail a run just because the baseline file doesn't parse.
#' @keywords internal
perf_read_baseline <- function(path) {
  if (!file.exists(path)) {
    return(list())
  }

  txt <- tryCatch(paste(readLines(path, warn = FALSE), collapse = "\n"), error = function(e) NULL)
  if (is.null(txt) || !nzchar(trimws(txt))) {
    return(list())
  }

  parsed <- if (requireNamespace("jsonlite", quietly = TRUE)) {
    tryCatch(
      jsonlite::fromJSON(txt, simplifyVector = FALSE),
      error = function(e) NULL
    )
  } else {
    tryCatch(perf_json_decode(txt), error = function(e) NULL)
  }

  if (is.null(parsed) || !is.list(parsed)) {
    return(list())
  }

  parsed
}

#' Write the baseline list back out as JSON
#' @keywords internal
perf_write_baseline <- function(path, data) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  if (requireNamespace("jsonlite", quietly = TRUE)) {
    txt <- jsonlite::toJSON(data, auto_unbox = TRUE, pretty = TRUE, null = "null")
    writeLines(as.character(txt), path)
  } else {
    writeLines(perf_json_encode(data), path)
  }

  invisible(path)
}

#' Minimal JSON encoder for the flat id -> record-of-scalars shape used here.
#' Not a general-purpose JSON writer -- only needs to pair with
#' perf_json_decode() below.
#' @keywords internal
perf_json_encode <- function(data) {
  if (length(data) == 0) {
    return("{}")
  }

  ids <- names(data)
  entries <- vapply(ids, function(id) {
    rec <- data[[id]]
    field_names <- names(rec)
    fields <- vapply(field_names, function(k) {
      v <- rec[[k]]
      if (is.character(v)) {
        sprintf('"%s": "%s"', k, gsub('"', '\\\\"', v))
      } else {
        sprintf('"%s": %s', k, format(as.numeric(v), scientific = FALSE, trim = TRUE))
      }
    }, character(1))
    sprintf('  "%s": {%s}', id, paste(fields, collapse = ", "))
  }, character(1))

  paste0("{\n", paste(entries, collapse = ",\n"), "\n}")
}

#' Minimal JSON decoder paired with perf_json_encode() above.
#'
#' Deliberately not a general JSON parser: it only needs to round-trip the
#' flat `{"id": {"key": value, ...}, ...}` structure this file writes, for
#' the fallback case where jsonlite isn't installed.
#' @keywords internal
perf_json_decode <- function(txt) {
  result <- list()

  # Match each top-level `"id": { ...fields... }` block.
  block_re <- '"([^"]+)"\\s*:\\s*\\{([^}]*)\\}'
  block_matches <- gregexpr(block_re, txt, perl = TRUE)
  blocks <- regmatches(txt, block_matches)[[1]]
  if (length(blocks) == 0) {
    return(result)
  }

  for (block in blocks) {
    m <- regmatches(block, regexec(block_re, block, perl = TRUE))[[1]]
    id <- m[2]
    body <- m[3]

    rec <- list()
    field_re <- '"([^"]+)"\\s*:\\s*("(?:[^"\\\\]|\\\\.)*"|[^,}]+)'
    field_matches <- gregexpr(field_re, body, perl = TRUE)
    fields <- regmatches(body, field_matches)[[1]]

    for (field in fields) {
      fm <- regmatches(field, regexec(field_re, field, perl = TRUE))[[1]]
      key <- fm[2]
      raw_val <- trimws(fm[3])
      if (startsWith(raw_val, "\"") && endsWith(raw_val, "\"")) {
        rec[[key]] <- gsub('\\\\"', '"', substr(raw_val, 2, nchar(raw_val) - 1))
      } else {
        rec[[key]] <- suppressWarnings(as.numeric(raw_val))
      }
    }

    result[[id]] <- rec
  }

  result
}

# =============================================================================
# git commit helper
# =============================================================================

#' Short git commit hash for `recorded_at` provenance, "unknown" on failure
#' (e.g. not a git checkout, git not on PATH).
#' @keywords internal
perf_git_commit <- function() {
  out <- tryCatch(
    system2("git", c("rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
    error = function(e) NULL,
    warning = function(w) NULL
  )
  status <- attr(out, "status")
  if (is.null(out) || length(out) == 0 || !is.null(status) && status != 0) {
    return("unknown")
  }
  trimws(out[1])
}

# =============================================================================
# Test-facing entry point
# =============================================================================

#' Assert that a benchmark hasn't regressed beyond max_ratio vs its baseline
#'
#' Skipped entirely unless CUPLYR_PERF=1 (perf checks are opt-in: they are
#' slow and machine-specific). With CUPLYR_PERF_RECORD=1, times `fn` and
#' (re)writes the baseline entry for `id` instead of checking anything.
#'
#' @param id Stable, descriptive benchmark identifier (used as the JSON key).
#' @param fn Zero-argument closure representing the full pipeline, ending in
#'   collect() (or compute()) so lazy pipelines actually execute.
#' @param max_ratio Maximum allowed current/baseline ratio before failing.
#' @param iterations Timed iterations passed through to perf_time().
expect_no_perf_regression <- function(id, fn, max_ratio = 1.5, iterations = 5) {
  if (!identical(Sys.getenv("CUPLYR_PERF"), "1")) {
    testthat::skip("perf checks are opt-in; set CUPLYR_PERF=1 to run them")
  }

  baseline_path <- perf_baseline_path()

  if (identical(Sys.getenv("CUPLYR_PERF_RECORD"), "1")) {
    median_s <- perf_time(fn, iterations = iterations)

    baseline <- perf_read_baseline(baseline_path)
    baseline[[id]] <- list(
      median_s = median_s,
      recorded_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
      commit = perf_git_commit(),
      iterations = iterations
    )
    perf_write_baseline(baseline_path, baseline)

    testthat::skip(sprintf(
      "baseline recorded for '%s': %.4fs (%d iterations) -> %s",
      id, median_s, iterations, baseline_path
    ))
  }

  baseline <- perf_read_baseline(baseline_path)
  entry <- baseline[[id]]

  if (is.null(entry) || is.null(entry$median_s)) {
    testthat::skip(sprintf(
      "no baseline for '%s'; run with CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 to record one",
      id
    ))
  }

  baseline_s <- as.numeric(entry$median_s)
  current_s <- perf_time(fn, iterations = iterations)
  ratio <- current_s / baseline_s

  # Emit on pass as well as failure so perf runs are auditable from logs.
  message(sprintf(
    "[perf] %s: current=%.4fs baseline=%.4fs ratio=%.3fx (max_ratio=%.2fx, commit=%s)",
    id, current_s, baseline_s, ratio, max_ratio, entry$commit %||% "unknown"
  ))

  testthat::expect_lt(
    current_s, baseline_s * max_ratio,
    label = sprintf("%s current time (%.4fs, %d iters)", id, current_s, iterations),
    expected.label = sprintf(
      "baseline (%.4fs) * max_ratio (%.2fx) = %.4fs [ratio would be %.3fx]",
      baseline_s, max_ratio, baseline_s * max_ratio, ratio
    )
  )
}
