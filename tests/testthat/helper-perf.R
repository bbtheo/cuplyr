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
# Process-local state
# =============================================================================
#
# GPU clock state (P8 idle vs boosted) drifts by ~2-3x between test-runner
# sessions independent of any code change, which used to make the perf gate
# fire on ambient drift alone. Two mechanisms address this:
#
#   - perf_warm_gpu(): force the clocks out of idle before anything is timed.
#   - perf_current_calibration(): measure a fixed, deterministic workload
#     alongside the real benchmarks and use current/baseline drift on *that*
#     workload to scale the acceptable threshold for every other benchmark.
#
# Both are cached per R process (via perf_state) so repeated expect_no_perf_
# regression() calls within one `testthat::test_file()` run only pay the
# warm-up/calibration cost once, keeping the ambient conditions the same for
# every benchmark. Use `force = TRUE` (or reset perf_state) to bypass.
perf_state <- new.env(parent = emptyenv())
perf_state$warmed <- FALSE
perf_state$calib_median <- NULL

# =============================================================================
# GPU warm-up
# =============================================================================

#' Data for the GPU warm-up loop
#'
#' Deliberately separate from the benchmark data (test-perf-regression.R's
#' perf_big_df) and the calibration data below -- it exists purely to keep
#' the GPU busy, not to be timed or compared against a baseline.
#' @keywords internal
perf_warmup_data <- function() {
  set.seed(20260101)
  n <- 2e5
  data.frame(
    x = runif(n),
    y = runif(n),
    z = rnorm(n)
  )
}

#' Force GPU clocks out of idle (P8) state before timing anything
#'
#' Runs real cuplyr GPU work (filter + mutate + collect) in a busy loop for
#' at least `min_seconds` of sustained kernel activity. GPUs left idle
#' between test sessions clock down to a power-saving state (P8) and only
#' ramp up under sustained load; timing benchmarks against a cold GPU
#' produces wall times 2-3x slower than a warmed one, for zero code-related
#' reason. This is idempotent per-process (see perf_state$warmed) so it's
#' cheap to call defensively from expect_no_perf_regression().
#'
#' @param min_seconds Minimum wall-clock duration to keep the GPU busy.
#' @param force Re-warm even if this process already warmed once.
#' @return Invisibly, TRUE if warm-up actually ran, FALSE if skipped (already warm).
#' @keywords internal
perf_warm_gpu <- function(min_seconds = 2, force = FALSE) {
  if (!force && isTRUE(perf_state$warmed)) {
    return(invisible(FALSE))
  }

  gpu_df <- tbl_gpu(perf_warmup_data(), lazy = FALSE)

  t0 <- proc.time()[["elapsed"]]
  repeat {
    gpu_df |>
      dplyr::filter(x > 0.2, y < 0.8) |>
      dplyr::mutate(a = x + y, b = a * z) |>
      collect()
    if (proc.time()[["elapsed"]] - t0 >= min_seconds) break
  }

  perf_state$warmed <- TRUE
  invisible(TRUE)
}

# =============================================================================
# Ambient calibration
# =============================================================================

#' Data for the ambient-calibration workload
#'
#' Distinct from both the warm-up data above and every benchmark pipeline in
#' test-perf-regression.R -- this workload's only job is to be timed
#' identically at record time and check time so its current/baseline drift
#' can serve as a proxy for ambient GPU clock state. Sized (5e6 rows) so the
#' filter+mutate+arrange pipeline below lands at ~70ms warm on this machine's
#' GPU, comfortably inside the ~50-100ms target range.
#' @keywords internal
perf_calibration_data <- function() {
  set.seed(20260202)
  n <- 5e6
  data.frame(
    id = sample.int(500L, n, replace = TRUE),
    v = runif(n),
    w = rnorm(n)
  )
}

#' Time the fixed ambient-calibration workload (median of 5, like perf_time() defaults)
#'
#' A filter/mutate/arrange pipeline, warm (the table is preallocated once,
#' outside the timed closure) and ~50-100ms per run on a boosted GPU -- long
#' enough to be a stable timing signal, short enough not to meaningfully add
#' to suite runtime.
#' @keywords internal
perf_calibration_time <- function() {
  gpu_df <- tbl_gpu(perf_calibration_data(), lazy = FALSE)

  perf_time(function() {
    gpu_df |>
      dplyr::filter(v > 0.1) |>
      dplyr::mutate(z = v + w) |>
      dplyr::arrange(id, dplyr::desc(z)) |>
      collect()
  })
}

#' Current-session calibration median, cached per process
#'
#' @param force Re-measure even if already cached this process.
#' @keywords internal
perf_current_calibration <- function(force = FALSE) {
  if (force || is.null(perf_state$calib_median)) {
    perf_state$calib_median <- perf_calibration_time()
  }
  perf_state$calib_median
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
#' (re)writes the baseline entry for `id` (plus the shared `_calibration`
#' entry, see below) instead of checking anything.
#'
#' Ambient robustness: GPU clock state (P8 idle vs boosted) can swing every
#' timing in a session by ~2-3x with zero code change, which would otherwise
#' make this gate fire on drift alone. Two defenses:
#'
#'   1. perf_warm_gpu() runs before any timing (cached per-process) to force
#'      clocks out of idle.
#'   2. A fixed ambient-calibration workload (perf_current_calibration()) is
#'      timed the same way `fn` is. `scale = current_calib / baseline_calib`
#'      estimates how much faster/slower *this session* is than the session
#'      the baseline was recorded in, purely from ambient state, and the
#'      threshold is widened/narrowed by that scale before comparing to
#'      `fn`'s time. A scale outside [0.5, 4] is clamped (with a warning); a
#'      scale outside [0.25, 8] means the machine state is too different to
#'      judge anything, so the check is skipped entirely rather than risking
#'      a false pass/fail.
#'
#' @param id Stable, descriptive benchmark identifier (used as the JSON key).
#'   Must not be "_calibration" -- that key is reserved for the calibration
#'   baseline entry.
#' @param fn Zero-argument closure representing the full pipeline, ending in
#'   collect() (or compute()) so lazy pipelines actually execute.
#' @param max_ratio Maximum allowed scale-adjusted current/baseline ratio
#'   before failing.
#' @param iterations Timed iterations passed through to perf_time().
expect_no_perf_regression <- function(id, fn, max_ratio = 1.5, iterations = 5) {
  if (!identical(Sys.getenv("CUPLYR_PERF"), "1")) {
    testthat::skip("perf checks are opt-in; set CUPLYR_PERF=1 to run them")
  }
  if (identical(id, "_calibration")) {
    stop("'_calibration' is a reserved benchmark id", call. = FALSE)
  }

  perf_warm_gpu()

  baseline_path <- perf_baseline_path()

  if (identical(Sys.getenv("CUPLYR_PERF_RECORD"), "1")) {
    median_s <- perf_time(fn, iterations = iterations)
    calib_median <- perf_current_calibration()

    baseline <- perf_read_baseline(baseline_path)
    baseline[[id]] <- list(
      median_s = median_s,
      recorded_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
      commit = perf_git_commit(),
      iterations = iterations
    )
    baseline[["_calibration"]] <- list(
      median_s = calib_median,
      recorded_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
      commit = perf_git_commit(),
      iterations = 5
    )
    perf_write_baseline(baseline_path, baseline)

    testthat::skip(sprintf(
      "baseline recorded for '%s': %.4fs (%d iterations, calibration=%.4fs) -> %s",
      id, median_s, iterations, calib_median, baseline_path
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

  calib_entry <- baseline[["_calibration"]]
  if (is.null(calib_entry) || is.null(calib_entry$median_s)) {
    testthat::skip(sprintf(
      "no '_calibration' baseline (baseline file predates ambient-calibration support); re-record with CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 for '%s'",
      id
    ))
  }

  baseline_calib <- as.numeric(calib_entry$median_s)
  current_calib <- perf_current_calibration()
  raw_scale <- current_calib / baseline_calib

  if (raw_scale < 0.25 || raw_scale > 8) {
    testthat::skip(sprintf(
      "[perf] %s: ambient calibration scale %.3fx (current=%.4fs, baseline=%.4fs) is far outside [0.25x, 8x] -- machine state too different from the baseline session to judge; skipping",
      id, raw_scale, current_calib, baseline_calib
    ))
  }

  scale <- raw_scale
  if (raw_scale < 0.5 || raw_scale > 4) {
    warning(sprintf(
      "[perf] %s: ambient calibration scale %.3fx is outside the normal [0.5x, 4x] range (current=%.4fs, baseline=%.4fs); clamping scale for this check",
      id, raw_scale, current_calib, baseline_calib
    ), call. = FALSE)
    scale <- max(0.5, min(4, raw_scale))
  }

  baseline_s <- as.numeric(entry$median_s)
  current_s <- perf_time(fn, iterations = iterations)
  raw_ratio <- current_s / baseline_s
  adjusted_ratio <- raw_ratio / scale
  threshold <- baseline_s * scale * max_ratio

  # Emit on pass as well as failure so perf runs are auditable from logs.
  message(sprintf(
    "[perf] %s: current=%.4fs baseline=%.4fs raw_ratio=%.3fx scale=%.3fx adjusted_ratio=%.3fx (max_ratio=%.2fx, commit=%s)",
    id, current_s, baseline_s, raw_ratio, scale, adjusted_ratio, max_ratio, entry$commit %||% "unknown"
  ))

  testthat::expect_lt(
    current_s, threshold,
    label = sprintf("%s current time (%.4fs, %d iters)", id, current_s, iterations),
    expected.label = sprintf(
      "baseline (%.4fs) * scale (%.3fx) * max_ratio (%.2fx) = %.4fs [raw_ratio=%.3fx, adjusted_ratio=%.3fx]",
      baseline_s, scale, max_ratio, threshold, raw_ratio, adjusted_ratio
    )
  )
}
