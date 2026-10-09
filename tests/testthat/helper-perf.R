# Performance-regression harness
#
# This complements the correctness oracles in helper-oracle.R: those check
# that GPU results match dplyr-on-tibble; this checks that GPU results
# don't get *slower* over time as core-execution code is rewritten.
#
# Three-piece API:
#
#   perf_time(fn, iterations, warmup)        - median per-call wall time of fn()
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

#' Time a zero-arg closure, returning the median per-call time
#'
#' proc.time() ticks in 1 ms steps, which is 10-50% of a 2-10 ms benchmark. So
#' each timed sample runs `fn` back to back `reps` times, with `reps` chosen so
#' the sample lasts at least `min_sample_s` (~50 ms; one tick is then <= 2%),
#' and the sample's elapsed time is divided by `reps`. `reps` is 1 for calls
#' that already take that long, so slow benchmarks are timed exactly as
#' before. Batched calls run without gc_gpu() between them; the GPU is
#' collected once before each sample. That back-to-back GC pressure adds
#' 2-4% to a collect()-heavy call (transfer probe, 3 calls per sample:
#' 0.0183-0.0187 s batched vs 0.0180 s single-call, 4/4 gate-like sessions),
#' which is why the calibration probes opt out with `batch = FALSE`.
#'
#' @param fn A zero-argument function. For lazy pipelines it must call
#'   collect() (or compute()) internally so the pipeline actually executes;
#'   otherwise only AST construction would be timed.
#' @param iterations Number of timed samples (median is reported).
#' @param warmup Number of untimed runs executed first (JIT/caching/allocator
#'   warmup), discarded.
#' @param min_sample_s Minimum duration of one timed sample, in seconds.
#' @param batch If FALSE, never batch: every sample is one call (`reps` = 1),
#'   timed the way the pre-F00 baselines were recorded. Used by the calibration
#'   probes (see perf_calibration_time()).
#' @return Median per-call elapsed time in seconds (numeric scalar) with
#'   attribute `reps`, the number of calls batched into each sample.
perf_time <- function(fn, iterations = 5, warmup = 2, min_sample_s = 0.05, batch = TRUE) {
  stopifnot(is.function(fn))
  stopifnot(is.numeric(iterations), length(iterations) == 1, iterations >= 1)
  stopifnot(is.numeric(warmup), length(warmup) == 1, warmup >= 0)
  stopifnot(is.numeric(min_sample_s), length(min_sample_s) == 1, min_sample_s > 0)
  stopifnot(is.logical(batch), length(batch) == 1, !is.na(batch))

  # proc.time()["elapsed"] is a monotonic wall-clock reading (unaffected by
  # system clock adjustments), unlike Sys.time().
  now <- function() proc.time()[["elapsed"]]
  run_batch <- function(n) {
    t0 <- now()
    for (j in seq_len(n)) fn()
    now() - t0
  }

  for (i in seq_len(warmup)) {
    fn()
  }
  if (warmup > 0) gc_gpu()

  # Probe the per-call time with batches of doubling size until one batch
  # lasts >= 10 ms (10 ticks, so <= 10% error on the estimate).
  reps <- 1
  if (batch) {
    n <- 1
    repeat {
      elapsed <- run_batch(n)
      if (elapsed >= 0.01 || n >= 2^20) break
      n <- n * 2
    }
    per_call <- max(elapsed / n, 1e-9)
    reps <- if (per_call >= min_sample_s) 1 else min(ceiling(min_sample_s / per_call), 1e5)
  }

  times <- numeric(iterations)
  for (i in seq_len(iterations)) {
    gc_gpu()
    times[i] <- run_batch(reps) / reps
  }

  structure(stats::median(times), reps = reps)
}

# =============================================================================
# Process-local state
# =============================================================================
#
# GPU clock state (P8 idle vs boosted) and host contention can shift timings
# between test-runner sessions independent of any code change. Mechanisms:
#
#   - perf_warm_gpu(): force the clocks out of idle before anything is timed.
#   - perf_current_calibration_transfer(): a fixed tbl_gpu()/collect() round
#     trip whose current/baseline drift widens the thresholds of the
#     transfer-bound benchmarks (`calibration = "transfer"`), up to
#     perf_calibration_cap.
#   - perf_current_calibration(): the "kernel" probe. It is NOT used to scale
#     anything: its reading has two modes that depend on R process state, not
#     on the machine (see perf_calibration_time()). Kernel-calibrated
#     benchmarks always use scale 1, and the probe is only a health check that
#     errors above perf_kernel_probe_bound, far above both modes.
#
# (An earlier note here attributed 2-5x transfer-benchmark slowdowns to PCIe
# drift. Those runs were -O0 builds: the R<->cudf conversion loops are
# package C++, the kernels are always-optimized libcudf, so only
# transfer-bound benchmarks and the transfer probe slowed down. See
# perf_check_build().)
#
# A scale only ever widens a threshold, never tightens it.
#
# Both probes are cached per R process (via perf_state) so repeated
# expect_no_perf_regression() calls within one `testthat::test_file()` run
# only pay the calibration cost once. `force = TRUE` (or resetting perf_state)
# re-measures; for the kernel probe that typically moves the reading from its
# fast mode to its slow mode, which is harmless because neither mode changes
# a threshold.
perf_state <- new.env(parent = emptyenv())
perf_state$warmed <- FALSE
perf_state$calib_median <- NULL
perf_state$calib_transfer_median <- NULL

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

#' Data for the kernel-probe workload
#'
#' Distinct from both the warm-up data above and every benchmark pipeline in
#' test-perf-regression.R; this workload's only job is to be timed the same
#' way at record time and check time. 5e6 rows; the pipeline below reads
#' 0.043-0.047 s or 0.075-0.080 s per call depending on process state (see
#' perf_calibration_time()).
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

#' Time the fixed kernel-probe workload (median of 5, one call per sample)
#'
#' filter/mutate/arrange on a table preallocated on the GPU (outside the
#' timed closure), then collect(): the ~4.5e6 surviving rows x 4 columns,
#' ~126 MB, are copied device->host and built into an R data frame on every
#' call. So this is not a kernel-only or transfer-light workload: the
#' device->host copy and the host-side allocation are a large part of it.
#'
#' One call per sample (`batch = FALSE`), the way the `_calibration`
#' baseline (0.065 s) was recorded.
#'
#' Measured on the -O2 build with the GPU warmed, its reading has two modes
#' that depend on the state of the R process, with nothing else changed:
#'   - fast, 0.043-0.047 s (scale 0.66-0.72): the first measurement in a
#'     process in which the perf file's fixtures already exist. That is the
#'     gate's state (10/10 perf runs on record).
#'   - slow, 0.075-0.080 s (scale 1.154-1.231): a bare session without the
#'     fixtures, and any second measurement in the same process, even with
#'     the fixtures alive (2/2).
#' Neither mode is an ambient GPU reading, the mechanism is unexplained, and
#' the 0.065 s baseline matches neither. The slow mode straddles any cap
#' placed between its readings (0.078 s is 1.2000 +/- 1e-12 against 0.065 s,
#' and floating-point noise in the proc.time() difference picked the side),
#' so using this probe to widen thresholds made the gate's outcome random.
#' That is why kernel-calibrated benchmarks are never scaled by it: see
#' perf_effective_scale(), which only checks it against
#' perf_kernel_probe_bound.
#' @keywords internal
perf_calibration_time <- function() {
  gpu_df <- tbl_gpu(perf_calibration_data(), lazy = FALSE)

  perf_time(function() {
    gpu_df |>
      dplyr::filter(v > 0.1) |>
      dplyr::mutate(z = v + w) |>
      dplyr::arrange(id, dplyr::desc(z)) |>
      collect()
  }, batch = FALSE)
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
# Ambient calibration -- transfer probe
# =============================================================================
#
# Benchmarks dominated by host<->device copies (a fresh tbl_gpu()
# construction, a collect() of a large result) scale against this probe,
# which times exactly that shape: transfer in, transfer out, no compute. It
# is the only probe that widens thresholds (see perf_effective_scale()).

#' Data for the transfer-calibration workload
#'
#' Preallocated once (outside the timed closure, like perf_calibration_data())
#' so only the tbl_gpu()/collect() round trip is measured. Distinct seed/size
#' from every other fixture in this file so it can't accidentally alias one.
#' @keywords internal
perf_calibration_transfer_data <- function() {
  set.seed(20260203)
  n <- 2e6
  data.frame(
    id = sample.int(500L, n, replace = TRUE),
    v = runif(n),
    w = rnorm(n)
  )
}

#' Time the fixed transfer-calibration workload (median of 5, one call per sample)
#'
#' tbl_gpu() construction (host->device) immediately followed by collect()
#' (device->host), with no compute step in between, so the workload is
#' dominated by transfer bandwidth/latency and host-side conversion, not
#' kernel execution.
#'
#' One call per sample (`batch = FALSE`), like the `_calibration_transfer`
#' baseline and like transfer_roundtrip / join_left_*, which take well over
#' 50 ms per call and so are never batched. Batching this ~18 ms probe (3
#' calls per sample) adds GC pressure between back-to-back calls: 0.0183-0.0187 s
#' batched against 0.0180 s single-call in 4/4 gate-like sessions
#' (+1.7-3.9%), 0.0193-0.0197 s in earlier merge-gate runs. That would widen
#' the transfer thresholds by a measurement artifact. The price is
#' resolution: single-call readings are whole 1 ms ticks, 5.3% of the
#' 0.019 s baseline each.
#' @keywords internal
perf_calibration_transfer_time <- function() {
  df <- perf_calibration_transfer_data()

  perf_time(function() {
    tbl_gpu(df) |>
      collect()
  }, batch = FALSE)
}

#' Current-session transfer-calibration median, cached per process
#'
#' @param force Re-measure even if already cached this process.
#' @keywords internal
perf_current_calibration_transfer <- function(force = FALSE) {
  if (force || is.null(perf_state$calib_transfer_median)) {
    perf_state$calib_transfer_median <- perf_calibration_transfer_time()
  }
  perf_state$calib_transfer_median
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
#'
#' `digits = NA` writes doubles at full precision: toJSON()'s default of 4
#' decimal places (not significant digits) would round batched sub-ms timings
#' (e.g. 0.0016xx s for summarise_grouped_*, written as 0.0016) by up to ~3%.
#' @keywords internal
perf_write_baseline <- function(path, data) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  if (requireNamespace("jsonlite", quietly = TRUE)) {
    txt <- jsonlite::toJSON(data, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
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
# Build check
# =============================================================================

#' Path of the active cuplyr shared object, NA if none is loaded
#'
#' After an in-process pkgload::load_all() reload, pkgload loads the rebuilt
#' DLL from a new private temp copy but leaves the old one loaded, so the
#' list holds several "cuplyr" entries and `[["cuplyr"]]` returns the first,
#' stale one: after a reload to a partial -O0 build, the check would scan the
#' previous, optimized DLL and pass. R appends new DLLs to the end of its table
#' and keeps the order when it drops entries, so the last entry is the most
#' recently loaded one, which is the one the reloaded namespace uses.
#' @param dlls A DLLInfoList; defaults to getLoadedDLLs().
#' @keywords internal
perf_loaded_dll_path <- function(dlls = getLoadedDLLs()) {
  dlls <- unclass(dlls)
  dlls <- dlls[names(dlls) == "cuplyr"]
  if (length(dlls) == 0) NA_character_ else unclass(dlls[[length(dlls)]])[["path"]]
}

#' Was the loaded cuplyr DLL compiled with optimization?
#'
#' pkgbuild/pkgload compile with `-O0` by default (`debug = TRUE`). That is
#' invisible to correctness tests but makes the package-side conversion loops
#' 2.5-5x slower than the -O2 build the baselines were recorded with, so every
#' perf ratio would be meaningless.
#'
#' The answer comes from the compiler flags recorded in the loaded .so's DWARF
#' producer strings (the same scan as cuplyr_so_build_state() in
#' tools/ensure-optimized.R, duplicated here so the helper does not depend on
#' tools/). It has to read the file, not ask the package: `__OPTIMIZE__` in a
#' package translation unit only describes that one unit, and a load_all()
#' after editing a single .cpp recompiles just that unit at -O0 while the rest
#' stay -O2 (a "partial" build, which a check inside any one unit misses). Any
#' `-O0` unit means timings of the code paths it holds are inflated, so any
#' `-O0` fails.
#'
#' The absence of " -O0 " only means something if the flags were recorded at
#' all, so at least one GCC producer string ("GNU C", which also matches
#' "GNU C++20 ...") must be present. A build without `-g`, with compressed
#' debug sections, or from clang (no flags in its producer) records none and
#' is reported unoptimized rather than passing vacuously (finding no " -O0 "
#' in a file that records no flags proves nothing).
#' On this toolchain all 13 package .cpp units record a "GNU C++20" producer.
#' A DLL that is not loaded or cannot be read cannot be vouched for: FALSE.
#' @param path The shared object to scan; defaults to the active cuplyr DLL.
#' @keywords internal
perf_build_is_optimized <- function(path = perf_loaded_dll_path()) {
  if (is.na(path) || !file.exists(path)) {
    return(FALSE)
  }
  bytes <- readBin(path, "raw", file.size(path))
  has_producer <- length(grepRaw("GNU C", bytes, fixed = TRUE)) > 0
  has_producer && length(grepRaw(" -O0 ", bytes, fixed = TRUE)) == 0
}

#' Refuse to time an unoptimized build
#' @param optimized Whether the build is optimized; defaults to asking the
#'   loaded DLL. Overridable so the check itself is testable.
#' @keywords internal
perf_check_build <- function(optimized = perf_build_is_optimized()) {
  if (!isTRUE(optimized)) {
    rlang::abort(
      paste0(
        "The active cuplyr DLL (", perf_loaded_dll_path(), ") has -O0 compilation units (or records no ",
        "GCC compiler flags, or could not be read), so perf ratios are meaningless: -O0 code is 2.5-5x slower ",
        "than the -O2 build the baseline was recorded with. Run tools/build (it rebuilds stale -O0 objects) or load through ",
        "tools/test-file or tools/r, which do. When loading by hand, call ",
        "cuplyr_ensure_optimized_dll() (tools/ensure-optimized.R) and pkgload::load_all(debug = FALSE); ",
        "the default load_all() recompiles edited files at -O0."
      ),
      class = "cuplyr_perf_unoptimized_build"
    )
  }
  invisible(TRUE)
}

# =============================================================================
# Calibration scale and threshold
# =============================================================================

# Largest transfer-probe scale (current probe time / baseline probe time)
# that still widens the transfer thresholds. 1.2 is the lowest value the gate
# contract allows (a 1.2 scale must still widen). Evidence, on the -O2 build
# on this machine with the GPU warmed, the probe timed one call per sample
# (the way its 0.019 s baseline was recorded):
#   - 0.017-0.018 s (0.895-0.947) in 7/7 gate-like sessions and in earlier
#     merge-gate runs; 0.017 and 0.020 s (0.895, 1.053) in two record-order sessions.
#     Single-call readings are whole 1 ms ticks, 5.3% of the baseline each.
#   - batched readings, which run 2-4% high (see perf_calibration_transfer_time()):
#     1.018-1.035 in earlier merge-gate runs, 1.07-1.09 in five fresh standalone
#     sessions.
#   - an unoptimized (-O0) build read 3.0x.
# Against the 0.019 s baseline the cap falls at 0.0228 s, between two ticks:
# 0.022 s (1.158) widens, 0.023 s (1.211) errors, so no reading sits on it.
# That leaves 2 ticks above the highest single-call reading (0.020 s) and
# sits far below the -O0 symptom. The widest transfer threshold is
# 1.2 * 1.5 = 1.8x baseline. If a legitimate environment ever reads above it,
# fix the probe or re-record its baseline (Theo's call); do not raise the cap.
perf_calibration_cap <- 1.2

# Health-check bound for the kernel probe, which never widens anything (see
# perf_calibration_time()). It must sit clear of both of the probe's modes,
# measured against its 0.065 s baseline: fast 0.043-0.047 s (0.66-0.72),
# slow 0.075-0.080 s (1.154-1.231). 1.5:
#   - is 22% above the slow mode's highest reading (1.231). The bound falls at
#     0.0975 s, 17.5 ms above it (0.080 s) and between two ticks, so a
#     median of whole-tick readings is 0.097 s (1.492, passes) or 0.098 s
#     (1.508, errors) and floating-point noise cannot decide the outcome.
#   - equals the gate's max_ratio: a probe 1.5x slower than its baseline is
#     already more slowdown than the gate tolerates for any benchmark, so the
#     error flags a session that is unusable for timing (other GPU load,
#     clocks stuck low, host contention) or a regression in the probe's own
#     code paths (filter/mutate/arrange/collect), instead of a pile of
#     unexplained benchmark failures.
#   - is well below the 3x the end-to-end tests pin as an error.
# Between 1.231 and 1.5 a kernel reading neither widens nor errors. The probe
# does not detect an -O0 build (it read 0.83 on one, because its
# work is mostly libcudf); perf_check_build() does.
perf_kernel_probe_bound <- 1.5

# A raw scale within this relative distance of a bound counts as on it, i.e.
# accepted. proc.time() differences carry ~1e-12 relative error (a 78 ms
# reading came back as 0.077999999999999847 s), so without it a reading of
# exactly the bound in whole ticks (possible after a re-record) would land on
# either side at random.
perf_scale_tolerance <- 1e-9

#' Turn a raw probe ratio into the factor applied to the baseline
#'
#' Transfer probe (and the default, `context = NULL`): a session faster than
#' the baseline session (raw < 1) never tightens the threshold below
#' `baseline * max_ratio`, which is the gate's stated contract. A slower
#' session widens it by exactly the measured slowdown, up to
#' `perf_calibration_cap`.
#'
#' Kernel probe (`context$probe == "kernel"`): always 1. The probe's reading
#' depends on R process state (perf_calibration_time()), so it is only a
#' health check against `perf_kernel_probe_bound`.
#'
#' Beyond the bound, or for a nonsensical value (NA, non-positive, infinite),
#' there is no honest threshold to give: an error is raised rather than a
#' skip (a skipped perf test passes the gate) or a silent clamp (which hides
#' real regressions behind a sick-machine allowance).
#' @param raw_scale current probe time / baseline probe time.
#' @param context Optional list(id, probe, current_s, baseline_s) naming the
#'   benchmark, the probe ("kernel" or "transfer") and its two timings.
#'   `probe` selects the rule above; the rest only explains the error.
#' @keywords internal
perf_effective_scale <- function(raw_scale, context = NULL) {
  kernel <- identical(context$probe, "kernel")
  bound <- if (kernel) perf_kernel_probe_bound else perf_calibration_cap
  valid <- is.numeric(raw_scale) && length(raw_scale) == 1 && is.finite(raw_scale) && raw_scale > 0
  if (!valid || raw_scale > bound * (1 + perf_scale_tolerance)) {
    where <- NULL
    if (!is.null(context)) {
      where <- sprintf(
        "Benchmark '%s', %s probe: current %s s, baseline %s s, raw scale %s, %s %.2f.",
        context$id %||% "?", context$probe %||% "?",
        format(context$current_s %||% NA, digits = 4), format(context$baseline_s %||% NA, digits = 4),
        format(raw_scale), if (kernel) "health-check bound" else "cap", bound
      )
    }
    hint <- if (kernel) {
      paste0(
        "The kernel probe never widens thresholds; this is its health check. Its two process-state modes ",
        "read 0.043-0.047 s and 0.075-0.080 s (scale 0.66-0.72 and 1.154-1.231), both well below the bound, ",
        "so this reading is not one of them."
      )
    } else {
      paste0(
        "The transfer probe widens transfer thresholds up to the cap. Gate sessions read ",
        "0.017-0.020 s against its 0.019 s baseline."
      )
    }
    rlang::abort(
      c(
        sprintf(
          "Calibration scale %s is not usable (valid: > 0 and <= %.2f).",
          format(raw_scale), bound
        ),
        where,
        hint,
        "i" = paste0(
          "Either the machine state (GPU clocks, other GPU load, host contention) or the probe's code paths ",
          "are far from the baseline session. Investigate; do not raise the bound to make this pass."
        )
      ),
      class = "cuplyr_perf_calibration_error"
    )
  }
  if (kernel) 1 else max(1, raw_scale)
}

#' Time limit for a benchmark: baseline * effective calibration scale * max_ratio
#' @param context See perf_effective_scale().
#' @keywords internal
perf_threshold <- function(baseline_s, raw_scale, max_ratio = 1.5, context = NULL) {
  baseline_s * perf_effective_scale(raw_scale, context) * max_ratio
}

# =============================================================================
# Test-facing entry point
# =============================================================================

#' Assert that a benchmark hasn't regressed beyond max_ratio vs its baseline
#'
#' Skipped entirely unless CUPLYR_PERF=1 (perf checks are opt-in: they are
#' slow and machine-specific). With CUPLYR_PERF_RECORD=1, times `fn` and
#' (re)writes the baseline entry for `id` (plus the shared `_calibration` and
#' `_calibration_transfer` entries, see below) instead of checking anything.
#'
#' Ambient robustness: GPU clock state (P8 idle vs boosted) and host/PCIe
#' speed can shift every timing in a session with zero code change.
#' Defenses:
#'
#'   0. perf_check_build() fails the check on an unoptimized (-O0) build.
#'   1. perf_warm_gpu() runs before any timing (cached per-process) to force
#'      clocks out of idle.
#'   2. Two fixed calibration probes are timed one call per sample (the way
#'      their baselines were recorded): perf_current_calibration() ("kernel")
#'      and perf_current_calibration_transfer() ("transfer"). Each benchmark
#'      names its probe via `calibration`, and `raw_scale = current_calib /
#'      baseline_calib` for that probe goes through perf_effective_scale():
#'        - transfer: 1 if the session is faster (never tighter than
#'          baseline * max_ratio), the raw value up to `perf_calibration_cap`;
#'        - kernel: always 1 (the probe is bimodal by R process state, see
#'          perf_calibration_time()), checked against `perf_kernel_probe_bound`;
#'      and an error beyond the bound (or if the scale is NA/non-positive/
#'      infinite) instead of a skip or a silent clamp.
#'
#' @param id Stable, descriptive benchmark identifier (used as the JSON key).
#'   Must not be "_calibration" or "_calibration_transfer" -- those keys are
#'   reserved for the calibration baseline entries.
#' @param fn Zero-argument closure representing the full pipeline, ending in
#'   collect() (or compute()) so lazy pipelines actually execute.
#' @param max_ratio Maximum allowed scale-adjusted current/baseline ratio
#'   before failing.
#' @param iterations Timed iterations passed through to perf_time().
#' @param calibration Which probe this benchmark is checked against:
#'   `"kernel"` (default) for GPU-compute-heavy/transfer-light benchmarks
#'   (filter/mutate/select/arrange/summarise chains, fused pipelines), whose
#'   threshold is never widened, or `"transfer"` for benchmarks dominated by
#'   host<->device copies (a raw round trip, or a join whose collected result
#'   is much larger than its inputs), whose threshold widens with the
#'   transfer probe.
expect_no_perf_regression <- function(id, fn, max_ratio = 1.5, iterations = 5,
                                       calibration = c("kernel", "transfer")) {
  calibration <- match.arg(calibration)

  if (!identical(Sys.getenv("CUPLYR_PERF"), "1")) {
    testthat::skip("perf checks are opt-in; set CUPLYR_PERF=1 to run them")
  }
  if (identical(id, "_calibration") || identical(id, "_calibration_transfer")) {
    stop("'_calibration'/'_calibration_transfer' are reserved benchmark ids", call. = FALSE)
  }

  perf_check_build()
  perf_warm_gpu()

  baseline_path <- perf_baseline_path()

  if (identical(Sys.getenv("CUPLYR_PERF_RECORD"), "1")) {
    median_s <- as.numeric(perf_time(fn, iterations = iterations))
    calib_median <- perf_current_calibration()
    calib_transfer_median <- perf_current_calibration_transfer()

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
    baseline[["_calibration_transfer"]] <- list(
      median_s = calib_transfer_median,
      recorded_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
      commit = perf_git_commit(),
      iterations = 5
    )
    perf_write_baseline(baseline_path, baseline)

    testthat::skip(sprintf(
      "baseline recorded for '%s': %.4fs (%d iterations, calibration=%.4fs, calibration_transfer=%.4fs) -> %s",
      id, median_s, iterations, calib_median, calib_transfer_median, baseline_path
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

  if (calibration == "transfer") {
    calib_key <- "_calibration_transfer"
    current_calib_fn <- perf_current_calibration_transfer
    missing_calib_msg <- sprintf(
      "baseline lacks transfer calibration; re-record with CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 for '%s'",
      id
    )
  } else {
    calib_key <- "_calibration"
    current_calib_fn <- perf_current_calibration
    missing_calib_msg <- sprintf(
      "no '_calibration' baseline (baseline file predates ambient-calibration support); re-record with CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 for '%s'",
      id
    )
  }

  calib_entry <- baseline[[calib_key]]
  if (is.null(calib_entry) || is.null(calib_entry$median_s)) {
    testthat::skip(missing_calib_msg)
  }

  baseline_calib <- as.numeric(calib_entry$median_s)
  current_calib <- current_calib_fn()
  raw_scale <- as.numeric(current_calib) / baseline_calib
  context <- list(
    id = id, probe = calibration, current_s = as.numeric(current_calib), baseline_s = baseline_calib
  )
  scale <- perf_effective_scale(raw_scale, context)

  baseline_s <- as.numeric(entry$median_s)
  timed <- perf_time(fn, iterations = iterations)
  current_s <- as.numeric(timed)
  reps <- attr(timed, "reps")
  raw_ratio <- current_s / baseline_s
  adjusted_ratio <- raw_ratio / scale
  threshold <- perf_threshold(baseline_s, raw_scale, max_ratio, context)

  # Emit on pass as well as failure so perf runs are auditable from logs.
  message(sprintf(
    "[perf] %s (calibration=%s): current=%.4fs baseline=%.4fs raw_ratio=%.3fx raw_scale=%.3fx scale=%.3fx adjusted_ratio=%.3fx (max_ratio=%.2fx, reps=%d, commit=%s)",
    id, calibration, current_s, baseline_s, raw_ratio, raw_scale, scale, adjusted_ratio, max_ratio, reps, entry$commit %||% "unknown"
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
