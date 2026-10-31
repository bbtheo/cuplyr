# Rolling-window functions (Theo-requested benchmark task,
# scratchpad/todo.md "Rolling-window benchmark target").
#
# roll_mean()/roll_sum()/roll_min()/roll_max()/roll_sd() are GPU-native
# inside mutate()/filter() (see R/ir.R's ir_parse_roll_call()/
# ir_parse_froll_call(), the "rolling" window kind, and
# src/ops_window.cpp's rolling branch) -- lowering happens through the
# exact same window-decomposition machinery every other window function
# (lag(), cumsum(), row_number(), ...) already uses, so no verb-specific
# wiring was needed in R/mutate.R/R/filter.R at all.
#
# roll_median() is NOT GPU-native: cudf 25.12's rolling_window()/
# grouped_rolling_window() have no MEDIAN aggregation instantiation in
# this environment (verified via `nm -DC libcudf.so`), so
# window_spec_lowerable() (R/lower.R) routes it through the existing
# mutate_window_fallback() CPU path, which re-runs the user's ORIGINAL
# `dplyr::mutate(tbl, roll_median(x, n))` call on the collected tibble --
# that re-run needs `roll_median()` to be a real, correct, exported R
# function. roll_median() uses the internal roll_median_sorted()
# (src/roll_median.cpp, O(n log n) whatever the width) by default; it equals
# the reference below (stats::median of each window as double). F66 made that the default: the opt-in
# option cuplyr.roll_median = "frollmedian" uses data.table::frollmedian()
# (1.18.4 or later; it stays in Suggests; a NaN is nulled first, and every
# position where data.table returns +-Inf takes the exact engine's value),
# which is faster on wide windows but can differ from the reference by one
# rounding step on windows with an even number of non-missing values. The
# other five functions keep the plain R implementation: they are the
# semantics oracle that GPU results are compared against (and the engine for
# standalone use on a vector).
#
# These are plain (non-generic) functions, like RcppRoll's/slider's own
# rolling-window surface -- not S3 methods, since there is exactly one
# implementation (a numeric vector in, a numeric vector out), matching
# data.table::frollmean()'s own non-generic signature.

#' Rolling-window aggregates
#'
#' Compute a right-aligned rolling (moving) aggregate over a numeric
#' vector: `n`-wide windows ending at (and including) each element. The
#' leading `n - 1` elements have no complete window and are always `NA`.
#' Matches `data.table::frollmean()`/`frollsum()`/etc.'s own
#' `align = "right"` default exactly (see `Details`).
#'
#' @details
#' `na.rm = FALSE` (the default): any window containing at least one `NA`
#'   produces `NA` for that position ("NA poisoning").
#' `na.rm = TRUE`: a window is only ever `NA` because it doesn't yet have
#'   `n` physical elements available (the leading `n - 1` positions,
#'   always). Once a window has `n` elements, `NA`s within it are dropped
#'   before computing (an all-`NA` window still yields `NA`). This
#'   `na.rm = TRUE` behavior was verified empirically against
#'   `data.table::frollmean(x, n, na.rm = TRUE)` (not assumed): the leading
#'   incomplete positions stay `NA` even though na.rm is set.
#'
#' Inside `mutate()`/`filter()` on a `tbl_gpu`, `roll_mean()`/`roll_sum()`/
#' `roll_min()`/`roll_max()`/`roll_sd()` (not `roll_median()`, see below)
#' run GPU-natively, in time that does not depend on the window width
#' (segmented scans; cudf's own rolling kernel is kept for
#' narrow windows, where it is faster, and for min/max of timestamps and
#' logicals) -- these R implementations are the CPU semantics reference and
#' the engine for standalone use. cudf 25.12 has
#' no MEDIAN rolling aggregation, so `roll_median()` always runs on the CPU
#' (see "Median engines").
#' `data.table::frollmean(x, n)` (and its five siblings) work as
#' equivalent GPU-native aliases inside `mutate()`/`filter()` too, as long
#' as `align`/`fill`/`adaptive` are left at their defaults (`"right"`,
#' `NA`, `FALSE`) -- any other value falls back / errors, matching this
#' task's explicit "right-aligned only" scope.
#'
#' **Median engines.** By default `roll_median()` uses a built-in exact
#' engine that returns exactly `stats::median()` of each window converted to
#' double (so integer and logical input give double, as `median()` of the
#' window cast to double would): the same NA/`na.rm` handling, `NaN` treated
#' as missing, `Inf`, and grouped windows. It runs in O(n log n) whatever the
#' width (roughly 0.2 s per 1e6 rows). `options(cuplyr.roll_median = "frollmedian")`
#' (or the environment variable `CUPLYR_ROLL_MEDIAN`; the option wins) opts
#' in to `data.table::frollmedian()`, which needs data.table >= 1.18.4
#' (Suggests). Its result equals `data.table::frollmedian(x, n, na.rm = na.rm)`
#' with `NaN` treated as `NA` first, except that every position where
#' data.table returns `Inf` or `-Inf` takes the exact engine's value (so a
#' midpoint of two finite values that overflows comes out finite, as in
#' `stats::median()`). Everywhere else it can differ from `stats::median()`
#' by one rounding step (1 ulp) on windows with an even number of non-missing
#' values. It is faster on wide windows without `NA`s, but slower than the
#' default with `NA`s and `na.rm = TRUE` on wide windows (about 4x slower at
#' width 1001 with 1% `NA`) and on thousands of small groups (several times
#' slower). If it is requested but data.table is missing or too
#' old, the exact engine is used and a message says so once per session. Any
#' other value of the option warns once (class `cuplyr_warning_invalid_option`)
#' and the exact engine is used; see [cuplyr-options]. `data.table::frollmedian()` written
#' inside `mutate()` keeps data.table's own result.
#'
#' The internal option `cuplyr.roll_path` (else the environment variable
#' `CUPLYR_ROLL_PATH`), one of `"auto"` (the default), `"scans"` or `"cudf"`,
#' forces one of the two GPU paths. It exists for tests and benchmarks and
#' changes speed, not results.
#'
#' @param x A numeric vector.
#' @param n A single positive whole number: the window width.
#' @param na.rm Logical scalar, default `FALSE`. See Details.
#' @return A numeric vector the same length as `x`.
#' @name roll
NULL

#' @rdname roll
#' @export
roll_mean <- function(x, n, na.rm = FALSE) {
  roll_apply_generic(x, n, base::mean, na.rm)
}

#' @rdname roll
#' @export
roll_sum <- function(x, n, na.rm = FALSE) {
  roll_apply_generic(x, n, base::sum, na.rm)
}

#' @rdname roll
#' @export
roll_min <- function(x, n, na.rm = FALSE) {
  roll_apply_generic(x, n, base::min, na.rm)
}

#' @rdname roll
#' @export
roll_max <- function(x, n, na.rm = FALSE) {
  roll_apply_generic(x, n, base::max, na.rm)
}

#' @rdname roll
#' @export
roll_sd <- function(x, n, na.rm = FALSE) {
  roll_apply_generic(x, n, stats::sd, na.rm)
}

#' @rdname roll
#' @export
roll_median <- function(x, n, na.rm = FALSE) {
  roll_check_n(n)
  n <- as.integer(n)
  nx <- length(x)
  engine <- roll_median_engine() # an invalid option warns even on short input
  if (nx == 0 || n > nx) {
    return(rep(NA_real_, nx))
  }
  # base R's rule, as in roll_apply_generic()'s `if (na.rm)`: 1 and "TRUE"
  # are TRUE; NA and a vector of length != 1 are errors. Both engines then
  # get a strict flag, so neither silently reads a non-flag as FALSE.
  na.rm <- if (na.rm) TRUE else FALSE
  if (identical(engine, "frollmedian")) {
    if (roll_has_frollmedian()) {
      return(roll_median_dt(x, n, na.rm))
    }
    roll_median_note_exact()
  }
  roll_median_sorted(x, n, na.rm)
}

# Which engine roll_median() uses: the option cuplyr.roll_median, else the
# environment variable CUPLYR_ROLL_MEDIAN, read per call (like
# roll_path_option()). "exact" (the default) or "frollmedian"; an invalid value
# warns once and means "exact" (registry: R/options.R).
roll_median_engine <- function() {
  # Per group, so the valid and unset paths cost what a bare getOption() does
  # (like resolve_exec_mode()); anything else goes to the registry, which warns
  # once about an invalid value and gives "exact".
  opt <- getOption("cuplyr.roll_median")
  if (is.null(opt)) {
    opt <- Sys.getenv("CUPLYR_ROLL_MEDIAN", "")
    if (identical(opt, "")) {
      return("exact")
    }
  }
  if (is.character(opt) && length(opt) == 1L && opt %in% cuplyr_roll_medians) {
    return(opt)
  }
  cuplyr_opt("roll_median")
}

# Per-session state of the opt-in engine: `noted` is the one-time message flag
# and `has_dt` caches the data.table availability check (see
# roll_has_frollmedian()). Fields of an environment so tests can reset them.
roll_median_state <- new.env(parent = emptyenv())

roll_median_note_exact <- function() {
  if (isTRUE(roll_median_state$noted)) {
    return(invisible())
  }
  roll_median_state$noted <- TRUE
  rlang::inform(
    paste0("The \"frollmedian\" engine (option `cuplyr.roll_median` or ",
           "environment variable CUPLYR_ROLL_MEDIAN) needs data.table >= 1.18.4, ",
           "which is not available; roll_median() uses the exact engine instead."),
    class = "cuplyr_message_roll_median_exact")
}

# The internal switch forcing gpu_window()'s rolling path (see ?roll): the
# option cuplyr.roll_path, else the environment variable CUPLYR_ROLL_PATH,
# read when a plan is lowered (like cuplyr.memory_pool in R/gpu-memory.R).
roll_path_option <- function() {
  # The same fast path as roll_median_engine().
  opt <- getOption("cuplyr.roll_path")
  if (is.null(opt)) {
    opt <- Sys.getenv("CUPLYR_ROLL_PATH", "")
    if (identical(opt, "")) {
      return("auto")
    }
  }
  if (is.character(opt) && length(opt) == 1L && opt %in% cuplyr_roll_paths) {
    return(opt)
  }
  cuplyr_opt("roll_path")
}

# data.table gained frollmedian() in 1.18.0; 1.18.4 is the release its output
# was checked against, so older versions (and no data.table at all) can't
# serve the opt-in "frollmedian" engine. A function so tests can force
# the fallback. Cached per session: packageVersion() costs ~0.1 ms, which
# adds up over thousands of groups. The cache never refreshes, so installing
# data.table mid-session still leaves the exact engine in use until restart;
# that is by design, since exact is the safe direction.
roll_has_frollmedian <- function() {
  if (is.null(roll_median_state$has_dt)) {
    roll_median_state$has_dt <- requireNamespace("data.table", quietly = TRUE) &&
      utils::packageVersion("data.table") >= "1.18.4"
  }
  roll_median_state$has_dt
}

# data.table::frollmedian() with F55's NaN rule (a NaN is missing, so it is
# nulled first). It matches roll_apply_generic(x, n, stats::median, na.rm)
# except (a) by one rounding step on windows with an even number of
# non-missing values, since data.table averages the two middle values in
# double and R's mean() in long double, and (b) where that midpoint
# overflows ((1e308 + 1.5e308) / 2 = Inf), whether or not the window also
# holds +-Inf. Every (b) position comes back +-Inf, so all +-Inf positions
# take the exact engine's value from one O(n log n) pass; where the true
# median is infinite that is the same +-Inf.
roll_median_dt <- function(x, n, na.rm) {
  x <- as.double(x)
  x[is.nan(x)] <- NA_real_
  out <- data.table::frollmedian(x, n, na.rm = na.rm)
  bad <- which(is.infinite(out))
  if (length(bad) > 0) {
    out[bad] <- roll_median_sorted(x, n, na.rm)[bad]
  }
  out
}

# The exact engine, data.table-free: O(n log n) setup, O(log n) per row (see
# src/roll_median.cpp). Same semantics as roll_apply_generic(x, n,
# stats::median, na.rm); `na.rm` must already be a strict TRUE/FALSE (Rcpp
# would read NA as true), which roll_median() guarantees.
roll_median_sorted <- function(x, n, na.rm = FALSE) {
  if (!isTRUE(na.rm) && !isFALSE(na.rm)) {
    stop("roll_median_sorted(): `na.rm` must be TRUE or FALSE.", call. = FALSE)
  }
  roll_check_n(n)
  n <- as.integer(n)
  nx <- length(x)
  if (nx == 0 || n > nx) {
    return(rep(NA_real_, nx))
  }
  roll_median_cpp(as.double(x), n, na.rm)
}

roll_check_n <- function(n) {
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n != trunc(n) || n < 1) {
    stop("`n` must be a single positive whole number.", call. = FALSE)
  }
  invisible(n)
}

# Shared CPU engine behind roll_mean/sum/min/max/sd above. Not
# performance-tuned (O(n * w)): the GPU path is native for these five and
# roll_median() has its own fast CPU path, so this is only hot for a user
# calling roll_mean()/etc. directly on a long plain R vector.
#
# @param x A numeric vector
# @param n A single positive whole number (the window width)
# @param stat_fn A summary function taking a numeric vector and returning
#   a single number (e.g. base::mean, stats::median)
# @param na.rm Logical scalar
# @return A numeric vector the same length as `x`
# @keywords internal
roll_apply_generic <- function(x, n, stat_fn, na.rm = FALSE) {
  roll_check_n(n)
  n <- as.integer(n)
  nx <- length(x)
  out <- rep(NA_real_, nx)
  if (nx == 0 || n > nx) {
    return(out)
  }

  x <- as.double(x)
  # embed(x, n)[i, ] == x[i + n - 1], x[i + n - 2], ..., x[i] (reverse
  # order within the row; irrelevant to every stat_fn used here, all of
  # which are order-invariant).
  m <- stats::embed(x, n)
  for (i in seq_len(nrow(m))) {
    w <- m[i, ]
    if (na.rm) {
      w <- w[!is.na(w)]
      if (length(w) == 0) {
        next
      }
    } else if (anyNA(w)) {
      next
    }
    out[i + n - 1L] <- stat_fn(w)
  }
  out
}
