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
# function. F59 made that CPU path fast: roll_median() uses
# data.table::frollmedian() when the installed data.table has it (1.18.4 or
# later; it stays in Suggests; a NaN is nulled first, and the few windows
# whose midpoint of two huge values overflows are recomputed exactly), else
# the internal roll_median_sorted() (src/roll_median.cpp, O(n log n) whatever the width).
# Both are identical() to the reference below. The other five functions keep
# the plain R implementation: they are the semantics oracle that GPU results
# are compared against (and the engine for standalone use on a vector).
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
#' no MEDIAN rolling aggregation, so `roll_median()` always runs on the CPU:
#' `data.table::frollmedian()` when the installed data.table has it (1.18.4 or
#' later), otherwise a built-in O(n log n) sliding median; both give exactly the result of
#' `stats::median()` over each window.
#' `data.table::frollmean(x, n)` (and its five siblings) work as
#' equivalent GPU-native aliases inside `mutate()`/`filter()` too, as long
#' as `align`/`fill`/`adaptive` are left at their defaults (`"right"`,
#' `NA`, `FALSE`) -- any other value falls back / errors, matching this
#' task's explicit "right-aligned only" scope.
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
  if (nx == 0 || n > nx) {
    return(rep(NA_real_, nx))
  }
  if (roll_has_frollmedian()) {
    roll_median_dt(x, n, na.rm)
  } else {
    roll_median_sorted(x, n, na.rm)
  }
}

# The internal switch forcing gpu_window()'s rolling path (see ?roll): the
# option cuplyr.roll_path, else the environment variable CUPLYR_ROLL_PATH,
# read when a plan is lowered (like cuplyr.memory_pool in R/gpu-memory.R).
roll_path_option <- function() {
  opt <- getOption("cuplyr.roll_path")
  if (is.null(opt)) opt <- Sys.getenv("CUPLYR_ROLL_PATH", "")
  if (identical(opt, "")) {
    return("auto")
  }
  if (!is.character(opt) || length(opt) != 1 || is.na(opt) ||
      !opt %in% c("auto", "scans", "cudf")) {
    stop("`cuplyr.roll_path` (or CUPLYR_ROLL_PATH) must be \"auto\", \"scans\" or ",
         "\"cudf\", not ", paste(deparse(opt), collapse = " "), ".", call. = FALSE)
  }
  opt
}

# data.table gained frollmedian() in 1.18.0; 1.18.4 is the release its output
# was checked identical() to the reference on, so older versions (and no
# data.table at all) take the built-in engine. A function so tests can force
# the fallback.
roll_has_frollmedian <- function() {
  requireNamespace("data.table", quietly = TRUE) &&
    utils::packageVersion("data.table") >= "1.18.4"
}

# data.table::frollmedian() with F55's NaN rule (a NaN is missing, so it is
# nulled first) -- identical() to roll_apply_generic(x, n, stats::median, na.rm)
# except where the two middle values' midpoint overflows: data.table averages
# in double ((1e308 + 1.5e308) / 2 = Inf), R's mean() in long double. Those
# are exactly the positions that come back +-Inf from a window holding no
# +-Inf input, so they are recomputed with the reference.
roll_median_dt <- function(x, n, na.rm) {
  x <- as.double(x)
  x[is.nan(x)] <- NA_real_
  out <- data.table::frollmedian(x, n, na.rm = na.rm)
  bad <- which(is.infinite(out))
  if (length(bad) > 0) {
    ci <- c(0L, cumsum(is.infinite(x)))
    bad <- bad[ci[bad + 1L] - ci[bad - n + 1L] == 0L]
    for (i in bad) {
      w <- x[(i - n + 1L):i]
      if (na.rm) w <- w[!is.na(w)]
      out[i] <- stats::median(w)
    }
  }
  out
}

# data.table-free sliding median: O(n log n) setup, O(log n) per row (see
# src/roll_median.cpp). Same semantics as roll_apply_generic(x, n,
# stats::median, na.rm).
roll_median_sorted <- function(x, n, na.rm = FALSE) {
  roll_check_n(n)
  n <- as.integer(n)
  nx <- length(x)
  if (nx == 0 || n > nx) {
    return(rep(NA_real_, nx))
  }
  roll_median_cpp(as.double(x), n, isTRUE(na.rm))
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
