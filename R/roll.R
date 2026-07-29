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
# function, which is what this file provides for all six names (so
# roll_mean()/etc. also work standalone, outside any tbl_gpu pipeline, and
# so a hypothetical future cudf version that adds MEDIAN support only
# needs a C++/registry change, never an R-level fallback rewrite).
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
#' run GPU-natively via `cudf::rolling_window()`/`grouped_rolling_window()`
#' -- these R implementations are the CPU semantics reference and the
#' engine for standalone use and for `roll_median()`'s CPU fallback (cudf
#' 25.12 has no MEDIAN rolling aggregation in this environment).
#' `data.table::frollmean(x, n)` (and its five siblings) work as
#' equivalent GPU-native aliases inside `mutate()`/`filter()` too, as long
#' as `align`/`fill`/`adaptive` are left at their defaults (`"right"`,
#' `NA`, `FALSE`) -- any other value falls back / errors, matching this
#' task's explicit "right-aligned only" scope.
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
  roll_apply_generic(x, n, stats::median, na.rm)
}

# Shared CPU engine behind all six roll_*() functions above. Not
# performance-tuned (O(n * w)): the GPU path is native for everything
# except roll_median(), so this is only ever hot for roll_median() itself
# (small oracle-test-sized data) or a user calling roll_*() directly on a
# plain R vector outside any tbl_gpu pipeline.
#
# @param x A numeric vector
# @param n A single positive whole number (the window width)
# @param stat_fn A summary function taking a numeric vector and returning
#   a single number (e.g. base::mean, stats::median)
# @param na.rm Logical scalar
# @return A numeric vector the same length as `x`
# @keywords internal
roll_apply_generic <- function(x, n, stat_fn, na.rm = FALSE) {
  if (!is.numeric(n) || length(n) != 1 || is.na(n) || n != trunc(n) || n < 1) {
    stop("`n` must be a single positive whole number.", call. = FALSE)
  }
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
