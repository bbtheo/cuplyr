# Shared timing for the suite's sub-millisecond timing tests (F106).
#
# Three timing tests failed the suite and a merge gate at random on unchanged
# code. Each flake had its own cause:
#   - F68a and F68b: proc.time() is rounded down to whole milliseconds
#     (?proc.time), and each closure's batch size (`reps`) was sized from ONE
#     call timed with that clock. A sub-ms call reads 0 or 1 ticks, so the
#     batches came out ~25 or ~250 ms instead of the documented 50, and one
#     slow probe call (a GC, a descheduling) shortened them further. Now:
#     time with the wall clock at sub-microsecond resolution (timing_now()),
#     size each batch from a probe that itself lasts long enough to read, and
#     re-size when the timed batches come in short (timing_medians()).
#   - F68b's remaining risk is its 2-5% headroom: a near miss is re-measured a
#     bounded number of times (timing_retry()); a far miss, and a sustained
#     slowdown that reads the same every time, still fails.
#   - F00 (perf_time() resolution test): Sys.sleep() overshoots by a variable
#     amount. Fixed by the timing_spin() fixture below; perf_time() itself is
#     unchanged.
#
# CUPLYR_TIMING_LOG=<file> appends one TSV row per sample (call id, label,
# closure, sample, per-call seconds, reps) so a future flake can be read
# without editing tests. The call id counts timing_medians() calls in the
# process. helper-perf.R and the perf gate do not use this file.

#' Busy-wait on the wall clock for `s` seconds; returns invisible NULL.
#' Unlike Sys.sleep() it lasts its time on any core at any speed: only a
#' descheduling across the deadline can overshoot it.
timing_spin <- function(s) {
  end <- timing_now() + s
  while (timing_now() < end) NULL
  invisible(NULL)
}

#' Wall clock in seconds, sub-microsecond resolution.
#' A step of the wall clock during a test is harmless: every statistic built
#' on it is a median over batches. Byte-compiled up front: the JIT would
#' otherwise compile it on its second call, a ~4 ms stall inside a timed span.
timing_now <- compiler::cmpfun(function() as.numeric(Sys.time()))

# per-process counter behind the call id column of CUPLYR_TIMING_LOG
.timing_state <- new.env()
.timing_state$calls <- 0L

#' One median per-call time (seconds) per closure.
#'
#' Each closure is warmed up `warmup` times, then sized: a doubling probe
#' batch lasting >= min(5 ms, `target`) gives its per-call time, and `reps`
#' is `ceiling(target / per_call)`. Samples are taken in interleaved rounds
#' (closure order fixed) so drift hits every closure alike; each is one batch
#' of `reps` calls, with an untimed gc() before it and a gpu_sync() fence
#' around it when `sync`. If a closure's median batch is under half its
#' target (the probe was unrepresentative: slow first calls), its `reps` is
#' re-sized from the median and all samples are re-taken, at most 3 times.
#'
#' @return Named numeric vector of medians, with attributes `reps` (named)
#'   and `samples` (samples x closures matrix of per-call seconds, last pass).
timing_medians <- function(fns, samples = 15L, target = 0.05, warmup = 3L, sync = TRUE,
                           label = NULL) {
  fence <- if (sync) gpu_sync else function() invisible(NULL)
  batch <- function(fn, reps) {
    fence()
    t0 <- timing_now()
    for (j in seq_len(reps)) fn()
    fence()
    timing_now() - t0
  }
  nms <- names(fns)
  for (fn in fns) for (i in seq_len(warmup)) fn()
  probe_min <- min(5e-3, target)
  reps <- vapply(fns, function(fn) {
    n <- 1L
    repeat {
      gc(verbose = FALSE)
      el <- batch(fn, n)
      if (el >= probe_min || n >= 16384L) break
      n <- n * 2L
    }
    max(1L, as.integer(ceiling(target / (el / n))))
  }, integer(1))
  for (pass in 1:4) {
    t <- matrix(NA_real_, samples, length(fns), dimnames = list(NULL, nms))
    for (s in seq_len(samples)) {
      for (k in seq_along(fns)) {
        gc(verbose = FALSE)
        t[s, k] <- batch(fns[[k]], reps[[k]]) / reps[[k]]
      }
    }
    med <- apply(t, 2, stats::median)
    short <- med * reps < 0.5 * target
    if (!any(short) || pass == 4L) break
    reps[short] <- pmax(reps[short], as.integer(ceiling(target / med[short])))
  }
  log <- Sys.getenv("CUPLYR_TIMING_LOG", "")
  if (nzchar(log)) {
    .timing_state$calls <- .timing_state$calls + 1L
    rows <- data.frame(
      call = .timing_state$calls,
      label = if (is.null(label)) "" else label,
      closure = rep(nms, each = samples),
      sample = rep(seq_len(samples), times = length(fns)),
      per_call_s = as.vector(t),
      reps = rep(unname(reps), each = samples)
    )
    utils::write.table(rows, log, append = TRUE, sep = "\t", quote = FALSE,
                       row.names = FALSE, col.names = FALSE)
  }
  structure(unname(med), names = nms, reps = stats::setNames(as.integer(reps), nms), samples = t)
}

#' Re-measure near misses, a bounded number of times.
#'
#' Calls `measure()`; stops at the first value with `ok(v)`, or the first
#' with `!retry_if(v)` (a far miss is a real regression, not noise), or after
#' `attempts` calls. `pause` seconds pass before each re-measure.
#' Callers assert on the last value and put every attempt in the label.
#'
#' @return List of the measured values, in order.
timing_retry <- function(measure, ok, retry_if, attempts = 3L, pause = 0) {
  out <- list()
  for (i in seq_len(attempts)) {
    if (i > 1L && pause > 0) Sys.sleep(pause)
    v <- measure()
    out[[i]] <- v
    if (ok(v) || !retry_if(v)) break
  }
  out
}
