# Helpers for the rolling-window benchmark (bench/rolling-window.R, F54).
#
# Pure definitions only: sourcing this file changes no search path, attaches
# no package and does no GPU work. Every package call is namespaced, so the
# driver decides which libraries are visible (bench/rolling-window.R prepends
# the benchmark library to .libPaths() before sourcing this).

rw_stats <- c("mean", "median", "min", "max", "sum", "sd")

# The post's data: set.seed(12345); tibble(x = rnorm(n_rows)).
rw_data <- function(n_rows, seed = 12345) {
  set.seed(seed)
  data.frame(x = rnorm(n_rows))
}

# Output of any rolling function -> plain double vector of length `len`, with
# NA padded on the left (the post compares right-aligned outputs; some
# packages return only the complete windows, as a matrix, or with names).
# Output tagged attr(, "rw_align") == "left" (runstats' Running*(), whose
# windows start at i and leave the last n - 1 values NA) is first shifted to
# right-aligned by the window width stored in attr(, "rw_window"). The shift
# happens here, outside timing; the timed closure only sets the two attributes.
rw_normalize <- function(out, len) {
  left <- identical(attr(out, "rw_align", exact = TRUE), "left")
  n <- attr(out, "rw_window", exact = TRUE)
  out <- as.numeric(out)
  if (left && length(n) == 1L && n > 1 && length(out) >= n) {
    out <- c(rep(NA_real_, n - 1), out[seq_len(length(out) - n + 1)])
  }
  if (length(out) < len) out <- c(rep(NA_real_, len - length(out)), out)
  out
}

# Compare a result with a reference: same length, same NA positions, and
# all.equal() on the rest. `max_abs_diff` is over positions where both are
# non-NA (NA when the lengths differ or nothing is comparable).
rw_compare <- function(res, ref, tol = 1e-8) {
  res <- as.numeric(res)
  ref <- as.numeric(ref)
  if (length(res) != length(ref)) {
    return(list(ok = FALSE, n_na_mismatch = NA_integer_, max_abs_diff = NA_real_))
  }
  na_res <- is.na(res)
  na_ref <- is.na(ref)
  n_na_mismatch <- sum(na_res != na_ref)
  both <- !na_res & !na_ref
  max_abs_diff <- if (any(both)) max(abs(res[both] - ref[both])) else NA_real_
  vals_ok <- !any(both) || isTRUE(all.equal(res[both], ref[both], tolerance = tol))
  list(
    ok = n_na_mismatch == 0L && vals_ok,
    n_na_mismatch = n_na_mismatch,
    max_abs_diff = max_abs_diff
  )
}

# Millisecond clock: microbenchmark's nanosecond timer when installed.
rw_clock <- function() {
  if (requireNamespace("microbenchmark", quietly = TRUE)) {
    function() as.numeric(microbenchmark::get_nanotime()) / 1e6
  } else {
    function() unname(proc.time()[["elapsed"]]) * 1e3
  }
}

# `warmup` untimed calls, then `times` timed ones; returns the elapsed ms of
# each timed call. An untimed gc() runs before every call (timed or not) with
# the previous result already dropped: cuplyr frees its GPU intermediates only
# when R finalizes their pointers, so without it back-to-back calls exhaust
# the card. Every function gets the same treatment.
rw_time <- function(fn, times, warmup = 2) {
  clock <- rw_clock()
  res <- NULL
  for (i in seq_len(warmup)) {
    res <- NULL
    gc(verbose = FALSE)
    res <- fn()
  }
  out <- numeric(times)
  for (i in seq_len(times)) {
    res <- NULL
    gc(verbose = FALSE)
    t0 <- clock()
    res <- fn()
    out[[i]] <- clock() - t0
  }
  res <- NULL
  out
}

# How many samples fit in a cell, from the duration of its first (cold) call.
# `probe_s` is that call's duration in seconds. The cold call is never one of
# the recorded samples, except when it alone exceeds `probe_limit`. Returns
# list(k, warmup, probe_is_sample, note): `k` timed samples after `warmup`
# untimed calls, and a note saying exactly which calls were recorded.
rw_plan_cell <- function(probe_s, times, cell_budget, probe_limit) {
  if (probe_s > probe_limit) {
    return(list(k = 1, warmup = 0, probe_is_sample = TRUE,
                note = sprintf("one sample, and it is the cold first call (%.0f s > %g s limit); no warmup, no repeat",
                               probe_s, probe_limit)))
  }
  k <- max(1, min(times, floor(cell_budget / max(probe_s, 1e-9))))
  warmup <- 2
  if (probe_s > 5) warmup <- 1
  if (probe_s > 15) warmup <- 0
  note <- if (k < times) {
    sprintf("times cut %d -> %d (cold probe call %.1f s, not recorded; %d untimed warmup call(s); %d timed call(s), all recorded)",
            times, k, probe_s, warmup, k)
  } else ""
  list(k = k, warmup = warmup, probe_is_sample = FALSE, note = note)
}

# Set a package's thread count to `n`; returns a function that restores the
# previous setting. The report measures what each setting does, because the
# settings do not all mean what their names suggest:
#   data.table: setDTthreads(n)
#   RcppRoll:   options(RcppRoll.threads = n) (OpenMP; read at call time)
#   roll:       RcppParallel::setThreadOptions(numThreads = n)
rw_set_threads <- function(pkg, n) {
  switch(pkg,
    data.table = {
      old <- data.table::setDTthreads(n)
      function() data.table::setDTthreads(old)
    },
    RcppRoll = {
      old <- options(RcppRoll.threads = n)
      function() options(old)
    },
    roll = {
      RcppParallel::setThreadOptions(numThreads = n)
      function() RcppParallel::setThreadOptions(numThreads = "auto")
    },
    stop("no thread control for package: ", pkg)
  )
}

# Which package's thread setting governs a benchmark entry (NA: none).
rw_threaded_pkg <- function(fn_name) {
  if (grepl("^data\\.table", fn_name)) return("data.table")
  if (identical(fn_name, "RcppRoll")) return("RcppRoll")
  if (identical(fn_name, "roll")) return("roll")
  NA_character_
}

# Entry label for the single-thread control of `fn_name`.
rw_one_thread_label <- function(fn_name) {
  if (grepl("\\)$", fn_name)) sub("\\)$", ", 1 thread)", fn_name) else paste0(fn_name, " (1 thread)")
}

# Inverse of rw_one_thread_label().
rw_base_label <- function(label) {
  if (grepl(", 1 thread\\)$", label)) sub(", 1 thread\\)$", ")", label) else sub(" \\(1 thread\\)$", "", label)
}

# data.table's right-aligned froll*() (the reference for the correctness
# table). "custom" is a rolling sum.
rw_reference <- function(stat, x, n) {
  switch(stat,
    mean = data.table::frollmean(x, n, align = "right"),
    median = data.table::frollmedian(x, n, align = "right"),
    min = data.table::frollmin(x, n, align = "right"),
    max = data.table::frollmax(x, n, align = "right"),
    sum = data.table::frollsum(x, n, align = "right"),
    sd = data.table::frollsd(x, n, align = "right"),
    custom = data.table::frollsum(x, n, align = "right"),
    stop("unknown statistic: ", stat)
  )
}

# The post's unoptimised custom function: a plain window sum.
rw_sum_custom <- function(x) sum(x)

# The post's calls and arguments, per statistic, for the packages that are
# installed. Each element is function(x, n). Output may need rw_normalize().
rw_functions <- function() {
  has <- function(pkg) requireNamespace(pkg, quietly = TRUE)
  keep <- function(l) l[!vapply(l, is.null, logical(1))]
  when <- function(pkg, f) if (has(pkg)) f else NULL

  # runstats' Running*() are left-aligned (the window starts at i, the last
  # n - 1 values are NA). The closures below only tag the raw result in place
  # (attributes set on the local fresh vector, no copy); rw_normalize() shifts
  # it to right-aligned outside timing, so the timed call is the package's own.
  dt <- function(fun) function(x, n) fun(x, n, align = "right")

  list(
    mean = keep(list(
      "data.table" = when("data.table", dt(data.table::frollmean)),
      "roll" = when("roll", function(x, n) roll::roll_mean(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingMean(x, n)),
      "runstats" = when("runstats", function(x, n) {
        out <- runstats::RunningMean(x, n)
        attr(out, "rw_align") <- "left"; attr(out, "rw_window") <- n
        out
      }),
      "slider" = when("slider", function(x, n) slider::slide_mean(x, before = n - 1, complete = TRUE)),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_meanr(x, n, fill = NA)),
      "zoo" = when("zoo", function(x, n) zoo::rollmeanr(x, n, fill = NA)),
      "runner" = when("runner", function(x, n) runner::mean_run(x, k = n, na_pad = TRUE))
    )),
    median = keep(list(
      "data.table" = when("data.table", dt(data.table::frollmedian)),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_medianr(x, n, fill = NA)),
      "roll" = when("roll", function(x, n) roll::roll_median(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingMedian(x, n)),
      "zoo" = when("zoo", function(x, n) zoo::rollmedianr(x, n, fill = NA))
    )),
    min = keep(list(
      "data.table" = when("data.table", dt(data.table::frollmin)),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_minr(x, n, fill = NA)),
      "roll" = when("roll", function(x, n) roll::roll_min(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingMin(x, n)),
      "slider" = when("slider", function(x, n) slider::slide_min(x, before = n - 1, complete = TRUE)),
      "runner" = when("runner", function(x, n) runner::min_run(x, k = n, na_pad = TRUE))
    )),
    max = keep(list(
      "data.table" = when("data.table", dt(data.table::frollmax)),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_maxr(x, n, fill = NA)),
      "roll" = when("roll", function(x, n) roll::roll_max(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingMax(x, n)),
      "slider" = when("slider", function(x, n) slider::slide_max(x, before = n - 1, complete = TRUE)),
      "runner" = when("runner", function(x, n) runner::max_run(x, k = n, na_pad = TRUE)),
      "zoo" = when("zoo", function(x, n) zoo::rollmaxr(x, n, fill = NA))
    )),
    sum = keep(list(
      "data.table" = when("data.table", function(x, n) data.table::frollsum(x, n, align = "right")),
      "data.table (exact)" = when("data.table", function(x, n) data.table::frollsum(x, n, algo = "exact", align = "right")),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_sumr(x, n, fill = NA)),
      "roll" = when("roll", function(x, n) roll::roll_sum(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingSum(x, n)),
      "runner" = when("runner", function(x, n) runner::sum_run(x, k = n, na_pad = TRUE)),
      "slider" = when("slider", function(x, n) slider::slide_sum(x, before = n - 1, complete = TRUE)),
      "zoo" = when("zoo", function(x, n) zoo::rollsumr(x, n, fill = NA))
    )),
    sd = keep(list(
      "data.table" = when("data.table", dt(data.table::frollsd)),
      "RcppRoll" = when("RcppRoll", function(x, n) RcppRoll::roll_sdr(x, n, fill = NA)),
      "roll" = when("roll", function(x, n) roll::roll_sd(x, n)),
      "RollingWindow" = when("RollingWindow", function(x, n) RollingWindow::RollingStd(x, n)),
      "runstats" = when("runstats", function(x, n) {
        out <- runstats::RunningSd(x, n)
        attr(out, "rw_align") <- "left"; attr(out, "rw_window") <- n
        out
      })
    )),
    custom = keep(list(
      "data.table::frollapply" = when("data.table", function(x, n) data.table::frollapply(x, n, rw_sum_custom)),
      "zoo::rollapply" = when("zoo", function(x, n) zoo::rollapply(x, n, rw_sum_custom, fill = NA, align = "right"))
    ))
  )
}

# Device-to-host fence. The cuplyr API gives no "work finished" guarantee
# (nothing calls cudaDeviceSynchronize()); copying one row back is a pageable
# cudaMemcpy, which synchronizes the default stream, and costs well under a
# millisecond. tbl_gpu's `$` goes through tibble from outside the namespace
# and returns NULL, hence .subset2(). Returns `r` unchanged.
rw_gpu_finish <- function(r) {
  schema <- .subset2(r, "schema")
  cuplyr:::gpu_head(.subset2(r, "ptr"), 1L, schema$names)
  r
}

# The cuplyr timed closure: function(gt, n) -> tbl_gpu on the GPU, finished.
# `gt` was uploaded once outside timing. alias = TRUE spells the statistic as
# data.table::froll<stat>() (cuplyr maps it to the native kernel). The median
# has no cudf kernel and runs as a CPU fallback.
rw_cuplyr_fun <- function(stat, alias = FALSE) {
  stat <- match.arg(stat, rw_stats)
  if (alias) {
    f <- switch(stat,
      mean = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollmean(x, n))),
      sum = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollsum(x, n))),
      min = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollmin(x, n))),
      max = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollmax(x, n))),
      sd = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollsd(x, n))),
      median = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = data.table::frollmedian(x, n)))
    )
  } else {
    f <- switch(stat,
      mean = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_mean(x, n))),
      sum = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_sum(x, n))),
      min = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_min(x, n))),
      max = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_max(x, n))),
      sd = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_sd(x, n))),
      median = function(gt, n) dplyr::compute(dplyr::mutate(gt, r = roll_median(x, n)))
    )
  }
  function(gt, n) {
    r <- f(gt, n)
    rw_gpu_finish(r)
    r
  }
}

# Grouped extension data: interleaved (unsorted) group ids.
rw_grouped_data <- function(n_rows, n_groups, seed = 12345) {
  set.seed(seed)
  g <- sample.int(n_groups, n_rows, replace = TRUE)
  x <- rnorm(n_rows)
  data.frame(g = g, x = x)
}

# Grouped rolling mean of width n for each engine. All preparation (upload,
# group_by, data.table copy) happens here, outside timing. `run` is the timed
# zero-argument closure; `extract(res)` gives the numeric result in the
# original row order (called outside timing).
rw_grouped_setup <- function(df, n) {
  out <- list()

  gtg <- dplyr::group_by(cuplyr::tbl_gpu(df), g)
  out$cuplyr <- list(
    run = function() {
      r <- dplyr::compute(dplyr::mutate(gtg, r = roll_mean(x, n)))
      rw_gpu_finish(r)
      r
    },
    extract = function(res) dplyr::collect(res)$r
  )

  dt <- data.table::as.data.table(df)
  out$data.table <- list(
    run = function() {
      dt[, r := data.table::frollmean(x, n), by = g]
      dt
    },
    extract = function(res) res$r
  )

  if (requireNamespace("slider", quietly = TRUE)) {
    gdf <- dplyr::group_by(tibble::as_tibble(df), g)
    out$slider <- list(
      run = function() dplyr::mutate(gdf, r = slider::slide_mean(x, before = n - 1, complete = TRUE)),
      extract = function(res) res$r
    )
  }
  out
}
