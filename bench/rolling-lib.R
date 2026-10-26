# Helpers for the rolling-window benchmark (bench/rolling-window.qmd, F54/F63).
#
# Pure definitions only: sourcing this file changes no search path, attaches
# no package and does no GPU work. Every package call is namespaced, so the
# document decides which libraries are visible (its setup chunk prepends the
# benchmark library to .libPaths() and load_all()s cuplyr before sourcing
# this). The file's contents are part of the knitr cache key
# (rw_cache_key()), so editing it reruns every timing chunk.

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
# `reset`, when given, is called before every call (after the gc(), outside the
# timed region): an engine that keeps its result inside itself (DuckDB's
# CREATE TEMP TABLE) drops the previous one there, untimed.
rw_time <- function(fn, times, warmup = 2, reset = NULL) {
  clock <- rw_clock()
  res <- NULL
  for (i in seq_len(warmup)) {
    res <- NULL
    gc(verbose = FALSE)
    if (!is.null(reset)) reset()
    res <- fn()
  }
  out <- numeric(times)
  for (i in seq_len(times)) {
    res <- NULL
    gc(verbose = FALSE)
    if (!is.null(reset)) reset()
    t0 <- clock()
    res <- fn()
    out[[i]] <- clock() - t0
  }
  res <- NULL
  out
}

# Untimed warmup calls after the cold probe: fewer as the cell gets slower.
# `above_s` is the probe duration (s) above which a tier starts.
rw_warmup_tiers <- data.frame(above_s = c(-Inf, 5, 15), warmup = c(2, 1, 0))
rw_warmup_for <- function(probe_s) rw_warmup_tiers$warmup[sum(probe_s > rw_warmup_tiers$above_s)]

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
  warmup <- rw_warmup_for(probe_s)
  note <- if (k < times) {
    sprintf("times cut %d -> %d (cold probe call %.1f s, not recorded; %d untimed warmup call(s); %d timed call(s), all recorded)",
            times, k, probe_s, warmup, k)
  } else ""
  list(k = k, warmup = warmup, probe_is_sample = FALSE, note = note)
}

# Like rw_time() for several closures at once: `fns` is a named list; each of
# `times` rounds times every closure once (same gc-before-every-call treatment),
# alternating the order from round to round, so a slow drift or a change of
# process state hits all of them alike. `warmup` untimed calls per closure come
# first. `prepare`, when given, is a list of functions named like `fns`:
# prepare[[nm]]() runs before every call of fns[[nm]] (after the gc, outside the
# timed region, warmup calls included), e.g. to pin the process to a CPU.
# Returns a data frame (label, sample, ms).
rw_time_interleaved <- function(fns, times, warmup = 2, prepare = NULL) {
  clock <- rw_clock()
  res <- NULL
  before <- function(nm) if (!is.null(prepare[[nm]])) prepare[[nm]]()
  for (nm in names(fns)) {
    for (i in seq_len(warmup)) {
      res <- NULL
      gc(verbose = FALSE)
      before(nm)
      res <- fns[[nm]]()
    }
  }
  out <- vector("list", times * length(fns))
  k <- 0L
  for (i in seq_len(times)) {
    for (nm in if (i %% 2L == 1L) names(fns) else rev(names(fns))) {
      res <- NULL
      gc(verbose = FALSE)
      before(nm)
      t0 <- clock()
      res <- fns[[nm]]()
      k <- k + 1L
      out[[k]] <- data.frame(label = nm, sample = i, ms = clock() - t0, stringsAsFactors = FALSE)
    }
  }
  res <- NULL
  do.call(rbind, out)
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

# GPU completion fence: cuplyr calls return once the work is queued, so a timing
# needs the device to finish. gpu_sync() (cudaDeviceSynchronize) waits for all of
# it, with no copy back to the host; it works on any table, including one with no
# columns. Returns `r` unchanged.
rw_gpu_finish <- function(r) {
  cuplyr::gpu_sync()
  r
}

# The cuplyr timed closure: function(gt, n) -> tbl_gpu on the GPU, finished.
# `gt` was uploaded once outside timing. alias = TRUE spells the statistic as
# data.table::froll<stat>() (cuplyr maps it to the native kernel). The median
# has no cudf kernel and runs as a CPU fallback, so the report never times it:
# every cuplyr cell goes through rw_native_probe() first.
# fence = FALSE drops the gpu_sync() wait (used once, to measure what the
# fence costs).
rw_cuplyr_fun <- function(stat, alias = FALSE, fence = TRUE) {
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
    if (fence) rw_gpu_finish(r)
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

# ---- cells, controls and the cuplyr measurements (moved from the F54 driver) -------

# Progress line on stderr (visible in the render log; not captured into the
# document by knitr).
rw_say <- function(...) {
  cat(format(Sys.time(), "%H:%M:%S"), " ", ..., "\n", sep = "", file = stderr())
}

# ---- no timed CPU fallback ---------------------------------------------------------

# Run `call()` once, untimed, with options(cuplyr.fallback = "error"), so that a
# cuplyr call that would fall back to the CPU stops instead. Returns
# list(native = TRUE/FALSE, reason = "" or the condition message, fallback =
# TRUE when the error is cuplyr's CPU-fallback error). The previous value of
# the option is restored, also on error. A cuplyr cell is timed only when this
# says native = TRUE: a CPU fallback is never timed by accident. Only a
# fallback (fallback = TRUE) is a reason to skip a cell; any other error is a
# failure of the cell (rw_probe_error_cell()).
rw_native_probe <- function(call) {
  old <- options(cuplyr.fallback = "error")
  on.exit(options(old), add = TRUE)
  tryCatch({
    call()
    list(native = TRUE, reason = "", fallback = FALSE)
  }, error = function(e) list(native = FALSE, reason = conditionMessage(e), fallback = rw_is_fallback(e)))
}

# Is this error cuplyr's CPU-fallback error? With cuplyr.fallback = "error",
# cuplyr stops with "cuplyr: <verb>() fell back to CPU evaluation for: ...";
# a condition of class cuplyr_cpu_fallback counts too.
rw_is_fallback <- function(cnd) {
  inherits(cnd, "cuplyr_cpu_fallback") ||
    grepl("fell back to CPU evaluation", conditionMessage(cnd), fixed = TRUE)
}

# The cell of a probe call that failed for another reason than a CPU fallback:
# the same error row safe_cell() records for a call that errors, so it counts
# as a failed cell in the correctness table instead of dropping out of it.
rw_probe_error_cell <- function(probe) {
  safe_cell(function() stop("probe call failed: ", probe$reason, call. = FALSE))
}

# One row of the skipped-cells table (the chunk that skipped it, the cell, and
# the probe's reason), and the table from a list of such rows.
rw_skip_row <- function(chunk, stat, fn, width, reason) {
  data.frame(chunk = chunk, stat = stat, fn = fn, width = as.numeric(width),
             reason = reason, stringsAsFactors = FALSE)
}
rw_skip_table <- function(rows) {
  if (length(rows)) return(do.call(rbind, rows))
  rw_skip_row(character(), character(), character(), numeric(), character())
}

# ---- one timed cell ---------------------------------------------------------
# `call` is the zero-argument timed closure; `extract` turns its result into a
# numeric vector for the correctness check (outside timing). The first (cold)
# call is a probe: its duration decides, through rw_plan_cell(), how many
# samples fit in `cell_budget`, and its output feeds the correctness check. The
# probe is not a recorded sample unless it alone exceeds `probe_limit`. `note`
# says exactly which calls were recorded. `reset` (see rw_time()) also runs
# before the probe call.
run_cell <- function(call, extract, times, ref = NULL, expect_fallback = FALSE,
                     cell_budget = 20, probe_limit = 60, reset = NULL) {
  note <- character()
  gc(verbose = FALSE)
  if (!is.null(reset)) reset()
  fell_back <- FALSE
  t0 <- microbenchmark::get_nanotime()
  out <- withCallingHandlers(call(), cuplyr_cpu_fallback = function(cnd) fell_back <<- TRUE)
  probe_s <- (microbenchmark::get_nanotime() - t0) / 1e9
  cmp <- NULL
  if (!is.null(ref)) {
    vec <- extract(out)
    cmp <- rw_compare(rw_normalize(vec, length(ref)), ref)
    rm(vec)
  }
  rm(out)
  if (fell_back && !expect_fallback) note <- c(note, "UNEXPECTED CPU fallback")
  if (!fell_back && expect_fallback) note <- c(note, "expected CPU fallback did not happen")
  plan <- rw_plan_cell(probe_s, times, cell_budget, probe_limit)
  if (plan$probe_is_sample) {
    ms <- probe_s * 1e3
  } else {
    ms <- rw_time(call, plan$k, warmup = plan$warmup, reset = reset)
  }
  note <- c(note, if (nzchar(plan$note)) plan$note)
  list(ms = ms, times = length(ms), note = paste(note, collapse = "; "), cmp = cmp, fell_back = fell_back)
}

# A cell whose call errors counts as a failed comparison (ok = FALSE) with the
# message in `note`; it must not drop out of the correctness count.
safe_cell <- function(thunk) {
  tryCatch(thunk(), error = function(e) {
    list(ms = NA_real_, times = 0, note = paste("error:", conditionMessage(e)),
         cmp = list(ok = FALSE, n_na_mismatch = NA_integer_, max_abs_diff = NA_real_),
         fell_back = NA)
  })
}

# Run a cell with a package's thread count set to 1, restoring it afterwards
# (also on error).
with_one_thread <- function(pkg, thunk) {
  restore <- rw_set_threads(pkg, 1)
  on.exit(restore(), add = TRUE)
  thunk()
}

cuplyr_extract <- function(r) dplyr::collect(r)$r

# Peak and held GPU bytes of one call (gc() first, so earlier garbage does not
# count).
rw_mem_of <- function(call) {
  gc(verbose = FALSE)
  m <- cuplyr:::gpu_alloc_tracking(call())
  list(peak = m$peak_bytes, held = m$held_bytes)
}

# The per-call overhead probes: a plain arithmetic mutate and two window
# mutates (cumsum, and the rolling mean of width `width`). Names are the row
# labels of the overhead table.
rw_overhead_calls <- function(width) {
  stats::setNames(list(
    function(g) dplyr::mutate(g, y = x + 1),
    function(g) dplyr::mutate(g, r = roll_mean(x, width)),
    function(g) dplyr::mutate(g, r = cumsum(x))
  ), c("plain mutate(y = x + 1)",
       sprintf("window mutate(r = roll_mean(x, %s))", format(width, scientific = FALSE)),
       "window mutate(r = cumsum(x))"))
}

# compute() + fence around one of those calls.
rw_overhead_fn <- function(f) function(g) {
  r <- dplyr::compute(f(g))
  rw_gpu_finish(r)
  r
}

# ---- polars and DuckDB (F73) --------------------------------------------------------
# Each engine gets its data once, outside timing, and the timed call is only
# the rolling computation, to a result that stays inside the engine (the same
# rule as cuplyr's, whose result stays on the GPU). The result is read into R
# only by extract(), outside timing, for the correctness check.

# The timed SQL of one DuckDB cell. A raw ROWS frame gives partial windows at
# the start (a value where data.table has NA), so the aggregate is guarded:
# only rows with a full window get a value, which makes the NA padding equal to
# data.table's. The guard is the row index `i` (ungrouped) or the row's
# position in its partition (grouped). The ORDER BY i sort is inside the timed
# query, since SQL has no row order without it. CREATE TEMP TABLE (not CREATE
# OR REPLACE) keeps the result inside DuckDB; rw_out must be dropped first, by
# the untimed reset.
rw_duckdb_sql <- function(stat, n, grouped = FALSE) {
  agg <- switch(stat,
    mean = "AVG", sum = "SUM", min = "MIN", max = "MAX", sd = "STDDEV_SAMP", median = "MEDIAN",
    stop("no DuckDB window aggregate for statistic: ", stat))
  n <- as.integer(n)
  guard <- if (grouped) "row_number() OVER w" else "i"
  sprintf(paste0("CREATE TEMP TABLE rw_out AS SELECT i, CASE WHEN %s >= %d THEN %s(x) OVER w END AS r ",
                 "FROM rw_x WINDOW w AS (%sORDER BY i ROWS BETWEEN %d PRECEDING AND CURRENT ROW)"),
          guard, n, agg, if (grouped) "PARTITION BY g " else "", n - 1L)
}

# An in-memory DuckDB holding `df` as the real table rw_x (i = 1..n, the
# columns of df), loaded once. `threads` = NULL leaves DuckDB's default, a
# number runs SET threads. Returns list(con, threads(), fun(stat, grouped) ->
# function(n), reset(), extract(res), close()). The timed closure returns a
# token, not data; threads() reads the setting back from DuckDB.
# shared_home = FALSE: nothing is written under the user's home directory.
rw_duckdb_engine <- function(df, threads = NULL) {
  con <- DBI::dbConnect(duckdb::duckdb(shared_home = FALSE))
  ok <- FALSE
  on.exit(if (!ok) DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
  if (!is.null(threads)) DBI::dbExecute(con, sprintf("SET threads = %d", as.integer(threads)))
  DBI::dbWriteTable(con, "rw_x", cbind(data.frame(i = seq_len(nrow(df))), df))
  ok <- TRUE
  list(
    con = con,
    threads = function() as.integer(DBI::dbGetQuery(con, "SELECT current_setting('threads') AS t")$t),
    version = function() DBI::dbGetQuery(con, "SELECT version() AS v")$v,
    fun = function(stat, grouped = FALSE) {
      rw_duckdb_sql(stat, 3, grouped)   # an unsupported statistic stops here
      function(n) {
        DBI::dbExecute(con, rw_duckdb_sql(stat, n, grouped))
        structure("rw_out", class = "rw_duckdb_result")
      }
    },
    reset = function() invisible(DBI::dbExecute(con, "DROP TABLE IF EXISTS rw_out")),
    extract = function(res) {
      force(res)
      DBI::dbGetQuery(con, "SELECT r FROM rw_out ORDER BY i")$r
    },
    close = function() DBI::dbDisconnect(con, shutdown = TRUE)
  )
}

# polars: `df` converted to a polars DataFrame once. The timed closure is
# DataFrame$with_columns(<rolling expression>) (eager, so the expression is
# evaluated when the call returns), with $over("g") when grouped; it returns the
# polars DataFrame. Incomplete windows are null (min_samples defaults to the
# window), like data.table's NA. The thread pool is fixed when polars first
# loads, so this process's pool is whatever it started with: use
# rw_polars_child() to choose it.
rw_polars_engine <- function(df) {
  pl <- polars::pl
  pdf <- polars::as_polars_df(df)
  expr <- function(stat, n) {
    col <- pl$col("x")
    switch(stat,
      mean = col$rolling_mean(n), sum = col$rolling_sum(n), min = col$rolling_min(n),
      max = col$rolling_max(n), sd = col$rolling_std(n), median = col$rolling_median(n),
      stop("no polars rolling expression for statistic: ", stat))
  }
  list(
    threads = function() as.integer(polars::polars_info()$thread_pool_size),
    fun = function(stat, grouped = FALSE) {
      expr(stat, 3)   # an unsupported statistic stops here
      function(n) {
        e <- expr(stat, n)
        if (grouped) e <- e$over("g")
        pdf$with_columns(e$alias("r"))
      }
    },
    reset = function() invisible(NULL),
    extract = function(res) as.vector(res$get_column("r")),
    close = function() invisible(NULL)
  )
}

# Run do.call(fn, args) in a fresh R process (callr) whose .libPaths() starts
# with `libs`, and return its value. polars reads POLARS_MAX_THREADS once, when
# it loads, so a thread count needs its own process: `threads` is set in the
# child's environment before anything loads; NULL leaves it unset (a value
# inherited from this process is removed too). No profile files are read and
# nothing is loaded for the child: cuplyr is not in it, so no CUDA context.
# `show` passes the child's output (its rw_say() progress lines) on to this
# process's stderr while it runs, which is the render log (rw_say() writes
# there too); it is not captured into a knitr chunk, as callr's own `show`
# would be. The output is drained either way, so the child never blocks on a
# full pipe.
rw_polars_child <- function(threads, fn, args = list(), libs = .libPaths(), show = FALSE) {
  env <- callr::rcmd_safe_env()
  env <- env[names(env) != "POLARS_MAX_THREADS"]
  if (!is.null(threads)) env <- c(env, POLARS_MAX_THREADS = as.character(threads))
  p <- callr::r_bg(function(fn, args, unset) {
                     if (unset) Sys.unsetenv("POLARS_MAX_THREADS")
                     do.call(fn, args)
                   }, list(fn = fn, args = args, unset = is.null(threads)),
                   libpath = libs, env = env, stdout = "|", stderr = "2>&1",
                   user_profile = FALSE, system_profile = FALSE)
  on.exit(if (p$is_alive()) p$kill(), add = TRUE)
  # TRUE when at least one complete line was read
  drain <- function() {
    got <- FALSE
    repeat {
      l <- p$read_output_lines()
      if (!length(l)) break
      got <- TRUE
      if (show) cat(l, sep = "\n", file = stderr())
    }
    got
  }
  while (p$is_alive()) {
    p$poll_io(500)   # returns as soon as the child has output (or exits), so a chatty child never waits on us
    # An unfinished line makes poll_io() report output ready while no complete line
    # can be read yet; without a pause the loop would spin on one core until the
    # child ends the line. The pause only delays this process's reading (by at
    # most 50 ms); the child keeps running, and its pipe holds what it writes.
    if (!drain()) Sys.sleep(0.05)
  }
  drain()
  p$get_result()
}

# ---- cache key for the rendered document ------------------------------------------

# Versions of installed packages, named; "not installed" when absent.
rw_versions <- function(pkgs) {
  out <- vapply(pkgs, function(p) {
    v <- tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
    if (is.na(v)) "not installed" else v
  }, character(1))
  stats::setNames(unname(out), pkgs)
}

# One string that changes when any file's CONTENT changes (not its mtime), any
# version changes, or the configuration changes. Files are taken in sorted
# path order; a missing file counts as "missing".
rw_cache_key <- function(files, versions, config) {
  files <- sort(unique(as.character(files)))
  sums <- unname(tools::md5sum(files))
  sums[is.na(sums)] <- "missing"
  txt <- c("files", sums, "versions", paste0(names(versions), "=", versions),
           "config", deparse(config))
  tf <- tempfile("rw-key-")
  on.exit(unlink(tf), add = TRUE)
  writeLines(txt, tf)
  unname(tools::md5sum(tf))
}

# cuplyr's own sources under `root`: R/*.R and src/*.{cpp,cu,cuh,h,hpp}.
rw_cuplyr_sources <- function(root) {
  r <- list.files(file.path(root, "R"), pattern = "\\.R$", full.names = TRUE)
  s <- list.files(file.path(root, "src"), pattern = "\\.(cpp|cu|cuh|h|hpp)$", full.names = TRUE)
  sort(c(r, s))
}
