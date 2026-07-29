# Benchmark: rolling-window operations, cuplyr GPU-native vs. data.table::froll*
#
# Spec: scratchpad/todo.md, section "Rolling-window benchmark target"
# Source workload design: Roald Arboel's rolling-window benchmark post
#   https://roald-arboel.com/blog/posts/2024/10/03/Benchmark-rolling/
#
# This script is a STANDALONE reference generator, not part of the
# tests/testthat perf-regression gate (CUPLYR_PERF): section 0-3 measure CPU
# rolling implementations (data.table::froll*, with a documented fallback
# chain for any froll* function missing from the installed data.table
# version) so the "cuplyr rolling vs best CPU" report below has an honest,
# locally-measured baseline (the blog's own numbers were measured on a
# different machine). Section 6 onward measures cuplyr's now-native rolling
# kernel (src/ops_window.cpp's "rolling" window family, wired in via
# R/ir.R's roll_mean()/roll_sum()/roll_min()/roll_max()/roll_sd()
# registrations and the data.table::froll*() alias -- roll_median() is the
# one exception, still CPU-fallback, see below) across THREE data-size
# tiers.
#
# Workloads: rolling mean / median / min / max / sum / sd
# Window sizes: 11, 101, 1001
# Data-size tiers: 1e6, 1e7, 1e8 iid Gaussian doubles (set.seed'd per tier
#   for reproducibility). The 1e8 tier uses fewer timed repeats (both sides)
#   and is entered defensively: a memory estimate is checked against
#   available system RAM / GPU memory before attempting it at all, and any
#   allocation/OOM failure at any point (R-side vector allocation, GPU
#   upload, or a cudf-side `Rcpp::stop()`/bad_alloc during the rolling
#   kernel itself) is caught and turns into a graceful "skipped" row in the
#   results table rather than crashing the whole script.
# Semantics (verified below, not assumed): right-aligned, output length ==
# input length, leading (window - 1) positions are NA, and ANY NA inside a
# window poisons that window's output to NA (data.table's own froll* default:
# align = "right", fill = NA, na.rm = FALSE).
#
# cuplyr timing methodology: EVERY cuplyr timed call measures BOTH
#   - "GPU-resident" time: `tbl_gpu(df, lazy = FALSE) |> mutate(roll = ...)`
#     ALONE, with no `collect()` -- eager mode means the rolling kernel has
#     already run by the time this call returns (push_op() ->
#     lower_and_execute() executes immediately, see R/execute.R), so this
#     number is the GPU-side compute cost without the device->host
#     transfer/tibble-construction overhead `collect()` adds.
#   - "e2e" (end-to-end) time: the same call immediately followed by
#     `collect()` -- this is what benchmark_rolling.R's OWN probe timed
#     before this task's scope addition (mutate()+collect() together), and
#     what most user code actually pays, so it's kept and reported
#     alongside the GPU-resident number rather than replaced by it (see
#     scratchpad/todo.md's "report e2e (with collect) honestly alongside").
#   Both numbers come from the SAME timed call (elapsed time is captured
#   right after `mutate()` returns, then again after `collect()` returns),
#   so this costs no extra GPU work compared to timing only one of the two.

# =============================================================================
# 0. Reference-engine setup (CPU only -- no GPU/cuplyr loaded in this section)
# =============================================================================

ensure_package <- function(pkg, install_if_missing = TRUE) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(TRUE)
  }
  if (!install_if_missing) {
    return(FALSE)
  }
  cat(sprintf("[setup] '%s' not installed; attempting install.packages(\"%s\")\n", pkg, pkg))
  ok <- tryCatch({
    install.packages(pkg, repos = "https://cran.r-project.org")
    requireNamespace(pkg, quietly = TRUE)
  }, error = function(e) {
    cat(sprintf("[setup] install of '%s' failed: %s\n", pkg, conditionMessage(e)))
    FALSE
  })
  ok
}

if (!ensure_package("data.table")) {
  stop("data.table is required as the reference rolling engine and could not be installed.")
}
if (!ensure_package("jsonlite")) {
  cat("[setup] jsonlite unavailable -- JSON dump step will be skipped.\n")
}

suppressMessages(library(data.table))

cat(sprintf("[setup] data.table version: %s\n", as.character(packageVersion("data.table"))))

# =============================================================================
# 1. Reference-engine resolution per workload
# =============================================================================
#
# The blog's ranking (data.table::froll* > RcppRoll > roll > RollingWindow >
# runner > runstats > slider > zoo) assumed frollmedian()/frollsd() did NOT
# exist yet in data.table (they were added later). THIS environment's
# installed data.table DOES have frollmedian/frollsd natively (verified
# below), so no fallback packages are needed here -- but the resolver still
# implements the fallback chain so this script keeps working (with clearly
# labeled, non-native engines) on an older data.table install.
#
# Every engine's `run(x, n)` wrapper is normalised to the same contract:
# right-aligned, length(x)-long output, leading (n-1) NAs, any-NA-in-window
# poisoning -- see the semantic self-check in section 2.

workloads <- c("mean", "median", "min", "max", "sum", "sd")

resolve_reference_engine <- function(workload) {
  dt_ns <- asNamespace("data.table")
  native_name <- paste0("froll", workload)

  # 1. Native data.table::froll<workload>, if this data.table version has it.
  if (exists(native_name, envir = dt_ns, inherits = FALSE)) {
    fn <- get(native_name, envir = dt_ns)
    return(list(
      engine = sprintf("data.table::%s (native)", native_name),
      native = TRUE,
      run = function(x, n) fn(x, n = n, align = "right", fill = NA_real_, na.rm = FALSE)
    ))
  }

  # 2. Specialised CRAN packages from the blog's ranking -- ONLY if already
  #    installed (this script never installs these; they're not needed here
  #    and some are heavier than justified for a fallback path).
  fallback_pkg_fns <- list(
    RcppRoll = c(mean = "roll_mean", median = "roll_median", min = "roll_min",
                 max = "roll_max", sum = "roll_sum", sd = "roll_sd"),
    roll     = c(mean = "roll_mean", median = "roll_median", sum = "roll_sum",
                 sd = "roll_sd")
  )
  for (pkg in names(fallback_pkg_fns)) {
    fn_name <- unname(fallback_pkg_fns[[pkg]][workload])
    if (!is.na(fn_name) && requireNamespace(pkg, quietly = TRUE)) {
      pkg_fn <- getExportedValue(pkg, fn_name)
      return(list(
        engine = sprintf("%s::%s (fallback, already installed)", pkg, fn_name),
        native = FALSE,
        run = function(x, n) {
          n_x <- length(x)
          raw <- pkg_fn(x, n = n, align = "right", fill = NA_real_)
          if (length(raw) != n_x) {
            out <- rep(NA_real_, n_x)
            out[(n_x - length(raw) + 1):n_x] <- raw
            raw <- out
          }
          raw
        }
      ))
    }
  }

  # 3. zoo::rollapply -- ONLY if already installed.
  if (requireNamespace("zoo", quietly = TRUE)) {
    stat_fn <- switch(workload,
      mean = base::mean, median = stats::median,
      min = base::min, max = base::max, sum = base::sum, sd = stats::sd
    )
    return(list(
      engine = "zoo::rollapply (fallback, already installed)",
      native = FALSE,
      run = function(x, n) as.numeric(zoo::rollapply(x, width = n, FUN = stat_fn, align = "right", fill = NA_real_))
    ))
  }

  # 4. Last resort: data.table::frollapply -- always present alongside
  #    data.table itself, generic (calls an R closure per window), much
  #    slower, but correct and always available.
  stat_fn <- switch(workload,
    mean = base::mean, median = stats::median,
    min = base::min, max = base::max, sum = base::sum, sd = stats::sd
  )
  list(
    engine = "data.table::frollapply (generic fallback, slow)",
    native = FALSE,
    run = function(x, n) data.table::frollapply(x, n, FUN = stat_fn, align = "right", fill = NA_real_)
  )
}

engines <- setNames(lapply(workloads, resolve_reference_engine), workloads)

cat("\n[setup] Reference engine per workload:\n")
for (wl in workloads) {
  cat(sprintf("  %-8s -> %s\n", wl, engines[[wl]]$engine))
}

# =============================================================================
# 2. Semantic self-check: right-alignment, leading NAs, NA-in-window poisoning
# =============================================================================
#
# Not assumed -- verified against a brute-force reference on a small vector
# with a deliberately injected NA, for every workload/engine pair resolved
# above, before any timing happens.

naive_roll <- function(x, n, stat_fn) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    if (i >= n) out[i] <- stat_fn(x[(i - n + 1):i])
  }
  out
}

naive_stat_fns <- list(
  mean = base::mean, median = stats::median,
  min = base::min, max = base::max, sum = base::sum, sd = stats::sd
)

run_semantic_check <- function() {
  set.seed(1)
  x_check <- rnorm(30)
  x_check[13] <- NA_real_  # deliberately poison one element mid-vector
  n_check <- 5

  cat("\n=== Semantic self-check (align=right, leading NA, NA-poisoning) ===\n")
  all_ok <- TRUE

  for (wl in workloads) {
    got <- engines[[wl]]$run(x_check, n_check)
    want <- naive_roll(x_check, n_check, naive_stat_fns[[wl]])

    len_ok <- length(got) == length(x_check)
    leading_ok <- all(is.na(got[seq_len(n_check - 1)]))
    poisoned_idx <- 13:(13 + n_check - 1)  # windows that include position 13
    poisoned_idx <- poisoned_idx[poisoned_idx <= length(x_check)]
    poison_ok <- all(is.na(got[poisoned_idx]))
    values_ok <- isTRUE(all.equal(got, want, tolerance = 1e-8, check.attributes = FALSE))

    ok <- len_ok && leading_ok && poison_ok && values_ok
    all_ok <- all_ok && ok

    cat(sprintf(
      "  %-8s [%s]: length=%s leading_na=%s na_poisoning=%s values_match_naive=%s -> %s\n",
      wl, engines[[wl]]$engine, len_ok, leading_ok, poison_ok, values_ok,
      if (ok) "PASS" else "FAIL"
    ))
  }

  if (!all_ok) {
    warning("Semantic self-check found a mismatch -- see PASS/FAIL detail above.")
  } else {
    cat("  All workloads confirmed: right-aligned, length(x)-long, leading (n-1) NAs,\n")
    cat("  and any-NA-in-window poisoning (data.table froll* defaults: align='right',\n")
    cat("  fill=NA, na.rm=FALSE).\n")
  }

  invisible(all_ok)
}

run_semantic_check()

# =============================================================================
# 3. Timing harnesses
# =============================================================================
#
# Mirrors the perf_time() idiom in tests/testthat/helper-perf.R (median of
# several timed runs, untimed warmup first) but is intentionally standalone:
# this script is not gated behind CUPLYR_PERF and has no GPU-clock ambient
# calibration.
#
# "Fewer for very slow cases": the first timed run also acts as a speed
# probe; slow workloads (frollapply-style fallbacks, huge windows, huge data
# tiers) get fewer repeats so the whole script still finishes in reasonable
# time, while fast calls get the full microbenchmark-style repeat count.

bench_time <- function(fn, max_iterations = 11L) {
  fn()  # untimed warmup

  t0 <- proc.time()[["elapsed"]]
  fn()
  first <- proc.time()[["elapsed"]] - t0

  iterations <- if (first > 2) 3L else if (first > 0.5) 5L else max_iterations
  iterations <- min(iterations, max_iterations)

  times <- numeric(iterations)
  times[1] <- first
  if (iterations > 1) {
    for (i in 2:iterations) {
      t0 <- proc.time()[["elapsed"]]
      fn()
      times[i] <- proc.time()[["elapsed"]] - t0
    }
  }

  list(median_s = stats::median(times), iterations = iterations, times = times)
}

# Two-phase timing for the cuplyr GPU probe: captures BOTH "GPU-resident"
# (build_fn() alone) and "e2e" (build_fn() + collect_fn()) elapsed time from
# the SAME call, so this costs no extra GPU work versus timing only one of
# the two (see the file-level comment). `after_each` runs after every timed
# call (NOT counted in the timing) -- used to gpu_gc() between iterations at
# the larger data tiers so GPU memory doesn't accumulate across repeats.
bench_time_two_phase <- function(build_fn, collect_fn, max_iterations = 11L, after_each = NULL) {
  measure_once <- function() {
    t0 <- proc.time()[["elapsed"]]
    obj <- build_fn()
    t_resident <- proc.time()[["elapsed"]] - t0
    res <- collect_fn(obj)
    t_e2e <- proc.time()[["elapsed"]] - t0
    if (!is.null(after_each)) after_each()
    list(resident = t_resident, e2e = t_e2e, result = res)
  }

  measure_once()  # untimed warmup

  first <- measure_once()
  iterations <- if (first$e2e > 2) 3L else if (first$e2e > 0.5) 5L else max_iterations
  iterations <- min(iterations, max_iterations)

  resident_times <- numeric(iterations)
  e2e_times <- numeric(iterations)
  resident_times[1] <- first$resident
  e2e_times[1] <- first$e2e
  if (iterations > 1) {
    for (i in 2:iterations) {
      m <- measure_once()
      resident_times[i] <- m$resident
      e2e_times[i] <- m$e2e
    }
  }

  list(
    resident_median_s = stats::median(resident_times),
    e2e_median_s = stats::median(e2e_times),
    iterations = iterations
  )
}

# =============================================================================
# 4. cuplyr setup (loaded ONCE, outside the per-size loop)
# =============================================================================

cat("\n=== cuplyr setup ===\n")

cuplyr_ready <- tryCatch({
  suppressMessages(devtools::load_all(quiet = TRUE))
  isTRUE(has_gpu())
}, error = function(e) {
  cat(sprintf("[cuplyr] could not load package / no GPU: %s\n", conditionMessage(e)))
  FALSE
})

cat(sprintf("[cuplyr] ready: %s\n", cuplyr_ready))

# CRITICAL for this benchmark's own correctness: roll_median() (and its
# data.table::frollmedian() alias) is registered and PARSES successfully,
# but silently falls back to a CPU re-run (mutate_window_fallback(), see
# R/roll.R) rather than erroring -- so a bare tryCatch(mutate(...)|>collect(),
# error=...) "succeeded, therefore native" check (this script's ORIGINAL
# pre-native-kernel probe design, back when EVERY rolling call was a hard
# error) is WRONG now: it would silently time the CPU fallback path and
# report it as "cuplyr (GPU) NATIVE". options(cuplyr.fallback = "error")
# makes any fallback raise instead of silently succeeding (see
# R/execute.R's cuplyr_fallback_notify(), exercised by
# test-fallback.R's "options(cuplyr.fallback = 'error') stops for a
# fallback verb"), so the probe below correctly reports roll_median() as
# "not native" via the same tryCatch()-catches-an-error path used for a
# genuinely unrecognized shape.
if (cuplyr_ready) {
  options(cuplyr.fallback = "error")
}

# Rough per-tier GPU memory requirement estimate: input column (8 bytes/row)
# + output rolling-result column (8 bytes/row) + gpu_window()'s current
# whole-input-table re-copy into the final output table (src/ops_window.cpp:
# every input column is deep-copied via `std::make_unique<cudf::column>
# (view.column(i))`, not just the new appended column) + headroom for the
# na.rm=TRUE path's extra COUNT-pass/mask/copy_if_else buffers. ~4.5x the
# raw column size is a conservative (over-, not under-) estimate.
estimate_gpu_bytes_needed <- function(n_rows) {
  as.numeric(n_rows) * 8 * 4.5
}

# Rough R-side memory requirement: the CPU reference vector itself, plus
# data.table::froll*()'s own internal working copy, plus the (never
# actually collected in full for the CPU side, but budgeted for anyway)
# result vector. ~3x the raw vector size is conservative.
estimate_r_bytes_needed <- function(n_rows) {
  as.numeric(n_rows) * 8 * 3
}

gpu_free_bytes <- function() {
  if (!cuplyr_ready) return(NA_real_)
  st <- tryCatch(gpu_memory_state(), error = function(e) NULL)
  if (is.null(st) || !isTRUE(st$available)) return(NA_real_)
  st$free_bytes
}

sys_free_bytes <- function() {
  # Linux-only best-effort read of MemAvailable from /proc/meminfo; returns
  # NA (i.e. "unknown, don't block on this check") on any other platform or
  # parse failure -- the tryCatch()-wrapped allocation attempt itself is the
  # real safety net regardless.
  tryCatch({
    lines <- readLines("/proc/meminfo", n = 5)
    avail_line <- grep("^MemAvailable:", lines, value = TRUE)
    if (length(avail_line) == 0) return(NA_real_)
    kb <- as.numeric(regmatches(avail_line, regexpr("[0-9]+", avail_line)))
    kb * 1024
  }, error = function(e) NA_real_)
}

# =============================================================================
# 5. Per-size-tier sweep: CPU reference + cuplyr GPU probe
# =============================================================================
#
# sizes: 1e6 (the original single-tier scope), 1e7, 1e8 (this task's scope
# addition). Larger tiers get fewer timed repeats (both CPU and GPU side)
# and a defensive memory check before ANY allocation is attempted; any
# allocation/OOM failure at any point downgrades gracefully to a "skipped"
# row instead of crashing the whole script.

sizes <- c(1e6, 1e7, 1e8)
window_sizes <- c(11L, 101L, 1001L)

max_iterations_for_size <- function(n_rows) {
  if (n_rows >= 1e8) return(3L)
  if (n_rows >= 1e7) return(7L)
  11L
}

results <- list()
add_result <- function(entry) {
  results[[length(results) + 1]] <<- entry
}

for (size_idx in seq_along(sizes)) {
  N <- sizes[size_idx]
  max_iter <- max_iterations_for_size(N)

  cat(sprintf(
    "\n############################################################\n"
  ))
  cat(sprintf(
    "# Size tier %d/%d: N = %s rows (max_iterations = %d)\n",
    size_idx, length(sizes), format(N, big.mark = ","), max_iter
  ))
  cat(sprintf(
    "############################################################\n"
  ))

  r_bytes_needed <- estimate_r_bytes_needed(N)
  r_free <- sys_free_bytes()
  if (!is.na(r_free) && r_bytes_needed > r_free * 0.9) {
    cat(sprintf(
      "[skip] size=%s: estimated R-side memory need (%.2f GB) exceeds 90%% of available system memory (%.2f GB) -- skipping this tier entirely.\n",
      format(N, big.mark = ","), r_bytes_needed / 1e9, r_free / 1e9
    ))
    add_result(list(size = N, workload = NA_character_, window = NA_integer_,
                     engine = "(tier skipped)", native = FALSE,
                     median_s = NA_real_, iterations = 0L,
                     skipped_reason = "estimated R-side memory exceeds available system memory"))
    next
  }

  gen_ok <- TRUE
  x <- tryCatch({
    set.seed(20260729 + size_idx)
    rnorm(N)
  }, error = function(e) {
    cat(sprintf("[skip] size=%s: R-side data generation failed: %s\n", format(N, big.mark = ","), conditionMessage(e)))
    gen_ok <<- FALSE
    NULL
  })

  if (!gen_ok) {
    add_result(list(size = N, workload = NA_character_, window = NA_integer_,
                     engine = "(tier skipped)", native = FALSE,
                     median_s = NA_real_, iterations = 0L,
                     skipped_reason = "R-side rnorm() allocation failed"))
    next
  }

  df <- data.frame(x = x)

  # --- 5a. CPU reference sweep ---
  cat(sprintf("\n=== CPU reference sweep (data.table::froll* et al.), size=%s ===\n", format(N, big.mark = ",")))

  for (wl in workloads) {
    eng <- engines[[wl]]
    for (w in window_sizes) {
      bt <- tryCatch(
        bench_time(function() eng$run(x, w), max_iterations = max_iter),
        error = function(e) {
          cat(sprintf("[skip] CPU %s window=%d size=%s: %s\n", wl, w, format(N, big.mark = ","), conditionMessage(e)))
          NULL
        }
      )
      if (is.null(bt)) {
        add_result(list(size = N, workload = wl, window = w, engine = eng$engine,
                         native = eng$native, median_s = NA_real_, iterations = 0L,
                         skipped_reason = "CPU reference run failed (see console)"))
        next
      }
      add_result(list(size = N, workload = wl, window = w, engine = eng$engine,
                       native = eng$native, median_s = bt$median_s, iterations = bt$iterations,
                       skipped_reason = NA_character_))
      cat(sprintf(
        "  %-8s window=%-5d engine=%-40s median=%.6fs (n=%d)\n",
        wl, w, eng$engine, bt$median_s, bt$iterations
      ))
    }
  }

  # --- 5b. cuplyr GPU probe ---
  cat(sprintf("\n=== cuplyr rolling-window probe, size=%s ===\n", format(N, big.mark = ",")))

  gpu_bytes_needed <- estimate_gpu_bytes_needed(N)
  gpu_free <- gpu_free_bytes()

  gpu_skip_reason <- NULL
  if (!cuplyr_ready) {
    gpu_skip_reason <- "cuplyr not loaded / no GPU"
  } else if (!is.na(gpu_free) && gpu_bytes_needed > gpu_free * 0.85) {
    gpu_skip_reason <- sprintf(
      "estimated GPU memory need (%.2f GB) exceeds 85%% of free GPU memory (%.2f GB)",
      gpu_bytes_needed / 1e9, gpu_free / 1e9
    )
  }

  gpu_df <- NULL
  if (is.null(gpu_skip_reason)) {
    gpu_df <- tryCatch(tbl_gpu(df, lazy = FALSE), error = function(e) {
      gpu_skip_reason <<- sprintf("GPU upload failed: %s", conditionMessage(e))
      NULL
    })
  }

  if (!is.null(gpu_skip_reason)) {
    cat(sprintf("[skip] cuplyr GPU probe, size=%s: %s -- skipping.\n", format(N, big.mark = ","), gpu_skip_reason))
    for (wl in workloads) {
      for (w in window_sizes) {
        add_result(list(size = N, workload = wl, window = w, engine = "cuplyr (GPU)",
                         native = FALSE, median_s = NA_real_, iterations = 0L,
                         gpu_resident_median_s = NA_real_, e2e_median_s = NA_real_,
                         skipped_reason = gpu_skip_reason))
      }
    }
  } else {
    for (wl in workloads) {
      for (w in window_sizes) {
        fn_name <- paste0("froll", wl)
        expr <- rlang::call2(fn_name, rlang::sym("x"), w, .ns = "data.table")

        probe_ok <- tryCatch({
          probe_result <- gpu_df |> dplyr::mutate(roll = !!expr) |> collect()
          is.data.frame(probe_result) && "roll" %in% names(probe_result)
        }, error = function(e) FALSE)

        if (!probe_ok) {
          cat(sprintf(
            "cuplyr: rolling %s (window=%d, size=%s) not native -- skipping (mutate() rejected the call, e.g. roll_median()'s cudf gap)\n",
            wl, w, format(N, big.mark = ",")
          ))
          add_result(list(size = N, workload = wl, window = w, engine = "cuplyr (GPU)",
                           native = FALSE, median_s = NA_real_, iterations = 0L,
                           gpu_resident_median_s = NA_real_, e2e_median_s = NA_real_,
                           skipped_reason = "mutate() rejected the rolling call (not GPU-native, e.g. roll_median())"))
          next
        }

        bt <- tryCatch(
          bench_time_two_phase(
            build_fn = function() gpu_df |> dplyr::mutate(roll = !!expr),
            collect_fn = function(obj) collect(obj),
            max_iterations = max_iter,
            after_each = function() gpu_gc(verbose = FALSE, aggressive = FALSE)
          ),
          error = function(e) {
            cat(sprintf(
              "[skip] cuplyr %s window=%d size=%s: %s\n",
              wl, w, format(N, big.mark = ","), conditionMessage(e)
            ))
            NULL
          }
        )

        if (is.null(bt)) {
          add_result(list(size = N, workload = wl, window = w, engine = "cuplyr (GPU)",
                           native = TRUE, median_s = NA_real_, iterations = 0L,
                           gpu_resident_median_s = NA_real_, e2e_median_s = NA_real_,
                           skipped_reason = "timed run failed (see console, likely OOM)"))
          next
        }

        cat(sprintf(
          "cuplyr: rolling %s (window=%d, size=%s) NATIVE -- gpu_resident=%.6fs e2e=%.6fs (n=%d)\n",
          wl, w, format(N, big.mark = ","), bt$resident_median_s, bt$e2e_median_s, bt$iterations
        ))
        add_result(list(size = N, workload = wl, window = w, engine = "cuplyr (GPU)",
                         native = TRUE, median_s = bt$e2e_median_s, iterations = bt$iterations,
                         gpu_resident_median_s = bt$resident_median_s, e2e_median_s = bt$e2e_median_s,
                         skipped_reason = NA_character_))
      }
    }
  }

  rm(gpu_df, x, df)
  if (cuplyr_ready) gpu_gc(verbose = FALSE, aggressive = TRUE)
  gc(verbose = FALSE, full = TRUE)
}

# =============================================================================
# 6. Tidy printed table + JSON dump
# =============================================================================

results_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    size = r$size,
    workload = r$workload,
    window = r$window,
    engine = r$engine,
    native = r$native,
    median_s = r$median_s,
    gpu_resident_median_s = if (!is.null(r$gpu_resident_median_s)) r$gpu_resident_median_s else NA_real_,
    e2e_median_s = if (!is.null(r$e2e_median_s)) r$e2e_median_s else NA_real_,
    iterations = r$iterations,
    skipped_reason = if (!is.null(r$skipped_reason)) r$skipped_reason else NA_character_,
    stringsAsFactors = FALSE
  )
}))
rownames(results_df) <- NULL

cat("\n=== Rolling-window benchmark reference table (all size tiers) ===\n")
print(results_df, row.names = FALSE)

if (requireNamespace("jsonlite", quietly = TRUE)) {
  out_path <- "scratchpad/rolling_reference.json"
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

  dump <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
    sizes = sizes,
    seed_base = 20260729,
    window_sizes = window_sizes,
    workloads = workloads,
    results = results
  )

  jsonlite::write_json(dump, out_path, auto_unbox = TRUE, pretty = TRUE, na = "null")
  cat(sprintf("\n[output] machine-readable dump written to %s\n", out_path))
} else {
  cat("\n[output] jsonlite unavailable -- skipped JSON dump.\n")
}

cat("\nDone.\n")
