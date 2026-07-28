# Benchmark: rolling-window operations reference (CPU baseline for a future
# cuplyr GPU-native rolling kernel)
#
# Spec: scratchpad/todo.md, section "Rolling-window benchmark target"
# Source workload design: Roald Arboel's rolling-window benchmark post
#   https://roald-arboel.com/blog/posts/2024/10/03/Benchmark-rolling/
#
# This script is a STANDALONE reference generator, not part of the
# tests/testthat perf-regression gate (CUPLYR_PERF): it measures CPU rolling
# implementations only (data.table::froll*, with a documented fallback chain
# for any froll* function missing from the installed data.table version) so
# a later "cuplyr rolling vs best CPU" report has an honest, locally-measured
# baseline (the blog's own numbers were measured on a different machine).
#
# Workloads: rolling mean / median / min / max / sum / sd
# Window sizes: 11, 101, 1001
# Data: 1,000,000 iid Gaussian doubles (set.seed'd for reproducibility)
# Semantics (verified below, not assumed): right-aligned, output length ==
# input length, leading (window - 1) positions are NA, and ANY NA inside a
# window poisons that window's output to NA (data.table's own froll* default:
# align = "right", fill = NA, na.rm = FALSE).
#
# cuplyr section: rolling windows are NOT yet GPU-native (see
# scratchpad/todo.md's "Rolling-window benchmark target" plan item). This
# script probes that via a real mutate() call and detects the hard error
# rather than timing dplyr's own CPU fallback as if it were a GPU number --
# mutate() has no CPU fallback (unlike filter()), so an unrecognized rolling
# call is a hard `stop()`, which is exactly the signal used to skip cleanly.
# Once a native kernel lands, the same probe should succeed and the timing
# loop below will "light up" automatically with no changes needed elsewhere
# in this file.

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
# 3. Timing harness
# =============================================================================
#
# Mirrors the perf_time() idiom in tests/testthat/helper-perf.R (median of
# several timed runs, untimed warmup first) but is intentionally standalone:
# this script is not gated behind CUPLYR_PERF and has no GPU-clock ambient
# calibration -- the reference section here is pure CPU, single-machine,
# single-session.
#
# "Fewer for very slow cases": the first timed run also acts as a speed
# probe; slow workloads (frollapply-style fallbacks, huge windows) get fewer
# repeats so the whole script still finishes quickly, while fast native
# froll* calls get the full microbenchmark-style 11 repeats.

bench_time <- function(fn, max_iterations = 11L) {
  fn()  # untimed warmup

  t0 <- proc.time()[["elapsed"]]
  fn()
  first <- proc.time()[["elapsed"]] - t0

  iterations <- if (first > 2) 3L else if (first > 0.5) 5L else max_iterations

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

# =============================================================================
# 4. Data
# =============================================================================

set.seed(20260729)
N <- 1e6
x <- rnorm(N)                 # plain vector -- what the CPU reference times
df <- data.frame(x = x)        # 1-column data.frame -- for the cuplyr path

window_sizes <- c(11L, 101L, 1001L)

cat(sprintf(
  "\n[data] %s Gaussian doubles (set.seed(20260729)), window sizes: %s\n",
  format(N, big.mark = ","), paste(window_sizes, collapse = ", ")
))

# =============================================================================
# 5. Reference timing sweep (CPU only)
# =============================================================================

cat("\n=== Reference timing sweep (CPU, data.table::froll* et al.) ===\n")

results <- list()

for (wl in workloads) {
  eng <- engines[[wl]]
  for (w in window_sizes) {
    bt <- bench_time(function() eng$run(x, w))
    results[[length(results) + 1]] <- list(
      workload = wl,
      window = w,
      engine = eng$engine,
      native = eng$native,
      median_s = bt$median_s,
      iterations = bt$iterations
    )
    cat(sprintf(
      "  %-8s window=%-5d engine=%-40s median=%.6fs (n=%d)\n",
      wl, w, eng$engine, bt$median_s, bt$iterations
    ))
  }
}

# =============================================================================
# 6. cuplyr native-support probe (placeholder until the GPU kernel lands)
# =============================================================================
#
# Deliberately does NOT time dplyr's/data.table's own CPU path as if it were
# a GPU number. mutate() has no CPU fallback (see CLAUDE.md's "Mutate
# Parsing" note): an unrecognized expression shape is a hard `stop()`, which
# is exactly the signal this probe uses to detect "not native yet". Once a
# rolling kernel is wired into the mutate() window-function registry, this
# same probe call should succeed and this loop will time it for real with no
# changes needed elsewhere in the script.

cat("\n=== cuplyr rolling-window support probe ===\n")

cuplyr_ready <- FALSE
gpu_df <- NULL

cuplyr_ready <- tryCatch({
  suppressMessages(devtools::load_all(quiet = TRUE))
  isTRUE(has_gpu())
}, error = function(e) {
  cat(sprintf("[cuplyr] could not load package / no GPU: %s\n", conditionMessage(e)))
  FALSE
})

if (cuplyr_ready) {
  gpu_df <- tbl_gpu(df, lazy = FALSE)
}

probe_rolling_native <- function(workload, window) {
  if (!cuplyr_ready) {
    return(list(native = FALSE, reason = "cuplyr not loaded / no GPU"))
  }
  fn_name <- paste0("froll", workload)
  expr <- rlang::call2(fn_name, rlang::sym("x"), window, .ns = "data.table")

  ok <- tryCatch({
    result <- gpu_df |>
      dplyr::mutate(roll = !!expr) |>
      collect()
    is.data.frame(result) && "roll" %in% names(result)
  }, error = function(e) FALSE)

  list(native = isTRUE(ok), reason = if (!isTRUE(ok)) "mutate() rejected the rolling call (hard error, as expected pre-kernel)" else NULL)
}

for (wl in workloads) {
  for (w in window_sizes) {
    probe <- probe_rolling_native(wl, w)

    if (!probe$native) {
      cat(sprintf("cuplyr: rolling %s (window=%d) not yet native -- skipping (%s)\n", wl, w, probe$reason))
      results[[length(results) + 1]] <- list(
        workload = wl, window = w, engine = "cuplyr (GPU)", native = FALSE,
        median_s = NA_real_, iterations = 0L
      )
    } else {
      # Rolling landed natively: time it for real, same harness as section 5.
      bt <- bench_time(function() {
        gpu_df |> dplyr::mutate(roll = !!rlang::call2(paste0("froll", wl), rlang::sym("x"), w, .ns = "data.table")) |> collect()
      })
      cat(sprintf(
        "cuplyr: rolling %s (window=%d) NATIVE -- median=%.6fs (n=%d)\n",
        wl, w, bt$median_s, bt$iterations
      ))
      results[[length(results) + 1]] <- list(
        workload = wl, window = w, engine = "cuplyr (GPU)", native = TRUE,
        median_s = bt$median_s, iterations = bt$iterations
      )
    }
  }
}

# =============================================================================
# 7. Tidy printed table + JSON dump
# =============================================================================

results_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    workload = r$workload,
    window = r$window,
    engine = r$engine,
    native = r$native,
    median_s = r$median_s,
    iterations = r$iterations,
    stringsAsFactors = FALSE
  )
}))
rownames(results_df) <- NULL

cat("\n=== Rolling-window benchmark reference table ===\n")
print(results_df, row.names = FALSE)

if (requireNamespace("jsonlite", quietly = TRUE)) {
  out_path <- "scratchpad/rolling_reference.json"
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

  dump <- list(
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
    n_rows = N,
    seed = 20260729,
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
