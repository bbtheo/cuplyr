# Rolling-window benchmark with cuplyr added (F54): a redo of
# https://roald-arboel.com/blog/posts/2024/10/03/Benchmark-rolling/
#
#   tools/r 'source("bench/rolling-window.R")'
#
# Variables set before source() override the defaults:
#   LIB          library holding the third-party packages (bench/install-deps.R)
#   OUT_DIR      where CSVs, PNGs and tables.md are written
#   N_ROWS       rows of the main data set (post: 1e6)
#   WIDTHS       window widths (post: 11, 101, 1001)
#   TIMES        samples per cell (post: 20); TIMES_CUSTOM for frollapply/rollapply (post: 5)
#   CELL_BUDGET  seconds of timed work allowed per cell; `times` is cut to fit
#                (default 20; the full run in the report uses the defaults)
#   PROBE_LIMIT  if the first (cold) call of a cell takes longer than this many
#                seconds, that call is the cell's only sample (rw_plan_cell())
#   GROUP_ROWS, N_GROUPS, EXT_ROWS, EXT_TIMES   extension sizes
#   STATS        subset of statistics to run
#   SKIP_EXT     TRUE to skip the extension

`%||%` <- function(a, b) if (is.null(a)) b else a
cfg <- function(name, default) if (exists(name, envir = globalenv(), inherits = FALSE)) get(name, envir = globalenv()) else default

LIB <- cfg("LIB", "/home/theo/cuplr/scratchpad/benchmarks/rlib")
OUT_DIR <- cfg("OUT_DIR", "/home/theo/cuplr/scratchpad/benchmarks/rolling")
N_ROWS <- cfg("N_ROWS", 1e6)
WIDTHS <- cfg("WIDTHS", c(11, 101, 1001))
TIMES <- cfg("TIMES", 20)
TIMES_CUSTOM <- cfg("TIMES_CUSTOM", 5)
CELL_BUDGET <- cfg("CELL_BUDGET", 20)
PROBE_LIMIT <- cfg("PROBE_LIMIT", 60)
GROUP_ROWS <- cfg("GROUP_ROWS", 1e6)
N_GROUPS <- cfg("N_GROUPS", 1e3)
GROUP_WIDTH <- cfg("GROUP_WIDTH", 101)
EXT_ROWS <- cfg("EXT_ROWS", 1e7)
EXT_TIMES <- cfg("EXT_TIMES", 10)
STATS <- cfg("STATS", c("mean", "median", "min", "max", "sum", "sd", "custom"))
SKIP_EXT <- cfg("SKIP_EXT", FALSE)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
if (dir.exists(LIB)) .libPaths(c(LIB, .libPaths()))
script_dir <- if (file.exists("bench/rolling-lib.R")) "bench" else dirname(sys.frame(1)$ofile %||% "bench/rolling-window.R")
source(file.path(script_dir, "rolling-lib.R"))

t_start <- Sys.time()
say <- function(...) message(format(Sys.time(), "%H:%M:%S"), " ", ...)
has_pkg <- function(p) requireNamespace(p, quietly = TRUE)
for (p in c("data.table", "microbenchmark")) if (!has_pkg(p)) stop("required package missing: ", p)
if (!requireNamespace("cuplyr", quietly = TRUE) && !"package:cuplyr" %in% search()) stop("cuplyr is not loaded")
have_gpu <- cuplyr::has_gpu()
if (!have_gpu) stop("no GPU")

# ---- one timed cell ---------------------------------------------------------
# `call` is the zero-argument timed closure; `extract` turns its result into a
# numeric vector for the correctness check (outside timing). The first (cold)
# call is a probe: its duration decides, through rw_plan_cell(), how many
# samples fit in CELL_BUDGET, and its output feeds the correctness check. The
# probe is not a recorded sample unless it alone exceeds PROBE_LIMIT. `note`
# says exactly which calls were recorded.
run_cell <- function(call, extract, times, ref = NULL, expect_fallback = FALSE) {
  note <- character()
  gc(verbose = FALSE)
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
  plan <- rw_plan_cell(probe_s, times, CELL_BUDGET, PROBE_LIMIT)
  if (plan$probe_is_sample) {
    ms <- probe_s * 1e3
  } else {
    ms <- rw_time(call, plan$k, warmup = plan$warmup)
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

# ---- thread settings as found (before any control changes them) -----------------
thread_info <- data.frame(
  item = c("OMP_NUM_THREADS (env)", "data.table getDTthreads()", "RcppRoll roll_threads()",
           "RcppParallel defaultNumThreads()"),
  version = c(if (nzchar(Sys.getenv("OMP_NUM_THREADS"))) Sys.getenv("OMP_NUM_THREADS") else "unset",
              data.table::getDTthreads(),
              if (has_pkg("RcppRoll")) RcppRoll::roll_threads() else NA,
              if (has_pkg("RcppParallel")) RcppParallel::defaultNumThreads() else NA),
  stringsAsFactors = FALSE)

# ---- main experiment ----------------------------------------------------
df <- rw_data(N_ROWS)
x <- df$x
gt <- cuplyr::tbl_gpu(df)
fns <- rw_functions()
timing_rows <- list()
corr_rows <- list()
add_cell <- function(stat, fn, width, n_rows, res, kind = "main") {
  timing_rows[[length(timing_rows) + 1]] <<- data.frame(
    stat = stat, fn = fn, width = width, n_rows = n_rows, sample = seq_along(res$ms),
    ms = res$ms, times = res$times, note = res$note, stringsAsFactors = FALSE)
  if (!is.null(res$cmp)) {
    corr_rows[[length(corr_rows) + 1]] <<- data.frame(
      stat = stat, fn = fn, width = width, ok = res$cmp$ok,
      n_na_mismatch = res$cmp$n_na_mismatch, max_abs_diff = res$cmp$max_abs_diff,
      note = res$note %||% "", stringsAsFactors = FALSE)
  }
}
flush_csv <- function() {
  if (length(timing_rows)) utils::write.csv(do.call(rbind, timing_rows), file.path(OUT_DIR, "timings.csv"), row.names = FALSE)
  if (length(corr_rows)) utils::write.csv(do.call(rbind, corr_rows), file.path(OUT_DIR, "correctness.csv"), row.names = FALSE)
}

for (stat in STATS) {
  say("== statistic: ", stat)
  stat_fns <- fns[[stat]]
  for (width in WIDTHS) {
    say("  width ", width)
    ref <- rw_reference(stat, x, width)
    times <- if (stat == "custom") TIMES_CUSTOM else TIMES

    for (fn_name in names(stat_fns)) {
      f <- stat_fns[[fn_name]]
      say("    ", fn_name)
      cell <- function() run_cell(function() f(x, width), identity, times, ref)
      add_cell(stat, fn_name, width, N_ROWS, safe_cell(cell))
      # Single-thread control for every entry whose package can use threads.
      pkg <- rw_threaded_pkg(fn_name)
      if (!is.na(pkg) && has_pkg(pkg)) {
        label1 <- rw_one_thread_label(fn_name)
        say("    ", label1)
        add_cell(stat, label1, width, N_ROWS, safe_cell(function() with_one_thread(pkg, cell)))
      }
    }

    if (stat != "custom") {
      variants <- list(list(label = "cuplyr", alias = FALSE))
      if (stat != "median") variants[[2]] <- list(label = "cuplyr (froll alias)", alias = TRUE)
      for (v in variants) {
        label <- if (stat == "median") "cuplyr (CPU fallback)" else v$label
        say("    ", label)
        cf <- rw_cuplyr_fun(stat, alias = v$alias)
        res <- safe_cell(function()
          run_cell(function() cf(gt, width), cuplyr_extract, times, ref, expect_fallback = stat == "median"))
        add_cell(stat, label, width, N_ROWS, res)
      }
    }
    flush_csv()
  }
}

# ---- per-call overhead: the same cuplyr calls on 1e3 rows -------------------------
# Kernel time is negligible on 1e3 rows, so these are the fixed cost of one
# eager mutate() + compute() + fence. A plain arithmetic mutate and a cumsum
# window mutate are timed next to the rolling ones to separate the window
# chain's extra cost from the bare mutate cost. Same harness as everything
# else (gc-separated, fenced).
say("== per-call overhead (1e3 rows)")
gt_small <- cuplyr::tbl_gpu(rw_data(1e3))
for (stat in intersect(STATS, c("mean", "sum", "min", "max", "sd"))) {
  cf <- rw_cuplyr_fun(stat)
  ms <- rw_time(function() cf(gt_small, 101), 50, warmup = 5)
  add_cell(stat, "cuplyr (1e3-row overhead floor)", 101, 1e3,
           list(ms = ms, times = length(ms), note = "same call on 1e3 rows; kernel time negligible", cmp = NULL))
}
overhead_calls <- list(
  "plain mutate(y = x + 1)" = function(g) dplyr::mutate(g, y = x + 1),
  "window mutate(r = roll_mean(x, 101))" = function(g) dplyr::mutate(g, r = roll_mean(x, 101)),
  "window mutate(r = cumsum(x))" = function(g) dplyr::mutate(g, r = cumsum(x))
)
overhead_fn <- function(f) function(g) { r <- dplyr::compute(f(g)); rw_gpu_finish(r); r }
overhead_inputs <- list(list(g = gt_small, n = 1e3, times = 50), list(g = gt, n = N_ROWS, times = TIMES))
for (inp in overhead_inputs) {
  for (nm in names(overhead_calls)) {
    say("    ", nm, " on ", inp$n, " rows")
    of <- overhead_fn(overhead_calls[[nm]])
    ms <- rw_time(function() of(inp$g), inp$times, warmup = 5)
    add_cell("overhead", nm, NA, inp$n,
             list(ms = ms, times = length(ms), note = "", cmp = NULL))
  }
}
flush_csv()

# ---- extension ------------------------------------------------------------------
ext_rows <- list()
if (!SKIP_EXT) {
  mem_of <- function(call) {
    gc(verbose = FALSE)
    m <- cuplyr:::gpu_alloc_tracking(call())
    list(peak = m$peak_bytes, held = m$held_bytes)
  }
  add_ext <- function(experiment, fn, width, n_rows, res, mem = NULL, ok = NA) {
    ext_rows[[length(ext_rows) + 1]] <<- data.frame(
      experiment = experiment, fn = fn, width = width, n_rows = n_rows,
      median_ms = stats::median(res$ms), q25_ms = unname(stats::quantile(res$ms, .25)),
      q75_ms = unname(stats::quantile(res$ms, .75)), min_ms = min(res$ms), max_ms = max(res$ms),
      times = res$times, peak_gpu_bytes = mem$peak %||% NA_real_, held_gpu_bytes_after_call = mem$held %||% NA_real_,
      correct = ok, note = res$note, stringsAsFactors = FALSE)
    utils::write.csv(do.call(rbind, ext_rows), file.path(OUT_DIR, "extension.csv"), row.names = FALSE)
  }

  # (a) grouped rolling mean
  say("== extension (a): grouped, ", GROUP_ROWS, " rows, ", N_GROUPS, " groups, width ", GROUP_WIDTH)
  dfg <- rw_grouped_data(GROUP_ROWS, N_GROUPS)
  setup <- rw_grouped_setup(dfg, GROUP_WIDTH)
  # data.table's own result is the reference
  g_ref <- setup$data.table$extract(setup$data.table$run())
  grouped_label <- function(nm) switch(nm, slider = "slider (grouped_df)", data.table = "data.table (by = g)", nm)
  for (nm in names(setup)) {
    say("    ", nm)
    e <- setup[[nm]]
    res <- safe_cell(function() run_cell(e$run, e$extract, EXT_TIMES, g_ref, expect_fallback = FALSE))
    mem <- if (nm == "cuplyr") mem_of(e$run) else NULL
    add_ext("grouped mean", grouped_label(nm), GROUP_WIDTH, GROUP_ROWS, res, mem, res$cmp$ok)
    if (nm == "data.table") {
      say("    data.table (1 thread)")
      res <- safe_cell(function() with_one_thread("data.table", function()
        run_cell(e$run, e$extract, EXT_TIMES, g_ref, expect_fallback = FALSE)))
      add_ext("grouped mean", "data.table (by = g, 1 thread)", GROUP_WIDTH, GROUP_ROWS, res, NULL, res$cmp$ok)
    }
  }
  rm(setup, dfg, g_ref); gc(verbose = FALSE)

  # (b) 1e7 rows, ungrouped mean
  say("== extension (b): ", EXT_ROWS, " rows, ungrouped mean")
  dfb <- rw_data(EXT_ROWS)
  xb <- dfb$x
  gtb <- cuplyr::tbl_gpu(dfb)
  engines <- list(
    cuplyr = NULL,
    data.table = function(x, n) data.table::frollmean(x, n, align = "right"),
    roll = if (has_pkg("roll")) function(x, n) roll::roll_mean(x, n),
    RcppRoll = if (has_pkg("RcppRoll")) function(x, n) RcppRoll::roll_meanr(x, n, fill = NA)
  )
  for (width in WIDTHS) {
    refb <- data.table::frollmean(xb, width, align = "right")
    for (nm in names(engines)) {
      say("    ", nm, " width ", width)
      if (nm == "cuplyr") {
        cf <- rw_cuplyr_fun("mean")
        call <- function() cf(gtb, width)
        extract <- cuplyr_extract
      } else {
        f <- engines[[nm]]
        if (is.null(f)) next
        call <- function() f(xb, width)
        extract <- identity
      }
      cell <- function() run_cell(call, extract, EXT_TIMES, refb)
      res <- safe_cell(cell)
      mem <- if (nm == "cuplyr") mem_of(call) else NULL
      add_ext("1e7 ungrouped mean", nm, width, EXT_ROWS, res, mem, res$cmp$ok %||% NA)
      if (nm != "cuplyr") {
        say("    ", rw_one_thread_label(nm), " width ", width)
        res <- safe_cell(function() with_one_thread(rw_threaded_pkg(nm), cell))
        add_ext("1e7 ungrouped mean", rw_one_thread_label(nm), width, EXT_ROWS, res, NULL, res$cmp$ok %||% NA)
      }
    }
    rm(refb); gc(verbose = FALSE)
  }
  rm(gtb, dfb, xb); gc(verbose = FALSE)
}

# ---- memory: what one eager call holds, and the no-gc() finding -------------------
# At EXT_ROWS: bytes allocated / held right after one call (the result is still
# referenced, so held = result + whatever intermediates have not been
# finalized yet), peak, and held after a gc() (the result alone), for a
# rolling mean, a plain arithmetic mutate and a cumsum window mutate. Then 20
# back-to-back rolling calls without gc(): the finding reported as F1.
mem_finding <- NULL
mem_breakdown <- NULL
if (!SKIP_EXT) {
  say("== memory (held bytes after one call; back-to-back calls without gc())")
  dfm <- rw_data(EXT_ROWS)
  gtm <- cuplyr::tbl_gpu(dfm)
  cf <- rw_cuplyr_fun("mean")
  mem_calls <- list(
    "window mutate(r = roll_mean(x, 101))" = function() cf(gtm, 101),
    "plain mutate(y = x + 1)" = function() overhead_fn(overhead_calls[["plain mutate(y = x + 1)"]])(gtm),
    "window mutate(r = cumsum(x))" = function() overhead_fn(overhead_calls[["window mutate(r = cumsum(x))"]])(gtm)
  )
  mem_breakdown <- do.call(rbind, lapply(names(mem_calls), function(nm) {
    gc(verbose = FALSE)
    one <- cuplyr:::gpu_alloc_tracking(mem_calls[[nm]]())
    row <- data.frame(call = nm, n_rows = EXT_ROWS, x_column_bytes = 8 * EXT_ROWS,
                      allocated_bytes = one$allocated_bytes, held_after_call_bytes = one$held_bytes,
                      peak_bytes = one$peak_bytes, stringsAsFactors = FALSE)
    rm(one); gc(verbose = FALSE)
    after_gc <- cuplyr:::gpu_alloc_tracking({ r <- mem_calls[[nm]](); gc(verbose = FALSE); r })
    row$held_after_gc_bytes <- after_gc$held_bytes
    rm(after_gc); gc(verbose = FALSE)
    row
  }))
  utils::write.csv(mem_breakdown, file.path(OUT_DIR, "memory-breakdown.csv"), row.names = FALSE)

  held_one <- mem_breakdown$held_after_call_bytes[1]
  peak_one <- mem_breakdown$peak_bytes[1]
  result_bytes <- mem_breakdown$held_after_gc_bytes[1]
  oom_at <- NA_integer_; err <- ""
  for (i in 1:20) {
    r <- tryCatch(cf(gtm, 101), error = function(e) { err <<- conditionMessage(e); NULL })
    if (is.null(r)) { oom_at <- i; break }
    rm(r) # the result is dropped; only unfinalized intermediates linger
  }
  gc(verbose = FALSE)
  mem_finding <- data.frame(
    n_rows = EXT_ROWS, held_after_one_call_bytes = held_one, peak_one_call_bytes = peak_one,
    result_table_bytes = result_bytes, failed_at_call = oom_at, error = substr(err, 1, 200))
  utils::write.csv(mem_finding, file.path(OUT_DIR, "memory-finding.csv"), row.names = FALSE)
  rm(gtm, dfm); gc(verbose = FALSE)
}

# ---- versions & hardware -----------------------------------------------------------
pkgs <- c("cuplyr", "data.table", "roll", "RollingWindow", "runstats", "slider", "RcppRoll", "zoo",
          "runner", "microbenchmark", "ggdist", "paletteer", "ggtext", "ggplot2", "scales", "patchwork",
          "dplyr", "RcppParallel")
ver <- data.frame(
  item = c("R", pkgs),
  version = c(paste(R.version$major, R.version$minor, sep = "."),
              vapply(pkgs, function(p) if (has_pkg(p)) as.character(utils::packageVersion(p)) else "not installed", "")),
  stringsAsFactors = FALSE)
cpu <- tryCatch(sub(".*: ", "", grep("model name", readLines("/proc/cpuinfo"), value = TRUE)[1]), error = function(e) NA)
gpu <- tryCatch(cuplyr::gpu_details(), error = function(e) NULL)
hw <- data.frame(
  item = c("CPU", "CPU logical cores", "GPU"),
  version = c(cpu, parallel::detectCores(),
              if (!is.null(gpu)) paste(utils::capture.output(print(gpu)), collapse = " ") else NA),
  stringsAsFactors = FALSE)
utils::write.csv(rbind(ver, hw, thread_info), file.path(OUT_DIR, "versions.csv"), row.names = FALSE)
flush_csv()

# ---- plots ---------------------------------------------------------------------
timings <- do.call(rbind, timing_rows)
timings <- timings[!is.na(timings$ms), ]
main <- timings[timings$n_rows == N_ROWS & timings$stat != "overhead", ]
width_cols <- c("#1b9e77", "#d95f02", "#7570b3")
if (has_pkg("paletteer")) {
  width_cols <- tryCatch(as.character(paletteer::paletteer_d("RColorBrewer::Dark2"))[1:3], error = function(e) width_cols)
}
lab_ms <- function(v) ifelse(v >= 1000, sprintf("%.1f s", v / 1000), sprintf("%.3g ms", v))

# The CPU-fallback entry downloads, loops in R and re-uploads; it is not
# "computation on data already on the GPU" and is labelled as such.
disp_label <- function(fn) ifelse(fn == "cuplyr (CPU fallback)",
                                  "cuplyr (CPU fallback; includes\ndownload, R loop, re-upload)", fn)

plot_stat <- function(stat) {
  d <- main[main$stat == stat, ]
  if (!nrow(d)) return(invisible(NULL))
  d$fn <- disp_label(d$fn)
  med101 <- tapply(d$ms[d$width == 101], d$fn[d$width == 101], stats::median)
  ord <- names(sort(med101, decreasing = TRUE))
  ord <- c(setdiff(unique(d$fn), ord), ord)
  d$fn <- factor(d$fn, levels = ord)
  d$width_f <- factor(d$width)
  lab <- data.frame(fn = factor(names(med101), levels = ord), ms = as.numeric(med101),
                    label = lab_ms(as.numeric(med101)))
  has_fb <- any(grepl("CPU fallback", ord))
  sub <- paste0("point = median, bars = 50% / 95% of samples; label = median at width 101\n",
                "(1 thread) = same call with that package's thread count set to 1\n",
                "cuplyr: computation only, data already on the GPU, no upload/download",
                if (has_fb) "\ncuplyr (CPU fallback) is NOT computation-only: it includes download, R loop and re-upload" else "")
  p <- ggplot2::ggplot(d, ggplot2::aes(x = ms, y = fn, colour = width_f))
  if (has_pkg("ggdist")) {
    p <- p + ggdist::stat_pointinterval(position = ggplot2::position_dodge(width = 0.6),
                                        .width = c(0.5, 0.95), point_size = 1.8)
  } else {
    p <- p + ggplot2::geom_boxplot(position = ggplot2::position_dodge(width = 0.6), width = 0.5)
  }
  p <- p +
    ggplot2::geom_text(data = lab, ggplot2::aes(x = ms, y = fn, label = label), inherit.aes = FALSE,
                       hjust = -0.35, size = 3) +
    ggplot2::scale_x_log10(labels = scales::label_number(drop0trailing = TRUE)) +
    ggplot2::scale_colour_manual(values = width_cols, name = "window width") +
    ggplot2::labs(x = "time per call, ms (log10)", y = NULL,
                  title = paste0("Rolling ", stat, ", n = ", format(N_ROWS, scientific = FALSE, big.mark = ",")),
                  subtitle = sub) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(legend.position = "bottom", plot.subtitle = ggplot2::element_text(size = 8),
                   panel.grid.minor = ggplot2::element_blank())
  ggplot2::ggsave(file.path(OUT_DIR, paste0("plot-", stat, ".png")), p,
                  width = 9, height = 2 + 0.55 * length(ord) + 0.2 * has_fb, dpi = 130, bg = "white")
}
for (s in unique(main$stat)) tryCatch(plot_stat(s), error = function(e) message("plot ", s, " failed: ", conditionMessage(e)))

if (length(ext_rows)) {
  ext <- do.call(rbind, ext_rows)
  ext_sub <- "computation only, data already on the GPU (cuplyr); no upload/download. Bars/points = median; error bars = IQR; (1 thread) = threads set to 1"
  d1 <- ext[ext$experiment == "grouped mean", ]
  d2 <- ext[ext$experiment != "grouped mean", ]
  d2$one_thread <- grepl("1 thread", d2$fn)
  panels <- list()
  if (nrow(d1)) panels[[1]] <- ggplot2::ggplot(d1, ggplot2::aes(x = median_ms, y = stats::reorder(fn, -median_ms))) +
      ggplot2::geom_col(fill = width_cols[1]) +
      ggplot2::geom_errorbar(ggplot2::aes(xmin = q25_ms, xmax = q75_ms), width = 0.2, orientation = "y") +
      ggplot2::geom_text(ggplot2::aes(label = lab_ms(median_ms)), hjust = -0.1, size = 3) +
      ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.25))) +
      ggplot2::labs(x = "median ms (IQR bars)", y = NULL, title = paste0("Grouped rolling mean, ", format(GROUP_ROWS, scientific = FALSE, big.mark = ","), " rows, ", N_GROUPS, " groups, w = ", GROUP_WIDTH),
                    subtitle = ext_sub) +
      ggplot2::theme_minimal(base_size = 10) + ggplot2::theme(plot.subtitle = ggplot2::element_text(size = 7))
  if (nrow(d2)) panels[[length(panels) + 1]] <- ggplot2::ggplot(d2, ggplot2::aes(x = width, y = median_ms, colour = fn, group = fn, linetype = one_thread)) +
      ggplot2::geom_line() + ggplot2::geom_point(size = 2.5) +
      ggplot2::scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "dashed"), guide = "none") +
      ggplot2::scale_x_log10(breaks = unique(d2$width)) + ggplot2::scale_y_log10() +
      ggplot2::labs(x = "window width (log10)", y = "median ms (log10)", colour = NULL, title = paste0("Ungrouped rolling mean, ", format(EXT_ROWS, scientific = FALSE, big.mark = ","), " rows"),
                    subtitle = ext_sub) +
      ggplot2::theme_minimal(base_size = 10) + ggplot2::theme(legend.position = "bottom", plot.subtitle = ggplot2::element_text(size = 7))
  if (length(panels)) {
    p <- Reduce(`/`, panels)
    ggplot2::ggsave(file.path(OUT_DIR, "plot-extension.png"), p, width = 9, height = 3.4 * length(panels) + 0.5, dpi = 130, bg = "white")
  }
}

# ---- markdown tables (pasted into report.md) -----------------------------------------
md_table <- function(df) {
  df[] <- lapply(df, as.character)
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|"),
    apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |")))
}
fmt <- function(v) formatC(v, digits = 3, format = "g")
med_iqr <- function(v) sprintf("%s (%s-%s)", fmt(stats::median(v)), fmt(unname(stats::quantile(v, .25))), fmt(unname(stats::quantile(v, .75))))
tab_lines <- character()
for (stat in unique(main$stat)) {
  d <- main[main$stat == stat, ]
  cells <- stats::aggregate(ms ~ fn + width, d, function(v) c(med = stats::median(v), q1 = unname(stats::quantile(v, .25)), q3 = unname(stats::quantile(v, .75))))
  tn <- stats::aggregate(times ~ fn + width, d, function(v) v[1])
  cells$txt <- sprintf("%s (%s-%s) n=%d", fmt(cells$ms[, "med"]), fmt(cells$ms[, "q1"]), fmt(cells$ms[, "q3"]), tn$times[match(paste(cells$fn, cells$width), paste(tn$fn, tn$width))])
  wide <- stats::reshape(data.frame(fn = cells$fn, width = paste0("w=", cells$width), txt = cells$txt),
                         idvar = "fn", timevar = "width", direction = "wide")
  names(wide) <- sub("^txt\\.", "", names(wide))
  med101 <- cells$ms[cells$width == 101, "med"]; names(med101) <- cells$fn[cells$width == 101]
  wide <- wide[order(med101[wide$fn]), ]
  names(wide)[1] <- "function"
  tab_lines <- c(tab_lines, "", paste0("### ", stat, " (median ms (IQR) n=samples)"), "", md_table(wide))
}
writeLines(c("# Median tables", tab_lines), file.path(OUT_DIR, "tables-median.md"))

# Threads actually used: default vs 1 thread, width 101, per entry that has a
# 1-thread twin. "scales" compares the two medians; it is what the call did,
# not what the package's settings say.
default_threads <- function(fn) {
  pkg <- rw_threaded_pkg(fn)
  if (is.na(pkg)) return(NA_character_)
  switch(pkg, data.table = paste0(thread_info$version[2], " (setDTthreads)"),
         RcppRoll = paste0(thread_info$version[3], " (OpenMP)"),
         roll = paste0(thread_info$version[4], " (RcppParallel)"))
}
# one row per (experiment, statistic, entry) at width 101: median and IQR
thr_cells <- do.call(rbind, lapply(split(main[main$width == 101, ], list(main$stat[main$width == 101], main$fn[main$width == 101]), drop = TRUE),
  function(v) data.frame(where = "1e6 rows, w=101", statistic = v$stat[1], fn = v$fn[1], median_ms = stats::median(v$ms),
                         q25_ms = unname(stats::quantile(v$ms, .25)), q75_ms = unname(stats::quantile(v$ms, .75)),
                         stringsAsFactors = FALSE)))
if (length(ext_rows)) {
  e0 <- do.call(rbind, ext_rows)
  e0 <- e0[e0$width == 101, ]
  thr_cells <- rbind(thr_cells, data.frame(where = paste0(e0$experiment, ", w=101"), statistic = "mean", fn = e0$fn,
                                           median_ms = e0$median_ms, q25_ms = e0$q25_ms, q75_ms = e0$q75_ms,
                                           stringsAsFactors = FALSE))
}
cell_txt <- function(r) sprintf("%s (%s-%s)", fmt(r$median_ms), fmt(r$q25_ms), fmt(r$q75_ms))
thr_rows <- list()
for (i in which(grepl("1 thread\\)$", thr_cells$fn))) {
  b <- thr_cells[i, ]
  a <- thr_cells[thr_cells$where == b$where & thr_cells$statistic == b$statistic & thr_cells$fn == rw_base_label(b$fn), ]
  if (nrow(a) != 1L) next
  ratio <- b$median_ms / a$median_ms
  thr_rows[[length(thr_rows) + 1]] <- data.frame(
    experiment = b$where, statistic = b$statistic, entry = a$fn, default_threads = default_threads(a$fn),
    default_ms = cell_txt(a), one_thread_ms = cell_txt(b), ratio = sprintf("%.2f", ratio),
    scales = if (ratio >= 1.5) sprintf("yes (%.1fx slower at 1 thread)", ratio) else if (ratio <= 1.15) "no" else "weakly",
    stringsAsFactors = FALSE)
}
if (length(thr_rows)) writeLines(md_table(do.call(rbind, thr_rows)), file.path(OUT_DIR, "tables-threads.md"))

# Per-call cost: the overhead calls plus the per-statistic 1e3-row floors.
oh <- timings[timings$stat == "overhead" | timings$n_rows == 1e3, ]
if (nrow(oh)) {
  oh$call <- ifelse(oh$stat == "overhead", oh$fn, paste0(oh$fn, " [", oh$stat, "]"))
  oh_tab <- do.call(rbind, lapply(split(oh, list(oh$call, oh$n_rows), drop = TRUE), function(v)
    data.frame(call = v$call[1], rows = format(v$n_rows[1], scientific = FALSE, big.mark = ","),
               median_ms_iqr = med_iqr(v$ms), min_max_ms = sprintf("%s-%s", fmt(min(v$ms)), fmt(max(v$ms))),
               n = nrow(v), stringsAsFactors = FALSE)))
  oh_tab <- oh_tab[order(oh_tab$rows, oh_tab$call), ]
  writeLines(md_table(oh_tab), file.path(OUT_DIR, "tables-overhead.md"))
}
if (!is.null(mem_breakdown)) {
  mb <- mem_breakdown
  for (cn in setdiff(names(mb), "call")) mb[[cn]] <- format(mb[[cn]], scientific = FALSE, big.mark = ",")
  writeLines(md_table(mb), file.path(OUT_DIR, "tables-memory.md"))
}

if (length(corr_rows)) {
  cr <- do.call(rbind, corr_rows)
  bad <- cr[!cr$ok, ]
  cr_lines <- c(sprintf("%d of %d (function x width) cells match data.table (all.equal tol 1e-8, same NA positions).", sum(cr$ok), nrow(cr)), "")
  if (nrow(bad)) {
    bad$max_abs_diff <- fmt(bad$max_abs_diff)
    cr_lines <- c(cr_lines, "Mismatches and errors:", "", md_table(bad[, c("stat", "fn", "width", "n_na_mismatch", "max_abs_diff", "note")]))
  }
  writeLines(cr_lines, file.path(OUT_DIR, "tables-correctness.md"))
}
if (length(ext_rows)) {
  e <- do.call(rbind, ext_rows)
  e$median_ms <- fmt(e$median_ms); e$IQR <- paste0(fmt(e$q25_ms), "-", fmt(e$q75_ms))
  e$peak_gpu_MB <- ifelse(is.na(e$peak_gpu_bytes), "", sprintf("%.0f", e$peak_gpu_bytes / 1e6))
  e$held_after_MB <- ifelse(is.na(e$held_gpu_bytes_after_call), "", sprintf("%.0f", e$held_gpu_bytes_after_call / 1e6))
  writeLines(md_table(e[, c("experiment", "fn", "width", "n_rows", "median_ms", "IQR", "times", "peak_gpu_MB", "held_after_MB", "correct", "note")]),
             file.path(OUT_DIR, "tables-extension.md"))
}

# Every cell whose sample count was cut or that carries a note.
nt <- unique(timings[nzchar(timings$note) & timings$stat != "overhead" & timings$n_rows == N_ROWS,
                     c("stat", "fn", "width", "times", "note")])
if (nrow(nt)) writeLines(md_table(nt), file.path(OUT_DIR, "tables-cuts.md"))

say("done in ", format(round(difftime(Sys.time(), t_start, units = "mins"), 1)))
