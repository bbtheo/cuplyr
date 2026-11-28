#!/usr/bin/env Rscript
# Test entry point used by tools/test, tools/test-file and tools/perf.
# Always runs against pkgload::load_all() of the current worktree; nothing is
# ever installed. Call it through tools/gpu-run, never directly.
#
#   Rscript tools/run-tests.R suite [--out results.tsv] [--shards N]
#       [--dir tests/testthat] [--serial-list tools/test-serial.txt]
#       [--durations tools/test-durations.tsv] [--record-durations f.tsv]
#   Rscript tools/run-tests.R file <test-file>... [--out results.tsv]
#   Rscript tools/run-tests.R perf [--out results.tsv]
#
# --out writes one row per test_that() block: file, test, status
# (pass/fail/error/skip), then seconds (the test's wall time), file_seconds
# (its file's wall time) and shard (which process ran it). Exit status is 1 if
# any test failed or errored.
#
# suite: with --shards 1 every file runs in this process, in name order. With
# more (default: CUPLYR_TEST_SHARDS, else tools/run-tests-lib.R's default) this
# process only supervises: the files are dealt out to that many child R
# processes by their recorded durations (--durations), the children run at once
# inside this one GPU lock hold, then the files named in --serial-list (timing-
# and memory-sensitive ones) run in one more child after all of them have
# exited. A child that dies or stops early fails the run; its files that did
# not finish get an error row and its output is printed with the rest.
# --record-durations writes `file<TAB>seconds` for every file run: refresh
# tools/test-durations.tsv from a `--shards 1` run.
#
# `shard` is an internal mode: one child of a sharded suite run.

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(a, b) if (is.null(a)) b else a

flags <- c("--out", "--shards", "--dir", "--serial-list", "--durations",
           "--record-durations", "--label", "--events", "--rows-dir")
opt <- list()
positional <- character()
i <- 1L
while (i <= length(args)) {
  a <- args[[i]]
  if (a %in% flags) {
    if (i == length(args)) stop(a, " needs a value")
    opt[[a]] <- args[[i + 1L]]
    i <- i + 2L
  } else if (a == "--small-pool") {
    opt[[a]] <- TRUE
    i <- i + 1L
  } else {
    positional <- c(positional, a)
    i <- i + 1L
  }
}
mode <- positional[[1]]
targets <- positional[-1]
out <- opt[["--out"]]

if (nzchar(Sys.getenv("CUPLYR_PERF_RECORD"))) {
  stop("CUPLYR_PERF_RECORD is set: re-baselining perf is Theo's decision, not a test run.")
}

# cwd is the worktree root: every caller (tools/test, tools/test-file,
# tools/perf and the merge gate script) cd's to it before running this.
source("tools/run-tests-lib.R")

load_package <- function(compile) {
  pkgload::load_all(quiet = TRUE, debug = FALSE, compile = compile)
}

# --- one file ----------------------------------------------------------------

# The suite runs each file through test_file() rather than test_dir() so a
# per-test results table can be written; helpers are re-sourced per file,
# which matches test_dir()'s isolation. A file with no test_that() blocks has
# no rows but still has a duration.
run_file <- function(f, shard) {
  t0 <- proc.time()[["elapsed"]]
  res <- testthat::test_file(f, reporter = "summary", stop_on_failure = FALSE)
  secs <- proc.time()[["elapsed"]] - t0
  df <- as.data.frame(res)
  rows <- NULL
  if (nrow(df) > 0) {
    status <- ifelse(df$error, "error",
      ifelse(df$failed > 0, "fail",
        ifelse(df$skipped & df$passed == 0, "skip", "pass")))
    rows <- data.frame(
      file = basename(f), test = df$test, status = status,
      seconds = round(df$real, 3), stringsAsFactors = FALSE
    )
  }
  list(file = basename(f), rows = rows, seconds = round(secs, 3), shard = shard)
}

# --- the report ----------------------------------------------------------------

# `entries` are run_file()-shaped lists. A suite's rows come out in byte order
# of the file name, tests in file order, whichever process ran what; `file` mode
# keeps the order of the command line.
merge_entries <- function(entries) {
  if (identical(mode, "suite")) {
    file <- vapply(entries, `[[`, "", "file")
    entries <- entries[order(factor(file, levels = sort_bytes(file)))]
  }
  rows <- lapply(entries, function(e) {
    if (is.null(e$rows)) return(NULL)
    e$rows$file_seconds <- e$seconds
    e$rows$shard <- e$shard
    e$rows
  })
  results <- do.call(rbind, rows)
  if (is.null(results)) {
    results <- data.frame(file = character(), test = character(), status = character(),
                          seconds = numeric(), file_seconds = numeric(),
                          shard = character(), stringsAsFactors = FALSE)
  }
  results
}

write_durations <- function(entries, path) {
  d <- data.frame(file = vapply(entries, `[[`, "", "file"),
                  seconds = vapply(entries, `[[`, 0, "seconds"),
                  stringsAsFactors = FALSE)
  d <- d[order(factor(d$file, levels = sort_bytes(d$file))), , drop = FALSE]
  utils::write.table(d, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

print_slowest <- function(entries, n = 15L) {
  if (length(entries) == 0L) return(invisible())
  secs <- vapply(entries, `[[`, 0, "seconds")
  shard <- vapply(entries, function(e) as.character(e$shard), "")
  file <- vapply(entries, `[[`, "", "file")
  top <- head(order(-secs, factor(file, levels = sort_bytes(file))), n)
  cat(sprintf("\nSLOWEST %d of %d files (seconds, shard):\n", length(top), length(entries)))
  cat(sprintf("  %8.1f  %-6s  %s\n", secs[top], shard[top], file[top]), sep = "")
}

finish <- function(results, entries, peak = NULL) {
  if (!is.null(opt[["--record-durations"]])) {
    write_durations(entries, opt[["--record-durations"]])
  }
  if (!is.null(out)) {
    results$test <- gsub("[\t\n\r]+", " ", results$test)
    utils::write.table(results, out, sep = "\t", quote = FALSE, row.names = FALSE)
  }

  if (identical(mode, "suite")) print_slowest(entries)
  if (!is.null(peak)) cat(peak, "\n", sep = "")

  counts <- table(factor(results$status, levels = c("pass", "fail", "error", "skip")))
  cat(sprintf("\nRESULT: %d pass, %d fail, %d error, %d skip\n",
    counts[["pass"]], counts[["fail"]], counts[["error"]], counts[["skip"]]))
  bad <- results[results$status %in% c("fail", "error"), ]
  if (nrow(bad) > 0) {
    cat("FAILING:\n")
    cat(sprintf("  [%s] %s :: %s\n", bad$status, bad$file, bad$test), sep = "")
  }
  quit(status = if (nrow(bad) > 0) 1L else 0L)
}

# --- shard child -----------------------------------------------------------------

# Writes "started <file>" before each file and, once the file is done, its
# entry (atomically) and "finished <file>", so the parent can tell what finished
# and what was running if this process dies.
if (identical(mode, "shard")) {
  label <- opt[["--label"]]
  events <- opt[["--events"]]
  rows_dir <- opt[["--rows-dir"]]
  event <- function(...) cat(..., "\n", sep = "\t", file = events, append = TRUE)

  # A parallel shard does not need the 1 GiB the default resource reserves at
  # load, and N of them would hold N GiB: reserve nothing up front and give
  # unused memory back above 512 MiB. Options, not environment variables: the
  # resource reads them once, at load, and an environment variable would also
  # reach the callr children some tests start. A value the caller set in the
  # environment is left to decide.
  if (isTRUE(opt[["--small-pool"]])) {
    small <- list()
    if (!nzchar(Sys.getenv("CUPLYR_POOL_INITIAL"))) small$cuplyr.pool_initial <- 0
    if (!nzchar(Sys.getenv("CUPLYR_RELEASE_THRESHOLD"))) small$cuplyr.release_threshold <- 512 * 2^20
    before <- options(small)
    load_package(compile = FALSE)
    options(before)
  } else {
    load_package(compile = FALSE)
  }

  bad <- FALSE
  for (f in targets) {
    event("started", basename(f))
    e <- run_file(f, label)
    path <- file.path(rows_dir, paste0(basename(f), ".rds"))
    saveRDS(e, paste0(path, ".tmp"))
    file.rename(paste0(path, ".tmp"), path)
    event("finished", basename(f), e$seconds)
    if (!is.null(e$rows) && any(e$rows$status %in% c("fail", "error"))) bad <- TRUE
  }
  quit(status = if (bad) 1L else 0L)
}

# --- parent of a sharded run -------------------------------------------------------

# GPU memory in use (MiB, device 0) and host memory in use (MiB: MemTotal minus
# MemAvailable), NA where they can't be read.
sample_memory <- function() {
  gpu <- c(used = NA_real_, total = NA_real_)
  if (nzchar(Sys.which("nvidia-smi"))) {
    line <- tryCatch(
      suppressWarnings(system2("nvidia-smi", c("--query-gpu=memory.used,memory.total",
                                               "--format=csv,noheader,nounits"),
                               stdout = TRUE, stderr = FALSE))[[1]],
      error = function(e) NA_character_)
    v <- suppressWarnings(as.numeric(strsplit(line, ",")[[1]]))
    if (length(v) >= 2L) gpu <- c(used = v[[1]], total = v[[2]])
  }
  host <- c(used = NA_real_, total = NA_real_)
  mi <- tryCatch(readLines("/proc/meminfo"), error = function(e) character())
  kb <- function(key) {
    x <- grep(paste0("^", key, ":"), mi, value = TRUE)
    if (length(x) == 0L) NA_real_ else as.numeric(gsub("[^0-9]", "", x[[1]])) / 1024
  }
  if (length(mi) > 0L) host <- c(used = kb("MemTotal") - kb("MemAvailable"), total = kb("MemTotal"))
  list(gpu = gpu, host = host)
}

# Start one child per spec, wait for all of them, sampling memory every second
# for the peaks. A spec is list(label, files, small_pool). Returns the specs
# with `status`, `seconds`, `log`, `events` and `rows_dir` filled in.
run_children <- function(specs, work, peaks) {
  rscript <- file.path(R.home("bin"), "Rscript")
  procs <- list()
  for (k in seq_along(specs)) {
    s <- specs[[k]]
    s$rows_dir <- file.path(work, paste0("rows-", s$label))
    dir.create(s$rows_dir)
    s$events <- file.path(work, paste0("events-", s$label, ".log"))
    s$log <- file.path(work, paste0("shard-", s$label, ".log"))
    child_args <- c("tools/run-tests.R", "shard", "--label", s$label,
                    "--events", s$events, "--rows-dir", s$rows_dir,
                    if (isTRUE(s$small_pool)) "--small-pool", s$paths)
    procs[[k]] <- processx::process$new(rscript, child_args, stdout = s$log,
                                        stderr = "2>&1", supervise = TRUE)
    s$started <- Sys.time()
    specs[[k]] <- s
  }
  update_peaks <- function() {
    m <- sample_memory()
    for (w in c("gpu", "host")) {
      if (is.finite(m[[w]][["used"]]) && (is.na(peaks[[w]]) || m[[w]][["used"]] > peaks[[w]])) {
        peaks[[w]] <- m[[w]][["used"]]
      }
      peaks[[paste0(w, "_total")]] <- m[[w]][["total"]]
    }
  }
  update_peaks()
  done <- rep(FALSE, length(specs))
  while (!all(done)) {
    Sys.sleep(1)
    update_peaks()
    for (k in which(!done)) {
      if (!procs[[k]]$is_alive()) {
        done[[k]] <- TRUE
        specs[[k]]$seconds <- as.numeric(difftime(Sys.time(), specs[[k]]$started, units = "secs"))
        specs[[k]]$status <- procs[[k]]$get_exit_status()
        cat(sprintf("[%s] finished after %.0f s, %s\n", specs[[k]]$label,
                    specs[[k]]$seconds, describe_status(specs[[k]]$status)))
      }
    }
  }
  specs
}

describe_status <- function(status) {
  if (is.na(status)) "no exit status"
  else if (status < 0) sprintf("killed by signal %d", -as.integer(status))
  else sprintf("exit status %d", as.integer(status))
}

# The entries of one finished child. Every assigned file either finished (its
# entry was written), was running when the process stopped, or never started;
# the last two get an error row, whatever the exit status was (a process that
# exits 0 early has not run its files).
collect_child <- function(s) {
  ev <- if (file.exists(s$events)) readLines(s$events, warn = FALSE) else character()
  started <- sub("^started\t", "", grep("^started\t", ev, value = TRUE))
  why <- describe_status(s$status)
  entries <- list()
  unfinished <- FALSE
  for (f in s$files) {
    rds <- file.path(s$rows_dir, paste0(f, ".rds"))
    e <- if (file.exists(rds)) tryCatch(readRDS(rds), error = function(err) NULL)
    if (is.null(e)) {
      unfinished <- TRUE
      msg <- if (f %in% started) {
        sprintf("shard %s stopped while running this file (%s)", s$label, why)
      } else {
        sprintf("not run: shard %s died (%s)", s$label, why)
      }
      e <- list(file = f, shard = s$label, seconds = 0,
                rows = data.frame(file = f, test = msg, status = "error", seconds = 0,
                                  stringsAsFactors = FALSE))
    }
    entries[[length(entries) + 1L]] <- e
  }
  # Every file finished, but the process did not end the way one that is done
  # does (0, or 1 when something failed).
  if (!unfinished && length(entries) > 0L && !(isTRUE(s$status %in% c(0L, 1L)))) {
    last <- entries[[length(entries)]]
    extra <- data.frame(file = last$file,
                        test = sprintf("shard %s ended abnormally after its files (%s)", s$label, why),
                        status = "error", seconds = 0, stringsAsFactors = FALSE)
    entries[[length(entries)]]$rows <- rbind(last$rows, extra)
  }
  entries
}

# --- main ----------------------------------------------------------------------------

files <- switch(mode,
  suite = {
    dir <- opt[["--dir"]] %||% "tests/testthat"
    sort_bytes(list.files(dir, pattern = "^test-.*\\.R$"))
  },
  file = targets,
  perf = {
    Sys.setenv(CUPLYR_PERF = "1")
    "tests/testthat/test-perf-regression.R"
  },
  stop("unknown mode: ", mode)
)

shards <- 1L
if (identical(mode, "suite")) {
  shards <- if (!is.null(opt[["--shards"]])) {
    n <- suppressWarnings(as.integer(opt[["--shards"]]))
    if (is.na(n) || n < 1L) stop("--shards must be a positive whole number")
    n
  } else {
    default_shards()
  }
}

if (shards == 1L) {
  # One process, every file in name order; what a run did before sharding.
  source("tools/ensure-optimized.R")
  cuplyr_ensure_optimized_dll(".")
  load_package(compile = TRUE)
  if (identical(mode, "suite")) files <- file.path(dir, files)
  entries <- lapply(files, run_file, shard = "1")
  finish(merge_entries(entries), entries)
}

# Sharded: this process never loads the package, so it holds no GPU context.
source("tools/ensure-optimized.R")
cuplyr_ensure_optimized_dll(".")  # build once, so no child races a rebuild

serial_list <- read_serial_list(opt[["--serial-list"]] %||% "tools/test-serial.txt")
unknown <- setdiff(serial_list, files)
if (length(unknown) > 0L) {
  message("note: the serial list names files that are not in ", dir, ": ",
          paste(unknown, collapse = ", "))
}
durations <- read_durations(opt[["--durations"]] %||% "tools/test-durations.tsv")
plan <- suite_plan(files, durations, serial_list, shards)

work <- tempfile("cuplyr-suite-")
dir.create(work)
peaks <- new.env()
peaks$gpu <- NA_real_
peaks$host <- NA_real_
peaks$gpu_total <- NA_real_
peaks$host_total <- NA_real_

make_spec <- function(label, fs, small_pool) {
  list(label = label, files = fs, paths = file.path(dir, fs), small_pool = small_pool)
}
shard_specs <- lapply(seq_along(plan$shards), function(k) make_spec(as.character(k), plan$shards[[k]], TRUE))
cat(sprintf("suite: %d files in %d parallel shard(s)%s\n", length(files), length(shard_specs),
            if (length(plan$serial) > 0L) sprintf(", then %d serial", length(plan$serial)) else ""))

finished <- list()
if (length(shard_specs) > 0L) {
  finished <- run_children(shard_specs, work, peaks)
}
# The serial phase: alone on the machine, with the default memory resource.
if (length(plan$serial) > 0L) {
  finished <- c(finished,
                run_children(list(make_spec("serial", plan$serial, FALSE)), work, peaks))
}

entries <- list()
for (s in finished) {
  entries <- c(entries, collect_child(s))
}

# Every child's full output, in shard order.
for (s in finished) {
  cat(sprintf("\n===== shard %s: %d file(s), %.0f s, %s =====\n", s$label, length(s$files),
              s$seconds, describe_status(s$status)))
  cat(paste(s$files, collapse = " "), "\n", sep = "")
  cat("-----\n")
  if (file.exists(s$log)) cat(paste0(readLines(s$log, warn = FALSE), "\n"), sep = "")
  cat("===== end of shard ", s$label, " =====\n", sep = "")
}

unlink(work, recursive = TRUE)

fmt <- function(used, total, unit) {
  if (is.na(used)) "n/a" else sprintf("%s of %s", unit(used), unit(total))
}
peak <- sprintf("PEAK: gpu %s, host %s (memory in use on the machine, sampled every second)",
                fmt(peaks$gpu, peaks$gpu_total, function(x) sprintf("%.0f MiB", x)),
                fmt(peaks$host, peaks$host_total, function(x) sprintf("%.1f GiB", x / 1024)))

finish(merge_entries(entries), entries, peak = peak)
