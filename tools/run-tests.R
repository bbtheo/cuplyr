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
# any test failed or errored. An unknown option is an error.
#
# suite: with --shards 1 every file runs in this process, in name order. With
# more (default: CUPLYR_TEST_SHARDS, else tools/run-tests-lib.R's default) this
# process only supervises: the files are dealt out to that many child R
# processes by their recorded durations (--durations), the children run at once
# inside this one GPU lock hold, then each file named in --serial-list (timing-
# and memory-sensitive ones) runs in a child of its own, one after another, once
# all of them have exited. A child that dies or stops early fails the run; its
# files that did not finish get an error row and its output is printed with the
# rest. --record-durations writes `file<TAB>seconds` for every file that ran:
# refresh tools/test-durations.tsv from a `--shards 1` run.
#
# The library is built once per top-level run, before anything is loaded: this
# process exports CUPLYR_TEST_BUILT (the worktree path), and every runner that
# sees it (shards, runs a test starts) skips the build and loads with
# compile = FALSE, so nothing replaces src/cuplyr.so under a running process.
# A run that finds the .so changed at the end of a phase reports an error row.
#
# `shard` is an internal mode: one child of a sharded suite run.

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(a, b) if (is.null(a)) b else a

flags <- c("--out", "--shards", "--dir", "--serial-list", "--durations",
           "--record-durations", "--label", "--events", "--rows-dir")
switches <- c("--parallel")
opt <- list()
positional <- character()
i <- 1L
while (i <= length(args)) {
  a <- args[[i]]
  if (a %in% flags) {
    if (i == length(args)) stop(a, " needs a value")
    opt[[a]] <- args[[i + 1L]]
    i <- i + 2L
  } else if (a %in% switches) {
    opt[[a]] <- TRUE
    i <- i + 1L
  } else if (startsWith(a, "--")) {
    stop("unknown option ", a, " (known: ", paste(c(flags, switches), collapse = " "), ")")
  } else {
    positional <- c(positional, a)
    i <- i + 1L
  }
}
if (length(positional) == 0L) stop("usage: Rscript tools/run-tests.R suite|file|perf [options]")
mode <- positional[[1]]
targets <- positional[-1]
out <- opt[["--out"]]

if (nzchar(Sys.getenv("CUPLYR_PERF_RECORD"))) {
  stop("CUPLYR_PERF_RECORD is set: re-baselining perf is Theo's decision, not a test run.")
}

# cwd is the worktree root: every caller (tools/test, tools/test-file,
# tools/perf and the merge gate script) cd's to it before running this.
source("tools/run-tests-lib.R")

# --- the library: built once, loaded without compiling ---------------------------

so_file <- function() file.path("src", paste0("cuplyr", .Platform$dynlib.ext))

# Build the library once per top-level run (a stale -O0 build is replaced, see
# tools/ensure-optimized.R) and tell every runner below this one not to. The
# marker is the worktree path, so one inherited from another tree means nothing.
build_once <- function() {
  root <- normalizePath(".")
  if (identical(Sys.getenv("CUPLYR_TEST_BUILT"), root)) return(invisible())
  source("tools/ensure-optimized.R")
  cuplyr_ensure_optimized_dll(".")
  Sys.setenv(CUPLYR_TEST_BUILT = root)
}

# compile = FALSE: pkgload's TRUE means "clean and rebuild" and NA can still
# rebuild; building is build_once()'s job. Without the .so, load_all() would
# silently load the R code with no native routines.
load_package <- function() {
  if (!file.exists(so_file())) {
    stop(so_file(), " is missing: the library is not built", call. = FALSE)
  }
  pkgload::load_all(quiet = TRUE, debug = FALSE, compile = FALSE)
}

so_state <- function() {
  if (!file.exists(so_file())) return(NULL)
  info <- file.info(so_file())
  c(size = info$size, mtime = as.numeric(info$mtime))
}

# An error row when the library was replaced since `before` was taken (the
# build the other processes of this run have loaded), else NULL.
so_changed_row <- function(before, phase) {
  now <- so_state()
  if (is.null(before) || identical(now, before)) return(NULL)
  describe <- function(x) {
    if (is.null(x)) "missing" else sprintf("%.0f bytes, mtime %.3f", x[["size"]], x[["mtime"]])
  }
  data.frame(
    file = "tools/run-tests.R",
    test = sprintf("src/cuplyr.so changed during the %s (%s -> %s): something rebuilt the library while the suite was using it",
                   phase, describe(before), describe(now)),
    status = "error", seconds = 0, file_seconds = 0, shard = "runner",
    stringsAsFactors = FALSE
  )
}

# --- one file ----------------------------------------------------------------

# The suite runs each file through test_file() rather than test_dir() so a
# per-test results table can be written; helpers are re-sourced per file,
# which matches test_dir()'s isolation. A file with no test_that() blocks has
# no rows but still has a duration.
#
# In a parallel shard (`memory_skip_error`) a test that skipped for lack of GPU
# memory (skip_if_insufficient_gpu_memory()) is an error: next to the other
# shards the card is fuller than it will be when the test runs alone, and a
# skip would drop the test without a trace. Its name is kept; the reason is in
# the log and in `notes`.
run_file <- function(f, shard, memory_skip_error = FALSE) {
  t0 <- proc.time()[["elapsed"]]
  res <- testthat::test_file(f, reporter = "summary", stop_on_failure = FALSE)
  secs <- proc.time()[["elapsed"]] - t0
  df <- as.data.frame(res)
  rows <- NULL
  notes <- character()
  if (nrow(df) > 0) {
    status <- ifelse(df$error, "error",
      ifelse(df$failed > 0, "fail",
        ifelse(df$skipped & df$passed == 0, "skip", "pass")))
    if (memory_skip_error && length(res) == nrow(df)) {
      lack <- function(r) {
        if (!inherits(r, "expectation_skip")) return(NA_character_)
        # testthat puts "Reason: " in front of a skip's message
        m <- sub("^Reason: ", "", conditionMessage(r))
        if (startsWith(m, "Insufficient GPU memory")) m else NA_character_
      }
      why <- vapply(res, function(t) {
        m <- stats::na.omit(vapply(t$results, lack, ""))
        if (length(m)) m[[1]] else NA_character_
      }, "")
      status[!is.na(why)] <- "error"
      for (j in which(!is.na(why))) {
        notes <- c(notes, sprintf("%s :: %s: skipped for lack of GPU memory next to the other parallel shards (%s); counted as an error",
                                  basename(f), df$test[[j]], why[[j]]))
      }
    }
    rows <- data.frame(
      file = basename(f), test = df$test, status = status,
      seconds = round(df$real, 3), stringsAsFactors = FALSE
    )
  }
  if (length(notes) > 0L) cat(paste0("RUNNER: ", notes, "\n"), sep = "")
  list(file = basename(f), rows = rows, seconds = round(secs, 3), shard = shard, notes = notes)
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

# Durations of files that really ran: the placeholder entries of files a dead
# shard never finished say 0 s and would unbalance the next plan.
write_durations <- function(entries, path) {
  entries <- Filter(function(e) !isTRUE(e$placeholder), entries)
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
  notes <- unlist(lapply(entries, `[[`, "notes"))
  if (length(notes) > 0L) cat(paste0("NOTE: ", notes, "\n"), sep = "")

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
# entry (atomically) and "finished <file> <seconds>", so the parent can tell
# what finished and what was running if this process dies. One tab-separated
# line per event.
if (identical(mode, "shard")) {
  label <- opt[["--label"]]
  events <- opt[["--events"]]
  rows_dir <- opt[["--rows-dir"]]
  event <- function(...) cat(paste(..., sep = "\t"), "\n", sep = "", file = events, append = TRUE)
  parallel <- isTRUE(opt[["--parallel"]])

  build_once()  # a no-op under a suite run, which built before starting us

  # A parallel shard does not need the 1 GiB the default resource reserves at
  # load, and N of them would hold N GiB: reserve nothing up front and give
  # unused memory back above 512 MiB. Options, not environment variables: the
  # resource reads them once, at load, and an environment variable would also
  # reach the callr children some tests start. A value the caller set in the
  # environment is left to decide.
  if (parallel) {
    small <- list()
    if (!nzchar(Sys.getenv("CUPLYR_POOL_INITIAL"))) small$cuplyr.pool_initial <- 0
    if (!nzchar(Sys.getenv("CUPLYR_RELEASE_THRESHOLD"))) small$cuplyr.release_threshold <- 512 * 2^20
    before <- options(small)
    load_package()
    options(before)
  } else {
    load_package()
  }

  bad <- FALSE
  for (f in targets) {
    event("started", basename(f))
    e <- run_file(f, label, memory_skip_error = parallel)
    path <- file.path(rows_dir, paste0(basename(f), ".rds"))
    saveRDS(e, paste0(path, ".tmp"))
    file.rename(paste0(path, ".tmp"), path)
    event("finished", basename(f), e$seconds)
    if (!is.null(e$rows) && any(e$rows$status %in% c("fail", "error"))) bad <- TRUE
  }
  quit(status = if (bad) 1L else 0L)
}

# --- parent of a sharded run -------------------------------------------------------

# GPU memory in use (MiB, device 0; only when `gpu`) and host memory in use
# (MiB: MemTotal minus MemAvailable), NA where they can't be read. nvidia-smi
# gets a timeout: a wedged driver must not hang the run.
sample_memory <- function(gpu = TRUE) {
  g <- c(used = NA_real_, total = NA_real_)
  if (gpu && nzchar(Sys.which("nvidia-smi"))) {
    r <- tryCatch(
      processx::run("nvidia-smi", c("--query-gpu=memory.used,memory.total",
                                    "--format=csv,noheader,nounits"),
                    timeout = 5, error_on_status = FALSE),
      error = function(e) NULL)
    if (!is.null(r) && identical(r$status, 0L)) {
      v <- suppressWarnings(as.numeric(strsplit(strsplit(r$stdout, "\n")[[1]][[1]], ",")[[1]]))
      if (length(v) >= 2L) g <- c(used = v[[1]], total = v[[2]])
    }
  }
  h <- c(used = NA_real_, total = NA_real_)
  mi <- tryCatch(readLines("/proc/meminfo"), error = function(e) character())
  kb <- function(key) {
    x <- grep(paste0("^", key, ":"), mi, value = TRUE)
    if (length(x) == 0L) NA_real_ else as.numeric(gsub("[^0-9]", "", x[[1]])) / 1024
  }
  if (length(mi) > 0L) h <- c(used = kb("MemTotal") - kb("MemAvailable"), total = kb("MemTotal"))
  list(gpu = g, host = h)
}

# Peaks live in an environment: keep the largest value seen under `key`.
bump <- function(peaks, key, value) {
  if (is.finite(value) && (is.na(peaks[[key]]) || value > peaks[[key]])) peaks[[key]] <- value
}

describe_status <- function(status) {
  if (is.na(status)) "no exit status"
  else if (status < 0) sprintf("killed by signal %d", -as.integer(status))
  else sprintf("exit status %d", as.integer(status))
}

# How a child is named in messages: "shard 2", or the serial phase's process.
who <- function(label) {
  if (identical(label, "serial")) "the serial-phase process" else sprintf("shard %s", label)
}

# Start one child per spec, wait for all of them, printing each child's
# "started"/"finished" events as they happen (a hang shows which file it is in).
# `phase` is "parallel" (memory sampled every second: GPU and host) or "serial"
# (host only: a nvidia-smi call every second is not what a file that times
# itself or reads the device's memory should run next to). A spec is
# list(label, id, files, paths, parallel); `label` is what the report calls the
# process, `id` names its files. Returns the specs with `status`, `seconds`,
# `log`, `events` and `rows_dir` filled in.
run_children <- function(specs, work, phase, peaks) {
  rscript <- file.path(R.home("bin"), "Rscript")
  procs <- list()
  for (k in seq_along(specs)) {
    s <- specs[[k]]
    s$rows_dir <- file.path(work, paste0("rows-", s$id))
    dir.create(s$rows_dir)
    s$events <- file.path(work, paste0("events-", s$id, ".log"))
    s$log <- file.path(work, paste0("shard-", s$id, ".log"))
    child_args <- c("tools/run-tests.R", "shard", "--label", s$label,
                    "--events", s$events, "--rows-dir", s$rows_dir,
                    if (isTRUE(s$parallel)) "--parallel", s$paths)
    procs[[k]] <- processx::process$new(rscript, child_args, stdout = s$log,
                                        stderr = "2>&1", supervise = TRUE)
    s$started <- Sys.time()
    specs[[k]] <- s
  }

  seen <- integer(length(specs))
  show_events <- function(k) {
    s <- specs[[k]]
    ev <- if (file.exists(s$events)) readLines(s$events, warn = FALSE) else character()
    if (length(ev) <= seen[[k]]) return(invisible())
    for (line in ev[seq.int(seen[[k]] + 1L, length(ev))]) {
      p <- strsplit(line, "\t", fixed = TRUE)[[1]]
      if (length(p) >= 2L && p[[1]] == "started") {
        cat(sprintf("[%s] started %s\n", s$label, p[[2]]))
      } else if (length(p) >= 3L && p[[1]] == "finished") {
        cat(sprintf("[%s] finished %s (%.0f s)\n", s$label, p[[2]], as.numeric(p[[3]])))
      }
    }
    seen[[k]] <<- length(ev)
  }
  sample_now <- function() {
    m <- sample_memory(gpu = identical(phase, "parallel"))
    if (identical(phase, "parallel")) {
      bump(peaks, "par_gpu", m$gpu[["used"]])
      bump(peaks, "par_host", m$host[["used"]])
      if (is.finite(m$gpu[["total"]])) peaks$gpu_total <- m$gpu[["total"]]
    } else {
      bump(peaks, "ser_host", m$host[["used"]])
    }
    peaks$host_total <- m$host[["total"]]
  }

  sample_now()
  done <- rep(FALSE, length(specs))
  while (!all(done)) {
    # Wakes when the first running child exits, or after a second.
    procs[[which(!done)[[1L]]]]$wait(1000)
    sample_now()
    for (k in which(!done)) {
      show_events(k)
      if (!procs[[k]]$is_alive()) {
        done[[k]] <- TRUE
        show_events(k)
        specs[[k]]$seconds <- as.numeric(difftime(Sys.time(), specs[[k]]$started, units = "secs"))
        specs[[k]]$status <- procs[[k]]$get_exit_status()
        cat(sprintf("[%s] exited after %.0f s, %s\n", specs[[k]]$label,
                    specs[[k]]$seconds, describe_status(specs[[k]]$status)))
      }
    }
  }
  specs
}

# The entries of one finished child. Every assigned file either finished (its
# entry was written), was running when the process stopped, or never started;
# the last two get an error row, whatever the exit status was (a process that
# exits 0 early has not run its files).
collect_child <- function(s) {
  ev <- if (file.exists(s$events)) readLines(s$events, warn = FALSE) else character()
  fields <- strsplit(ev, "\t", fixed = TRUE)
  started <- vapply(Filter(function(p) length(p) >= 2L && p[[1]] == "started", fields),
                    `[[`, "", 2L)
  why <- describe_status(s$status)
  entries <- list()
  unfinished <- FALSE
  for (f in s$files) {
    rds <- file.path(s$rows_dir, paste0(f, ".rds"))
    e <- if (file.exists(rds)) tryCatch(readRDS(rds), error = function(err) NULL)
    if (is.null(e)) {
      unfinished <- TRUE
      msg <- if (f %in% started) {
        sprintf("%s stopped while running this file (%s)", who(s$label), why)
      } else {
        sprintf("not run: %s died (%s)", who(s$label), why)
      }
      e <- list(file = f, shard = s$label, seconds = 0, placeholder = TRUE,
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
                        test = sprintf("%s ended abnormally after its files (%s)", who(s$label), why),
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
    check_shard_count(opt[["--shards"]], "--shards")
  } else {
    default_shards()
  }
}

build_once()
so_before <- so_state()

if (shards == 1L) {
  # One process, every file in name order; what a run did before sharding.
  load_package()
  if (identical(mode, "suite")) files <- file.path(dir, files)
  entries <- lapply(files, run_file, shard = "1")
  results <- merge_entries(entries)
  results <- rbind(results, so_changed_row(so_before, "run"))
  finish(results, entries)
}

# Sharded: this process never loads the package, so it holds no GPU context.
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
for (key in c("par_gpu", "par_host", "ser_host", "gpu_total", "host_total")) peaks[[key]] <- NA_real_

shard_specs <- lapply(seq_along(plan$shards), function(k) {
  list(label = as.character(k), id = as.character(k), files = plan$shards[[k]],
       paths = file.path(dir, plan$shards[[k]]), parallel = TRUE)
})
cat(sprintf("suite: %d files in %d parallel shard(s)%s\n", length(files), length(shard_specs),
            if (length(plan$serial) > 0L) sprintf(", then %d serial (one process each)", length(plan$serial)) else ""))

# What the machine uses before the first child starts: the baseline the peaks
# are read against.
before <- sample_memory()
before_gpu <- before$gpu[["used"]]
before_host <- before$host[["used"]]

finished <- list()
so_rows <- NULL
if (length(shard_specs) > 0L) {
  finished <- run_children(shard_specs, work, "parallel", peaks)
  so_rows <- rbind(so_rows, so_changed_row(so_before, "parallel phase"))
  so_before <- so_state()
}
# The serial phase: alone on the machine, with the default memory resource, each
# file in a fresh process so what it runs after does not depend on the list.
for (k in seq_along(plan$serial)) {
  one <- list(list(label = "serial", id = sprintf("serial-%03d", k), files = plan$serial[[k]],
                   paths = file.path(dir, plan$serial[[k]]), parallel = FALSE))
  finished <- c(finished, run_children(one, work, "serial", peaks))
}
if (length(plan$serial) > 0L) {
  so_rows <- rbind(so_rows, so_changed_row(so_before, "serial phase"))
}

entries <- list()
for (s in finished) {
  entries <- c(entries, collect_child(s))
}

# Every child's full output, in shard order.
for (s in finished) {
  cat(sprintf("\n===== %s: %d file(s), %.0f s, %s =====\n", who(s$label), length(s$files),
              s$seconds, describe_status(s$status)))
  cat(paste(s$files, collapse = " "), "\n", sep = "")
  cat("-----\n")
  if (file.exists(s$log)) cat(paste0(readLines(s$log, warn = FALSE), "\n"), sep = "")
  cat("===== end of ", who(s$label), " =====\n", sep = "")
}

unlink(work, recursive = TRUE)

mib <- function(x) if (is.na(x)) "n/a" else sprintf("%.0f MiB", x)
gib <- function(x) if (is.na(x)) "n/a" else sprintf("%.1f GiB", x / 1024)
peak <- sprintf(
  paste0("PEAK: gpu %s in use before the first child, %s peak in the parallel phase of %s ",
         "(serial phase not sampled); host %s in use before the first child, %s peak in the ",
         "parallel phase, %s peak in the serial phase, of %s (memory in use on the machine, ",
         "sampled every second)"),
  mib(before_gpu), mib(peaks$par_gpu), mib(peaks$gpu_total),
  gib(before_host), gib(peaks$par_host), gib(peaks$ser_host), gib(peaks$host_total))

results <- rbind(merge_entries(entries), so_rows)
finish(results, entries, peak = peak)
