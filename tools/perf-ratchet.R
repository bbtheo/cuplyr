#!/usr/bin/env Rscript
# CLI over tools/perf-ratchet-lib.R, behind tools/perf-ratchet (merge-queue)
# and tools/perf-baseline-set (Theo). Pure R: no package load, no GPU.
#
#   Rscript tools/perf-ratchet.R ratchet --observations <tsv> --summary <txt> --out <dir>
#                                        [--commit <sha>] [--baseline <json>]
#   Rscript tools/perf-ratchet.R set [--baseline <json>] <id>=<seconds> ...
#
# ratchet: baseline = --baseline, else $CUPLYR_PERF_BASELINE, else an error
# (never a default). Exit 0 when applied or when the gate did not pass (no-op).
# set: baseline = --baseline, else $CUPLYR_PERF_BASELINE, else
# $CUPLYR_MAIN/scratchpad/perf_baseline.json.
# Any error exits non-zero and leaves the baseline untouched.

args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
sys.source(file.path(script_dir, "perf-ratchet-lib.R"), envir = environment())

die <- function(...) {
  cat("perf-ratchet: ", ..., "\n", sep = "", file = stderr())
  quit(save = "no", status = 2L)
}

cmd <- if (length(args) > 0) args[[1]] else ""
rest <- args[-1]

# Pull "--name value" pairs out of `rest`; returns list(opts, positional).
parse_args <- function(rest, names) {
  opts <- list()
  pos <- character()
  i <- 1L
  while (i <= length(rest)) {
    a <- rest[[i]]
    if (startsWith(a, "--")) {
      if (!a %in% names) die("unknown option ", a)
      if (i == length(rest)) die(a, " needs a value")
      opts[[sub("^--", "", a)]] <- rest[[i + 1L]]
      i <- i + 2L
    } else {
      pos <- c(pos, a)
      i <- i + 1L
    }
  }
  list(opts = opts, pos = pos)
}

env_baseline <- function() {
  p <- Sys.getenv("CUPLYR_PERF_BASELINE", "")
  if (nzchar(p)) p else NULL
}

if (identical(cmd, "ratchet")) {
  a <- parse_args(rest, c("--baseline", "--observations", "--summary", "--out", "--commit"))
  o <- a$opts
  for (req in c("observations", "summary", "out")) if (is.null(o[[req]])) die("missing --", req)
  baseline <- o$baseline %||% env_baseline() %||% die("no baseline: pass --baseline or set CUPLYR_PERF_BASELINE")
  changes <- tryCatch(
    perf_ratchet_run(baseline, o$observations, o$summary, o$out, commit = o$commit %||% "unknown"),
    error = function(e) die(conditionMessage(e))
  )
  if (is.null(changes)) {
    cat("perf-ratchet: gate did not pass; baseline unchanged\n")
  } else {
    cat("perf-ratchet: ", nrow(changes), " change(s) written to ", baseline, "\n", sep = "")
  }
} else if (identical(cmd, "set")) {
  a <- parse_args(rest, "--baseline")
  baseline <- a$opts$baseline %||% env_baseline() %||%
    file.path(Sys.getenv("CUPLYR_MAIN"), "scratchpad", "perf_baseline.json")
  if (length(a$pos) == 0) die("usage: perf-baseline-set [--baseline <json>] <id>=<seconds> ...")
  parts <- strsplit(a$pos, "=", fixed = TRUE)
  if (!all(lengths(parts) == 2L)) die("arguments must look like <id>=<seconds>")
  values <- suppressWarnings(as.numeric(vapply(parts, `[`, "", 2L)))
  names(values) <- vapply(parts, `[`, "", 1L)
  tryCatch(
    perf_baseline_set(baseline, values),
    error = function(e) die(conditionMessage(e))
  )
  cat("perf-baseline-set: ", paste0(names(values), "=", format(values, digits = 6), collapse = " "),
    " -> ", baseline, "\n", sep = "")
} else {
  die("usage: perf-ratchet.R ratchet|set ...")
}
