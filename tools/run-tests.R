#!/usr/bin/env Rscript
# Test entry point used by tools/test, tools/test-file and tools/perf.
# Always runs against pkgload::load_all() of the current worktree; nothing is
# ever installed. Call it through tools/gpu-run, never directly.
#
#   Rscript tools/run-tests.R suite [--out results.tsv]
#   Rscript tools/run-tests.R file <test-file>... [--out results.tsv]
#   Rscript tools/run-tests.R perf [--out results.tsv]
#
# --out writes one row per test_that() block: file, test, status
# (pass/fail/error/skip). Exit status is 1 if any test failed or errored.

args <- commandArgs(trailingOnly = TRUE)
out_i <- match("--out", args)
out <- NULL
if (!is.na(out_i)) {
  out <- args[[out_i + 1L]]
  args <- args[-c(out_i, out_i + 1L)]
}
mode <- args[[1]]
targets <- args[-1]

if (nzchar(Sys.getenv("CUPLYR_PERF_RECORD"))) {
  stop("CUPLYR_PERF_RECORD is set: re-baselining perf is Theo's decision, not a test run.")
}

pkgload::load_all(quiet = TRUE)

run <- function(files) {
  rows <- list()
  for (f in files) {
    res <- testthat::test_file(f, reporter = "summary", stop_on_failure = FALSE)
    df <- as.data.frame(res)
    if (nrow(df) == 0) next
    status <- ifelse(df$error, "error",
      ifelse(df$failed > 0, "fail",
        ifelse(df$skipped & df$passed == 0, "skip", "pass")))
    rows[[length(rows) + 1L]] <- data.frame(
      file = basename(f), test = df$test, status = status,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

files <- switch(mode,
  suite = sort(list.files("tests/testthat", pattern = "^test-.*\\.R$", full.names = TRUE)),
  file = targets,
  perf = {
    Sys.setenv(CUPLYR_PERF = "1")
    "tests/testthat/test-perf-regression.R"
  },
  stop("unknown mode: ", mode)
)

# The suite runs each file through test_file() rather than test_dir() so a
# per-test results table can be written; helpers are re-sourced per file,
# which matches test_dir()'s isolation.
results <- run(files)
if (is.null(results)) {
  results <- data.frame(file = character(), test = character(), status = character())
}
if (!is.null(out)) {
  results$test <- gsub("[\t\n\r]+", " ", results$test)
  utils::write.table(results, out, sep = "\t", quote = FALSE, row.names = FALSE)
}

counts <- table(factor(results$status, levels = c("pass", "fail", "error", "skip")))
cat(sprintf("\nRESULT: %d pass, %d fail, %d error, %d skip\n",
  counts[["pass"]], counts[["fail"]], counts[["error"]], counts[["skip"]]))
bad <- results[results$status %in% c("fail", "error"), ]
if (nrow(bad) > 0) {
  cat("FAILING:\n")
  cat(sprintf("  [%s] %s :: %s\n", bad$status, bad$file, bad$test), sep = "")
}
quit(status = if (nrow(bad) > 0) 1L else 0L)
