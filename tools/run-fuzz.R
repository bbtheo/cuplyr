#!/usr/bin/env Rscript
# The R half of tools/fuzz (F118). Call it through tools/gpu-run, never directly.
#
#   Rscript tools/run-fuzz.R batch  --from S --to E --out batch.tsv --progress p --stats s.tsv
#   Rscript tools/run-fuzz.R shrink --sig-file sigs.tsv --line N --out shrunk.tsv
#   Rscript tools/run-fuzz.R report --findings f.tsv --shrunk s.tsv --stats s.tsv --out final.tsv
#
# batch   runs seeds S..E one case at a time, appends each case's findings to
#         --out and its stats (seed, native, native_ok, plan_break, masked) to --stats as it goes, and writes the seed to
#         --progress before the case: if the process dies, the driver reads the
#         seed that killed it and restarts after it.
# shrink  shrinks the case of line N of the signature file (seed, mode, kind,
#         signature) keeping that finding, and appends the result to --out. A
#         finding that doesn't reproduce is left unshrunk: exit status 3.
# report  joins the findings with their shrunk code, fills in the code of crash
#         rows, writes the final TSV and prints the findings grouped by
#         category and cause.

args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1L]]
opt <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) default else args[[i + 1L]]
}

source("tools/ensure-optimized.R")
cuplyr_ensure_optimized_dll(".")
pkgload::load_all(quiet = TRUE, debug = FALSE)
suppressPackageStartupMessages(library(dplyr))
source("tests/testthat/helper-fuzz.R")

write_rows <- function(df, file) {
  utils::write.table(df, file, sep = "\t", quote = FALSE, row.names = FALSE,
                     col.names = !file.exists(file), append = file.exists(file))
}
# Every column as character: read.delim makes an all-blank column logical NA
read_tsv <- function(file) {
  if (!file.exists(file) || file.size(file) == 0L) return(NULL)
  d <- utils::read.delim(file, quote = "", comment.char = "", colClasses = "character",
                         na.strings = character())
  d$seed <- as.integer(d$seed)
  d
}

if (mode == "batch") {
  out <- opt("out"); progress <- opt("progress"); stats <- opt("stats")
  # test hooks for the driver's recovery: CUPLYR_FUZZ_TEST_CRASH=<seed>[,<seed>...]
  # kills this process with SIGSEGV when it reaches one of those seeds,
  # ..._HANG=<seed> sleeps; ..._SHRINK_FAIL=1 makes every shrink fail,
  # ..._SHRINK_NOREPRO=1 makes every finding fail to reproduce when shrinking
  seed_list <- function(v) as.integer(strsplit(Sys.getenv(v, "-1"), ",", fixed = TRUE)[[1L]])
  test_crash <- seed_list("CUPLYR_FUZZ_TEST_CRASH")
  test_hang <- seed_list("CUPLYR_FUZZ_TEST_HANG")
  for (s in seq.int(as.integer(opt("from")), as.integer(opt("to")))) {
    if (s %in% test_crash || s %in% test_hang) {
      writeLines(as.character(s), progress)
      if (s %in% test_crash) tools::pskill(Sys.getpid(), 11L)
      Sys.sleep(1e5)
    }
    res <- fuzz_campaign(s, progress = progress)
    st <- attr(res, "stats")
    if (nrow(res)) write_rows(res, out)
    write(sprintf("%d\t%d\t%d\t%d\t%s", s, as.integer(st$native_share == 1),
                  as.integer(st$native_ok_share == 1), as.integer(st$plan_break_share > 0),
                  paste(sprintf("%s=%d", names(st$masked), st$masked), collapse = ";")),
          stats, append = TRUE)
  }
} else if (mode == "shrink") {
  if (nzchar(Sys.getenv("CUPLYR_FUZZ_TEST_SHRINK_FAIL"))) stop("test hook: shrink failure")
  sigs <- read_tsv(opt("sig-file"))
  row <- sigs[as.integer(opt("line")), ]
  case <- fuzz_case(row$seed)
  modes <- if (row$mode %in% c("eager", "lazy")) row$mode else c("eager", "lazy")
  fails <- function(k) {
    r <- fuzz_check(k, modes = modes)
    any(r$kind == row$kind & r$signature == row$signature)
  }
  if (nzchar(Sys.getenv("CUPLYR_FUZZ_TEST_SHRINK_NOREPRO")) || !fails(case)) {
    message("[shrink] seed ", row$seed, " did not reproduce (", row$kind, ", ", row$signature,
            "); left unshrunk")
    quit(save = "no", status = 3L)
  }
  small <- fuzz_shrink(case, fails = fails, max_attempts = 200L)
  left <- attr(small, "budget_exhausted")
  if (length(left)) {
    message("[shrink] attempt budget ran out in pass(es): ", paste(left, collapse = ", "),
            "; the case may not be minimal")
  }
  write_rows(data.frame(signature = row$signature, seed = row$seed,
                        shrunk = fz_tsv_text(fuzz_code(small)), stringsAsFactors = FALSE),
             opt("out"))
} else if (mode == "report") {
  f <- read_tsv(opt("findings"))
  if (is.null(f)) {
    f <- data.frame(seed = integer(), mode = character(), kind = character(), category = character(),
                    signature = character(), detail = character(), code = character(),
                    stringsAsFactors = FALSE)
  }
  sh <- read_tsv(opt("shrunk"))
  f$shrunk <- rep("", nrow(f))
  if (!is.null(sh) && nrow(f)) {
    hit <- match(f$signature, sh$signature)
    f$shrunk[!is.na(hit)] <- sh$shrunk[hit[!is.na(hit)]]
  }
  empty <- which(!nzchar(f$code))
  for (i in empty) f$code[i] <- fz_tsv_text(fuzz_code(fuzz_case(f$seed[i])))
  f <- f[order(f$category, f$signature, f$seed), ]
  utils::write.table(f, opt("out"), sep = "\t", quote = FALSE, row.names = FALSE)

  st <- if (file.exists(opt("stats")) && file.size(opt("stats")) > 0L) {
    utils::read.delim(opt("stats"), header = FALSE, quote = "", stringsAsFactors = FALSE,
                      col.names = c("seed", "native", "native_ok", "plan_break", "masked"), na.strings = character())
  }
  cat("\n== fuzz report ==\n")
  if (!is.null(st) && nrow(st)) {
    cat(sprintf("cases: %d, native share: %.3f (no CPU fallback), native-ok share: %.3f (also no side errored), plan-break share: %.3f\n",
                nrow(st), mean(st$native), mean(st$native_ok), mean(st$plan_break)))
    masked <- unlist(strsplit(as.character(st$masked), ";"))
    masked <- masked[grepl("=", masked, fixed = TRUE)]
    if (length(masked)) {
      kv <- do.call(rbind, strsplit(masked, "="))
      tot <- tapply(as.integer(kv[, 2]), kv[, 1], sum)
      cat("masked known differences:", paste(sprintf("%s=%d", names(tot), tot), collapse = ", "), "\n")
    }
  }
  cat(sprintf("findings: %d rows, %d seeds, %d distinct signatures\n",
              nrow(f), length(unique(f$seed)), length(unique(f$signature))))
  for (cat_name in c("silent_wrong", "crash_or_raw", "error_class", "warning_only")) {
    g <- f[f$category == cat_name, ]
    if (!nrow(g)) next
    cat(sprintf("\n-- %s: %d rows, %d seeds, %d causes\n", cat_name, nrow(g),
                length(unique(g$seed)), length(unique(g$signature))))
    for (sig in unique(g$signature)) {
      r <- g[g$signature == sig, ]
      cat(sprintf("  [%d seeds] %s\n      e.g. seed %d (%s): %s\n", length(unique(r$seed)), sig,
                  r$seed[1L], r$mode[1L], r$detail[1L]))
    }
  }
  cat(sprintf("\nwrote %s\n", opt("out")))
} else {
  stop("unknown mode: ", mode)
}
