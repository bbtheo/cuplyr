# F63 round 3 (review N12, N13): render bench/rolling-window.qmd at smoke size
# with probe outcomes injected into a copy of rolling-lib.R, so that the skip
# paths other than the median's run end to end:
#
#   - CPU fallbacks (skipped, not timed): cuplyr sd at every width, cuplyr mean
#     at the narrowest width, the froll alias of sum at the middle width, the
#     grouped cuplyr entry, the larger-table ungrouped cuplyr mean at the widest
#     width, and the cumsum overhead mutate;
#   - a probe error that is not a fallback: cuplyr max at the widest width,
#     which must become a failed cell in correctness.csv, not a skip.
#
# The document must render, list exactly those skips, count the failed cell,
# and not write a comparison word next to a missing figure. Opt-in like the
# acceptance render test: CUPLYR_BENCH_RENDER=1.

test_that("the report renders with non-median skips and a failed probe", {
  skip_if_no_gpu()
  skip_if(Sys.getenv("CUPLYR_BENCH_RENDER") != "1",
          "set CUPLYR_BENCH_RENDER=1 to render the F63 report at smoke size")
  qmd_src <- f63_require("bench/rolling-window.qmd")
  lib_src <- f63_require("bench/rolling-lib.R")
  skip_if(!nzchar(f63_quarto()), "quarto CLI not found")
  lib <- f63_bench_lib()
  withr::local_libpaths(lib, action = "prefix")
  skip_if_not(requireNamespace("knitr", quietly = TRUE) &&
                requireNamespace("rmarkdown", quietly = TRUE),
              "knitr/rmarkdown not available")

  tmp <- withr::local_tempdir("f63-skip-")
  file.copy(c(qmd_src, lib_src), tmp)
  qmd <- file.path(tmp, "rolling-window.qmd")
  out_dir <- file.path(tmp, "out")
  dir.create(out_dir)
  params_file <- file.path(tmp, "smoke.yml")
  writeLines(c(
    sprintf("root: \"%s\"", f63_root()),
    sprintf("lib: \"%s\"", lib),
    sprintf("out_dir: \"%s\"", out_dir),
    sprintf("cache_dir: \"%s\"", file.path(tmp, "knitr-cache")),
    "n_rows: 5000", "widths: [11, 101, 1001]", "times: 2", "times_custom: 1",
    "cell_budget: 1", "probe_limit: 5", "group_rows: 5000", "n_groups: 10",
    "group_width: 101", "ext_rows: 20000", "ext_times: 2"
  ), params_file)

  # The injection: every probe call is identified by the closure it gets (its
  # body and the variables of the loop that built it); the rest go to the real
  # probe.
  cat('
# ---- F63 skip-path test: injected probe outcomes ----
.rw_probe_real <- rw_native_probe
rw_native_probe <- function(call) {
  e <- environment(call)
  g <- function(nm) get0(nm, envir = e, inherits = FALSE)
  b <- paste(deparse(body(call)), collapse = " ")
  fb <- list(native = FALSE, fallback = TRUE,
             reason = "cuplyr: mutate() fell back to CPU evaluation for: injected by the F63 skip test")
  err <- list(native = FALSE, fallback = FALSE, reason = "injected by the F63 skip test: out_of_memory")
  if (b == "cf(gt, width)") {
    lab <- g("v")$label; stat <- g("stat"); w <- g("width")
    if (stat == "sd" && lab == "cuplyr") return(fb)
    if (stat == "mean" && lab == "cuplyr" && w == 11) return(fb)
    if (stat == "sum" && lab == "cuplyr (froll alias)" && w == 101) return(fb)
    if (stat == "max" && lab == "cuplyr" && w == 1001) return(err)
  }
  if (!is.null(g("gtg"))) return(fb)
  if (b == "cf(gtb, width)" && g("width") == 1001) return(fb)
  if (b == "of(inp$g)" && identical(g("nm"), "window mutate(r = cumsum(x))")) return(fb)
  .rw_probe_real(call)
}
', file = file.path(tmp, "rolling-lib.R"), append = TRUE)

  r <- f63_render(qmd, out_dir, params_file)
  expect_equal(r$status, 0L, label = paste0("render exit status; log tail:\n",
                                            paste(utils::tail(r$log, 40), collapse = "\n")))
  if (r$status != 0L) return(invisible())
  html <- paste(readLines(file.path(out_dir, "rolling-window.html"), warn = FALSE), collapse = "\n")
  expect_no_match(html, "cell-output-error", fixed = TRUE)
  body <- gsub("(?s)<(script|style)[^>]*>.*?</\\1>", " ", html, perl = TRUE)
  body <- gsub("src=\"data:[^\"]*\"", "", body)
  text <- gsub("\\s+", " ", gsub("<[^>]+>", " ", body))
  # the prose without the tables (a failed cell's table row shows NA figures)
  prose <- gsub("\\s+", " ", gsub("<[^>]+>", " ", gsub("(?s)<table.*?</table>", " ", body, perl = TRUE)))
  # the prose around each match, so a failure names the sentence
  around <- function(pattern) {
    m <- gregexpr(pattern, prose, fixed = TRUE)[[1]]
    if (m[1] < 0) return(character())
    vapply(m, function(i) substr(prose, max(1, i - 120), i + 60), "")
  }

  # exactly the injected fallbacks (and the real median ones) are skipped
  sk <- utils::read.csv(file.path(out_dir, "skipped.csv"))
  key <- function(d) sort(paste(d$chunk, d$stat, d$fn, ifelse(is.na(d$width), "NA", d$width)))
  expected <- data.frame(
    chunk = c(rep("bench-main", 8), "bench-overhead", "bench-overhead", "bench-ext", "bench-ext"),
    stat = c("median", "median", "median", "sd", "sd", "sd", "mean", "sum", "overhead", "overhead",
             "grouped mean", "ungrouped mean"),
    fn = c(rep("cuplyr", 7), "cuplyr (froll alias)", "window mutate(r = cumsum(x))",
           "window mutate(r = cumsum(x))", "cuplyr", "cuplyr"),
    width = c(11, 101, 1001, 11, 101, 1001, 11, 101, NA, NA, 101, 1001))
  expect_identical(key(sk), key(expected))
  expect_true(all(grepl("fell back to CPU evaluation", sk$reason, fixed = TRUE)))

  # the non-fallback probe error is a failed cell, counted, not a skip
  cr <- utils::read.csv(file.path(out_dir, "correctness.csv"))
  bad <- cr[cr$stat == "max" & cr$fn == "cuplyr" & cr$width == 1001, ]
  expect_equal(nrow(bad), 1L)
  expect_false(bad$ok)
  expect_match(bad$note, "^error: probe call failed: injected")
  expect_match(text, sprintf("%d of %d", sum(cr$ok), nrow(cr)), fixed = TRUE)
  expect_false(any(cr$stat == "sd" & cr$fn == "cuplyr"))

  # no skipped cell is timed
  tm <- utils::read.csv(file.path(out_dir, "timings.csv"))
  timed <- tm[!is.na(tm$ms), ]
  expect_false(any(timed$stat == "sd" & timed$fn == "cuplyr"))
  expect_false(any(timed$stat == "mean" & timed$fn == "cuplyr" & timed$width == 11))
  expect_false(any(timed$fn == "window mutate(r = cumsum(x))"))
  ext <- utils::read.csv(file.path(out_dir, "extension.csv"))
  expect_false(any(ext$fn == "cuplyr" & grepl("grouped", ext$experiment) & !grepl("ungrouped", ext$experiment)))
  expect_false(any(ext$fn == "cuplyr" & ext$width == 1001 & grepl("ungrouped", ext$experiment)))

  # no comparison word or figure pair built on a missing value
  for (p in c("level with NA", "NA NA", "NAx", "NA%")) {
    expect_identical(around(p), character(), label = paste0("text around \"", p, "\""))
  }
  # the tables mark the partial skip and the failed cell in the entry's own row
  expect_match(text, "not timed (CPU fallback)", fixed = TRUE)
  expect_match(text, "error (see Correctness)", fixed = TRUE)
  expect_match(text, "it was not timed at w=1001", fixed = TRUE)
})
