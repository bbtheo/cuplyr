# F63: render bench/rolling-window.qmd at smoke size, five times, in a temp
# directory, and check the caching contract:
#
#   1. clean cache: renders; self-contained HTML; the CSVs are written to
#      params$out_dir; the knitr cache lives in params$cache_dir; the prose's
#      correctness count matches correctness.csv; the run used the params.
#   2. prose-only edit: every timing chunk is a cache hit (same cache hashes,
#      timings.csv unchanged).
#   3. code edit in the bench-main chunk: bench-main reruns (new hash, new
#      timings).
#   4. edit of rolling-lib.R (sourced, not chunk code): every timing chunk
#      reruns, because the cache key covers the helper file.
#   5. params$rebuild = true (what FULL_RERUN passes): every timing chunk
#      re-runs, and nothing else gets cached or left next to the qmd.
#
# The render runs a (tiny) benchmark on the GPU through a child R process
# started by quarto, so it is opt-in: CUPLYR_BENCH_RENDER=1. It needs the
# quarto CLI (~/.local/bin/quarto, or QUARTO_PATH) and knitr/rmarkdown.

test_that("the report renders at smoke size and its knitr cache behaves", {
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

  tmp <- withr::local_tempdir("f63-render-")
  file.copy(c(qmd_src, lib_src), tmp)
  qmd <- file.path(tmp, "rolling-window.qmd")
  lib_copy <- file.path(tmp, "rolling-lib.R")
  out_dir <- file.path(tmp, "out")
  cache_dir <- file.path(tmp, "knitr-cache")
  dir.create(out_dir)
  params_file <- file.path(tmp, "smoke.yml")
  writeLines(c(
    sprintf("root: \"%s\"", f63_root()),
    sprintf("lib: \"%s\"", lib),
    sprintf("out_dir: \"%s\"", out_dir),
    sprintf("cache_dir: \"%s\"", cache_dir),
    "n_rows: 5000",
    "widths: [11, 101, 1001]",
    "times: 2",
    "times_custom: 1",
    "cell_budget: 1",
    "probe_limit: 5",
    "group_rows: 5000",
    "n_groups: 10",
    "group_width: 101",
    "ext_rows: 20000",
    "ext_times: 2"
  ), params_file)

  labels <- vapply(f63_parse_qmd(readLines(qmd))$chunks, function(ch) ch$label, "")
  bench_labels <- labels[startsWith(labels, "bench-")]
  html_path <- file.path(out_dir, "rolling-window.html")
  hashes_of <- function(h, lb) sort(unname(h[names(h) == lb]))
  timings_md5 <- function() unname(tools::md5sum(file.path(out_dir, "timings.csv")))

  # ---- 1. clean cache -------------------------------------------------------------
  r1 <- f63_render(qmd, out_dir, params_file)
  expect_equal(r1$status, 0L, label = paste0("render 1 exit status; log tail:\n",
                                             paste(utils::tail(r1$log, 40), collapse = "\n")))
  if (r1$status != 0L) return(invisible())
  expect_true(file.exists(html_path))
  html <- paste(readLines(html_path, warn = FALSE), collapse = "\n")

  # self-contained: no relative src/href to a side file, no *_files directory
  refs <- regmatches(html, gregexpr("(src|href)=\"[^\"]*\"", html))[[1]]
  vals <- sub("^(src|href)=\"(.*)\"$", "\\2", refs)
  side <- vals[!grepl("^(data:|https?:|#|mailto:)", vals) & grepl("\\.(png|svg|jpe?g|css|js)$", vals)]
  expect_length(side, 0L)
  expect_false(any(grepl("_files$", list.dirs(tmp, recursive = TRUE))))
  n_img <- lengths(regmatches(html, gregexpr("src=\"data:image", html, fixed = TRUE)))
  expect_gte(n_img, 8L) # one plot per statistic (7) plus the extension
  expect_no_match(html, "cell-output-error", fixed = TRUE)

  # side outputs and the cache location
  for (f in c("timings.csv", "correctness.csv", "extension.csv", "versions.csv",
              "memory-breakdown.csv", "memory-finding.csv")) {
    expect_true(file.exists(file.path(out_dir, f)), label = paste(f, "written to params$out_dir"))
  }
  expect_false(dir.exists(file.path(tmp, "rolling-window_cache")))
  h1 <- f63_cache_hashes(cache_dir)
  for (lb in bench_labels) {
    expect_true(lb %in% names(h1), label = paste0("chunk ", lb, " cached under params$cache_dir"))
  }

  # the run used the params, and the prose agrees with the CSVs
  tm <- utils::read.csv(file.path(out_dir, "timings.csv"))
  expect_true(5000 %in% tm$n_rows)
  expect_false(1e6 %in% tm$n_rows)
  cr <- utils::read.csv(file.path(out_dir, "correctness.csv"))
  expect_true(all(cr$ok))
  text <- gsub("<[^>]+>", " ", html)
  text <- gsub("\\s+", " ", text)
  expect_match(text, sprintf("%d of %d", sum(cr$ok), nrow(cr)), fixed = TRUE)

  # amendment (Theo, 2026-10-22): no cuplyr CPU-fallback cell is timed or
  # checked; the median panel says cuplyr is not timed instead of a number
  is_cuplyr <- function(fn) grepl("^cuplyr", fn)
  expect_false(any(is_cuplyr(tm$fn) & tm$stat == "median"),
               label = "a cuplyr median (CPU fallback) row in timings.csv")
  expect_false(any(is_cuplyr(tm$fn) & grepl("fallback", tm$fn, ignore.case = TRUE)))
  expect_false(any(is_cuplyr(tm$fn) & grepl("fallback", tm$note, ignore.case = TRUE) &
                     !is.na(tm$ms)),
               label = "a timed cuplyr row whose note mentions a fallback")
  expect_false(any(is_cuplyr(cr$fn) & cr$stat == "median"),
               label = "a cuplyr median row in correctness.csv")
  expect_match(text, "not timed", ignore.case = TRUE)
  t1 <- timings_md5()

  # ---- 2. prose-only edit: cache hit ---------------------------------------------
  cat("\nProse-only edit for the F63 cache test.\n", file = qmd, append = TRUE)
  r2 <- f63_render(qmd, out_dir, params_file)
  expect_equal(r2$status, 0L, label = paste0("render 2 exit status; log tail:\n",
                                             paste(utils::tail(r2$log, 40), collapse = "\n")))
  if (r2$status != 0L) return(invisible())
  h2 <- f63_cache_hashes(cache_dir)
  for (lb in bench_labels) {
    expect_identical(hashes_of(h2, lb), hashes_of(h1, lb),
                     label = paste0("chunk ", lb, " is a cache hit after a prose-only edit"))
  }
  expect_identical(timings_md5(), t1)
  expect_match(paste(readLines(html_path, warn = FALSE), collapse = "\n"),
               "Prose-only edit for the F63 cache test.", fixed = TRUE)

  # ---- 3. code edit in bench-main: that chunk reruns -----------------------------
  q <- readLines(qmd)
  p <- f63_parse_qmd(q)
  main <- p$chunks[[which(labels == "bench-main")[1]]]
  k <- main$line + 1L
  while (k <= length(q) && grepl("^#\\|", q[k])) k <- k + 1L
  q <- append(q, "invisible(\"F63 cache test: code edit\")", after = k - 1L)
  writeLines(q, qmd)
  r3 <- f63_render(qmd, out_dir, params_file)
  expect_equal(r3$status, 0L, label = paste0("render 3 exit status; log tail:\n",
                                             paste(utils::tail(r3$log, 40), collapse = "\n")))
  if (r3$status != 0L) return(invisible())
  h3 <- f63_cache_hashes(cache_dir)
  expect_true(length(setdiff(hashes_of(h3, "bench-main"), hashes_of(h2, "bench-main"))) > 0L,
               label = "bench-main got a new cache hash after its code changed")
  expect_false(identical(timings_md5(), t1), label = "timings.csv rewritten by the rerun")

  # ---- 4. edit of the sourced helper file: every timing chunk reruns -------------
  cat("\n# F63 cache test: helper edit\n", file = lib_copy, append = TRUE)
  r4 <- f63_render(qmd, out_dir, params_file)
  expect_equal(r4$status, 0L, label = paste0("render 4 exit status; log tail:\n",
                                             paste(utils::tail(r4$log, 40), collapse = "\n")))
  if (r4$status != 0L) return(invisible())
  h4 <- f63_cache_hashes(cache_dir)
  for (lb in bench_labels) {
    expect_true(length(setdiff(hashes_of(h4, lb), hashes_of(h3, lb))) > 0L,
                 label = paste0("chunk ", lb, " got a new cache hash after rolling-lib.R changed"))
  }

  # ---- 5. forced full rerun (params$rebuild): every timing chunk re-runs ---------
  # (round-1 review B2: quarto's --cache-refresh re-ran nothing and cached
  # every chunk; the rebuild switch is a param that bench/rolling-window.R's
  # FULL_RERUN passes)
  t4 <- timings_md5()
  rebuild_file <- file.path(tmp, "smoke-rebuild.yml")
  writeLines(c(readLines(params_file), "rebuild: true"), rebuild_file)
  rdb <- function() {
    f <- list.files(cache_dir, pattern = "\\.rdb$", full.names = TRUE, recursive = TRUE)
    f[startsWith(basename(f), "bench-")]
  }
  before5 <- Sys.time()
  Sys.sleep(1.1)
  r5 <- f63_render(qmd, out_dir, rebuild_file)
  expect_equal(r5$status, 0L, label = paste0("render 5 exit status; log tail:\n",
                                             paste(utils::tail(r5$log, 40), collapse = "\n")))
  if (r5$status != 0L) return(invisible())
  expect_false(identical(timings_md5(), t4), label = "timings.csv rewritten by the forced rerun")
  for (lb in bench_labels) {
    f <- rdb()[startsWith(basename(rdb()), paste0(lb, "_"))]
    expect_true(length(f) > 0L && any(file.mtime(f) > before5),
                label = paste0("chunk ", lb, " re-ran and re-saved its cache in the forced rerun"))
  }
  expect_false(dir.exists(file.path(tmp, "rolling-window_cache")))
  expect_false(any(grepl("_files$", list.dirs(tmp, recursive = TRUE))))
  # only bench-* chunks are cached, also in the forced rerun
  all_rdb <- basename(list.files(cache_dir, pattern = "\\.rdb$", recursive = TRUE))
  expect_true(all(startsWith(all_rdb, "bench-")),
              label = paste("only bench-* chunks cached; found:", paste(all_rdb, collapse = ", ")))
})
