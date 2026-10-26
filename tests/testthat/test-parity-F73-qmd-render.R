# F73: render bench/rolling-window.qmd at smoke size (a temp copy, clean
# cache) and check the polars and DuckDB results it writes: every statistic x
# width at default threads and one thread, correct against data.table, the
# grouped and larger-table extensions, the thread counts in effect, and the new
# section of the report. Opt-in, like the F63 render test: CUPLYR_BENCH_RENDER=1.

test_that("the report renders polars and DuckDB at default threads and one thread", {
  skip_if_no_gpu()
  skip_if(Sys.getenv("CUPLYR_BENCH_RENDER") != "1",
          "set CUPLYR_BENCH_RENDER=1 to render the F73 report at smoke size")
  qmd_src <- f63_require("bench/rolling-window.qmd")
  lib_src <- f63_require("bench/rolling-lib.R")
  skip_if(!nzchar(f63_quarto()), "quarto CLI not found")
  lib <- f63_bench_lib()
  withr::local_libpaths(lib, action = "prefix")
  for (p in c("knitr", "rmarkdown", "polars", "duckdb", "DBI", "callr")) {
    expect_true(requireNamespace(p, quietly = TRUE), label = paste(p, "available"))
  }

  tmp <- withr::local_tempdir("f73-render-")
  file.copy(c(qmd_src, lib_src), tmp)
  qmd <- file.path(tmp, "rolling-window.qmd")
  out_dir <- file.path(tmp, "out")
  cache_dir <- file.path(tmp, "knitr-cache")
  dir.create(out_dir)
  # the install record the report reads (a copy, so the test never writes the real one)
  real_out <- "/home/theo/cuplr/scratchpad/benchmarks/rolling"
  for (f in list.files(real_out, pattern = "^install.*\\.(log|csv)$", full.names = TRUE)) file.copy(f, out_dir)
  widths <- c(11, 101, 1001)
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

  r1 <- f63_render(qmd, out_dir, params_file)
  expect_equal(r1$status, 0L, label = paste0("render exit status; log tail:\n",
                                             paste(utils::tail(r1$log, 40), collapse = "\n")))
  if (r1$status != 0L) return(invisible())
  html_path <- file.path(out_dir, "rolling-window.html")
  expect_true(file.exists(html_path))
  html <- paste(readLines(html_path, warn = FALSE), collapse = "\n")
  expect_no_match(html, "cell-output-error", fixed = TRUE)
  text <- gsub("\\s+", " ", gsub("<[^>]+>", " ", html))

  engines <- c("polars", "polars (1 thread)", "DuckDB", "DuckDB (1 thread)")
  stats <- c("mean", "median", "min", "max", "sum", "sd")

  # every engine cell timed at the smoke size and correct
  tm <- utils::read.csv(file.path(out_dir, "timings.csv"))
  cr <- utils::read.csv(file.path(out_dir, "correctness.csv"))
  for (eng in engines) for (s in stats) for (w in widths) {
    lab <- sprintf("%s %s w=%s", eng, s, w)
    expect_true(any(tm$fn == eng & tm$stat == s & tm$width == w & tm$n_rows == 5000 & !is.na(tm$ms)),
                label = paste(lab, "timed"))
    hit <- cr$fn == eng & cr$stat == s & cr$width == w
    expect_identical(sum(hit), 1L, label = paste(lab, "in correctness.csv"))
    expect_true(all(cr$ok[hit]), label = paste(lab, "matches data.table"))
  }
  expect_false(any(tm$fn %in% engines & tm$stat == "custom"))
  expect_true(all(cr$ok))
  expect_match(text, sprintf("%d of %d", sum(cr$ok), nrow(cr)), fixed = TRUE)
  # no timed cuplyr CPU fallback (F63 amendment), unchanged
  expect_false(any(grepl("^cuplyr", tm$fn) & tm$stat == "median"))

  # the extensions: grouped and larger-table rows for every engine, correct
  ex <- utils::read.csv(file.path(out_dir, "extension.csv"))
  for (eng in engines) {
    g <- ex$fn == eng & grepl("^grouped", ex$experiment)
    expect_identical(sum(g), 1L, label = paste(eng, "grouped row"))
    expect_true(all(ex$correct[g] %in% TRUE), label = paste(eng, "grouped correct"))
    u <- ex$fn == eng & grepl("ungrouped", ex$experiment)
    expect_setequal(ex$width[u], widths)
    expect_true(all(ex$n_rows[u] == 20000))
    expect_true(all(ex$correct[u] %in% TRUE), label = paste(eng, "ungrouped correct"))
    expect_true(all(!is.na(ex$median_ms[g | u])))
  }

  # the thread counts in effect, read back from each engine
  th_path <- file.path(out_dir, "engine-threads.csv")
  expect_true(file.exists(th_path))
  th <- utils::read.csv(th_path)
  expect_true(all(c("engine", "config", "threads") %in% names(th)))
  get_th <- function(e, cfg) th$threads[th$engine == e & th$config == cfg]
  for (e in c("polars", "DuckDB")) {
    expect_identical(as.integer(get_th(e, "1 thread")), 1L, label = paste(e, "one-thread setting in effect"))
    if (parallel::detectCores() > 1) expect_gt(get_th(e, "default"), 1)
  }

  # the new section, the render time and the published path
  expect_match(text, "polars and DuckDB", fixed = TRUE)
  expect_match(text, "Where cuplyr wins and loses against every engine", fixed = TRUE)
  expect_match(text, "This render took", fixed = TRUE)
  expect_match(text, file.path(out_dir, "rolling-window.html"), fixed = TRUE)
  # the rendered document leaves no stray files next to the source
  expect_false(any(grepl("_files$|_cache$", list.dirs(tmp, recursive = FALSE))))
})
