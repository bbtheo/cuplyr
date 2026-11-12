# F121: end-to-end smoke run of the parquet pipeline benchmark at tiny sizes:
# every driver step (gen, check, readers, time, first, render) into a scratch
# output directory, then the CSV contracts and the rendered HTML. Opt-in
# (CUPLYR_BENCH_RENDER=1), like the F63/F73 render tests: it starts child R
# processes and the quarto CLI, so the merge-gate suite skips it.

f121_read_csv <- function(dir, name) {
  path <- file.path(dir, name)
  expect_true(file.exists(path), label = paste(name, "written"))
  if (!file.exists(path)) skip(paste(name, "missing (failure recorded)"))
  utils::read.csv(path, stringsAsFactors = FALSE)
}

f121_run_driver <- function(pp_list) {
  driver <- f121_file("bench/parquet-pipeline.R")
  assign("PP", pp_list, envir = globalenv())
  on.exit(rm("PP", envir = globalenv()), add = TRUE)
  withr::with_dir(dirname(dirname(driver)), source(driver, local = new.env(parent = globalenv())))
}

test_that("a smoke run writes every CSV, times only checked cells, and renders the report", {
  skip_if_no_gpu()
  if (!identical(Sys.getenv("CUPLYR_BENCH_RENDER"), "1")) skip("set CUPLYR_BENCH_RENDER=1 to run the smoke benchmark")
  if (!nzchar(f63_quarto())) skip("quarto CLI not found")
  f121_load(c("arrow", "duckdb", "DBI", "data.table", "nanoparquet", "callr", "knitr", "rmarkdown"))
  d <- f121_tmpdir()
  if (f121_is_tmpfs(d)) skip("scratch directory is on tmpfs; cold reps cannot be observed")
  root <- normalizePath(testthat::test_path("..", ".."))
  bench_status <- function() system2("git", c("-C", shQuote(root), "status", "--porcelain", "--", "bench"), stdout = TRUE)
  status_before <- bench_status()
  sizes <- c(20000L, 60000L)
  cfg <- list(
    steps = c("gen", "check", "readers", "time", "first", "render"),
    sizes = sizes, tables = c("narrow", "wide"), caches = c("warm", "cold"),
    reps = 2L, row_group_rows = 5000L,
    data_dir = file.path(d, "data"), out_dir = file.path(d, "out"),
    root = root, lib = f121_bench_lib(),
    busy_max = 1, first_rows = 20000L, first_calls = 3L
  )
  f121_run_driver(cfg)
  out <- cfg$out_dir
  engines <- c("cuplyr", "duckdb", "dplyr", "data.table")

  files <- f121_read_csv(out, "files.csv")
  expect_true(all(c("table", "rows", "path", "bytes", "row_groups", "codec", "columns") %in% names(files)))
  expect_identical(nrow(files), 4L)
  expect_setequal(paste(files$table, files$rows), paste(rep(c("narrow", "wide"), each = 2), sizes))
  expect_true(all(files$bytes > 0))
  expect_equal(files$row_groups, ceiling(files$rows / 5000))
  expect_true(all(toupper(files$codec) == "SNAPPY"))
  expect_equal(sort(unique(files$columns)), c(8, 40))

  cr <- f121_read_csv(out, "correctness.csv")
  expect_true(all(c("engine", "variant", "table", "rows", "status", "detail") %in% names(cr)))
  expect_identical(nrow(cr), length(engines) * 6L * 2L * length(sizes))
  expect_setequal(unique(cr$engine), engines)
  expect_true(all(cr$status == "ok"), label = paste(cr$detail[cr$status != "ok"], collapse = "; "))

  pd <- f121_read_csv(out, "pushdown.csv")
  expect_true(all(c("variant", "table", "rows", "columns_read", "predicates", "rows_read",
                    "row_groups_total", "row_groups_after_stats") %in% names(pd)))
  push <- pd[grepl("_push$", pd$variant) & !startsWith(pd$variant, "p2"), ]
  expect_gt(nrow(push), 0L)
  expect_true(all(push$row_groups_after_stats < push$row_groups_total))

  th <- f121_read_csv(out, "threads.csv")
  expect_true(all(c("engine", "setting", "value") %in% names(th)))
  expect_true(all(c("cuplyr", "duckdb", "dplyr", "data.table") %in% th$engine))

  rd <- f121_read_csv(out, "readers.csv")
  expect_true(all(c("reader", "table", "rows", "secs", "busy") %in% names(rd)))
  expect_setequal(unique(rd$reader), c("arrow", "nanoparquet"))

  tm <- f121_read_csv(out, "timings.csv")
  expect_true(all(c("engine", "variant", "table", "rows", "cache", "rep", "secs", "busy",
                    "resident", "quiet", "ok", "status", "started") %in% names(tm)))
  key <- function(df) paste(df$engine, df$variant, df$table, df$rows)
  expect_true(all(key(tm) %in% key(cr[cr$status == "ok", ])), label = "only checked cells are timed")
  cells <- unique(tm[c("engine", "variant", "table", "rows", "cache")])
  expect_identical(nrow(cells), length(engines) * 6L * 2L * length(sizes) * 2L)
  for (i in seq_len(nrow(cells))) {
    c1 <- merge(tm, cells[i, ])
    lab <- paste(unlist(cells[i, ]), collapse = " ")
    expect_true(0L %in% c1$rep, label = paste(lab, "has its warm-up rep"))
    expect_gte(sum(c1$rep >= 1L & as.logical(c1$quiet) & as.logical(c1$ok)), 2L, label = lab)
  }
  expect_false(anyNA(tm$busy[tm$status == "ok"]))
  expect_true(all(tm$resident[tm$cache == "cold"] <= 0.01), label = "cold calls start with the files evicted")
  expect_true(all(tm$resident[tm$cache == "warm"] >= 0.99), label = "warm calls start with the files cached")

  fc <- f121_read_csv(out, "firstcall.csv")
  expect_true(all(c("engine", "variant", "call", "secs", "load_secs", "pid") %in% names(fc)))
  expect_identical(nrow(fc), length(engines) * 6L * 3L)
  expect_identical(nrow(unique(fc[c("engine", "variant", "pid")])), length(engines) * 6L)

  html <- file.path(out, "parquet-pipeline.html")
  expect_true(file.exists(html))
  h <- paste(readLines(html, warn = FALSE), collapse = "\n")
  expect_match(h, "Correctness", fixed = TRUE)
  expect_match(h, "data:image/png;base64", fixed = TRUE)
  expect_false(grepl("<script src=\"[^\"h]", h))
  expect_false(dir.exists(file.path(out, "parquet-pipeline_files")))

  # re-running one step for a subset replaces exactly those rows
  started_before <- tm$started
  cfg2 <- modifyList(cfg, list(steps = "time", sizes = 20000L, tables = "narrow", caches = "warm"))
  f121_run_driver(cfg2)
  tm2 <- f121_read_csv(out, "timings.csv")
  sub1 <- tm$rows == 20000L & tm$table == "narrow" & tm$cache == "warm"
  sub2 <- tm2$rows == 20000L & tm2$table == "narrow" & tm2$cache == "warm"
  expect_identical(nrow(unique(tm2[c("engine", "variant", "table", "rows", "cache")])), nrow(cells))
  srt <- function(df) {
    df <- df[setdiff(names(df), "started")]
    df[do.call(order, unname(as.list(df[c("engine", "variant", "table", "rows", "cache", "rep")]))), ]
  }
  expect_equal(srt(tm2[!sub2, ]), srt(tm[!sub1, ]), ignore_attr = TRUE)
  # started is an ISO 8601 time stamp, so it sorts as text
  expect_true(min(tm2$started[sub2]) > max(tm$started[sub1]))

  # nothing stray left in bench/
  expect_identical(bench_status(), status_before)
})
