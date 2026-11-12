# F73: static checks of the rolling-window report with polars and DuckDB
# (bench/rolling-window.qmd, bench/install-deps.R). No GPU, no render; the
# parser and prose helpers come from helper-parity-F63.R. The F63 static test
# (no hand-typed number in the prose, cached bench-* chunks) still applies.

f73_prose_text <- function(lines) {
  p <- f63_parse_qmd(lines)
  paste(p$prose[!is.na(p$prose)], collapse = "\n")
}

test_that("install-deps.R installs duckdb, DBI and polars into the benchmark library", {
  path <- f63_require("bench/install-deps.R")
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  expect_no_error(parse(path, keep.source = FALSE))
  for (p in c("\"duckdb\"", "\"DBI\"", "\"polars\"")) expect_match(txt, p, fixed = TRUE)
  # the repository and switch the r-polars install docs name
  expect_match(txt, "https://community.r-multiverse.org", fixed = TRUE)
  expect_match(txt, "NOT_CRAN", fixed = TRUE)
  # build time per package in the summary
  expect_match(txt, "secs", fixed = TRUE)
})

test_that("a cached bench-engines chunk runs polars and DuckDB, keyed on their versions", {
  lines <- f63_qmd_lines()
  p <- f63_parse_qmd(lines)
  labels <- vapply(p$chunks, function(ch) ch$label, "")
  expect_true("bench-engines" %in% labels)
  if (!"bench-engines" %in% labels) return(invisible())
  ch <- p$chunks[[which(labels == "bench-engines")]]
  expect_true(f63_chunk_cached(ch))
  code <- paste(ch$code, collapse = "\n")
  expect_match(code, "rw_duckdb_engine(", fixed = TRUE)
  expect_match(code, "rw_polars_child(", fixed = TRUE)
  expect_match(code, "run_cell(", fixed = TRUE)
  expect_no_error(parse(text = code, keep.source = FALSE))

  setup <- p$chunks[[which(labels == "setup")]]
  setup_code <- paste(setup$code, collapse = "\n")
  for (pkg in c("polars", "duckdb", "DBI", "callr")) {
    expect_match(setup_code, paste0("\"", pkg, "\""), fixed = TRUE,
                 label = paste0("setup chunk names ", pkg, " (cache key and versions)"))
  }
})

test_that("the report has the polars and DuckDB section and documents the method", {
  lines <- f63_qmd_lines()
  expect_true(any(grepl("^## polars and DuckDB\\s*$", lines)))
  expect_true(any(grepl("^### Where cuplyr wins and loses against every engine\\s*$", lines)))
  prose <- f73_prose_text(lines)
  # how the engines are timed and threaded, stated in the prose (not only in code)
  for (s in c("POLARS_MAX_THREADS", "SET threads", "CREATE TEMP TABLE", "engine-threads.csv")) {
    expect_match(prose, s, fixed = TRUE, label = paste0("the prose mentions ", s))
  }
  expect_match(prose, "partial", fixed = TRUE, label = "the prose explains DuckDB's partial windows")
  expect_match(prose, "This render took", fixed = TRUE)
  # still no hand-typed number anywhere in the prose
  bad <- f63_prose_number_lines(lines)
  expect_equal(nrow(bad), 0L, label = paste("hand-typed numbers:", paste(bad$line, collapse = ", ")))
})
