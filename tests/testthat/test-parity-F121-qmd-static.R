# F121: static checks of the parquet pipeline report (bench/parquet-pipeline.qmd),
# its driver (bench/parquet-pipeline.R) and bench/install-deps.R. No GPU, no
# render. The qmd parser and the prose linter come from helper-parity-F63.R.

f121_qmd_lines <- function() readLines(f121_file("bench/parquet-pipeline.qmd"), warn = FALSE)

f121_prose_text <- function(lines) {
  p <- f63_parse_qmd(lines)
  paste(p$prose[!is.na(p$prose)], collapse = "\n")
}

test_that("install-deps.R lists the readers and engines the benchmark uses", {
  path <- f121_file("bench/install-deps.R")
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  expect_no_error(parse(path, keep.source = FALSE))
  for (p in c("\"nanoparquet\"", "\"arrow\"", "\"duckdb\"", "\"data.table\"", "\"DBI\"", "\"callr\"")) {
    expect_match(txt, p, fixed = TRUE)
  }
})

test_that("the driver parses and runs the steps in order from a PP list", {
  path <- f121_file("bench/parquet-pipeline.R")
  expect_no_error(parse(path, keep.source = FALSE))
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  for (s in c("\"gen\"", "\"check\"", "\"readers\"", "\"time\"", "\"first\"", "\"render\"")) {
    expect_match(txt, s, fixed = TRUE, label = paste("the driver has step", s))
  }
  expect_match(txt, "parquet-lib.R", fixed = TRUE)
  expect_match(txt, "PP", fixed = TRUE)
  expect_false(grepl("install.packages", txt, fixed = TRUE))
})

test_that("the report is a self-contained HTML document with the expected params", {
  lines <- f121_qmd_lines()
  front <- f63_front(lines)
  fmt <- front$format$html
  expect_true(isTRUE(fmt$`embed-resources`) || isTRUE(fmt$`self-contained`))
  for (nm in c("root", "lib", "out_dir")) {
    expect_false(is.null(f63_param(front, nm)), label = paste("param", nm))
  }
})

test_that("the report reads the measurements and does not take them", {
  lines <- f121_qmd_lines()
  p <- f63_parse_qmd(lines)
  expect_gt(length(p$chunks), 0L)
  code <- paste(unlist(lapply(p$chunks, function(ch) ch$code)), collapse = "\n")
  for (ch in p$chunks) {
    expect_no_error(parse(text = ch$code, keep.source = FALSE))
  }
  for (bad in c("pp_time_cell(", "pp_run(", "pp_first_call(", "read_parquet_gpu(", "pp_gen_fact(",
                "install.packages(")) {
    expect_false(grepl(bad, code, fixed = TRUE), label = paste("an executed chunk calls", bad))
  }
  for (csv in c("files.csv", "correctness.csv", "pushdown.csv", "threads.csv", "readers.csv",
                "timings.csv", "firstcall.csv")) {
    expect_match(code, csv, fixed = TRUE, label = paste("the report reads", csv))
  }
  expect_match(code, "parquet-lib.R", fixed = TRUE)
})

test_that("the report has its sections and states the method in the prose", {
  lines <- f121_qmd_lines()
  heads <- trimws(sub("^#+", "", grep("^#{2,3} ", lines, value = TRUE)))
  for (h in c("Setup", "Correctness", "Results", "Where cuplyr wins and loses", "First call",
              "Findings", "Caveats")) {
    expect_true(any(startsWith(heads, h)), label = paste("a section starting", h))
  }
  prose <- f121_prose_text(lines)
  for (s in c("posix_fadvise", "fincore", "LIBCUDF_USE_DEBUG_STREAM_POOL", "F128", "F86", "F87",
              "nanoparquet", "setDT", "read_parquet(", "col_select", "read_parquet_gpu",
              "snappy", "F123")) {
    expect_match(prose, s, fixed = TRUE, label = paste("the prose mentions", s))
  }
  expect_match(prose, "busy", ignore.case = TRUE)
  expect_match(prose, "warm", ignore.case = TRUE)
  expect_match(prose, "cold", ignore.case = TRUE)
  expect_match(prose, "does not fit|out of memory", ignore.case = TRUE)
  expect_match(prose, "pushed|pushdown", ignore.case = TRUE)
})

test_that("every number in the prose is computed", {
  lines <- f121_qmd_lines()
  bad <- f63_prose_number_lines(lines)
  expect_equal(nrow(bad), 0L, label = paste("hand-typed numbers on lines:", paste(bad$line, collapse = ", ")))
  expect_gte(f63_inline_r_count(lines), 40L)
})
