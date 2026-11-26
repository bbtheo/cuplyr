# Help pages are written for users (#159): no development-tracking ids
# ("Phase 5, task W6", "F24", "Phase 7 J2"), no pointers into design notes,
# and the exported pages don't send a reader to source files or libcudf
# function names. The checks read the source tree (R/, man/, NEWS.md), so
# they skip where it isn't there (an installed package under R CMD check).

p159_root <- function() testthat::test_path("..", "..")

p159_skip_if_no_dir <- function(...) {
  skip_if_not(dir.exists(file.path(p159_root(), ...)),
              paste(file.path(...), "not available in this test context"))
}

# Each pattern is a kind of development id. Matched with perl = TRUE.
p159_id_patterns <- c(
  phase = "(?i)\\bphase[ _-]?\\d",
  task = "(?i)\\btasks? [A-Z]?\\d",
  feature_id = "\\bF\\d{2,3}[a-z]?\\b",
  task_series = "\\b[ADJLNW]\\d{1,2}\\b",
  # T4 is also an NVIDIA GPU ("Turing (T4, RTX 20 series)")
  t_series = "(?<!Turing \\()\\bT\\d\\b",
  bug_id = "\\bBug \\d",
  design_section = "(?i)\\bsections? \\d|\u00a7",
  design_note = "(?i)scratchpad|workflow_state|_design\\.md|todo\\.md"
)

# "file:line: text [pattern]" for every line that matches a pattern.
p159_offenders <- function(files, patterns, keep = function(lines) rep(TRUE, length(lines))) {
  out <- character()
  for (f in files) {
    lines <- readLines(f, warn = FALSE, encoding = "UTF-8")
    use <- keep(lines)
    for (nm in names(patterns)) {
      hit <- which(use & grepl(patterns[[nm]], lines, perl = TRUE))
      if (length(hit) > 0L) {
        out <- c(out, sprintf("%s:%d: %s [%s]", basename(f), hit, trimws(lines[hit]), nm))
      }
    }
  }
  out
}

p159_rd_files <- function() {
  list.files(file.path(p159_root(), "man"), pattern = "\\.Rd$", full.names = TRUE)
}

p159_is_internal <- function(f) {
  any(grepl("\\keyword{internal}", readLines(f, warn = FALSE), fixed = TRUE))
}

test_that("no roxygen comment in R/ carries a development id", {
  p159_skip_if_no_dir("R")
  files <- list.files(file.path(p159_root(), "R"), pattern = "\\.R$", full.names = TRUE)
  expect_gt(length(files), 0L)
  roxygen <- function(lines) grepl("^\\s*#'", lines)
  bad <- p159_offenders(files, p159_id_patterns, keep = roxygen)
  expect(length(bad) == 0L,
         paste0(length(bad), " roxygen line(s) carry a development id:\n",
                paste(utils::head(bad, 40L), collapse = "\n")))
})

test_that("no help page in man/ carries a development id", {
  p159_skip_if_no_dir("man")
  files <- p159_rd_files()
  expect_gt(length(files), 0L)
  # skip the "% Please edit documentation in R/x.R" header
  body <- function(lines) !grepl("^%", lines)
  bad <- p159_offenders(files, p159_id_patterns, keep = body)
  expect(length(bad) == 0L,
         paste0(length(bad), " Rd line(s) carry a development id:\n",
                paste(utils::head(bad, 40L), collapse = "\n")))
})

test_that("exported help pages don't point at source files or libcudf functions", {
  p159_skip_if_no_dir("man")
  files <- Filter(Negate(p159_is_internal), p159_rd_files())
  expect_gt(length(files), 20L)
  patterns <- c(
    source_file = "\\b(R|src)/[A-Za-z0-9_.-]+\\.(R|cpp|hpp|cu|cuh|h)\\b",
    libcudf_function = "cudf::"
  )
  body <- function(lines) !grepl("^%", lines)
  bad <- p159_offenders(files, patterns, keep = body)
  expect(length(bad) == 0L,
         paste0(length(bad), " exported Rd line(s) cite internals:\n",
                paste(utils::head(bad, 40L), collapse = "\n")))
})

test_that("exported help pages don't name cuplyr's internal functions or fields", {
  p159_skip_if_no_dir("man")
  ns_file <- file.path(p159_root(), "NAMESPACE")
  skip_if_not(file.exists(ns_file), "NAMESPACE not available in this test context")
  ns_lines <- readLines(ns_file, warn = FALSE)
  exports <- sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", ns_lines, value = TRUE))
  exports <- gsub("^\"|\"$|`", "", exports)
  ns <- asNamespace("cuplyr")
  fns <- Filter(function(nm) is.function(get(nm, envir = ns)), ls(ns))
  # S3 methods (print.tbl_gpu()) are fine to name; so is anything another
  # package a user knows also defines
  others <- unlist(lapply(c("base", "stats", "utils", "methods", "dplyr", "tibble",
                            "vctrs", "rlang", "tidyselect"),
                          function(p) if (requireNamespace(p, quietly = TRUE)) getNamespaceExports(p)))
  internal <- setdiff(fns, c(exports, others))
  internal <- internal[grepl("^[A-Za-z][A-Za-z0-9_]*$", internal)]
  expect_gt(length(internal), 50L)
  alts <- paste(internal, collapse = "|")
  # a call `fn()`, or a link `\\link[=fn]{...}`
  call_pattern <- paste0("(?<![A-Za-z0-9_.])(", alts, ")\\(")
  link_pattern <- paste0("\\[=(", alts, ")\\]")
  files <- Filter(Negate(p159_is_internal), p159_rd_files())
  body <- function(lines) !grepl("^%", lines)
  fields <- "lazy_ops|(?<![.\\w])exec_mode\\b|\\$group_drop|\\$fixed_keys"
  bad <- p159_offenders(files, c(internal_function = call_pattern, internal_link = link_pattern,
                          internal_field = fields),
                        keep = body)
  expect(length(bad) == 0L,
         paste0(length(bad), " exported Rd line(s) name an internal function:\n",
                paste(utils::head(bad, 40L), collapse = "\n")))
})

test_that("exported help pages carry no development history", {
  p159_skip_if_no_dir("man")
  files <- Filter(Negate(p159_is_internal), p159_rd_files())
  patterns <- c(
    history = paste0("(?i)\\b(fixed in|older versions of this package|now-removed|",
                     "previously (a |CPU )?fallback|planned for a later phase|",
                     "as of (this|the) (phase|wave|task)|this wave|",
                     "historical(ly)?|verified empirically|not assumed|checked on \\d+)\\b")
  )
  body <- function(lines) !grepl("^%", lines)
  bad <- p159_offenders(files, patterns, keep = body)
  expect(length(bad) == 0L,
         paste0(length(bad), " exported Rd line(s) narrate development history:\n",
                paste(utils::head(bad, 40L), collapse = "\n")))
})

test_that("?tbl_gpu describes what a user can do with a GPU table, not its fields", {
  p159_skip_if_no_dir("man")
  rd_file <- file.path(p159_root(), "man", "tbl_gpu.Rd")
  expect_true(file.exists(rd_file))
  rd <- tools::parse_Rd(rd_file)
  tags <- vapply(rd, function(x) attr(x, "Rd_tag"), character(1))
  expect_equal(sum(tags == "\\value"), 1L)
  value <- paste(unlist(rd[tags == "\\value"]), collapse = "")
  expect_true(nzchar(value))
  # the internal fields of the list are not the user's interface
  for (field in c("ptr", "lazy_ops", "exec_mode", "External pointer", "external pointer")) {
    expect_false(grepl(field, value, fixed = TRUE), info = field)
  }

  txt <- paste(readLines(rd_file, warn = FALSE), collapse = "\n")
  # what a user does with one
  for (topic in c("collect()", "compute()", "explain()", "show_query()",
                  "cuplyr.exec_mode", "lazy = FALSE")) {
    expect_true(grepl(topic, txt, fixed = TRUE), info = topic)
  }
  # the type mapping covers every supported column type, one \item each
  items <- grep("^\\s*\\\\item", strsplit(txt, "\n", fixed = TRUE)[[1]], value = TRUE)
  for (type in c("FLOAT64", "INT32", "STRING", "BOOL8", "TIMESTAMP_DAYS",
                 "TIMESTAMP_MICROSECONDS", "factor", "integer64")) {
    expect_true(any(grepl(type, items, fixed = TRUE)), info = type)
  }
})

test_that("NEWS has a bullet for the help-page cleanup under 0.2.0", {
  news_file <- file.path(p159_root(), "NEWS.md")
  skip_if_not(file.exists(news_file), "NEWS.md not available in this test context")
  lines <- readLines(news_file, warn = FALSE, encoding = "UTF-8")
  heads <- grep("^# ", lines)
  start <- grep("^# cuplyr 0\\.2\\.0\\s*$", lines)
  expect_length(start, 1L)
  after <- heads[heads > start]
  section <- lines[seq(start, if (length(after)) after[1] - 1L else length(lines))]
  sub_heads <- grep("^## ", section)
  minor <- grep("^## Minor improvements and bug fixes", section)
  expect_length(minor, 1L)
  next_head <- sub_heads[sub_heads > minor]
  minor_block <- section[seq(minor, if (length(next_head)) next_head[1] - 1L else length(section))]
  expect_true(any(grepl("(#159)", minor_block, fixed = TRUE)))
})
