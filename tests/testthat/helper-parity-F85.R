# Helpers for the F85 acceptance tests (parquet read/write and pushdown).
#
# `parquet_last_read()` and the option `cuplyr.parquet_pushdown` are the
# internal hooks these tests rely on. `parquet_last_read()` is NULL before any
# read since the last `parquet_last_read(reset = TRUE)`; a scan's footer probe
# at read_parquet_gpu() time is not a read. After a read it is a list:
#   paths (chr), columns (chr, the columns asked of the reader as output),
#   predicates (chr, one entry per pushed conjunct, `character()` for none),
#   rows_read (num), row_groups_total (int),
#   row_groups_after_stats (int, NA when no filter was pushed).

# A parquet file with the rows of `df`, deleted with the calling frame.
pq_write <- function(df, row_group_size = NULL, ..., envir = parent.frame()) {
  path <- tempfile(fileext = ".parquet")
  withr::defer(unlink(path), envir = envir)
  write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path, row_group_size = row_group_size, ...)
  path
}

# The committed files written by arrow-R (not by cuplyr), see the F85 brief.
pq_fixture <- function(name) test_path("fixtures", "parquet", name)

# `pipeline` over a parquet copy of `df`: eager, lazy with pushdown, lazy
# without pushdown, and over tbl_gpu(df), each against dplyr on the tibble.
expect_parquet_same_as_dplyr <- function(df, pipeline, ..., arrange_by = NULL,
                                         ignore_col_types = FALSE,
                                         row_group_size = NULL, info = NULL) {
  path <- pq_write(df, row_group_size = row_group_size)
  oracle_result <- pipeline(tibble::as_tibble(df))
  oracle_groups <- oracle_group_vars(oracle_result)
  oracle_tbl <- oracle_normalize(oracle_result, arrange_by, ignore_col_types)

  check <- function(input, label) {
    result <- pipeline(input)
    lbl <- paste(c(info, label), collapse = " / ")
    testthat::expect_equal(oracle_group_vars(result), oracle_groups, info = lbl)
    got <- oracle_normalize(collect(result), arrange_by, ignore_col_types)
    testthat::expect_equal(names(got), names(oracle_tbl), info = lbl)
    testthat::expect_equal(got, oracle_tbl, info = lbl)
  }

  check(read_parquet_gpu(path, lazy = FALSE), "eager read")
  withr::with_options(list(cuplyr.parquet_pushdown = TRUE),
                      check(read_parquet_gpu(path, lazy = TRUE), "lazy, pushdown on"))
  withr::with_options(list(cuplyr.parquet_pushdown = FALSE),
                      check(read_parquet_gpu(path, lazy = TRUE), "lazy, pushdown off"))
  check(tbl_gpu(df, lazy = TRUE), "tbl_gpu(df), lazy")
  invisible(NULL)
}

# What the last scan pushed, after running `q`.
pq_run <- function(q) {
  parquet_last_read(reset = TRUE)
  out <- collect(q)
  list(result = out, read = parquet_last_read())
}

# The frame behind the semantics and pushdown tests: NA in every column, NaN
# and infinities in the doubles, an empty string, a non-ASCII string, unused
# factor levels.
pq_sem_df <- function() {
  tibble::tibble(
    id = 1:12,
    g = rep(c("a", "b", "c"), 4L),
    s = c("x", "y", NA, "x", "z", "", "y", "x", NA, "é", "x", "y"),
    i = c(1L, NA, 3L, 4L, -5L, 6L, NA, 8L, 2147483647L, 10L, 0L, 2L),
    d = c(1.5, NA, NaN, -2.5, 0, 1.5, Inf, -Inf, 1e300, 2, NA, 3.25),
    l = c(TRUE, NA, FALSE, TRUE, NA, FALSE, TRUE, TRUE, FALSE, NA, TRUE, FALSE),
    dt = as.Date("2020-01-01") + c(0, 1, NA, 3, 4, 5, 6, NA, 8, 9, 10, 11),
    ts = as.POSIXct("2021-03-04 05:06:07", tz = "UTC") +
      c(0, 1.5, NA, 3, 0.25, 5, 6, NA, 8, 9.75, 10, 11),
    f = factor(c("b", "a", NA, "c", "b", "b", "a", "c", NA, "a", "c", "b"),
               levels = c("c", "b", "a", "unused"))
  )
}
