# F124 (attempt 2): each min()/max() decision path (summarise, window mutate,
# window filter, `.by`, `.drop = FALSE`) on each source kind: an eager table, a
# lazy in-memory table (with and without pending steps) and a lazy parquet scan
# (bare, and with a filter the optimizer pushes into the scan). Every pipeline
# matches dplyr: values, column types, the lazy schema before collect(), and
# dplyr's folded warnings. On a parquet scan nothing runs at the verb that
# reads the file whole: no `cuplyr_plan_break`, and the read behind the result
# asks only for the columns the result needs.
# Helpers: helper-parity-F124.R (f124_capture(), f124_msgs(), f124_col_types()),
# helper-parity-F85.R (pq_write()).

f124g_df <- function() {
  set.seed(124)
  n <- 600L
  g <- rep(1:5, length.out = n)
  tibble::tibble(
    g = g,
    f = factor(rep(c("p", "q", "r"), length.out = n), levels = c("p", "q", "r", "z")),
    a = sample.int(1000L, n, replace = TRUE),
    an = ifelse(g == 5L, NA_integer_, sample.int(1000L, n, replace = TRUE)),
    b = runif(n),
    l = rep(c(TRUE, FALSE), length.out = n),
    s = sample(c("u", "v", "w"), n, replace = TRUE),
    v1 = runif(n), v2 = runif(n), v3 = runif(n)
  )
}

# `pipeline` on the tibble (dplyr) and on each source; `cols` are the columns
# the parquet read behind the result may ask for. `lazy_break = FALSE` also
# forbids a plan break on the lazy in-memory table (a dot whose type does not
# depend on the data has nothing to run early for). `parquet_break = TRUE` only
# where summarise() runs its plan at the verb by design (an integer min()/max(),
# F07): the summarise is part of what runs, so the read is still pruned. `cols =
# NULL` skips the column check (a path outside min()/max() reads more).
f124g_check <- function(df, path, pipeline, cols, info, lazy_break = TRUE, parquet_break = FALSE,
                        sources = c("eager", "lazy", "parquet")) {
  oracle <- f124_capture(pipeline(df))
  want <- oracle_normalize(oracle$value)
  want_groups <- dplyr::group_vars(oracle$value)
  make <- list(
    eager = function() tbl_gpu(df, lazy = FALSE),
    lazy = function() tbl_gpu(df, lazy = TRUE),
    parquet = function() read_parquet_gpu(path, lazy = TRUE)
  )
  for (s in sources) {
    lbl <- paste0(info, " [", s, "]")
    src <- make[[s]]()
    parquet_last_read(reset = TRUE)
    verb <- f124_capture(pipeline(src))
    expect_identical(verb$value$groups, want_groups, info = paste(lbl, "group vars"))
    planned <- if (s != "eager") f124_col_types(tidyselect_proxy(current_schema(verb$value)))
    col <- f124_capture(collect(verb$value))
    got <- oracle_normalize(col$value)
    expect_identical(f124_col_types(got), f124_col_types(want), info = paste(lbl, "collected types"))
    expect_equal(got, want, info = paste(lbl, "values"))
    if (!is.null(planned)) {
      expect_identical(planned, f124_col_types(want), info = paste(lbl, "lazy schema before collect()"))
    }
    expect_identical(f124_msgs(c(verb$warnings, col$warnings)), f124_msgs(oracle$warnings),
                     info = paste(lbl, "warnings"))
    expect_identical(c(verb$fallback, col$fallback), character(), info = paste(lbl, "CPU fallback"))
    if (s == "eager") {
      expect_length(col$warnings, 0L)
    }
    if ((s == "parquet" && !parquet_break) || (s == "lazy" && !lazy_break)) {
      expect_identical(c(verb$plan_break, col$plan_break), character(), info = paste(lbl, "plan break"))
    }
    if (s == "parquet") {
      rd <- parquet_last_read()
      expect_false(is.null(rd), info = paste(lbl, "a read happened"))
      if (!is.null(cols)) expect_setequal(rd$columns, cols)
    }
  }
}

test_that("F124 grid: window mutate() min()/max() with no empty group, every source", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  # (B11 P1, P4, P5: an integer window max() on a bare scan reads one column)
  f124g_check(df, path, function(d) select(mutate(d, m = max(a)), m), "a", "ungrouped int")
  f124g_check(df, path, function(d) select(mutate(group_by(d, g), m = min(a)), g, m), c("g", "a"),
              "grouped int")
  f124g_check(df, path, function(d) select(mutate(d, m = max(a), .by = g), g, m), c("g", "a"), ".by int")
  f124g_check(df, path, function(d) select(mutate(d, m = max(l)), m), "l", "logical")
  f124g_check(df, path, function(d) select(mutate(d, m = max(b)), m), "b", "double")
  f124g_check(df, path, function(d) select(mutate(d, r = runif(1) * 0, m = max(b)), m), "b",
              "next to an impure call")
  f124g_check(df, path, function(d) select(mutate(d, m = max(a) + 1L, k = m * 2L), k), "a",
              "an expression and a later dot")
  # row-preserving steps before the call keep the row count known
  f124g_check(df, path, function(d) select(mutate(arrange(mutate(d, z = a * 2L), b), m = max(z)), m),
              c("a", "b"), "after row-preserving steps", lazy_break = FALSE)
})

test_that("F124 grid: window mutate() after a filter (rows unknown at the verb), every source", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  # some rows left: integer type, no warning
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 0.5), m = max(a)), m), c("a", "b"),
              "filter, int")
  f124g_check(df, path, function(d) select(mutate(group_by(filter(d, b > 0.5), g), m = min(a)), g, m),
              c("g", "a", "b"), "filter, grouped int")
  # no row left: dplyr's double and warning
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 2), m = max(a)), m), c("a", "b"),
              "empty filter, int")
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 2), m = max(a), .by = g), g, m),
              c("g", "a", "b"), "empty filter, .by int")
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 2), m = max(b)), m), "b",
              "empty filter, double")
  # a dot whose type does not depend on the data never runs anything early
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 0.5), top = a == max(a)), top),
              c("a", "b"), "filter, comparison", lazy_break = FALSE)
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 2), top = a == max(a)), top),
              c("a", "b"), "empty filter, comparison", lazy_break = FALSE)
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 0.5), m = max(b)), m), "b",
              "filter, double", lazy_break = FALSE)
  f124g_check(df, path, function(d) select(mutate(filter(d, b > 0.5), r = runif(1) * 0, m = max(b)), m),
              "b", "filter, next to an impure call")
})

test_that("F124 grid: window filter() min()/max(), every source", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  f124g_check(df, path, function(d) select(filter(group_by(d, g), a == max(a)), g, a), c("g", "a"),
              "grouped filter", lazy_break = FALSE)
  f124g_check(df, path, function(d) select(filter(d, a == min(a), .by = g), g, a), c("g", "a"),
              ".by filter", lazy_break = FALSE)
  f124g_check(df, path, function(d) select(filter(group_by(filter(d, b > 2), g), a == max(a)), g, a),
              c("g", "a", "b"), "empty, grouped filter", lazy_break = FALSE)
})

test_that("F124 grid: .drop = FALSE groupings with an empty group, every source", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  # level "z" has no row: an integer max() is a double, one warning naming it
  f124g_check(df, path, function(d) select(mutate(group_by(d, f, .drop = FALSE), m = max(a)), f, m),
              c("f", "a"), ".drop = FALSE int")
  f124g_check(df, path, function(d) select(mutate(group_by(d, f, .drop = FALSE), m = max(b)), f, m),
              c("f", "b"), ".drop = FALSE double")
  f124g_check(df, path, function(d) select(mutate(group_by(d, f, .drop = FALSE), top = a == max(a)), f, top),
              c("f", "a"), ".drop = FALSE comparison")
  f124g_check(df, path, function(d) {
    select(mutate(group_by(filter(d, b > 0.5), f, .drop = FALSE), m = min(a)), f, m)
  }, c("f", "a", "b"), "filter, then .drop = FALSE int")
  # every level has a row once "z" is dropped from the levels: integer, no warning
  df2 <- dplyr::mutate(df, f = factor(as.character(f), levels = c("p", "q", "r")))
  path2 <- pq_write(df2)
  f124g_check(df2, path2, function(d) select(mutate(group_by(d, f, .drop = FALSE), m = max(a)), f, m),
              c("f", "a"), ".drop = FALSE, no empty group")
})

test_that("F124 grid: summarise() min()/max(), every source", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  withr::local_collate("C")
  # (summarise() of an integer min()/max() runs at the verb, F07, but the
  # summarise is part of what runs: the read stays pruned)
  chk <- function(pipeline, cols, info, ...) f124g_check(df, path, pipeline, cols, info, ...)
  chk(function(d) summarise(d, m = min(a)), "a", "ungrouped", parquet_break = TRUE)
  chk(function(d) summarise(group_by(d, g), m = max(an, na.rm = TRUE)), c("g", "an"), "all-NA group",
      parquet_break = TRUE)
  chk(function(d) summarise(d, m = max(an, na.rm = TRUE), .by = g), c("g", "an"), ".by all-NA group",
      parquet_break = TRUE)
  chk(function(d) summarise(group_by(d, g), m = max(b), n = min(s)), c("g", "b", "s"), "double, string")
  chk(function(d) summarise(filter(d, b > 2), m = min(a)), c("a", "b"), "empty filter", parquet_break = TRUE)
  # (a lazy summarise() over a .drop = FALSE grouping computes its whole input
  # first, F74, for any aggregation: not checked here)
  chk(function(d) summarise(group_by(d, f, .drop = FALSE), m = max(a)), NULL, ".drop = FALSE",
      parquet_break = TRUE)
})

test_that("F124 grid: a zero-row parquet file", {
  skip_if_no_gpu()
  df <- f124g_df()[0, ]
  path <- pq_write(df)
  f124g_check(df, path, function(d) select(mutate(d, m = max(a)), m), "a", "int")
  f124g_check(df, path, function(d) select(mutate(group_by(d, g), m = min(a)), g, m), c("g", "a"), "grouped int")
  f124g_check(df, path, function(d) select(mutate(d, m = max(b)), m), "b", "double")
  f124g_check(df, path, function(d) summarise(d, m = min(a)), "a", "summarise", parquet_break = TRUE)
})

test_that("F124 grid: a parquet file emptied after a min()/max() type was decided from its footer", {
  skip_if_no_gpu()
  df <- f124g_df()
  path <- pq_write(df)
  q <- mutate(read_parquet_gpu(path, lazy = TRUE), m = max(a))
  expect_identical(current_schema(q)$types[[match("m", current_schema(q)$names)]], "INT32")
  write_parquet_gpu(tbl_gpu(df[0, ], lazy = FALSE), path)
  # (the plan said <int>; dplyr gives a double for a zero-row file: refused)
  expect_error(collect(q), class = "cuplyr_error_parquet")
})
