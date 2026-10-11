# F40 acceptance tests: selections built from names resolve from the schema.
#
# A selection such as select(df, all_of(sort(names(df)))) only computes
# column names: tidyselect never shows it column values. It must resolve
# without reading the data (no CPU fallback, no collect), as it did before
# F14. Correctness must not change: a call that can hide a value predicate
# (a user helper returning where(~ anyNA(.x))) still resolves on the data.

f40n_df <- function() {
  data.frame(
    x1 = c(1L, NA, 3L),
    x2 = c(2.5, 3.5, NA),
    s = c("a", "b", "c"),
    b = c(TRUE, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
}

f40n_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f40n_native <- function(df, pipeline, ...) {
  withr::local_options(cuplyr.fallback = "error")
  f40n_both(df, pipeline, ...)
}

test_that("name-building calls in a selection resolve without the data", {
  skip_if_no_gpu()
  df <- f40n_df()
  nm <- c("s", "x1")
  f40n_native(df, function(d) select(d, sort(names(d))))
  f40n_native(df, function(d) select(d, all_of(sort(names(d)))))
  f40n_native(df, function(d) select(d, rev(names(d))))
  f40n_native(df, function(d) select(d, paste0("x", 1:2)))
  f40n_native(df, function(d) select(d, all_of(paste0("x", 2:1)), s))
  f40n_native(df, function(d) select(d, all_of(nm)))
  f40n_native(df, function(d) select(d, any_of(c(nm, "zz"))))
  f40n_native(df, function(d) select(d, -all_of(nm)))
  f40n_native(df, function(d) select(d, all_of(setdiff(names(d), "s"))))
  # tidyselect soft-deprecates .data in selections; muted on both sides
  f40n_native(df, function(d) suppressWarnings(select(d, .data$s, .data[["x2"]])))
  f40n_native(df, function(d) select(d, unique(c("s", "s", "b"))))
  f40n_native(df, function(d) select(d, seq_len(2)))
  f40n_native(df, function(d) relocate(d, all_of(rev(nm))))
  f40n_native(df, function(d) relocate(d, sort(names(d))))
  f40n_native(df, function(d) rename_with(d, toupper, .cols = all_of(sort(nm))))
  f40n_native(df, function(d) rename(d, any_of(c(z = "s", w = "zz"))))
  f40n_native(df, function(d) d |> group_by(s) |> select(all_of(c("x1", "b"))) |> suppressMessages())
})

test_that("calls that may hide a value predicate still resolve on the data", {
  skip_if_no_gpu()
  df <- f40n_df()
  na_cols <- function() tidyselect::where(~ anyNA(.x))
  f40n_both(df, function(d) select(d, na_cols()))
  f40n_both(df, function(d) select(d, c(s, na_cols())))
  f40n_both(df, function(d) relocate(d, na_cols(), .after = b))
  # a whitelisted name builder wrapping something else is not trusted blindly
  f40n_both(df, function(d) select(d, c(na_cols(), all_of(sort(c("s", "b"))))))
})

test_that("select(all_of(sort(names()))) on 1e6 rows is metadata-only", {
  skip_if_no_gpu()
  n <- 1e6
  df <- data.frame(b = seq_len(n), a = as.double(seq_len(n)), c = seq_len(n) %% 7L)
  g <- as_lazy(tbl_gpu(df))
  withr::local_options(cuplyr.fallback = "error")
  plain <- perf_time(function() select(g, a, b, c), iterations = 5)
  # resolving names must not read or transfer the table: same order of cost
  # as a plain by-name select (a collect of 1e6 rows costs orders more)
  for (sel in list(function() select(g, all_of(sort(names(g)))),
                   function() select(g, sort(names(g))),
                   function() select(g, paste0(c("a", "b"), ""), rev(names(g))[1]))) {
    built <- perf_time(sel, iterations = 5)
    expect_lt(as.numeric(built), max(5 * as.numeric(plain), as.numeric(plain) + 0.002))
  }
  expect_identical(names(select(g, all_of(sort(names(g))))), c("a", "b", "c"))
})
