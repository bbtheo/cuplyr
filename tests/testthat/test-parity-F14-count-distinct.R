# F14 acceptance tests: count()/add_count()/distinct() drop NULL dots, and
# dots the GPU expression IR can't take (incl. column-free calls that only
# work inside dplyr's data mask) go to dplyr instead of a generic error.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - count(df, NULL) == count(df); count(df, s, NULL) == count(df, s);
#   count(df, {{ grp }}) with grp = NULL == count(df); same for add_count().
# - count(df, k = my_n()) with my_n <- function() dplyr::n() gives k = n.
# - count(df, k = h(x)) / distinct(df, k = h(x)) with a user function works.
# - An expression that errors raises dplyr's own error
#   (class dplyr:::mutate_error, message carrying the cause).

f14c_df <- function() {
  data.frame(
    x = c(1L, 2L, 2L, NA, 5L, 2L),
    s = c("a", "b", "b", NA, "a", "b"),
    v = c(1.5, NA, 2.5, 2.5, 0, 1),
    stringsAsFactors = FALSE
  )
}

f14c_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f14c_count_by <- function(data, grp) count(data, {{ grp }})
f14c_add_count_by <- function(data, grp) add_count(data, {{ grp }})
f14c_distinct_by <- function(data, grp) distinct(data, {{ grp }})
f14c_my_n <- function() dplyr::n()
f14c_h <- function(z) z * 2
f14c_boom <- function() stop("boom")

# dplyr and cuplyr (eager and lazy) both error with the same message and
# dplyr's condition class.
f14c_same_error <- function(df, pipeline) {
  cnd <- function(x) tryCatch({
    out <- pipeline(x)
    if (inherits(out, "tbl_gpu")) out <- dplyr::collect(out)
    NULL
  }, error = function(e) e)
  cpu <- cnd(df)
  expect_false(is.null(cpu), info = "dplyr itself should error here")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- cnd(tbl_gpu(df, lazy = lazy))
    expect_false(is.null(gpu), info = "cuplyr should error too")
    expect_identical(conditionMessage(gpu), conditionMessage(cpu))
    expect_true(inherits(gpu, class(cpu)[1]))
  }
}

# =============================================================================
# NULL dots
# =============================================================================

test_that("count() ignores NULL dots", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) count(d, NULL))
  f14c_both(df, function(d) count(d, s, NULL), arrange_by = "s")
  f14c_both(df, function(d) count(d, NULL, s, NULL, sort = TRUE))
  f14c_both(df, function(d) count(d, NULL, wt = v))
  f14c_both(df, function(d) d |> group_by(s) |> count(NULL), arrange_by = "s")
})

test_that("count(df, {{ grp }}) with grp = NULL counts all rows", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) f14c_count_by(d, NULL))
  f14c_both(df, function(d) f14c_count_by(d, s), arrange_by = "s")
  f14c_both(df, function(d) d |> group_by(x) |> f14c_count_by(NULL), arrange_by = "x")
})

test_that("add_count() ignores NULL dots", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) add_count(d, NULL))
  f14c_both(df, function(d) f14c_add_count_by(d, NULL))
  f14c_both(df, function(d) add_count(d, NULL, s))
  f14c_both(df, function(d) d |> group_by(s) |> add_count(NULL))
})

test_that("distinct() ignores NULL dots among real keys", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) distinct(d, s, NULL))
  f14c_both(df, function(d) f14c_distinct_by(d, s))
  f14c_both(df, function(d) distinct(d, NULL, x, s))
  f14c_both(df, function(d) d |> group_by(s) |> distinct(x, NULL))
})

# =============================================================================
# Dots the IR can't take go to dplyr
# =============================================================================

test_that("count() with a column-free call that needs the data mask works", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) count(d, k = f14c_my_n()))
  f14c_both(df, function(d) count(d, s, k = f14c_my_n()), arrange_by = "s")
  f14c_both(df, function(d) add_count(d, k = f14c_my_n()))
  f14c_both(df, function(d) d |> group_by(s) |> add_count(k = f14c_my_n()))
})

test_that("distinct() with a column-free call that needs the data mask works", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) distinct(d, k = f14c_my_n()))
  f14c_both(df, function(d) distinct(d, s, k = f14c_my_n()))
  f14c_both(df, function(d) d |> group_by(s) |> distinct(k = f14c_my_n()))
  f14c_both(df, function(d) distinct(d, k = f14c_my_n(), .keep_all = TRUE))
})

test_that("count()/distinct() with a user function of a column work", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_both(df, function(d) count(d, k = f14c_h(x)), arrange_by = "k")
  f14c_both(df, function(d) distinct(d, k = f14c_h(x)))
  f14c_both(df, function(d) distinct(d, s, k = f14c_h(v), .keep_all = TRUE))
  f14c_both(df, function(d) d |> filter(!is.na(x)) |> distinct(k = f14c_h(x)))
})

test_that("count()/distinct() errors in a dot are dplyr's own", {
  skip_if_no_gpu()
  df <- f14c_df()
  f14c_same_error(df, function(d) count(d, k = f14c_boom()))
  f14c_same_error(df, function(d) add_count(d, k = f14c_boom()))
  f14c_same_error(df, function(d) distinct(d, k = f14c_boom()))
  f14c_same_error(df, function(d) distinct(d, k = f14c_h(no_such_col)))
})

test_that("native count()/distinct() calls stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f14c_df()
  f14c_both(df, function(d) count(d, s, NULL), arrange_by = "s")
  f14c_both(df, function(d) distinct(d, s, k = x + 1L))
  f14c_both(df, function(d) count(d, k = x * 2L), arrange_by = "k")
})
