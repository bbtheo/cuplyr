# F82 round 2: guards for the review findings that the acceptance tests don't
# pin down: a raw `.data[[k]]` on the other CPU routes and the order of notices
# for a failing column-free predicate.

r2_df <- function() {
  tibble::tibble(x = c(1, 5, 10, NA, 3), s = c("a", "zz", "b", NA, "zz"), g = c(1L, 1L, 2L, 2L, 2L))
}

test_that("F82: an injected raw .data[[k]] works in distinct() and count() on the CPU route", {
  skip_if_no_gpu()
  df <- r2_df()
  k <- "g"
  twice <- function(v) v * 2
  expect_same_as_dplyr(df, function(d) distinct(d, z = !!quote(twice(.data[[k]]))))
  expect_same_as_dplyr(df, function(d) count(d, z = !!quote(twice(.data[[k]]))), arrange_by = "z")
  expect_same_as_dplyr(df, function(d) group_by(d, z = !!quote(.data[[k]] + 1L)) |> summarise(n = n()),
                       arrange_by = "z")
})

test_that("F82: a typo in a column-free predicate errors before anything runs", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- r2_df()
  # dev raised the parser's error, with no table copied to the host
  for (lazy in c(FALSE, TRUE)) {
    err <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), made_up_f82 == 1, x > 1)), error = identity)
    expect_false(inherits(err, "cuplyr_error_fallback"), info = conditionMessage(err))
    expect_match(conditionMessage(err), "made_up_f82")
    err <- tryCatch(collect(filter_out(tbl_gpu(df, lazy = lazy), made_up_f82)), error = identity)
    expect_false(inherits(err, "cuplyr_error_fallback"), info = conditionMessage(err))
    expect_match(conditionMessage(err), "made_up_f82")
  }
})

test_that("F82: a name bound in the quosure env or in a function literal is not a typo", {
  skip_if_no_gpu()
  df <- r2_df()
  lim <- 2
  flag <- TRUE
  f82_unbound_ok <- function(d) filter(d, flag, (function(v) v > lim)(3), x > lim)
  expect_same_as_dplyr(df, f82_unbound_ok)
  expect_same_as_dplyr_lazy(df, f82_unbound_ok)
})

test_that("F82: an earlier dot's notice wins over a later column-free dot that fails", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- r2_df()
  fail <- function() stop("f82 boom")
  keep <- function(v) !is.na(v) & v > 2
  note <- tryCatch(filter(tbl_gpu(df), keep(x), fail()), error = identity)
  expect_s3_class(note, "cuplyr_error_fallback")
  expect_match(conditionMessage(note), "for: keep(x)", fixed = TRUE)
})
