# F82 round 3: guards the acceptance tests don't pin down: if_any()/if_all()
# date filters stay native (round-2 review N1), a failing column-free
# `FALSE || nope` gets dplyr's error, and a raw `.data[[k]]` reads the
# quosure's `k` on the routes that build a call (filter's call2(), sample_n()),
# next to a same-named outer variable.

r3_df <- function() {
  tibble::tibble(x = c(1, 5, 10, NA, 3), g = c(5L, 4L, 3L, 2L, 1L), w = c(0, 0, 1, 0, 0))
}

test_that("F82: an if_any() date filter with a column-free bound stays native", {
  skip_if_no_gpu()
  today <- as.Date("2026-11-01")
  df <- tibble::tibble(d1 = today - c(0, 40, 100, 10), d2 = today - c(50, 5, 60, NA), v = 1:4)
  pipes <- list(
    function(d) filter(d, if_any(c(d1, d2), ~ .x >= Sys.Date() - 30)),
    function(d) filter(d, if_all(c(d1, d2), ~ !is.na(.x) & .x < Sys.Date())),
    function(d) mutate(d, recent = dplyr::if_any(c(d1, d2), ~ .x >= Sys.Date() - 30))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_no_error(collect(p(tbl_gpu(df))))
      expect_no_error(collect(p(tbl_gpu(df, lazy = TRUE))))
    })
  }
})

test_that("F82: a failing column-free short-circuit dot gives dplyr's error", {
  skip_if_no_gpu()
  df <- r3_df()
  ref <- tryCatch(filter(df, FALSE || not_defined_f82r3), error = identity)
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), FALSE || not_defined_f82r3)), error = identity)
    expect_identical(class(got), class(ref))
    expect_identical(conditionMessage(got$parent), conditionMessage(ref$parent))
  }
})

test_that("F82: a raw .data[[k]] reads the quosure's k where the fallback builds a call", {
  skip_if_no_gpu()
  df <- r3_df()
  k <- "g"
  twice <- function(v) v * 2
  both <- function(p) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
  both(function(d) { k <- "x"; filter(d, !!quote(.data[[k]] > 2), twice(x) > 0) })
  both(function(d) { k <- "x"; filter_out(d, !!quote(.data[[k]] > 2), twice(x) > 100) })
  both(function(d) { k <- "w"; sample_n(d, 1, weight = !!quote(.data[[k]])) })
  both(function(d) { k <- "w"; sample_frac(d, 0.2, weight = !!quote(.data[[k]])) })
  both(function(d) { k <- "x"; arrange(d, !!quote(twice(.data[[k]]))) })
})

test_that("F82: a raw .data[[k]] count() weight stays native", {
  skip_if_no_gpu()
  df <- r3_df()
  k <- "g"
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    f <- function(d) { k <- "x"; count(d, wt = !!quote(.data[[k]])) }
    expect_equal(collect(f(tbl_gpu(df, lazy = lazy)))$n, 19)
    f <- function(d) { k <- "x"; tally(group_by(d, g), wt = !!quote(.data[[k]])) }
    expect_equal(dplyr::arrange(collect(f(tbl_gpu(df, lazy = lazy))), g)$n, c(3, 0, 10, 5, 1))
  }
})

test_that("F82: a CPU-route error names the dot as dplyr does", {
  skip_if_no_gpu()
  df <- dplyr::mutate(r3_df(), s = c("a", "b", "c", "d", "e"))
  s <- "x"
  twice <- function(v) v * 2
  # (`s` is also a column, so the subscript is dplyr's error; its label must
  # not be a rewritten call)
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), !!quote(.data[[s]] > 2), twice(x) > 0)),
                    error = identity)
    expect_s3_class(got, "error")
    expect_match(conditionMessage(got), ".data[[s]] > 2", fixed = TRUE)
  }
})
