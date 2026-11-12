# Oracle parity tests for pick() (Phase 10).
#
# pick() returns a genuine data-frame-of-columns value for use INSIDE
# another expression (`rowSums(pick(...))`, `arrange(pick(...))`, ...) --
# there is no GPU (cuDF/IR) representation for "a handful of columns as
# one value" in this package, so every pick()-bearing dot is routed
# through the whole-call CPU fallback (R/across.R's dots_need_fallback()/
# verb_across_fallback()), which simply re-runs the real dplyr call.
# These tests exist to pin that the fallback produces EXACTLY dplyr's own
# results across the shapes dplyr's own test-pick.R (~/R/*/dplyr/tests/
# testthat/, read directly) exercises most: plain tidyselect, tidyselect
# helpers, and pick()'s own group-column-exclusion rule (identical to
# across()'s, verified empirically -- see R/across.R's module docs).

test_that("pick() with an explicit column list, inside rowSums()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, total = rowSums(dplyr::pick(x, y)))

  expect_warning(result <- tbl_gpu(df) |> pipeline() |> collect(), "fell back to CPU evaluation")
  expect_equal(result, pipeline(df))
})

test_that("pick(everything()) excludes grouping columns", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3, y = 4:6)
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::mutate(total = rowSums(dplyr::pick(dplyr::everything())))
  }

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(as.data.frame(result), as.data.frame(dplyr::ungroup(pipeline(df))))
})

test_that("pick() explicitly referencing a grouping column errors (matches real dplyr)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gt <- tbl_gpu(tibble::tibble(g = 1, x = 2)) |> dplyr::group_by(g)
  oracle <- tibble::tibble(g = 1, x = 2) |> dplyr::group_by(g)

  gpu_err <- tryCatch({
    dplyr::mutate(gt, total = rowSums(dplyr::pick(g, x)))
    NULL
  }, error = function(e) e)
  oracle_err <- tryCatch({
    dplyr::mutate(oracle, total = rowSums(dplyr::pick(g, x)))
    NULL
  }, error = function(e) e)

  expect_false(is.null(gpu_err))
  expect_false(is.null(oracle_err))
})

test_that("pick() with a tidyselect helper (starts_with)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x1 = c(1, 2), x2 = c(3, 4), y = c(5, 6))
  pipeline <- function(d) dplyr::mutate(d, total = rowSums(dplyr::pick(dplyr::starts_with("x"))))

  expect_warning(result <- tbl_gpu(df) |> pipeline() |> collect(), "fell back to CPU evaluation")
  expect_equal(result, pipeline(df))
})

test_that("pick() with an empty match is a 0-column data frame (rowSums -> 0)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3))
  pipeline <- function(d) {
    dplyr::mutate(d, total = rowSums(dplyr::pick(dplyr::starts_with("zzz"))))
  }

  expect_warning(result <- tbl_gpu(df) |> pipeline() |> collect(), "fell back to CPU evaluation")
  expect_equal(result, pipeline(df))
})

test_that("pick() used in summarise()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::summarise(total = sum(rowSums(dplyr::pick(x, y))))
  }

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(
    dplyr::arrange(result, g),
    dplyr::arrange(tibble::as_tibble(dplyr::ungroup(pipeline(df))), g)
  )
})
