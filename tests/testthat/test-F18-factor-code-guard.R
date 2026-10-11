# F18 round 4: a factor's integer codes never decide a comparison or a
# logical test, whatever expression the factor sits under. Complements the
# protected acceptance file (test-parity-F18-type-fidelity.R) with the shapes
# ir_factor_rule() sends to the CPU beyond B3, and pins the native
# shapes that must stay on the GPU.

f18g_df <- function() {
  tibble::tibble(
    f = factor(c("1", "2", "10", NA, "2", "1"), levels = c("10", "1", "2")),
    f2 = factor(c("1", "2", "10", NA, "2", "10"), levels = c("10", "1", "2")),
    n = c(1L, 2L, 3L, 4L, 2L, 10L),
    g = c("a", "a", "b", "b", "b", "a")
  )
}

test_that("logical operators on a factor warn and give NA like base R", {
  skip_if_no_gpu()
  df <- f18g_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    for (pipe in list(
      function(d) dplyr::filter(d, !f),
      function(d) dplyr::filter(d, f & TRUE),
      function(d) dplyr::filter(d, f | n > 100)
    )) {
      expected <- suppressWarnings(pipe(df))
      expect_warning(out <- dplyr::collect(pipe(g)), "not meaningful for factors")
      expect_equal(tibble::as_tibble(out), expected)
    }
  }
})

test_that("comparisons on computed factors match dplyr", {
  skip_if_no_gpu()
  df <- f18g_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    # factor arithmetic under a comparison in mutate: dplyr warns, gives NA
    expect_warning(out <- dplyr::collect(dplyr::mutate(g, y = f - 1 == 0)),
                   "not meaningful for factors")
    expect_equal(tibble::as_tibble(out), suppressWarnings(dplyr::mutate(df, y = f - 1 == 0)))
    expect_warning(out <- dplyr::collect(dplyr::mutate(g, y = dplyr::near(f, 1))),
                   "not meaningful for factors")
    expect_equal(tibble::as_tibble(out), suppressWarnings(dplyr::mutate(df, y = dplyr::near(f, 1))))
    # a window over a factor, compared with the factor itself
    expect_equal(tibble::as_tibble(dplyr::collect(dplyr::mutate(g, y = dplyr::lag(f) == f))),
                 dplyr::mutate(df, y = dplyr::lag(f) == f))
    expect_equal(tibble::as_tibble(dplyr::collect(dplyr::mutate(g, y = dplyr::consecutive_id(f) == 1))),
                 dplyr::mutate(df, y = dplyr::consecutive_id(f) == 1))
    expect_equal(tibble::as_tibble(dplyr::collect(dplyr::filter(g, dplyr::if_else(n > 2, f, f2) == f))),
                 dplyr::filter(df, dplyr::if_else(n > 2, f, f2) == f))
    expect_error(dplyr::collect(dplyr::filter(g, f == max(f))), "not meaningful for factors")
    # a string function of a factor works on its labels
    expect_equal(tibble::as_tibble(dplyr::collect(dplyr::filter(g, toupper(f) == "10"))),
                 dplyr::filter(df, toupper(f) == "10"))
    # grouped
    expect_equal(
      tibble::as_tibble(dplyr::collect(dplyr::summarise(dplyr::group_by(g, g),
                                                        m = sum(dplyr::lag(f) == f, na.rm = TRUE)))),
      dplyr::summarise(dplyr::group_by(df, g), m = sum(dplyr::lag(f) == f, na.rm = TRUE))
    )
  }
})

test_that("factor shapes whose codes are harmless stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f18g_df()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    for (pipe in list(
      function(d) dplyr::filter(d, f == f2),
      function(d) dplyr::filter(d, f != f2 & n > 1),
      function(d) dplyr::filter(d, is.na(f) == TRUE),
      function(d) dplyr::filter(d, !is.na(f) & n > 1),
      function(d) dplyr::filter(d, (f == f2) | is.na(f)),
      function(d) dplyr::mutate(d, y = f == f2, z = is.na(f)),
      function(d) dplyr::mutate(d, h = f),
      function(d) dplyr::summarise(d, k = dplyr::n_distinct(f))
    )) {
      expect_equal(tibble::as_tibble(dplyr::collect(pipe(g))), pipe(df))
    }
  }
})
