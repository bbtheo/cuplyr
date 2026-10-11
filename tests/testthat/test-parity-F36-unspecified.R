# F36 acceptance tests: `vctrs::unspecified()` columns and joins (#6804).
#
# dplyr (verified with tools/r, scratchpad/parity/features/F36/probe/p1.out,
# p4.out):
#   - filtering joins return x verbatim, so a pre-existing unspecified() column
#     in x keeps class `vctrs_unspecified` (semi and anti, any na_matches, any
#     y key type);
#   - mutating joins take the common type of the keys, so an unspecified KEY is
#     finalized (logical with an unspecified/logical partner, double with a
#     double partner);
#   - row-only verbs (filter, arrange) keep the class.
# cuplyr used to refuse to upload an unspecified column at all.

f36u_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

f36u_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f36u_x0 <- tibble::tibble(x = vctrs::unspecified())
f36u_x2 <- tibble::tibble(x = vctrs::unspecified(2), k = 1:2)

test_that("an unspecified column survives upload and collect", {
  skip_if_no_gpu()
  for (df in list(f36u_x0, f36u_x2)) {
    for (lazy in c(FALSE, TRUE)) {
      out <- collect(tbl_gpu(df, lazy = lazy))
      expect_identical(tibble::as_tibble(out), df)
    }
  }
})

test_that("semi_join keeps x's unspecified column (conformance #6804 case)", {
  skip_if_no_gpu()
  y <- tibble::tibble(x = NA)
  f36u_both(f36u_x0, function(d) semi_join(d, f36u_like(d, y), by = dplyr::join_by(x)))
  out <- collect(semi_join(tbl_gpu(f36u_x0), tbl_gpu(y), by = dplyr::join_by(x)))
  expect_identical(tibble::as_tibble(out), tibble::tibble(x = vctrs::unspecified()))
})

test_that("semi/anti keep the class when rows match (NA matches NA)", {
  skip_if_no_gpu()
  y <- tibble::tibble(x = NA)
  for (join in list(semi_join, anti_join)) {
    f36u_both(f36u_x2, function(d) join(d, f36u_like(d, y), by = "x"))
    f36u_both(f36u_x2, function(d) join(d, f36u_like(d, y), by = "x", na_matches = "never"))
  }
})

test_that("semi/anti keep the class against a double key", {
  skip_if_no_gpu()
  y <- tibble::tibble(x = c(1, NA))
  for (join in list(semi_join, anti_join)) {
    f36u_both(f36u_x2, function(d) join(d, f36u_like(d, y), by = "x"))
  }
})

test_that("semi/anti keep the class of a non-key unspecified column, grouped too", {
  skip_if_no_gpu()
  y <- tibble::tibble(k = 2L)
  for (join in list(semi_join, anti_join)) {
    f36u_both(f36u_x2, function(d) join(d, f36u_like(d, y), by = "k"))
    f36u_both(f36u_x2, function(d) join(group_by(d, k), f36u_like(d, y), by = "k"))
  }
})

test_that("a plain logical x stays logical when y's key is unspecified", {
  skip_if_no_gpu()
  x <- tibble::tibble(x = NA)
  y <- tibble::tibble(x = vctrs::unspecified(1))
  f36u_both(x, function(d) semi_join(d, f36u_like(d, y), by = "x"))
})

test_that("filter and arrange keep the class", {
  skip_if_no_gpu()
  f36u_both(f36u_x2, function(d) filter(d, k > 1L))
  f36u_both(f36u_x2, function(d) arrange(d, dplyr::desc(k)))
})

test_that("mutating joins finalize an unspecified key to the common type", {
  skip_if_no_gpu()
  # conformance "mutating joins finalize unspecified columns (#6804)"
  f36u_both(f36u_x0, function(d) {
    inner_join(d, f36u_like(d, tibble::tibble(x = vctrs::unspecified())), by = dplyr::join_by(x))
  })
  f36u_both(f36u_x2, function(d) {
    left_join(d, f36u_like(d, tibble::tibble(x = NA, w = 1)), by = "x")
  })
  f36u_both(f36u_x2, function(d) {
    inner_join(d, f36u_like(d, tibble::tibble(x = c(1, NA))), by = "x")
  })
})
