# Issue #19, follow-up cases: rowwise class on every zero-column result, the
# filter() identity, joins of two zero-column tables answered from counts, and
# the join stand-in.

r2_zero <- function(n) tibble::new_tibble(list(), nrow = n)

test_that("a rowwise mutate() that keeps no column stays rowwise and summarises per row", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1L, 2L, NA, 4L), g = c("a", "a", "b", NA), v = c(1.5, NA, 2.5, 0))
  for (lz in c(FALSE, TRUE)) {
    for (res in list(
      function(t) mutate(t, x = NULL, g = NULL, v = NULL),
      function(t) mutate(t, .keep = "none")
    )) {
      want <- res(dplyr::rowwise(df))
      got <- collect(res(rowwise(tbl_gpu(df, lazy = lz))))
      expect_s3_class(got, "rowwise_df")
      expect_equal(dim(got), dim(want))
      expect_equal(nrow(collect(summarise(res(rowwise(tbl_gpu(df, lazy = lz))), n = n()))), 4L)
    }
  }
})

test_that("a join with a rowwise zero-column x keeps the rowwise class", {
  skip_if_no_gpu()
  z <- r2_zero(2)
  y <- tibble::tibble(k = 1:3)
  for (lz in c(FALSE, TRUE)) {
    cj <- collect(cross_join(rowwise(tbl_gpu(z, lazy = lz)), tbl_gpu(y, lazy = lz)))
    expect_s3_class(cj, "rowwise_df")
    expect_equal(dim(cj), c(6L, 1L))
    sj <- collect(semi_join(rowwise(tbl_gpu(z, lazy = lz)), tbl_gpu(y, lazy = lz), by = character()))
    expect_s3_class(sj, "rowwise_df")
    expect_equal(dim(sj), c(2L, 0L))
    zz <- collect(cross_join(rowwise(tbl_gpu(z, lazy = lz)), tbl_gpu(z, lazy = lz)))
    expect_s3_class(zz, "rowwise_df")
    expect_equal(dim(zz), c(4L, 0L))
  }
})

test_that("filter() with no dots on a zero-column table is the identity, no fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  for (lz in c(FALSE, TRUE)) {
    g <- tbl_gpu(r2_zero(3), lazy = lz)
    expect_no_error(out <- filter(g))
    expect_identical(out, g)
    expect_equal(dim(collect(out)), c(3L, 0L))
  }
})

test_that("joins of two zero-column tables follow dplyr's counts", {
  skip_if_no_gpu()
  ns <- c(0L, 1L, 3L)
  for (lz in c(FALSE, TRUE)) {
    for (nx in ns) for (ny in ns) {
      x <- r2_zero(nx); y <- r2_zero(ny)
      gx <- tbl_gpu(x, lazy = lz); gy <- tbl_gpu(y, lazy = lz)
      expect_equal(dim(collect(cross_join(gx, gy))), dim(dplyr::cross_join(x, y)))
      suppressWarnings({
        expect_equal(dim(collect(left_join(gx, gy, by = character()))),
                     dim(dplyr::left_join(x, y, by = character())))
        expect_equal(dim(collect(full_join(gx, gy, by = character()))),
                     dim(dplyr::full_join(x, y, by = character())))
        expect_equal(dim(collect(semi_join(gx, gy, by = character()))),
                     dim(dplyr::semi_join(x, y, by = character())))
        expect_equal(dim(collect(anti_join(gx, gy, by = character()))),
                     dim(dplyr::anti_join(x, y, by = character())))
      })
    }
  }
})

test_that("a cardinality argument on a zero-column join is still checked", {
  skip_if_no_gpu()
  x <- r2_zero(2); y <- r2_zero(3)
  want <- tryCatch(
    suppressWarnings(dplyr::left_join(x, y, by = character(), relationship = "one-to-one")),
    error = function(e) e
  )
  got <- tryCatch(
    suppressWarnings(left_join(tbl_gpu(x), tbl_gpu(y), by = character(), relationship = "one-to-one")),
    error = function(e) e
  )
  expect_identical(inherits(got, "error"), inherits(want, "error"))
})

test_that("the lazy mode of a zero-column join follows either side", {
  skip_if_no_gpu()
  out <- cross_join(tbl_gpu(r2_zero(2), lazy = FALSE), tbl_gpu(r2_zero(2), lazy = TRUE))
  expect_identical(.subset2(out, "exec_mode"), "lazy")
  expect_equal(dim(collect(out)), c(4L, 0L))
})

test_that("vec_restore() of a rowwise zero-column table keeps the class", {
  skip_if_no_gpu()
  g <- rowwise(tbl_gpu(r2_zero(3)))
  out <- vctrs::vec_slice(g, c(1L, 1L, 2L))
  expect_true(isTRUE(.subset2(out, "rowwise")))
  expect_s3_class(collect(out), "rowwise_df")
  expect_equal(dim(collect(out)), c(3L, 0L))
})
