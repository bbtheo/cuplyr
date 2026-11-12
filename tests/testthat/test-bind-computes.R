# bind_rows()/bind_cols() and the set ops built on them run each lazy input
# once, and evaluate their arguments once.

test_that("bind_rows() and union_all() compute each lazy input once", {
  skip_if_no_gpu()
  mk <- function(cut) {
    dplyr::filter(as_lazy(tbl_gpu(data.frame(x = 1:10, s = letters[1:10]))), x > cut)
  }
  lz1 <- mk(2L)
  lz2 <- mk(5L)
  # compute() runs the optimizer once per lazy pipeline (eager verbs, like
  # union()'s distinct(), never do), so optimize_ast() calls count computes
  n <- 0L
  orig <- optimize_ast
  local_mocked_bindings(optimize_ast = function(...) {
    n <<- n + 1L
    orig(...)
  })

  invisible(compute(lz1))
  expect_identical(n, 1L)
  per <- n

  n <- 0L
  out <- cuplyr::bind_rows(lz1, lz2)
  expect_identical(n, 2L * per)
  expect_identical(collect(out)$x, c(3:10, 6:10))

  n <- 0L
  out <- cuplyr::bind_rows(list(lz1, lz2), .id = "id")
  expect_identical(n, 2L * per)

  n <- 0L
  out <- dplyr::union_all(lz1, lz2)
  expect_identical(n, 2L * per)
  expect_identical(collect(out)$x, c(3:10, 6:10))

  n <- 0L
  out <- dplyr::union(lz1, lz2)
  expect_identical(n, 2L * per)
  expect_identical(sort(collect(out)$x), 3:10)
})

test_that("bind_rows() refuses named non-table inputs next to a tbl_gpu", {
  skip_if_no_gpu()
  g1 <- tbl_gpu(data.frame(x = 1:3))
  # dplyr packs these into a tibble with a data frame column, which a
  # tbl_gpu can't hold
  expect_error(
    cuplyr::bind_rows(a = g1, b = 1),
    class = "cuplyr_error_bind_unsupported"
  )
  expect_error(
    cuplyr::bind_rows(list(a = g1, b = 1:3)),
    class = "cuplyr_error_bind_unsupported"
  )
})

test_that("bind_rows()/bind_cols() evaluate spliced arguments once", {
  skip_if_no_gpu()
  t1 <- tibble::tibble(x = 1:3)
  n <- 0L
  f <- function(...) {
    n <<- n + 1L
    list(...)
  }

  # no tbl_gpu: the call falls through to dplyr
  out <- cuplyr::bind_rows(t1, !!!f(tibble::tibble(x = 4L)))
  expect_identical(n, 1L)
  expect_identical(out, tibble::tibble(x = 1:4))

  n <- 0L
  out <- cuplyr::bind_cols(t1, !!!f(tibble::tibble(y = 4:6)))
  expect_identical(n, 1L)
  expect_identical(out, tibble::tibble(x = 1:3, y = 4:6))

  # a tbl_gpu among the spliced inputs
  n <- 0L
  out <- cuplyr::bind_rows(t1, !!!f(tbl_gpu(data.frame(x = 4L))))
  expect_identical(n, 1L)
  expect_identical(collect(out), tibble::tibble(x = 1:4))
})
