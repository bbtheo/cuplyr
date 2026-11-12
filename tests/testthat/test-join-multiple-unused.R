# F24: capped-out matches under multiple = "first"/"last"/"any" reappear as
# unmatched y rows in right/full joins only when NO kept pair uses that y row,
# and once each (dplyr); multiple = "error"/"warning" go to the CPU fallback;
# unmatched = "error" sees y rows that multiple= capped out.

test_that("right/full joins with multiple= list each unused y row once", {
  skip_if_no_gpu()
  x <- data.frame(k = c(1, 1, 2), xv = 1:3)
  y <- data.frame(k = c(1, 1, 1, 2, 2), yv = 1:5)
  a <- data.frame(a = 1:2)
  b <- data.frame(b = c(1, 2, 2, 3), bv = 1:4)
  for (m in c("first", "last", "any")) {
    for (join in list(dplyr::right_join, dplyr::full_join)) {
      expect_same_as_dplyr(x, function(d) {
        join(d, if (is_tbl_gpu(d)) tbl_gpu(y) else tibble::as_tibble(y), by = "k", multiple = m)
      })
      expect_same_as_dplyr(a, function(d) {
        join(d, if (is_tbl_gpu(d)) tbl_gpu(b) else tibble::as_tibble(b),
             by = dplyr::join_by(a <= b), multiple = m)
      })
    }
  }
})

# Did evaluating `expr` go through the notified CPU fallback?
join_fell_back <- function(expr) {
  fell <- FALSE
  withCallingHandlers(
    force(expr),
    cuplyr_cpu_fallback = function(cnd) fell <<- TRUE
  )
  fell
}

test_that("multiple = 'error'/'warning' run on the CPU fallback for every join shape", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  x <- tibble::tibble(a = c(1, 5), xv = 1:2)
  y <- tibble::tibble(b = c(0, 2, 3), yv = 1:3)
  xk <- tibble::tibble(k = c(2, 1, 1), xv = 1:3)
  yk <- tibble::tibble(k = c(1, 1, 2, 2), yv = 1:4)
  xs <- tibble::tibble(s = c("b", "z"), xv = 1:2)
  ys <- tibble::tibble(r = c("a", "c", "d"), yv = 1:3)
  xt <- tibble::tibble(t = c(5, 9), xv = 1:2)
  yt <- tibble::tibble(ty = c(4, 4, 8), yv = 1:3)
  shapes <- list(
    equi = list(xk, yk, "k"),
    nonequi = list(x, y, dplyr::join_by(a >= b)),
    string = list(xs, ys, dplyr::join_by(s >= r)),
    rolling = list(xt, yt, dplyr::join_by(closest(t >= ty)))
  )
  for (shape in shapes) {
    for (m in c("error", "warning")) {
      for (lazy in c(FALSE, TRUE)) {
        res <- NULL
        expect_true(join_fell_back(tryCatch(
          suppressWarnings(res <- collect(dplyr::right_join(
            tbl_gpu(shape[[1]], lazy = lazy), tbl_gpu(shape[[2]], lazy = lazy),
            by = shape[[3]], multiple = m))),
          error = function(e) NULL
        )))
        expected <- tryCatch(
          suppressWarnings(dplyr::right_join(shape[[1]], shape[[2]], by = shape[[3]], multiple = m)),
          error = function(e) NULL
        )
        if (!is.null(expected)) expect_equal(res, expected, ignore_attr = TRUE)
      }
    }
  }
  # A non-equi join under multiple = "warning" keeps every match.
  res <- suppressWarnings(collect(dplyr::inner_join(
    tbl_gpu(x), tbl_gpu(y), by = dplyr::join_by(a >= b), multiple = "warning")))
  expect_equal(nrow(res), 4L)
  # The native paths never see them: all/any/first/last do not fall back.
  withr::local_options(cuplyr.fallback = "error")
  for (m in c("all", "any", "first", "last")) {
    expect_no_error(collect(dplyr::inner_join(
      tbl_gpu(x), tbl_gpu(y), by = dplyr::join_by(a >= b), multiple = m)))
  }
})

test_that("unmatched = 'error' counts y rows that multiple = 'first'/'last' capped out (equi)", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1, xv = 1L)
  y <- tibble::tibble(k = c(1, 1), yv = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    for (case in list(list(dplyr::left_join, "first", 2L), list(dplyr::inner_join, "last", 1L),
                      list(dplyr::inner_join, "any", 2L))) {
      join <- case[[1]]
      expected <- tryCatch(join(x, y, by = "k", multiple = case[[2]], unmatched = "error"),
                           error = function(e) e)
      expect_s3_class(expected, "dplyr_error_join_matches_remaining")
      actual <- tryCatch(
        collect(join(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy), by = "k",
                     multiple = case[[2]], unmatched = "error")),
        error = function(e) e
      )
      expect_identical(class(actual), class(expected))
      expect_identical(rlang::cnd_message(actual), rlang::cnd_message(expected))
    }
  }
  # Nothing capped out: no error.
  x2 <- tibble::tibble(k = c(1, 2), xv = 1:2)
  y2 <- tibble::tibble(k = c(2, 1), yv = 1:2)
  expect_equal(
    collect(dplyr::left_join(tbl_gpu(x2), tbl_gpu(y2), by = "k", multiple = "first",
                             unmatched = "error")),
    dplyr::left_join(x2, y2, by = "k", multiple = "first", unmatched = "error"),
    ignore_attr = TRUE
  )
})

test_that("relationship= still checks the raw match set when multiple= caps matches", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = c(1, 1), xv = 1:2)
  y <- tibble::tibble(k = c(1, 1), yv = 1:2)
  for (rel in c("one-to-one", "one-to-many", "many-to-one", "many-to-many")) {
    expected <- tryCatch(dplyr::left_join(x, y, by = "k", multiple = "first", relationship = rel),
                         error = function(e) e)
    actual <- tryCatch(
      collect(dplyr::left_join(tbl_gpu(x), tbl_gpu(y), by = "k", multiple = "first",
                               relationship = rel)),
      error = function(e) e
    )
    expect_identical(class(actual), class(expected))
  }
})
