# F37 attempt 3 acceptance tests (B15): a column-free VECTOR constant dot
# is evaluated exactly once, also when a later dot of the same call reads it
# and so sends the call to the CPU route.
#
# dplyr evaluates each dot once (dplyr-1.2.1 R/mutate.R::mutate_col()). A
# fractional-Date vector constant read by a later dot must reach dplyr with
# the value cuplyr already computed, as scalar constants already do
# (test-parity-F37-evaluate-once.R, test-parity-F11-mutate-shapes.R). The
# upload of the CPU result floors the Date column with
# `cuplyr_warning_fractional_date` (F37 decision 2).

f37v_df <- function() tibble::tibble(x = 1:3, k = c(7L, 8L, 9L))

f37v_floor <- function(v) as.Date(floor(unclass(v)))

test_that("F37: a fractional-Date vector constant read by a later dot draws RNG once (mutate)", {
  skip_if_no_gpu()
  df <- f37v_df()

  set.seed(1)
  expected <- dplyr::mutate(df, dd = as.Date(19723 + runif(3) * 10), z = dd > as.Date(19727))
  expected_next <- runif(1)
  expected$dd <- f37v_floor(expected$dd)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   dd = as.Date(19723 + runif(3) * 10), z = dd > as.Date(19727))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a fractional-Date vector constant read by a later dot draws RNG once (transmute)", {
  skip_if_no_gpu()
  df <- f37v_df()

  set.seed(1)
  expected <- dplyr::transmute(df, dd = as.Date(19723 + runif(3) * 10), z = dd > as.Date(19727))
  expected_next <- runif(1)
  expected$dd <- f37v_floor(expected$dd)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy),
                                      dd = as.Date(19723 + runif(3) * 10), z = dd > as.Date(19727))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a side-effecting fractional-Date vector dot read by a later dot runs once", {
  skip_if_no_gpu()
  df <- f37v_df()
  vals <- as.Date(c(19723.5, 19724.5, 19725.5))

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f_vec <- function() {
      cnt <<- cnt + 1
      vals
    }

    expected <- dplyr::mutate(df, dd = vals, z = as.numeric(dd), w = dd > as.Date(19724))
    expected$dd <- f37v_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   dd = f_vec(), z = as.numeric(dd), w = dd > as.Date(19724))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    cnt <- 0
    expected <- dplyr::transmute(df, dd = vals, z = as.numeric(dd))
    expected$dd <- f37v_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), dd = f_vec(), z = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    # .keep / .before still behave like dplyr
    cnt <- 0
    expected <- dplyr::mutate(df, dd = vals, z = as.numeric(dd), .keep = "none", .before = 1)
    expected$dd <- f37v_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   dd = f_vec(), z = as.numeric(dd), .keep = "none", .before = 1)),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: an unnamed vector constant on the CPU route runs once and keeps dplyr's name", {
  skip_if_no_gpu()
  df <- f37v_df()
  vals <- as.Date(c(19723.5, 19724.5, 19725.5))

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f_vec <- function() {
      cnt <<- cnt + 1
      c(0.25, 0.5, 0.75)
    }
    g_vec <- function() vals

    expected <- dplyr::transmute(df, f_vec(), dd = g_vec(), z = as.numeric(dd))
    expected$dd <- f37v_floor(expected$dd)
    cnt <- 0
    expect_warning(
      res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), f_vec(), dd = g_vec(), z = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(names(res), names(expected))
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a numeric vector constant next to a read fractional-Date dot draws RNG once", {
  skip_if_no_gpu()
  df <- f37v_df()

  set.seed(1)
  expected <- dplyr::mutate(df, v = runif(3), dd = as.Date(19723.5), zz = as.numeric(dd))
  expected_next <- runif(1)
  expected$dd <- f37v_floor(expected$dd)

  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   v = runif(3), dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(runif(1), expected_next)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a wrong-size vector constant read by a later dot keeps dplyr's error", {
  skip_if_no_gpu()
  df <- f37v_df()

  expected <- tryCatch(
    dplyr::mutate(df, dd = as.Date(c(19723.5, 19724.5)), z = as.numeric(dd)),
    error = function(e) e
  )
  expect_s3_class(expected, "dplyr:::mutate_error")
  expected_t <- tryCatch(
    dplyr::transmute(df, dd = as.Date(c(19723.5, 19724.5)), z = as.numeric(dd)),
    error = function(e) e
  )

  for (lazy in c(FALSE, TRUE)) {
    res <- tryCatch(
      collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), dd = as.Date(c(19723.5, 19724.5)), z = as.numeric(dd))),
      error = function(e) e
    )
    expect_s3_class(res, "dplyr:::mutate_error")
    # dplyr names the user's expression, not a deparsed value
    expect_equal(conditionMessage(res), conditionMessage(expected))

    res <- tryCatch(
      collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), dd = as.Date(c(19723.5, 19724.5)), z = as.numeric(dd))),
      error = function(e) e
    )
    expect_s3_class(res, "dplyr:::mutate_error")
    expect_equal(conditionMessage(res), conditionMessage(expected_t))
  }
})

test_that("F37: a column-free dot evaluating to NULL runs once on the CPU route (B16)", {
  skip_if_no_gpu()
  df <- f37v_df()

  for (lazy in c(FALSE, TRUE)) {
    cnt <- 0
    f_null <- function() {
      cnt <<- cnt + 1
      invisible(NULL)
    }

    expected <- dplyr::mutate(df, y = NULL, zz = rev(x))
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), y = f_null(), zz = rev(x)))
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    cnt <- 0
    expected <- dplyr::transmute(df, x, y = NULL, zz = rev(x))
    res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), x, y = f_null(), zz = rev(x)))
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)

    # next to a fractional-Date dot read by a later dot (F37's own CPU route)
    cnt <- 0
    expected <- dplyr::mutate(df, y = NULL, dd = as.Date(19723.5), zz = as.numeric(dd))
    expected$dd <- f37v_floor(expected$dd)
    expect_warning(
      res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy),
                                   y = f_null(), dd = as.Date(19723.5), zz = as.numeric(dd))),
      class = "cuplyr_warning_fractional_date"
    )
    expect_equal(cnt, 1)
    expect_equal(tibble::as_tibble(res), expected)
  }
})

test_that("F37: a wrong-size matrix constant keeps dplyr's error (B17)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4)

  expected <- tryCatch(dplyr::mutate(df, m = matrix(1:4, 2)), error = function(e) e)
  expect_s3_class(expected, "dplyr:::mutate_error")
  expected_t <- tryCatch(dplyr::transmute(df, m = matrix(1:4, 2)), error = function(e) e)

  for (lazy in c(FALSE, TRUE)) {
    res <- tryCatch(collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), m = matrix(1:4, 2))),
                    error = function(e) e)
    expect_s3_class(res, "dplyr:::mutate_error")
    expect_equal(conditionMessage(res), conditionMessage(expected))

    res <- tryCatch(collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), m = matrix(1:4, 2))),
                    error = function(e) e)
    expect_s3_class(res, "dplyr:::mutate_error")
    expect_equal(conditionMessage(res), conditionMessage(expected_t))
  }
})

test_that("F37: an inlined vector constant keeps its element names for later dots", {
  skip_if_no_gpu()
  df <- f37v_df()

  expected <- dplyr::mutate(df, v = c(a = 1.5, b = 2.5, c = 3.5), w = names(v))
  expect_equal(expected$w, c("a", "b", "c"))
  expected$v <- unname(expected$v)

  for (lazy in c(FALSE, TRUE)) {
    res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), v = c(a = 1.5, b = 2.5, c = 3.5), w = names(v)))
    expect_equal(tibble::as_tibble(res), expected)

    res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), v = c(a = 1.5, b = 2.5, c = 3.5), w = names(v)))
    expect_equal(tibble::as_tibble(res), expected[c("v", "w")])
  }
})

test_that("F37: a character vector constant doesn't warn", {
  skip_if_no_gpu()
  df <- f37v_df()

  expected <- dplyr::mutate(df, lab = paste0("s", 1:3))
  expected_t <- dplyr::transmute(df, lab = paste0("s", 1:3))
  for (lazy in c(FALSE, TRUE)) {
    expect_no_warning(res <- collect(dplyr::mutate(tbl_gpu(df, lazy = lazy), lab = paste0("s", 1:3))))
    expect_equal(tibble::as_tibble(res), expected)
    expect_no_warning(res <- collect(dplyr::transmute(tbl_gpu(df, lazy = lazy), lab = paste0("s", 1:3))))
    expect_equal(tibble::as_tibble(res), expected_t)
  }
})
