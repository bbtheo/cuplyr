# Acceptance tests for #70, item 3: binding a data frame that has a list (or
# data frame) column to a tbl_gpu works through the CPU, like dplyr: the result
# keeps the list column (as a plain tibble, since a list column has no GPU
# form). A matrix column raises a classed error naming the column.

p70b_like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

p70b_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

p70b_gpu <- tibble::tibble(a = 3:4)
p70b_list <- tibble::tibble(a = 1:2, l = list(1, "x"))

test_that("bind_rows() of a tbl_gpu and a tibble with a list column, either order", {
  skip_if_no_gpu()
  p70b_both(p70b_gpu, function(d) bind_rows(d, p70b_list))
  p70b_both(p70b_gpu, function(d) bind_rows(p70b_list, d))
  p70b_both(p70b_gpu, function(d) bind_rows(d, p70b_list, d))
  out <- bind_rows(tbl_gpu(p70b_gpu, lazy = FALSE), p70b_list)
  expect_identical(tibble::as_tibble(collect(out)), dplyr::bind_rows(p70b_gpu, p70b_list))
})

test_that("dplyr::bind_rows() takes the same route", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(p70b_gpu, lazy = lazy)
    expect_identical(tibble::as_tibble(collect(dplyr::bind_rows(g, p70b_list))),
                     dplyr::bind_rows(p70b_gpu, p70b_list))
    expect_identical(tibble::as_tibble(collect(dplyr::bind_rows(p70b_list, g))),
                     dplyr::bind_rows(p70b_list, p70b_gpu))
  }
})

test_that("groups of the first input, .id and rbind()", {
  skip_if_no_gpu()
  p70b_both(p70b_gpu, function(d) bind_rows(group_by(d, a), p70b_list))
  p70b_both(p70b_gpu, function(d) bind_rows(d, p70b_list, .id = "src"))
  p70b_both(p70b_gpu, function(d) bind_rows(first = d, second = p70b_list, .id = "src"))
  for (lazy in c(FALSE, TRUE)) {
    out <- rbind(tbl_gpu(p70b_gpu, lazy = lazy), p70b_list)
    expect_identical(tibble::as_tibble(collect(out)), dplyr::bind_rows(p70b_gpu, p70b_list))
  }
})

test_that("data frame columns bind too", {
  skip_if_no_gpu()
  packed <- tibble::tibble(a = 1:2, p = tibble::tibble(x = 1:2))
  p70b_both(p70b_gpu, function(d) bind_rows(d, packed))
})

test_that("bind_cols() with a list column", {
  skip_if_no_gpu()
  p70b_both(p70b_gpu, function(d) bind_cols(d, tibble::tibble(l = list(1, "x"))))
})

test_that("a matrix column raises a classed error naming it", {
  skip_if_no_gpu()
  mat <- tibble::tibble(a = 1:2, m = matrix(1:4, ncol = 2))
  for (lazy in c(FALSE, TRUE)) {
    expect_error(collect(bind_rows(tbl_gpu(p70b_gpu, lazy = lazy), mat)),
                 "`m`", fixed = TRUE, class = "cuplyr_error_unsupported_class")
  }
})

# Added after the third review: the class and groups of the result are dplyr's.

test_that("a named list input, NA names with .id, grouped and bare data.frame first inputs", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 3:4)
  x <- list(a = 1, l = list(2))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_identical(bind_rows(g, x), dplyr::bind_rows(df, x))
    expect_identical(dplyr::bind_rows(g, x), dplyr::bind_rows(df, x))
    expect_identical(bind_rows(group_by(g, a), x), dplyr::bind_rows(dplyr::group_by(df, a), x))

    xs <- list(tibble::tibble(a = 1:2), tibble::tibble(a = 3L, l = list(1)))
    names(xs) <- c("p", NA)
    gs <- list(tbl_gpu(xs[[1]], lazy = lazy), xs[[2]])
    names(gs) <- c("p", NA)
    expect_identical(bind_rows(gs, .id = "id"), dplyr::bind_rows(xs, .id = "id"))

    fd <- tibble::tibble(f = factor(c("a", "a"), levels = c("a", "b")), v = 1:2)
    fl <- tibble::tibble(f = factor("a", levels = c("a", "b")), l = list(1))
    expect_identical(bind_rows(group_by(tbl_gpu(fd, lazy = lazy), f, .drop = FALSE), fl),
                     dplyr::bind_rows(dplyr::group_by(fd, f, .drop = FALSE), fl))

    bdf <- data.frame(a = 1:2)
    bl <- tibble::tibble(a = 3L, l = list(1))
    expect_identical(bind_rows(tbl_gpu(bdf, lazy = lazy), bl), dplyr::bind_rows(bdf, bl))
  }
})

test_that("bind_cols() and cbind() keep a grouped first input's groups", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 1L, 2L))
  l <- tibble::tibble(l = list(1, "x", NULL))
  expected <- dplyr::bind_cols(dplyr::group_by(df, a), l)
  for (lazy in c(FALSE, TRUE)) {
    g <- group_by(tbl_gpu(df, lazy = lazy), a)
    expect_identical(dplyr::bind_cols(g, l), expected)
    expect_identical(cbind(g, l), expected)
  }
})
