# F36 round 2: the `$unspecified` marker must follow its column.

f36m_class <- function(x) vapply(x, function(col) class(col)[[1]], "")

test_that("vec_cast() to a logical target does not keep the unspecified class", {
  skip_if_no_gpu()
  u <- tibble::tibble(x = vctrs::unspecified(1))
  l <- tibble::tibble(x = NA)
  ref <- vctrs::vec_cast(u, l)
  expect_identical(class(ref$x), "logical")
  out <- collect(vctrs::vec_cast(tbl_gpu(u), tbl_gpu(l)))
  expect_identical(class(out$x), "logical")

  # A target that is itself unspecified keeps the class (dplyr does too).
  ref2 <- vctrs::vec_cast(u, u)
  out2 <- collect(vctrs::vec_cast(tbl_gpu(u), tbl_gpu(u)))
  expect_identical(class(out2$x), class(ref2$x))
})

test_that("join_cast_common() finalizes unspecified columns (#6804)", {
  skip_if_no_gpu()
  u <- tibble::tibble(x = vctrs::unspecified(1))
  vars <- dplyr:::join_cols("x", "x", by = dplyr::join_by(x))
  ref <- dplyr:::join_cast_common(u, u, vars)
  out <- dplyr:::join_cast_common(tbl_gpu(u), tbl_gpu(u), vars)
  expect_identical(class(collect(out$y)$x), class(ref$y$x))
  expect_identical(class(collect(out$y)$x), "logical")
})

test_that("names<- moves the unspecified class with its column", {
  skip_if_no_gpu()
  df <- tibble::tibble(u = vctrs::unspecified(2), flag = c(NA, NA))
  ref <- df
  names(ref) <- c("flag", "u")
  for (lazy in c(FALSE, TRUE)) {
    t <- tbl_gpu(df, lazy = lazy)
    names(t) <- c("flag", "u")
    out <- collect(t)
    expect_identical(f36m_class(out), f36m_class(ref))
    expect_identical(tibble::as_tibble(out), tibble::as_tibble(ref))
  }
})

test_that("names<- drops the marker for a name that ends up duplicated", {
  skip_if_no_gpu()
  df <- tibble::tibble(u = vctrs::unspecified(2), flag = c(NA, NA))
  t <- tbl_gpu(df)
  names(t) <- c("a", "a")
  out <- collect(t)
  expect_identical(unname(f36m_class(out)), c("logical", "logical"))
})

test_that("join_by() checks run in dplyr's order", {
  skip_if_no_gpu()
  x <- tibble::tibble(a = 1, b = 2)
  y <- tibble::tibble(a = 1, b = 2)
  msg <- function(expr) tryCatch({expr; ""}, error = function(e) conditionMessage(e))
  by <- dplyr::join_by(a == a, a >= b)
  expect_identical(
    msg(left_join(tbl_gpu(x), tbl_gpu(y), by = by, keep = FALSE)),
    msg(left_join(x, y, by = by, keep = FALSE))
  )
  # Missing column in x is reported before a duplicate in y.
  by2 <- dplyr::join_by(zz == a, b == a)
  expect_match(msg(left_join(x, y, by = by2)), "`x` must be present")
  expect_identical(
    msg(left_join(tbl_gpu(x), tbl_gpu(y), by = by2)) != "",
    TRUE
  )
  expect_match(msg(left_join(tbl_gpu(x), tbl_gpu(y), by = by2)), "zz")
  expect_no_match(msg(left_join(tbl_gpu(x), tbl_gpu(y), by = by2)), "must be unique")
})
