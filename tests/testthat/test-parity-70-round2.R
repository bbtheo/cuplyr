# #70 round 2: the key-order error on every join shape, the verb-time raise
# with pending steps, and the upload hints.

r2_y <- tibble::tibble(k1 = c("a", "b"), k2 = c("1", "2"))

r2_raises <- function(x, y, f, regexp, class = "dplyr_error_join_incompatible_type") {
  for (lazy in c(FALSE, TRUE)) {
    expect_error(f(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy)),
                 regexp = regexp, class = class, fixed = TRUE, info = paste("lazy =", lazy))
  }
}

test_that("the first bad pair is named for keep = TRUE and filtering joins too", {
  skip_if_no_gpu()
  first <- "Can't join `x$k1` with `y$k1` due to incompatible types."
  second <- "Can't join `x$k2` with `y$k2` due to incompatible types."
  bad1 <- tibble::tibble(k1 = c(TRUE, NA), k2 = 1:2)
  bad2 <- tibble::tibble(k1 = c(NA, NA), k2 = 1:2)
  r2_raises(bad1, r2_y, function(x, y) semi_join(x, y, by = c("k1", "k2")), first)
  r2_raises(bad1, r2_y, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), first)
  r2_raises(bad2, r2_y, function(x, y) anti_join(x, y, by = c("k1", "k2")), second)
  r2_raises(bad2, r2_y, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), second)
})

test_that("the verb-time raise holds with steps pending on either side", {
  skip_if_no_gpu()
  bad <- tbl_gpu(tibble::tibble(k = c(NA, TRUE), a = 1:2), lazy = TRUE)
  chr <- tbl_gpu(tibble::tibble(k = c("u", NA), b = 2:3), lazy = TRUE)
  expect_error(left_join(filter(bad, a > 0L), chr, by = "k"),
               class = "dplyr_error_join_incompatible_type")
  expect_error(left_join(bad, mutate(chr, b = b + 1L), by = "k"),
               class = "dplyr_error_join_incompatible_type")
  ok <- tbl_gpu(tibble::tibble(k = c(NA, NA), a = 1:2), lazy = TRUE)
  out <- left_join(filter(ok, a > 0L), mutate(chr, b = b + 1L), by = "k")
  expect_equal(collect(out)$b, c(4L, 4L))
  # an unspecified() key needs no check and stays lazy
  un <- tbl_gpu(tibble::tibble(k = vctrs::unspecified(2), a = 1:2), lazy = TRUE)
  lazy_out <- left_join(un, chr, by = "k")
  expect_true(has_pending_ops(lazy_out))
})

test_that("the zone in a join error is the key's own, `local` for an empty zone", {
  skip_if_no_gpu()
  local_key <- tibble::tibble(k = as.POSIXct("2020-01-01", tz = ""), b = 2L)
  expected <- tryCatch(inner_join(tibble::tibble(k = NA, a = 1), local_key, by = "k"),
                       error = conditionMessage)
  r2_raises(tibble::tibble(k = NA, a = 1), local_key,
            function(x, y) inner_join(x, y, by = "k"), expected, class = "vctrs_error_cast")
})

test_that("the vctrs_error_cast does not report an internal call", {
  skip_if_no_gpu()
  e <- tryCatch(inner_join(tbl_gpu(tibble::tibble(k = NA, a = 1)),
                           tbl_gpu(tibble::tibble(k = "u", b = 2L)), by = "k"),
                error = identity)
  expect_s3_class(e, "vctrs_error_cast")
  expect_null(e$call)
})

test_that("complex and raw columns get the classed upload error", {
  skip_if_no_gpu()
  expect_error(tbl_gpu(tibble::tibble(z = complex(2))), "`z`", fixed = TRUE,
               class = "cuplyr_error_unsupported_class")
  expect_error(tbl_gpu(tibble::tibble(r = as.raw(1:2))), "`r`", fixed = TRUE,
               class = "cuplyr_error_unsupported_class")
})

test_that("the upload hint fits the kind of column", {
  skip_if_no_gpu()
  hint <- function(col) {
    e <- tryCatch(tbl_gpu(tibble::tibble(c = col)), error = identity)
    paste(conditionMessage(e), collapse = " ")
  }
  expect_match(hint(cbind(1:2, 3:4)), "one plain column per matrix column", fixed = TRUE)
  expect_match(hint(tibble::tibble(a = 1:2)), "tidyr::unpack()", fixed = TRUE)
  expect_match(hint(list(1, 2)), "Make a list column atomic", fixed = TRUE)
  expect_no_match(hint(list(1, 2)), "matrix column", fixed = TRUE)
})
