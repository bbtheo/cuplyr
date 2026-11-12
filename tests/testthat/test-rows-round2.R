# F26 review round 2: pieces the acceptance tests do not pin.

test_that("rows_delete() on a .drop = FALSE x keeps emptied groups for a later summarise()", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:3, g = c("a", "b", "a"))
  y <- tibble::tibble(id = 2L)
  want <- dplyr::summarise(dplyr::rows_delete(dplyr::group_by(x, g, .drop = FALSE), y, by = "id"),
                           n = dplyr::n())
  for (lazy in c(FALSE, TRUE)) {
    gx <- dplyr::group_by(tbl_gpu(x, lazy = lazy), g, .drop = FALSE)
    out <- dplyr::rows_delete(gx, tbl_gpu(y), by = "id")
    expect_equal(dplyr::n_groups(collect(out)), 2L)
    # summarise() on kept empty groups is a CPU step by design
    withr::local_options(cuplyr.fallback = "silent")
    expect_equal(tibble::as_tibble(collect(dplyr::summarise(out, n = dplyr::n()))), want)
  }
})

test_that("in_place = NA is accepted like dplyr and prints no stray message", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:3, v = c(1, 2, 3))
  y <- tibble::tibble(id = 2L, v = 20)
  expect_no_message(
    got <- dplyr::rows_update(tbl_gpu(x), tbl_gpu(y), by = "id", in_place = NA)
  )
  expect_equal(tibble::as_tibble(collect(got)), dplyr::rows_update(x, y, by = "id", in_place = NA))
})

test_that("errors carry the verb's name and not the data", {
  skip_if_no_gpu()
  x <- tibble::tibble(id = 1:3, v = c(1, 2, 3))
  y <- tibble::tibble(id = 3L, v = 20)
  err <- tryCatch(dplyr::rows_insert(tbl_gpu(x), tbl_gpu(y), by = "id"), error = function(e) e)
  expect_s3_class(err, "error")
  expect_match(format(conditionCall(err)), "rows_insert", fixed = TRUE)
  expect_lt(nchar(paste(deparse(conditionCall(err)), collapse = "")), 200)
  # an error from dplyr's checks on the 0-row prototypes (a factor `v` can't
  # be cast to x's double) keeps the call short too; `err` above is the one
  # raised through the CPU fallback (a conflicting key)
  y2 <- tibble::tibble(id = 3L, v = factor("a"))
  err2 <- tryCatch(dplyr::rows_update(tbl_gpu(x), tbl_gpu(y2), by = "id"), error = function(e) e)
  expect_lt(nchar(paste(deparse(conditionCall(err2)), collapse = "")), 200)
})
