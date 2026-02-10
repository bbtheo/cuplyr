# Tests for join operations

test_that("left_join() works", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::left_join(gpu_left, gpu_right, by = "id") |> collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id")

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("eager join result has no pending lazy ops", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  out <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id")
  expect_null(out$lazy_ops)
  expect_false(has_pending_ops(out))
})

test_that("inner_join() works", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::inner_join(gpu_left, gpu_right, by = "id") |> collect()
  expected <- dplyr::inner_join(left_df, right_df, by = "id")

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("left_join() works in lazy mode", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  gpu_left <- tbl_gpu(left_df, lazy = TRUE)
  gpu_right <- tbl_gpu(right_df, lazy = TRUE)

  result <- dplyr::left_join(gpu_left, gpu_right, by = "id") |> collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id")

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("left_join() works with multiple keys", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, 1, 2), k2 = c(10, 20, 10), x = c(5, 6, 7))
  right_df <- data.frame(k1 = c(1, 2, 2), k2 = c(10, 10, 30), y = c(50, 70, 80))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::left_join(gpu_left, gpu_right, by = c("k1", "k2")) |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = c("k1", "k2"))

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("left_join() with multiple keys works in lazy mode", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, 1, 2), k2 = c(10, 20, 10), x = c(5, 6, 7))
  right_df <- data.frame(k1 = c(1, 2, 2), k2 = c(10, 10, 30), y = c(50, 70, 80))

  gpu_left <- tbl_gpu(left_df, lazy = TRUE)
  gpu_right <- tbl_gpu(right_df, lazy = TRUE)

  result <- dplyr::left_join(gpu_left, gpu_right, by = c("k1", "k2")) |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = c("k1", "k2"))

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("left_join() works with different key names", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  result <- dplyr::left_join(gpu_left, gpu_right, by = c("a" = "b")) |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = c("a" = "b"))

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("left_join() with different key names works in lazy mode", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  gpu_left <- tbl_gpu(left_df, lazy = TRUE)
  gpu_right <- tbl_gpu(right_df, lazy = TRUE)

  result <- dplyr::left_join(gpu_left, gpu_right, by = c("a" = "b")) |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = c("a" = "b"))

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("join results match dplyr in eager and lazy modes (edge cases)", {
  skip_if_no_gpu()

  compare_join <- function(left_df, right_df, join_fn, by, ...) {
    expected <- join_fn(left_df, right_df, by = by, ...)

    eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
      collect()
    lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                    by = by, ...) |>
      collect()

    expect_equal(as.data.frame(eager), as.data.frame(expected))
    expect_equal(as.data.frame(lazy), as.data.frame(expected))
  }

  # Many-to-many keys
  left_df <- data.frame(k = c(1, 1, 2), x = c(10, 20, 30))
  right_df <- data.frame(k = c(1, 1, 2), y = c(100, 200, 300))
  compare_join(left_df, right_df, dplyr::inner_join, by = "k",
               relationship = "many-to-many")

  # Different key names
  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))
  compare_join(left_df, right_df, dplyr::left_join, by = c("a" = "b"))

  # NULLs in keys
  left_df <- data.frame(k = c(1, NA, 2), x = c(10, 20, 30))
  right_df <- data.frame(k = c(NA, 2, 3), y = c(100, 200, 300))
  compare_join(left_df, right_df, dplyr::left_join, by = "k")
})
