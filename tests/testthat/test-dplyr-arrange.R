# Mirrored dplyr-oracle tests for arrange()'s NA placement (Phase 6, task
# 6.1, Fix B).
#
# Real dplyr 1.2.1 places NA LAST for BOTH `arrange(x)` (ascending) and
# `arrange(desc(x))` (descending) -- verified empirically. cuplyr's own
# gpu_arrange() (src/ops_arrange.cpp) used `cudf::null_order::AFTER`
# unconditionally for every key regardless of direction; cudf's null_order is
# a physical-placement convention (AFTER = "nulls compare as larger than
# everything"), so for a DESCENDING column this actually placed NAs FIRST,
# not last -- a live parity divergence from dplyr, pre-existing since before
# Phase 5 (root-caused during Phase 5's W7 grouped-slice work, which had to
# build its own is.na()-based workaround to avoid relying on arrange()'s
# broken desc()-NA behavior -- see R/slice.R's module docs). The fix mirrors
# the same technique already applied to the window rank kernel (W7) and
# ops_slice.cpp's gpu_slice_rank(): flip null_order with direction (BEFORE
# for descending, AFTER for ascending) so NA always sorts as "last" in VALUE
# terms regardless of physical sort direction.
#
# test-arrange.R's own OLD pinned tests ("places NA first for descending
# sort") encoded the wrong (buggy) behavior as intended and are updated
# alongside this fix.

test_that("arrange(desc(x)) places NA last, matching dplyr (eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3, NA))

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(x)))
})

test_that("arrange(desc(x)) places NA last, matching dplyr (lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3, NA))

  expect_same_as_dplyr_lazy(df, function(d) dplyr::arrange(d, dplyr::desc(x)))
})

test_that("arrange(-x) (unary minus shorthand) places NA last, matching dplyr", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3, NA))

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, -x))
})

test_that("arrange() places NA last for ascending sort (oracle)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(2, NA, 1, 3, NA))

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x))
})

test_that("arrange() with mixed asc/desc multi-key sort places NA last per key (eager)", {
  skip_if_no_gpu()

  df <- data.frame(
    g = c(1, 1, 1, 2, 2, 2),
    x = c(NA, 1, 1, 2, NA, 1)
  )

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, g, dplyr::desc(x)))
})

test_that("arrange() with mixed asc/desc multi-key sort places NA last per key (lazy)", {
  skip_if_no_gpu()

  df <- data.frame(
    g = c(1, 1, 1, 2, 2, 2),
    x = c(NA, 1, 1, 2, NA, 1)
  )

  expect_same_as_dplyr_lazy(df, function(d) dplyr::arrange(d, g, dplyr::desc(x)))
})

test_that("arrange() with desc() primary and ascending secondary key places NA last on both", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, 1, NA, NA, 2, 2),
    y = c(NA, 1, 2, NA, 3, NA)
  )

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(x), y))
})

test_that("arrange(desc(x)) with .by_group=TRUE places NA last within each group", {
  skip_if_no_gpu()

  df <- data.frame(
    grp = c(1, 1, 1, 2, 2, 2),
    x = c(2, NA, 1, NA, 3, 1)
  )

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(grp) |>
        dplyr::arrange(dplyr::desc(x), .by_group = TRUE)
    }
  )
})

test_that("arrange(desc(x)) with string column places NA last (eager)", {
  skip_if_no_gpu()

  df <- data.frame(x = c("b", NA, "a", "c", NA), stringsAsFactors = FALSE)

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(x)))
})

test_that("arrange(desc(x)) with string column places NA last (lazy)", {
  skip_if_no_gpu()

  df <- data.frame(x = c("b", NA, "a", "c", NA), stringsAsFactors = FALSE)

  expect_same_as_dplyr_lazy(df, function(d) dplyr::arrange(d, dplyr::desc(x)))
})

test_that("arrange() all-NA column with desc() is unaffected (all rows NA either way)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(NA_real_, NA_real_, NA_real_))

  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, dplyr::desc(x)))
})
