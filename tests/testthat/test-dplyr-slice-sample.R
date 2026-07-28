# Mirrored dplyr-oracle tests for slice_sample() (Phase 5, task W8)
#
# Unlike every other verb's oracle tests, these do NOT use
# expect_same_as_dplyr() (helper-oracle.R): that helper calls the oracle
# pipeline and the GPU pipeline back-to-back without resetting R's global
# RNG state in between, so it can't check bit-for-bit RNG parity. Every
# test here instead calls `set.seed(<n>)` immediately before EACH of the
# two calls (real dplyr on a tibble, cuplyr on a tbl_gpu) and asserts
# EXACT equality of the collected result -- not a statistical property.
#
# GPU-native (R/slice.R::slice_sample_native()): ungrouped, group_by()-
# grouped, and by= all run natively -- CPU-side index generation (to
# reproduce dplyr's own `sample.int()` call sequence exactly), GPU-side
# gather via the existing `gpu_slice_indices()` primitive (no new C++ at
# all). Still-fallback: `weight_by=` (needs actual column values, not just
# row counts/group columns -- gpu_fallback() re-running real dplyr already
# gets this right for free).
#
# Group-ordering findings (verified empirically against dplyr 1.2.1 before
# implementing, see R/slice.R::slice_sample_native()'s own docs for the
# full write-up):
#   - group_by(): RNG draws AND output row order both follow ASCENDING
#     group-key order (NA-key group last) -- same as W7's grouped slice_*
#     family, and the same order dplyr::group_by()/group_rows() itself
#     produces.
#   - by=: RNG draws AND output row order both follow FIRST-APPEARANCE
#     group order (via dplyr's own `vctrs::vec_group_loc()`, which is
#     exactly what `dplyr:::compute_by_groups()` calls internally) -- NOT
#     ascending group-key order, and NOT a final sort-by-original-row-
#     position either. This is a correction to
#     scratchpad/phase5_window_design.md §7's own text ("`by=` => sort()
#     final indices (original row order)"), which doesn't match either
#     dimension of dplyr's actual by= behavior.

# =============================================================================
# Ungrouped
# =============================================================================

test_that("ungrouped slice_sample(n=) matches dplyr's RNG stream exactly", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:10)
  gt <- tbl_gpu(df)

  set.seed(101)
  oracle <- dplyr::slice_sample(df, n = 4)
  set.seed(101)
  gpu <- collect(dplyr::slice_sample(gt, n = 4))

  expect_equal(gpu, oracle)
})

test_that("ungrouped slice_sample(prop=) matches dplyr's RNG stream exactly", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:20)
  gt <- tbl_gpu(df)

  set.seed(202)
  oracle <- dplyr::slice_sample(df, prop = 0.3)
  set.seed(202)
  gpu <- collect(dplyr::slice_sample(gt, prop = 0.3))

  expect_equal(gpu, oracle)
})

test_that("ungrouped slice_sample(replace = TRUE) matches, including oversampling past nrow", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  gt <- tbl_gpu(df)

  set.seed(303)
  oracle <- dplyr::slice_sample(df, n = 12, replace = TRUE)
  set.seed(303)
  gpu <- collect(dplyr::slice_sample(gt, n = 12, replace = TRUE))

  expect_equal(nrow(oracle), 12L)
  expect_equal(gpu, oracle)
})

test_that("ungrouped slice_sample(n > nrow, replace = FALSE) clamps (no error) and matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  gt <- tbl_gpu(df)

  set.seed(404)
  oracle <- dplyr::slice_sample(df, n = 100)
  set.seed(404)
  gpu <- collect(dplyr::slice_sample(gt, n = 100))

  expect_equal(nrow(oracle), 5L)
  expect_equal(gpu, oracle)
})

test_that("slice_sample(n = 0) returns 0 rows and matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:5)
  gt <- tbl_gpu(df)

  set.seed(1)
  oracle <- dplyr::slice_sample(df, n = 0)
  set.seed(1)
  gpu <- collect(dplyr::slice_sample(gt, n = 0))

  expect_equal(nrow(oracle), 0L)
  expect_equal(gpu, oracle)
})

test_that("slice_sample() on a 0-row table matches (default n, and n = 3 no-replace clamp)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = integer(0))
  gt <- tbl_gpu(df)

  set.seed(1)
  oracle1 <- dplyr::slice_sample(df)
  set.seed(1)
  gpu1 <- collect(dplyr::slice_sample(gt))
  expect_equal(gpu1, oracle1)
  expect_equal(nrow(gpu1), 0L)

  set.seed(1)
  oracle2 <- dplyr::slice_sample(df, n = 3)
  set.seed(1)
  gpu2 <- collect(dplyr::slice_sample(gt, n = 3))
  expect_equal(gpu2, oracle2)
  expect_equal(nrow(gpu2), 0L)
})

test_that("slice_sample() default (no n/prop) draws exactly 1 row and matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:8)
  gt <- tbl_gpu(df)

  set.seed(55)
  oracle <- dplyr::slice_sample(df)
  set.seed(55)
  gpu <- collect(dplyr::slice_sample(gt))

  expect_equal(nrow(oracle), 1L)
  expect_equal(gpu, oracle)
})

test_that("negative n clamps like slice_head()/slice_tail() and matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:6)
  gt <- tbl_gpu(df)

  set.seed(7)
  oracle <- dplyr::slice_sample(df, n = -2)
  set.seed(7)
  gpu <- collect(dplyr::slice_sample(gt, n = -2))

  expect_equal(nrow(oracle), 4L)
  expect_equal(gpu, oracle)
})

# =============================================================================
# group_by()-grouped: ascending group-key RNG/output order
# =============================================================================

test_that("group_by()-grouped slice_sample(n=) draws/orders by ascending group key and matches", {
  skip_if_no_gpu()
  # Unsorted/interleaved groups so a passing test actually exercises the
  # group-key reordering rule.
  df <- tibble::tibble(g = c(3, 1, 1, 2, 3, 2, 1), x = c(10, 20, 30, 40, 50, 60, 70))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  set.seed(11)
  oracle <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 1)
  set.seed(11)
  gpu_res <- dplyr::slice_sample(gt, n = 1)
  expect_equal(gpu_res$groups, "g")
  gpu <- collect(gpu_res)

  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))
  # Output really is reordered into ascending group-key order, not input order.
  expect_equal(gpu$g, sort(unique(df$g)))
})

test_that("group_by()-grouped slice_sample(replace = TRUE) oversampling per group matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 2), x = c(10, 20, 30, 40, 50))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  set.seed(22)
  oracle <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 4, replace = TRUE)
  set.seed(22)
  gpu <- collect(dplyr::slice_sample(gt, n = 4, replace = TRUE))

  expect_equal(nrow(oracle), 8L)
  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("group_by()-grouped slice_sample() with single-row groups matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 2, 2, 3), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  set.seed(33)
  oracle <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 1)
  set.seed(33)
  gpu <- collect(dplyr::slice_sample(gt, n = 1))

  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))

  # replace = TRUE oversampling a single-row group: every draw is that one row.
  set.seed(44)
  oracle2 <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 3, replace = TRUE)
  set.seed(44)
  gpu2 <- collect(dplyr::slice_sample(gt, n = 3, replace = TRUE))
  expect_equal(gpu2, tibble::as_tibble(dplyr::ungroup(oracle2)))
})

test_that("group_by()-grouped slice_sample() with an NA group key matches (NA group last)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(2, NA, 1, 2, NA, 1), x = 1:6)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  set.seed(66)
  oracle <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 1)
  set.seed(66)
  gpu <- collect(dplyr::slice_sample(gt, n = 1))

  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))
})

# =============================================================================
# by=: FIRST-APPEARANCE group RNG/output order (not ascending key, not
# sort-by-original-position -- see this file's own top-of-file docs)
# =============================================================================

test_that("by= slice_sample(n=) draws/orders by first-appearance group order and matches", {
  skip_if_no_gpu()
  # Ascending key order (a, c) deliberately differs from first-appearance
  # order (c, a) so a passing test can't be explained by either group
  # coinciding with the other's order.
  df <- tibble::tibble(g = c("c", "a", "a", "c", "c"), x = 1:5)
  gt <- tbl_gpu(df)

  set.seed(7)
  oracle <- dplyr::slice_sample(df, n = 1, by = g)
  set.seed(7)
  gpu_res <- dplyr::slice_sample(gt, n = 1, by = g)
  expect_equal(gpu_res$groups, character())
  gpu <- collect(gpu_res)

  expect_equal(gpu, oracle)
  # Output order really is first-appearance (c, a), not ascending key (a, c)
  # and not a plain sort of the selected rows' original positions.
  expect_equal(gpu$g, c("c", "a"))
})

test_that("by= slice_sample(n=, replace = TRUE) with differing group sizes matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c("c", "a", "a", "c", "c"), x = 1:5)
  gt <- tbl_gpu(df)

  set.seed(11)
  oracle <- dplyr::slice_sample(df, n = 2, replace = TRUE, by = g)
  set.seed(11)
  gpu <- collect(dplyr::slice_sample(gt, n = 2, replace = TRUE, by = g))

  expect_equal(nrow(oracle), 4L)
  expect_equal(gpu, oracle)
})

test_that("by= slice_sample() on a table with a longer, more realistic group layout matches", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c("b", "b", "a", "a", "a", "c", "a", "b"),
    x = c(1, 2, 3, 4, 5, 6, 7, 8)
  )
  gt <- tbl_gpu(df)

  set.seed(42)
  oracle <- dplyr::slice_sample(df, n = 2, replace = TRUE, by = g)
  set.seed(42)
  gpu <- collect(dplyr::slice_sample(gt, n = 2, replace = TRUE, by = g))

  expect_equal(gpu, oracle)
})

# =============================================================================
# Lazy input: compute() first, result remains a valid tbl_gpu (still lazy)
# =============================================================================

test_that("slice_sample() on a lazy tbl_gpu with pending ops still matches dplyr exactly", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 2), x = c(10, 20, 30, 40, 50))
  gt <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 0) |> dplyr::group_by(g)

  set.seed(5)
  oracle <- df |> dplyr::filter(x > 0) |> dplyr::group_by(g) |> dplyr::slice_sample(n = 1)

  set.seed(5)
  gpu_res <- dplyr::slice_sample(gt, n = 1)
  expect_equal(gpu_res$exec_mode, "lazy")
  gpu <- collect(gpu_res)

  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))
})

# =============================================================================
# Native-vs-fallback dispatch: weight_by= is the only shape that still
# falls back to CPU (and notifies); everything else above never does.
# =============================================================================

test_that("slice_sample() never triggers a fallback notification for native shapes", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df)
  ggt <- gt |> dplyr::group_by(g)

  expect_no_error(collect(dplyr::slice_sample(gt, n = 2)))
  expect_no_error(collect(dplyr::slice_sample(gt, prop = 0.5)))
  expect_no_error(collect(dplyr::slice_sample(gt, n = 5, replace = TRUE)))
  expect_no_error(collect(dplyr::slice_sample(ggt, n = 1)))
  expect_no_error(collect(dplyr::slice_sample(gt, n = 1, by = g)))
})

test_that("slice_sample(weight_by=) falls back to CPU, notifies, and still matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:10, w = c(rep(1, 5), rep(10, 5)))
  gt <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")

  set.seed(9)
  oracle <- dplyr::slice_sample(df, n = 3, weight_by = w)
  set.seed(9)
  expect_warning(
    gpu <- collect(dplyr::slice_sample(gt, n = 3, weight_by = w)),
    "slice_sample.*fell back to CPU evaluation"
  )

  expect_equal(gpu, oracle)
})

test_that("grouped slice_sample(weight_by=) falls back to CPU, notifies, and still matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 1, 2, 2), x = 1:5, w = c(1, 1, 10, 1, 5))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  withr::local_options(cuplyr.fallback = "warn")

  set.seed(13)
  oracle <- df |> dplyr::group_by(g) |> dplyr::slice_sample(n = 1, weight_by = w)
  set.seed(13)
  expect_warning(
    gpu <- collect(dplyr::slice_sample(gt, n = 1, weight_by = w)),
    "slice_sample.*fell back to CPU evaluation"
  )

  expect_equal(gpu, tibble::as_tibble(dplyr::ungroup(oracle)))
})

test_that("supplying by= on an already-grouped tbl_gpu errors, matching dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2), x = c(10, 20, 30, 40))
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  expect_error(
    dplyr::slice_sample(gt, n = 1, by = g),
    "Can't supply `by`"
  )
  expect_error(
    dplyr::slice_sample(dplyr::group_by(df, g), n = 1, by = g),
    "Can't supply `by`"
  )
})
