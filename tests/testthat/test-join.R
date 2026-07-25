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

# =============================================================================
# full_join() / right_join() coverage
#
# Neither had any test coverage before this file (see scratchpad/todo.md
# Phase 0). cuDF's full/right join primitives don't guarantee dplyr's row
# order (dplyr keeps left-table order for full_join, matched rows first, then
# unmatched right rows appended; right_join preserves right-table order), so
# comparisons below arrange() both the oracle and the GPU result by a
# deterministic key before comparing values. This is noted per the roadmap
# instructions and matches the ordering caveat already documented for joins
# in CLAUDE.md ("cuDF join outputs are unordered").
# =============================================================================

# Shared helper: run `join_fn` (full_join/right_join/etc.) eagerly AND lazily
# on tbl_gpu, and compare both against the dplyr oracle. `arrange_by`, when
# given, sorts all three results before comparing so row-order differences
# (legitimate for full/right joins) don't cause spurious failures. `ignore_cols`
# drops columns from the comparison entirely -- used to work around the known
# join-key-coalescing bug documented below (search this file for "BUG:").
compare_join_modes <- function(left_df, right_df, join_fn, by, ...,
                                arrange_by = NULL, ignore_cols = NULL) {
  expected <- join_fn(left_df, right_df, by = by, ...)

  eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
    collect()
  lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                  by = by, ...) |>
    collect()

  expected_df <- as.data.frame(expected)
  eager_df <- as.data.frame(eager)
  lazy_df <- as.data.frame(lazy)

  if (!is.null(arrange_by)) {
    ord <- function(df) do.call(order, c(as.list(df[arrange_by]), list(na.last = TRUE)))
    expected_df <- expected_df[ord(expected_df), , drop = FALSE]
    eager_df <- eager_df[ord(eager_df), , drop = FALSE]
    lazy_df <- lazy_df[ord(lazy_df), , drop = FALSE]
    rownames(expected_df) <- NULL
    rownames(eager_df) <- NULL
    rownames(lazy_df) <- NULL
  }

  if (!is.null(ignore_cols)) {
    expected_df <- expected_df[setdiff(names(expected_df), ignore_cols)]
    eager_df <- eager_df[setdiff(names(eager_df), ignore_cols)]
    lazy_df <- lazy_df[setdiff(names(lazy_df), ignore_cols)]
  }

  expect_equal(eager_df, expected_df)
  expect_equal(lazy_df, expected_df)
}

# BUG: gpu full_join() does not coalesce the join-key column for rows that
# only exist in the right table. dplyr's contract (and the default `keep =
# FALSE` docs) is that unmatched right-only rows get the join key filled in
# from the right table; cuplyr instead leaves it NA.
#
# Root cause (src/ops_join.cpp, gpu_full_join()/build_join_result()): the
# right table's join-key columns are unconditionally dropped via
# `right_drop_cols` before the gather step (this is correct for left/inner
# join, where the left key column is always non-null), then the left table's
# columns -- including the key -- are gathered via `left_map` with
# `out_of_bounds_policy::NULLIFY`. For a right-only row, `left_map` points
# out of bounds, so the key column comes back NULL with no fallback source
# once the right key was already dropped.
#
# This is a real, pre-existing bug discovered while adding this test-first
# coverage (scratchpad/todo.md Phase 0). Per the task's test-first mandate,
# it is documented here and NOT fixed in this task -- see the dedicated BUG
# test immediately below, which pins the current (wrong) behavior.
test_that("full_join() BUG: join key is not coalesced for right-only rows", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  result <- result[order(result$x, result$y, na.last = TRUE), ]

  # Correct dplyr behavior would give id 1, 2, 3, 4 here (see expected below).
  expected <- dplyr::full_join(left_df, right_df, by = "id")
  expect_equal(expected$id, c(1, 2, 3, 4))

  # BUG: cuplyr currently produces NA instead of the coalesced value (4) for
  # the right-only row. When this is fixed, this expectation should be
  # updated to `c(1, 2, 3, 4)` and the coalescing note above removed.
  expect_equal(result$id, c(1, 2, 3, NA))
})

test_that("full_join() basic case matches dplyr oracle", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  # BUG: the "id" column is excluded from comparison due to the join-key
  # coalescing bug documented above (unmatched right-only row gets NA instead
  # of 4). Non-key columns are still compared exactly.
  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = c("x", "y"), ignore_cols = "id")
})

test_that("full_join() fills unmatched rows with NA on both sides", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  # Select rows by non-key columns (id has the coalescing bug documented
  # above for the right-only row, so it can't be used as a reliable filter).
  row1 <- result[!is.na(result$x) & result$x == 10, ]
  expect_equal(row1$x, 10)
  expect_true(is.na(row1$y))

  row4 <- result[!is.na(result$y) & result$y == 400, ]
  expect_true(is.na(row4$x))
  expect_equal(row4$y, 400)

  # id 2 and 3: matched, no NAs introduced
  matched <- result[!is.na(result$x) & !is.na(result$y), ]
  expect_equal(nrow(matched), 2)
  expect_false(any(is.na(matched$x)))
  expect_false(any(is.na(matched$y)))
})

test_that("full_join() works with multi-column keys", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, 1, 2), k2 = c(10, 20, 10), x = c(5, 6, 7))
  right_df <- data.frame(k1 = c(1, 2, 2), k2 = c(10, 10, 30), y = c(50, 70, 80))

  # BUG: same join-key coalescing bug as above, applied to both key columns.
  compare_join_modes(left_df, right_df, dplyr::full_join, by = c("k1", "k2"),
                     arrange_by = c("x", "y"), ignore_cols = c("k1", "k2"))
})

test_that("full_join() works with renamed keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  # BUG: same join-key coalescing bug; output key column is "a" (left's name).
  compare_join_modes(left_df, right_df, dplyr::full_join, by = c("a" = "b"),
                     arrange_by = c("x", "y"), ignore_cols = "a")
})

test_that("full_join() applies suffixes on non-key name collisions", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), val = c(200, 300, 400))

  expected <- dplyr::full_join(left_df, right_df, by = "id")
  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  expect_equal(sort(names(result)), sort(names(expected)))
  expect_true(all(c("val.x", "val.y") %in% names(result)))

  # BUG: same join-key coalescing bug as above.
  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = c("val.x", "val.y"), ignore_cols = "id")
})

test_that("full_join() with keep = TRUE retains both key columns", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df),
                             by = c("a" = "b"), keep = TRUE) |>
    collect()
  expected <- dplyr::full_join(left_df, right_df, by = c("a" = "b"), keep = TRUE)

  expect_equal(sort(names(result)), sort(names(expected)))
  expect_true(all(c("a", "b") %in% names(result)))
})

test_that("full_join() with empty inputs returns expected shape", {
  skip_if_no_gpu()

  # Left side empty: every output row is right-only, so this hits the
  # join-key coalescing BUG documented above for every single row.
  left_df <- data.frame(id = numeric(0), x = numeric(0))
  right_df <- data.frame(id = c(1, 2), y = c(10, 20))

  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = "y", ignore_cols = "id")

  # Right side empty: every output row is left-only, so the key column is
  # sourced entirely from the left table and the coalescing bug doesn't
  # apply here -- this direction can be compared exactly, id included.
  left_df2 <- data.frame(id = c(1, 2), x = c(10, 20))
  right_df2 <- data.frame(id = numeric(0), y = numeric(0))

  compare_join_modes(left_df2, right_df2, dplyr::full_join, by = "id",
                     arrange_by = "id")
})

# BUG: right_join.tbl_gpu() currently errors on EVERY call, for any input.
#
# Root cause (R/join.R, right_join.tbl_gpu(), ~line 420 before the dedup
# refactor): it builds `by_swapped <- list(left = ..., right = ...)` -- a
# plain list -- and passes it as the `by` argument to `left_join()`. Since
# `left_join` (imported from dplyr, S3-dispatched to `left_join.tbl_gpu`)
# calls `parse_join_by(by, x, y)` internally, and `parse_join_by()` only
# understands `NULL`, a `dplyr_join_by` object, an unnamed character vector,
# or a named character vector (R/join.R, parse_join_by()) -- a plain list
# falls through to `stop("Invalid \`by\` specification. ...")`. So
# `right_join()` never gets past its first internal call.
#
# This means every one of the intended right_join() tests below (basic case,
# NA-fill, multi-column keys, renamed keys, suffixes, keep = TRUE, empty
# inputs, column order) can't currently run at all. Per the task's test-first
# mandate, this is documented and NOT fixed here -- a dedicated test pins the
# current (broken) behavior, and the intended coverage is kept as skipped
# placeholders (with the assertions commented out) so it can be reactivated
# once `right_join.tbl_gpu()` is fixed to build a proper `by` specification
# (e.g. a named character vector) for its internal `left_join()` call.
test_that("right_join() BUG: errors on every call instead of joining", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  expect_error(
    dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id"),
    "Invalid `by` specification"
  )
})

test_that("right_join() basic case matches dplyr oracle", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  # right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))
  #
  # compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
  #                    arrange_by = "id")
})

test_that("right_join() fills unmatched left rows with NA", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  # right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))
  #
  # result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
  #   collect()
  #
  # # Every row in right_df must appear (right_join preserves right rows)
  # expect_equal(nrow(result), nrow(right_df))
  #
  # result <- result[order(result$id), ]
  # row4 <- result[result$id == 4, ]
  # expect_true(is.na(row4$x))
  # expect_equal(row4$y, 400)
})

test_that("right_join() works with multi-column keys", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(k1 = c(1, 1, 2), k2 = c(10, 20, 10), x = c(5, 6, 7))
  # right_df <- data.frame(k1 = c(1, 2, 2), k2 = c(10, 10, 30), y = c(50, 70, 80))
  #
  # compare_join_modes(left_df, right_df, dplyr::right_join, by = c("k1", "k2"),
  #                    arrange_by = c("k1", "k2"))
})

test_that("right_join() works with renamed keys", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  # right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))
  #
  # compare_join_modes(left_df, right_df, dplyr::right_join, by = c("a" = "b"),
  #                    arrange_by = "b")
})

test_that("right_join() applies suffixes on non-key name collisions", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  # right_df <- data.frame(id = c(2, 3, 4), val = c(200, 300, 400))
  #
  # expected <- dplyr::right_join(left_df, right_df, by = "id")
  # result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
  #   collect()
  #
  # expect_equal(sort(names(result)), sort(names(expected)))
  # expect_true(all(c("val.x", "val.y") %in% names(result)))
  #
  # compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
  #                    arrange_by = "id")
})

test_that("right_join() with keep = TRUE retains both key columns", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  # right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))
  #
  # result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df),
  #                             by = c("a" = "b"), keep = TRUE) |>
  #   collect()
  # expected <- dplyr::right_join(left_df, right_df, by = c("a" = "b"), keep = TRUE)
  #
  # expect_equal(sort(names(result)), sort(names(expected)))
  # expect_true(all(c("a", "b") %in% names(result)))
  # expect_equal(nrow(result), nrow(expected))
})

test_that("right_join() with empty inputs returns expected shape", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(id = numeric(0), x = numeric(0))
  # right_df <- data.frame(id = c(1, 2), y = c(10, 20))
  #
  # compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
  #                    arrange_by = "id")
  #
  # left_df2 <- data.frame(id = c(1, 2), x = c(10, 20))
  # right_df2 <- data.frame(id = numeric(0), y = numeric(0))
  #
  # compare_join_modes(left_df2, right_df2, dplyr::right_join, by = "id",
  #                    arrange_by = "id")
})

test_that("right_join() column order matches build_join_schema (left cols then right)", {
  skip_if_no_gpu()
  skip("BUG: right_join.tbl_gpu() errors on every call -- see dedicated BUG test above")

  # left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  # right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))
  #
  # result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
  #   collect()
  #
  # expect_equal(names(result), c("id", "x", "y"))
})
