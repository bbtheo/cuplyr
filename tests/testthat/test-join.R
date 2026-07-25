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
# drops columns from the comparison entirely -- available for callers that
# need to exclude a column for reasons unrelated to join-key coalescing (that
# gap is fixed; see src/ops_join.cpp and R/join.R).
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

# join-key coalescing: full_join() must fill the join-key column for rows
# that only exist in the right table, exactly like dplyr does. dplyr's
# contract (and the default `keep = FALSE` docs) is that unmatched
# right-only rows get the join key filled in from the right table.
#
# Fix (src/ops_join.cpp, build_join_result()): for full_join() only, the
# right table's join-key columns are gathered via `right_map` *before*
# `right_drop_cols` drops them, and `cudf::replace_nulls()` coalesces each
# gathered left-key column with its corresponding right-key column. Right-only
# rows (where `left_map` points out of bounds and the left-gathered key is
# NULL) get the key filled in from the right side; left/inner joins and
# `keep = TRUE` full joins are unaffected.
test_that("full_join() coalesces the join key for right-only rows", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  result <- result[order(result$x, result$y, na.last = TRUE), ]

  expected <- dplyr::full_join(left_df, right_df, by = "id")
  expect_equal(expected$id, c(1, 2, 3, 4))

  # The right-only row (id = 4) must be coalesced from the right table's key,
  # not left NA.
  expect_equal(result$id, c(1, 2, 3, 4))
})

test_that("full_join() basic case matches dplyr oracle", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = c("x", "y"))
})

test_that("full_join() fills unmatched rows with NA on both sides", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  row1 <- result[result$id == 1, ]
  expect_equal(row1$x, 10)
  expect_true(is.na(row1$y))

  row4 <- result[result$id == 4, ]
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

  compare_join_modes(left_df, right_df, dplyr::full_join, by = c("k1", "k2"),
                     arrange_by = c("x", "y"))
})

test_that("full_join() works with renamed keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  compare_join_modes(left_df, right_df, dplyr::full_join, by = c("a" = "b"),
                     arrange_by = c("x", "y"))
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

  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = c("val.x", "val.y"))
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

  # Left side empty: every output row is right-only.
  left_df <- data.frame(id = numeric(0), x = numeric(0))
  right_df <- data.frame(id = c(1, 2), y = c(10, 20))

  compare_join_modes(left_df, right_df, dplyr::full_join, by = "id",
                     arrange_by = "y")

  # Right side empty: every output row is left-only.
  left_df2 <- data.frame(id = c(1, 2), x = c(10, 20))
  right_df2 <- data.frame(id = numeric(0), y = numeric(0))

  compare_join_modes(left_df2, right_df2, dplyr::full_join, by = "id",
                     arrange_by = "id")
})

# FIXED: right_join.tbl_gpu() used to error on EVERY call, for any input.
#
# Root cause (R/join.R, right_join.tbl_gpu(), before this fix): the eager
# implementation built `right_join(x, y)` as a swapped `left_join(y, x, by =
# swapped_by, ...)`, where `swapped_by <- list(left = ..., right = ...)` is a
# plain list. `left_join()` (S3-dispatched to `left_join.tbl_gpu`) calls
# `parse_join_by(by, x, y)` internally, and `parse_join_by()` only understood
# `NULL`, a `dplyr_join_by` object, an unnamed character vector, or a named
# character vector -- a plain list fell through to
# `stop("Invalid \`by\` specification. ...")`. So `right_join()` never got
# past its first internal call.
#
# Fix: `parse_join_by()` now also accepts an already-parsed spec (a plain
# list with character `left`/`right` elements, as produced by
# `build_right_join_via_left()`) and returns it as-is. `right_join.tbl_gpu()`
# was also reworked to build its own `ast_join("right", ...)` node for the
# lazy path (mirroring `left_join.tbl_gpu()`/`inner_join.tbl_gpu()`/
# `full_join.tbl_gpu()`) and, for both paths, to resolve the swapped join's
# output columns via `resolve_right_join_select_idx()`, which matches by
# *origin table + raw source column name* rather than by literal output
# name -- necessary because the swapped join is always run with
# `keep = TRUE` internally, so shared key names (e.g. `"id"` on both sides)
# come back suffixed (`"id.x"`/`"id.y"`) and never literally match the
# single unsuffixed name a `keep = FALSE` right join wants.
#
# Also fixed here: the `keep = FALSE` key column's *values*. Because the
# swapped join keeps every row of the *original right* table and NA-fills
# unmatched columns from the *original left* table, naively sourcing the
# single output key column from the *left* table's raw key copy (dplyr's
# naming convention: `by = c(x_col = y_col)` displays it under `x_col`'s
# name) would leave it `NA` for right-only rows -- the same shape of gap as
# `full_join()`'s join-key coalescing bug. Since a right_join is driven
# entirely by the right table, `build_right_join_via_left()` re-points the
# `keep = FALSE` key column's value source to the right (y) table's raw key
# copy, which is always present, while keeping the left-derived display name.
test_that("right_join() basic case matches dplyr oracle", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
                     arrange_by = c("x", "y"))
})

test_that("right_join() fills unmatched left rows with NA", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  # Every row in right_df must appear (right_join preserves right rows)
  expect_equal(nrow(result), nrow(right_df))

  # Identify the unmatched row by id (now correctly coalesced from y's key).
  row4 <- result[result$id == 4, ]
  expect_equal(nrow(row4), 1)
  expect_true(is.na(row4$x))
  expect_equal(row4$y, 400)

  # Matched rows: no NAs introduced, ids line up correctly
  matched <- result[!is.na(result$x), ]
  expect_equal(nrow(matched), 2)
  expect_equal(sort(matched$id), c(2, 3))
})

test_that("right_join() works with multi-column keys", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, 1, 2), k2 = c(10, 20, 10), x = c(5, 6, 7))
  right_df <- data.frame(k1 = c(1, 2, 2), k2 = c(10, 10, 30), y = c(50, 70, 80))

  compare_join_modes(left_df, right_df, dplyr::right_join, by = c("k1", "k2"),
                     arrange_by = c("x", "y"))
})

test_that("right_join() works with renamed keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  compare_join_modes(left_df, right_df, dplyr::right_join, by = c("a" = "b"),
                     arrange_by = c("x", "y"))
})

test_that("right_join() applies suffixes on non-key name collisions", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), val = c(200, 300, 400))

  expected <- dplyr::right_join(left_df, right_df, by = "id")
  result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  expect_equal(sort(names(result)), sort(names(expected)))
  expect_true(all(c("val.x", "val.y") %in% names(result)))

  compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
                     arrange_by = c("val.x", "val.y"))
})

test_that("right_join() with keep = TRUE retains both key columns", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df),
                              by = c("a" = "b"), keep = TRUE) |>
    collect()
  expected <- dplyr::right_join(left_df, right_df, by = c("a" = "b"), keep = TRUE)

  expect_equal(sort(names(result)), sort(names(expected)))
  expect_true(all(c("a", "b") %in% names(result)))
  expect_equal(nrow(result), nrow(expected))

  # keep = TRUE retains both tables' raw key columns uncombined, so there's
  # no coalescing needed/possible here -- unlike the keep = FALSE cases
  # above, this can be compared exactly, key columns included.
  compare_join_modes(left_df, right_df, dplyr::right_join, by = c("a" = "b"),
                     keep = TRUE, arrange_by = c("x", "y"))
})

test_that("right_join() with empty inputs returns expected shape", {
  skip_if_no_gpu()

  # Left side empty: every output row is right-only.
  left_df <- data.frame(id = numeric(0), x = numeric(0))
  right_df <- data.frame(id = c(1, 2), y = c(10, 20))

  compare_join_modes(left_df, right_df, dplyr::right_join, by = "id",
                     arrange_by = "y")

  # Right side empty: right_join keeps only right's rows, so the result is
  # empty too.
  left_df2 <- data.frame(id = c(1, 2), x = c(10, 20))
  right_df2 <- data.frame(id = numeric(0), y = numeric(0))

  compare_join_modes(left_df2, right_df2, dplyr::right_join, by = "id",
                     arrange_by = "id")
})

test_that("right_join() column order matches build_join_schema (left cols then right)", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(2, 3, 4), y = c(200, 300, 400))

  result <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()

  expect_equal(names(result), c("id", "x", "y"))
})
