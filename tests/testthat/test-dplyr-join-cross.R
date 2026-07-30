# Oracle parity tests for Phase 7 J3: cross_join() native via
# cudf::cross_join (mirrors dplyr's own tests/testthat/test-join-cross.R
# shape/naming).
#
# Empirically verified against dplyr 1.2.1 (see scratchpad/phase7_joins_design.md
# J3 + the task's own empirical-verification step) before implementation:
#   - args(dplyr::cross_join): x, y, ..., copy = FALSE, suffix = c(".x", ".y")
#     -- no by/keep/na_matches/multiple/unmatched/relationship at all.
#   - Output column order is left-major: all of x's columns (original
#     order), then all of y's (original order).
#   - EVERY common column name gets both suffixes, not just would-be join
#     keys (there are none) -- e.g. two tables sharing "id" AND "val" produce
#     "id.x"/"val.x"/"id.y"/"val.y", never a bare shared name.
#   - The result is always ungrouped, even when x is grouped (a
#     column-merging mutating join, unlike semi_join()/anti_join()'s
#     row-filter grouping preservation).
#   - 0-row x or y -> 0-row result with the correct merged schema, no error.
#   - copy = TRUE with a plain data.frame y works like every other join.

# Shared helper: run cross_join() eagerly AND lazily on tbl_gpu, and compare
# both against the dplyr-on-data.frame oracle, after sorting all three by
# `arrange_by` (cross_join()'s left-major row order is exact and already
# covered by the dedicated "row order" test below; this helper is for cases
# where we mainly care about values/shape, e.g. 0-row edge cases).
compare_cross_join <- function(left_df, right_df, ..., arrange_by = NULL) {
  expected <- dplyr::cross_join(left_df, right_df, ...)

  eager <- dplyr::cross_join(tbl_gpu(left_df), tbl_gpu(right_df), ...) |> collect()
  lazy <- dplyr::cross_join(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE), ...) |>
    collect()

  norm <- function(df) {
    df <- as.data.frame(df)
    if (!is.null(arrange_by)) {
      df <- df[do.call(order, as.list(df[arrange_by])), , drop = FALSE]
      rownames(df) <- NULL
    }
    df
  }

  expect_equal(norm(eager), norm(as.data.frame(expected)))
  expect_equal(norm(lazy), norm(as.data.frame(expected)))
}

test_that("cross_join() basic row order matches dplyr (left-major)", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(id = 10:12, val2 = c("x", "y", "z"))

  expected <- dplyr::cross_join(x, y)

  eager <- dplyr::cross_join(tbl_gpu(x), tbl_gpu(y)) |> collect()
  lazy <- dplyr::cross_join(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE)) |> collect()

  # No arrange_by: row order itself is the contract under test here.
  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

test_that("cross_join() suffixes EVERY common column name, not just keys", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"), extra = c("p", "q"))
  y <- data.frame(id = 10:11, val = c("m", "n"), extra = c("r", "s"))

  expected <- dplyr::cross_join(x, y)
  expect_equal(names(expected), c("id.x", "val.x", "extra.x", "id.y", "val.y", "extra.y"))

  eager <- dplyr::cross_join(tbl_gpu(x), tbl_gpu(y)) |> collect()
  lazy <- dplyr::cross_join(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE)) |> collect()

  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

test_that("cross_join() with a custom suffix matches dplyr", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(id = 10:12, val = c("x", "y", "z"))

  expected <- dplyr::cross_join(x, y, suffix = c("_L", "_R"))
  expect_equal(names(expected), c("id_L", "val_L", "id_R", "val_R"))

  eager <- dplyr::cross_join(tbl_gpu(x), tbl_gpu(y), suffix = c("_L", "_R")) |> collect()
  lazy <- dplyr::cross_join(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE),
                            suffix = c("_L", "_R")) |> collect()

  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

test_that("cross_join() with 0-row left produces a 0-row result with the right schema", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(id = 10:12, val = c("x", "y", "z"))

  compare_cross_join(x[0, ], y)
})

test_that("cross_join() with 0-row right produces a 0-row result with the right schema", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(id = 10:12, val = c("x", "y", "z"))

  compare_cross_join(x, y[0, ])
})

test_that("cross_join() with 1-row inputs matches dplyr", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(id = 10:12, val = c("x", "y", "z"))

  compare_cross_join(x[1, , drop = FALSE], y[1, , drop = FALSE])
})

test_that("cross_join() errors clearly on 0-column inputs", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2)
  y <- data.frame(id = 10:12)

  expect_error(
    dplyr::cross_join(tbl_gpu(x[, 0, drop = FALSE]), tbl_gpu(y)),
    "at least one column"
  )
  expect_error(
    dplyr::cross_join(tbl_gpu(x), tbl_gpu(y[, 0, drop = FALSE])),
    "at least one column"
  )
})

test_that("cross_join() preserves x's grouping when the group var's name doesn't collide", {
  skip_if_no_gpu()

  # dplyr's own cross_join.data.frame() ends with dplyr_reconstruct(x_out, x)
  # -- verified empirically against dplyr 1.2.1 -- so a group column whose
  # name is unique to x (no collision with y) survives under its own name
  # and STAYS grouped in the result (unlike left/inner/full/right_join(),
  # which always clear groups per D3).
  x <- data.frame(g = c(1, 1, 2), id = 1:3)
  y <- data.frame(k = 1:2)

  oracle <- dplyr::cross_join(dplyr::group_by(x, g), y)
  expect_equal(dplyr::group_vars(oracle), "g")

  gt <- tbl_gpu(x) |> dplyr::group_by(g)
  eager_result <- dplyr::cross_join(gt, tbl_gpu(y))
  expect_equal(eager_result$groups, "g")

  lazy_result <- dplyr::cross_join(as_lazy(gt), tbl_gpu(y, lazy = TRUE))
  expect_equal(lazy_result$groups, "g")

  eager <- collect(eager_result)
  lazy <- collect(lazy_result)
  expected <- as.data.frame(dplyr::ungroup(oracle))
  expect_equal(as.data.frame(eager), expected)
  expect_equal(as.data.frame(lazy), expected)
})

test_that("cross_join() drops a group var from grouping when its name collides (gets suffixed)", {
  skip_if_no_gpu()

  # Both g and id collide with y's names here, so BOTH get suffixed away
  # (e.g. "g" -> "g.x") and grouping drops entirely -- verified empirically.
  x <- data.frame(g = c(1, 1, 2), id = c(1, 2, 3))
  y <- data.frame(g = c(9, 9), id = c(5, 6))

  oracle <- dplyr::cross_join(dplyr::group_by(x, g, id), y)
  expect_equal(dplyr::group_vars(oracle), character(0))

  gt <- tbl_gpu(x) |> dplyr::group_by(g, id)
  eager_result <- dplyr::cross_join(gt, tbl_gpu(y))
  expect_equal(eager_result$groups, character(0))

  lazy_result <- dplyr::cross_join(as_lazy(gt), tbl_gpu(y, lazy = TRUE))
  expect_equal(lazy_result$groups, character(0))

  expect_equal(as.data.frame(collect(eager_result)), as.data.frame(oracle))
  expect_equal(as.data.frame(collect(lazy_result)), as.data.frame(oracle))
})

test_that("cross_join() partial grouping: only the non-colliding group var survives", {
  skip_if_no_gpu()

  # g is unique to x (stays grouped); id collides with y and gets suffixed
  # away (drops from grouping) -- verified empirically.
  x <- data.frame(g = c(1, 1, 2), id = c(1, 2, 3))
  y <- data.frame(id = c(5, 6))

  oracle <- dplyr::cross_join(dplyr::group_by(x, g, id), y)
  expect_equal(dplyr::group_vars(oracle), "g")

  gt <- tbl_gpu(x) |> dplyr::group_by(g, id)
  eager_result <- dplyr::cross_join(gt, tbl_gpu(y))
  expect_equal(eager_result$groups, "g")

  expect_equal(as.data.frame(collect(eager_result)), as.data.frame(oracle))
})

test_that("cross_join() never uses y's grouping (only x's own is the template)", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2)
  y <- data.frame(k = c(1, 1, 2))

  oracle <- dplyr::cross_join(x, dplyr::group_by(y, k))
  expect_equal(dplyr::group_vars(oracle), character(0))

  eager_result <- dplyr::cross_join(tbl_gpu(x), tbl_gpu(y) |> dplyr::group_by(k))
  expect_equal(eager_result$groups, character(0))
  expect_equal(as.data.frame(collect(eager_result)), as.data.frame(oracle))
})

test_that("cross_join() works with copy = TRUE and a plain data.frame y", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2, val = c("a", "b"))
  y <- data.frame(k = 1:2)

  expected <- dplyr::cross_join(x, y)
  result <- dplyr::cross_join(tbl_gpu(x), y, copy = TRUE) |> collect()

  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("cross_join() requires y to be a tbl_gpu unless copy = TRUE", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:2)
  y <- data.frame(k = 1:2)

  expect_error(
    dplyr::cross_join(tbl_gpu(x), y),
    "must be a tbl_gpu"
  )
})

test_that("cross_join() size warning fires when the estimated output exceeds free GPU memory", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:100)
  y <- data.frame(k = 1:100)

  testthat::local_mocked_bindings(
    gpu_memory_state = function() list(available = TRUE, free_bytes = 1),
    .package = "cuplyr"
  )

  expect_warning(
    dplyr::cross_join(tbl_gpu(x), tbl_gpu(y)),
    "estimated at"
  )
})

test_that("cross_join() size warning does not fire when there is ample free GPU memory", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:10)
  y <- data.frame(k = 1:10)

  testthat::local_mocked_bindings(
    gpu_memory_state = function() list(available = TRUE, free_bytes = 1e12),
    .package = "cuplyr"
  )

  expect_no_warning(
    dplyr::cross_join(tbl_gpu(x), tbl_gpu(y))
  )
})

test_that("cross_join() size warning is skipped (not lowered) for a lazy join", {
  skip_if_no_gpu()

  x <- data.frame(id = 1:100)
  y <- data.frame(k = 1:100)

  testthat::local_mocked_bindings(
    gpu_memory_state = function() list(available = TRUE, free_bytes = 1),
    .package = "cuplyr"
  )

  # warn_if_join_too_large() only runs on the eager path (push_join()); a
  # lazy cross_join() defers execution entirely, so building the AST node
  # itself must not warn.
  expect_no_warning(
    dplyr::cross_join(tbl_gpu(x, lazy = TRUE), tbl_gpu(y, lazy = TRUE))
  )
})
