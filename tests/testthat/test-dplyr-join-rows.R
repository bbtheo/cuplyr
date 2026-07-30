# Oracle parity tests for Phase 7 J5: multiple=/relationship=/unmatched=
# activation (mirrors dplyr's own tests/testthat/test-join-rows.R).
#
# check_join_cardinality() (R/join.R) raises dplyr's EXACT rlang classes and
# message/bullet texts for the many-to-many auto-warning, the three
# relationship= cardinality violations (one-to-one both directions,
# one-to-many, many-to-one), and the two unmatched= violations (x-side
# "matches_nothing", y-side "matches_remaining") -- every text/class was
# empirically transcribed against dplyr 1.2.1 by walking dplyr's own
# internal condition-building functions (see check_join_cardinality()'s own
# roxygen, R/join.R, for the full list) and cross-checked live via
# `Rscript` against real `left_join()`/`inner_join()`/`full_join()`/
# `right_join()` calls.
#
# `multiple=` filtering itself (src/ops_join.cpp apply_multiple_filter(),
# called from build_join_result() between the row-order stable sort and the
# final gather) is verified separately below against dplyr's actual output
# rows (not just conditions) -- including the surprising full_join()/
# right_join() "excess match reclassified as an additional unmatched-right
# row" contract (verified empirically: capping x's matches down to one
# does NOT drop the extra right row entirely for those two join types,
# unlike left_join()/inner_join(), which do drop it).

# =============================================================================
# Shared helpers
# =============================================================================

#' Capture the first error or (muffled) warning `expr` signals.
#'
#' Every condition this file tests is exclusively an error OR a warning,
#' never both from the same call, so "whichever fires" is unambiguous.
#' Warnings are muffled (not just caught) so evaluation of `expr` completes
#' normally afterward -- needed because some scenarios wrap a `collect()`
#' that must finish running for its side effects (none currently) or
#' simply so a stray downstream warning-vs-error ordering can't matter.
#' @keywords internal
capture_join_condition <- function(expr) {
  warning_cnd <- NULL
  error_cnd <- tryCatch(
    withCallingHandlers(
      {
        expr
        NULL
      },
      warning = function(w) {
        warning_cnd <<- w
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  if (!is.null(error_cnd)) error_cnd else warning_cnd
}

#' Assert cuplyr's condition for `cuplyr_expr` matches dplyr's own for
#' `dplyr_expr` (identical class vector, identical `cnd_message()` text),
#' and pin both texts side by side via `expect_snapshot()`.
#' @keywords internal
expect_join_condition_matches_dplyr <- function(cuplyr_expr, dplyr_expr) {
  cu_cnd <- capture_join_condition(cuplyr_expr)
  dp_cnd <- capture_join_condition(dplyr_expr)

  expect_false(is.null(dp_cnd), info = "dplyr oracle raised no condition -- test fixture is wrong")
  expect_false(is.null(cu_cnd), info = "cuplyr raised no condition")

  expect_identical(class(cu_cnd), class(dp_cnd))
  expect_identical(rlang::cnd_message(cu_cnd), rlang::cnd_message(dp_cnd))

  expect_snapshot({
    cat("dplyr:\n")
    cat(rlang::cnd_message(dp_cnd), "\n")
    cat("cuplyr:\n")
    cat(rlang::cnd_message(cu_cnd), "\n")
  })

  invisible(list(cuplyr = cu_cnd, dplyr = dp_cnd))
}

#' Run `join_fn` eagerly AND lazily on tbl_gpu, comparing row-for-row
#' (order included -- `multiple=` result correctness is a row-order-
#' sensitive contract, same as plain join row order) against the
#' dplyr-on-data.frame oracle.
#' @keywords internal
compare_multiple_join <- function(left_df, right_df, join_fn, by, ...) {
  expected <- join_fn(left_df, right_df, by = by, ...)

  eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
    collect()
  lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                  by = by, ...) |>
    collect()

  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
}

# =============================================================================
# Snapshot + class/message parity: many-to-many auto-warning
# =============================================================================

test_that("many-to-many auto-warning matches dplyr exactly", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(9, 9, 1, 1, 2))
  right_df <- data.frame(a = c(9, 3, 1, 1, 1))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a")
  )
})

test_that("many-to-many auto-warning fires for every mutating equi join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 2))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  for (join_fn in list(dplyr::left_join, dplyr::inner_join, dplyr::full_join, dplyr::right_join)) {
    w <- capture_join_condition(join_fn(gx, gy, by = "a") |> collect())
    expect_s3_class(w, "dplyr_warning_join_relationship_many_to_many")
  }
})

test_that("many-to-many auto-warning does NOT fire when multiple != 'all'", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 2))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  for (m in c("first", "last", "any")) {
    w <- capture_join_condition(dplyr::left_join(gx, gy, by = "a", multiple = m) |> collect())
    expect_null(w)
  }
})

test_that("relationship = 'many-to-many' silences the auto-warning (both dplyr and cuplyr)", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 2))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  dp <- capture_join_condition(
    dplyr::left_join(left_df, right_df, by = "a", relationship = "many-to-many")
  )
  cu <- capture_join_condition(
    dplyr::left_join(gx, gy, by = "a", relationship = "many-to-many") |> collect()
  )
  expect_null(dp)
  expect_null(cu)
})

# =============================================================================
# Snapshot + class/message parity: relationship= violations
# =============================================================================

test_that("relationship = 'one-to-one' violation (y-side duplicate) matches dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-one") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-one")
  )
})

test_that("relationship = 'one-to-one' violation (x-side duplicate) matches dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 3))
  right_df <- data.frame(a = c(1, 2, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-one") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-one")
  )
})

test_that("relationship = 'one-to-one' tie-break (both sides violate at the same row) matches dplyr", {
  skip_if_no_gpu()

  # x rows 1,2 ("A") each match BOTH y rows 1,2 ("A") -- a genuine tie:
  # x row 1 is the first x row matching multiple y rows (needles-side
  # violation at row 1), AND y row 1 is the first y row matching multiple x
  # rows (haystack-side violation, also at row 1). Empirically verified
  # directly against vctrs::vec_locate_matches(relationship = "one-to-one")
  # (dplyr's own underlying engine): needles/x wins ties.
  left_df <- data.frame(a = c("A", "A", "B", "C"))
  right_df <- data.frame(a = c("A", "A", "C", "D"))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  result <- expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-one") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-one")
  )
  expect_s3_class(result$dplyr, "dplyr_error_join_relationship_one_to_one")
  expect_match(rlang::cnd_message(result$dplyr), "Each row in `x` must match at most 1 row in `y`",
              fixed = TRUE)
})

test_that("relationship = 'one-to-many' violation (x-side duplicate) matches dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 3))
  right_df <- data.frame(a = c(1, 2, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-many") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-many")
  )
})

test_that("relationship = 'one-to-many' allows y-side duplication", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_null(capture_join_condition(
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-many")
  ))
  expect_null(capture_join_condition(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-many") |> collect()
  ))
})

test_that("relationship = 'many-to-one' violation (y-side duplicate) matches dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 1, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "many-to-one") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "many-to-one")
  )
})

test_that("relationship = 'many-to-one' allows x-side duplication", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 1, 3))
  right_df <- data.frame(a = c(1, 2, 3))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_null(capture_join_condition(
    dplyr::left_join(left_df, right_df, by = "a", relationship = "many-to-one")
  ))
  expect_null(capture_join_condition(
    dplyr::left_join(gx, gy, by = "a", relationship = "many-to-one") |> collect()
  ))
})

# =============================================================================
# Snapshot + class/message parity: unmatched= violations
# =============================================================================

test_that("unmatched='error' x-side (matches_nothing) matches dplyr for inner_join()/right_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 3, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::inner_join(gx, gy, by = "a", unmatched = c("error", "drop")) |> collect(),
    dplyr::inner_join(left_df, right_df, by = "a", unmatched = c("error", "drop"))
  )
  expect_join_condition_matches_dplyr(
    dplyr::right_join(gx, gy, by = "a", unmatched = "error") |> collect(),
    dplyr::right_join(left_df, right_df, by = "a", unmatched = "error")
  )
})

test_that("unmatched='error' y-side (matches_remaining) matches dplyr for left_join()/inner_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 3, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", unmatched = "error") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", unmatched = "error")
  )
  expect_join_condition_matches_dplyr(
    dplyr::inner_join(gx, gy, by = "a", unmatched = c("drop", "error")) |> collect(),
    dplyr::inner_join(left_df, right_df, by = "a", unmatched = c("drop", "error"))
  )
})

test_that("right_join() never checks y-side unmatched, left_join() never checks x-side unmatched", {
  skip_if_no_gpu()

  # right_join(unmatched = "error"): x-side only -- every x row (1, 2) has a
  # y match here, so the extra y-only row (9) never errors (only an x row
  # with no y match would).
  left_df <- data.frame(a = c(1, 2))
  right_df <- data.frame(a = c(1, 2, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  expect_null(capture_join_condition(
    dplyr::right_join(left_df, right_df, by = "a", unmatched = "error")
  ))
  expect_null(capture_join_condition(
    dplyr::right_join(gx, gy, by = "a", unmatched = "error") |> collect()
  ))

  # left_join(unmatched = "error"): y-side only -- an x row with no y match
  # (id 2) never errors here (only a y row with no x match would).
  left_df2 <- data.frame(a = c(1, 2))
  right_df2 <- data.frame(a = c(1))
  gx2 <- tbl_gpu(left_df2); gy2 <- tbl_gpu(right_df2)

  expect_null(capture_join_condition(
    dplyr::left_join(left_df2, right_df2, by = "a", unmatched = "error")
  ))
  expect_null(capture_join_condition(
    dplyr::left_join(gx2, gy2, by = "a", unmatched = "error") |> collect()
  ))
})

test_that("full_join() never raises matches_nothing/matches_remaining (no unmatched= at all)", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2))
  right_df <- data.frame(a = c(1, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  # `unmatched=` isn't a formal of full_join.tbl_gpu() (matching dplyr's own
  # full_join() signature) -- passing it is silently absorbed by `...` and
  # never reaches check_join_cardinality() with anything but the hardcoded
  # "drop"/"drop" default, so neither matches_nothing nor matches_remaining
  # can ever fire. (NOTE: this is NOT compared against the dplyr oracle
  # here -- dplyr's own full_join(unmatched = "error") actually hits its
  # `...` "must be empty" dots-validation error, a narrower, pre-existing,
  # documented divergence from Phase 7 J1 that this task does not close;
  # see test-dplyr-join.R's "full_join() has no unmatched=" test.)
  expect_no_error(dplyr::full_join(gx, gy, by = "a", unmatched = "error") |> collect())
  expect_no_error(dplyr::full_join(gx, gy, by = "a") |> collect())
})

# =============================================================================
# Precedence: multiplicity (relationship=) fires before unmatched=, and
# x-side unmatched fires before y-side unmatched.
# =============================================================================

test_that("precedence: relationship violation fires before unmatched= violation", {
  skip_if_no_gpu()

  # x rows 1,2 (a = 1) both match y's two a = 1 rows (relationship
  # violation); x row 3 (a = 5) has no y match at all AND y row 3 (a = 9)
  # has no x match at all (both would trigger unmatched= if reached).
  left_df <- data.frame(a = c(1, 1, 5))
  right_df <- data.frame(a = c(1, 1, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  result <- expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-one", unmatched = "error") |>
      collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-one", unmatched = "error")
  )
  expect_s3_class(result$dplyr, "dplyr_error_join_relationship_one_to_one")

  # inner_join with the length-2 unmatched= form: same precedence holds.
  result2 <- expect_join_condition_matches_dplyr(
    dplyr::inner_join(gx, gy, by = "a", relationship = "one-to-one",
                      unmatched = c("error", "error")) |> collect(),
    dplyr::inner_join(left_df, right_df, by = "a", relationship = "one-to-one",
                      unmatched = c("error", "error"))
  )
  expect_s3_class(result2$dplyr, "dplyr_error_join_relationship_one_to_one")
})

test_that("precedence: x-side unmatched= fires before y-side unmatched= (inner_join)", {
  skip_if_no_gpu()

  # x row 2 (a = 2) has no y match (x-side violation, at index 2); y row 3
  # (a = 9) has no x match (y-side violation, at index 3) -- x-side wins.
  left_df <- data.frame(a = c(1, 2, 3))
  right_df <- data.frame(a = c(1, 3, 9))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  result <- expect_join_condition_matches_dplyr(
    dplyr::inner_join(gx, gy, by = "a", unmatched = c("error", "error")) |> collect(),
    dplyr::inner_join(left_df, right_df, by = "a", unmatched = c("error", "error"))
  )
  expect_s3_class(result$dplyr, "dplyr_error_join_matches_nothing")
  expect_match(rlang::cnd_message(result$dplyr), "Row 2 of `x`", fixed = TRUE)
})

# =============================================================================
# First-offender correctness on shuffled inputs
# =============================================================================

test_that("first-offender row numbers stay correct under a shuffled row order", {
  skip_if_no_gpu()

  set.seed(20260730)
  # 12 rows: value 7 appears TWICE in x (positions 4 and 9), matching a
  # SINGLE y row -- a one-to-many violation (a y row matching multiple x
  # rows, i.e. right_multi_first); value 11 has no y match at all.
  left_df <- data.frame(a = c(1, 2, 3, 7, 5, 6, 8, 9, 7, 10, 11, 12))
  right_df <- data.frame(a = c(1, 2, 3, 7, 5, 6, 8, 9, 10, 12))

  expect_join_condition_matches_dplyr(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "a",
                     relationship = "one-to-many") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-many")
  )

  # Shuffle both sides -- dplyr's first-offender index is defined by ROW
  # POSITION in the (possibly shuffled) input, not by key value, so
  # shuffling changes which row number is reported; the point is that
  # cuplyr's reported index still matches dplyr's on this SAME shuffled
  # data (both computed from the same permutation).
  perm_x <- sample(nrow(left_df))
  perm_y <- sample(nrow(right_df))
  shuffled_left <- left_df[perm_x, , drop = FALSE]
  shuffled_right <- right_df[perm_y, , drop = FALSE]
  rownames(shuffled_left) <- NULL
  rownames(shuffled_right) <- NULL

  expect_join_condition_matches_dplyr(
    dplyr::left_join(tbl_gpu(shuffled_left), tbl_gpu(shuffled_right), by = "a",
                     relationship = "one-to-many") |> collect(),
    dplyr::left_join(shuffled_left, shuffled_right, by = "a", relationship = "one-to-many")
  )
})

test_that("first-offender row numbers stay correct for unmatched= under a shuffled row order", {
  skip_if_no_gpu()

  set.seed(20260731)
  left_df <- data.frame(a = 1:15)
  right_df <- data.frame(a = c(1:6, 8:15))  # y is missing a = 7 (x row 7 unmatched)

  perm_x <- sample(nrow(left_df))
  perm_y <- sample(nrow(right_df))
  shuffled_left <- left_df[perm_x, , drop = FALSE]
  shuffled_right <- right_df[perm_y, , drop = FALSE]
  rownames(shuffled_left) <- NULL
  rownames(shuffled_right) <- NULL

  result <- expect_join_condition_matches_dplyr(
    dplyr::inner_join(tbl_gpu(shuffled_left), tbl_gpu(shuffled_right), by = "a",
                      unmatched = c("error", "drop")) |> collect(),
    dplyr::inner_join(shuffled_left, shuffled_right, by = "a", unmatched = c("error", "drop"))
  )
  expect_s3_class(result$dplyr, "dplyr_error_join_matches_nothing")
})

# =============================================================================
# multiple= result correctness vs dplyr oracle: first/last/any x
# left/inner/right/full joins, eager + lazy
# =============================================================================

test_that("multiple= 'first'/'last'/'any' match dplyr row-for-row: left_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), row_x = 1:3)
  right_df <- data.frame(a = c(2, 1, 1, 2, 1), row_y = 1:5)  # shuffled key order

  for (m in c("first", "last", "any")) {
    compare_multiple_join(left_df, right_df, dplyr::left_join, by = "a", multiple = m)
  }
})

test_that("multiple= 'first'/'last'/'any' match dplyr row-for-row: inner_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), row_x = 1:3)
  right_df <- data.frame(a = c(2, 1, 1, 2, 1), row_y = 1:5)

  for (m in c("first", "last", "any")) {
    compare_multiple_join(left_df, right_df, dplyr::inner_join, by = "a", multiple = m)
  }
})

test_that("multiple= 'first'/'last'/'any' match dplyr row-for-row: right_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2), row_x = 1:2)
  right_df <- data.frame(a = c(1, 1, 3), row_y = 1:3)

  for (m in c("first", "last", "any")) {
    compare_multiple_join(left_df, right_df, dplyr::right_join, by = "a", multiple = m)
  }
})

test_that("multiple= 'first'/'last'/'any' match dplyr row-for-row: full_join()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2), row_x = 1:2)
  right_df <- data.frame(a = c(1, 1, 3), row_y = 1:3)

  for (m in c("first", "last", "any")) {
    compare_multiple_join(left_df, right_df, dplyr::full_join, by = "a", multiple = m)
  }
})

test_that("multiple= 'first'/'last' on full_join()/right_join() reclassify excess matches as unmatched-right rows, not drop them", {
  skip_if_no_gpu()

  # dplyr's own (surprising) contract: capping x's matches down to one does
  # NOT drop the extra right row entirely for full_join()/right_join() --
  # it reappears as an ADDITIONAL unmatched-right row (NA on the left side)
  # -- unlike left_join()/inner_join(), which drop it outright. Pin the row
  # COUNT explicitly (not just row-for-row equality above) so a regression
  # that silently drops instead of reclassifying is caught unambiguously.
  left_df <- data.frame(a = c(1, 2), row_x = 1:2)
  right_df <- data.frame(a = c(1, 1, 3), row_y = 1:3)

  expected_full <- dplyr::full_join(left_df, right_df, by = "a", multiple = "first")
  expect_equal(nrow(expected_full), 4L)  # NOT 3 -- see roxygen above

  gpu_full <- dplyr::full_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "a",
                               multiple = "first") |> collect()
  expect_equal(as.data.frame(gpu_full), as.data.frame(expected_full))

  expected_right <- dplyr::right_join(left_df, right_df, by = "a", multiple = "first")
  expect_equal(nrow(expected_right), 3L)

  gpu_right <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "a",
                                 multiple = "first") |> collect()
  expect_equal(as.data.frame(gpu_right), as.data.frame(expected_right))
})

# =============================================================================
# na_matches= interaction with cardinality checks
# =============================================================================

test_that("na_matches = 'na' (default): NA keys count toward multiplicity/relationship checks", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, NA, NA, 3))
  right_df <- data.frame(a = c(1, NA, NA, 5))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  # Two NA rows on each side mutually match each other (na_matches = "na"),
  # producing a genuine many-to-many pattern at those NA rows.
  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a")
  )

  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", relationship = "one-to-one") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", relationship = "one-to-one")
  )
})

test_that("na_matches = 'never': NA keys never match, so they surface as unmatched= violations instead", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, NA, NA, 3))
  right_df <- data.frame(a = c(1, NA, NA, 5))
  gx <- tbl_gpu(left_df); gy <- tbl_gpu(right_df)

  # No many-to-many warning at all under "never" -- NA rows never match
  # anything, on either side, so there's no multiplicity to detect.
  expect_null(capture_join_condition(
    dplyr::left_join(left_df, right_df, by = "a", na_matches = "never")
  ))
  expect_null(capture_join_condition(
    dplyr::left_join(gx, gy, by = "a", na_matches = "never") |> collect()
  ))

  # But the NA-keyed y rows are now genuinely unmatched (no x row can ever
  # match them under "never"), so unmatched = "error" (y-side, left_join)
  # does fire.
  expect_join_condition_matches_dplyr(
    dplyr::left_join(gx, gy, by = "a", na_matches = "never", unmatched = "error") |> collect(),
    dplyr::left_join(left_df, right_df, by = "a", na_matches = "never", unmatched = "error")
  )
})
