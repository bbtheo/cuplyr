# Oracle parity tests for Phase 7 J6: the join_by() DSL, equi subset.
#
# join_by() objects (class `dplyr_join_by`) are PRE-NORMALIZED by dplyr
# itself into `list(exprs, condition, filter, x, y)` -- `condition`/
# `filter`/`x`/`y` are parallel, per-condition character vectors (verified
# empirically against dplyr 1.2.1 via `str(join_by(...))`, see
# `join_by_is_non_equi()`'s own roxygen, R/join.R, for the full derivation):
#   join_by(a)                     -> x = "a",        y = "a",        condition = "=="
#   join_by(a == b, c >= d)        -> x = c("a","c"),  y = c("b","d"),  condition = c("==",">=")
#   join_by(x$a == y$b)            -> x = "a",         y = "b",         condition = "=="
#   join_by(between(a, lo, hi))    -> x = c("a","a"),  y = c("lo","hi"), condition = c(">=","<=")
#   join_by(closest(a >= b))       -> x = "a", y = "b", condition = ">=", filter = "max"
#
# An ALL-equi join_by() (every condition == "==" AND filter == "none") is a
# genuine 4-vector spec (parse_join_by()) and routes through the existing
# native hash-join path -- exactly like a plain `by = "col"`/`by = c(a =
# "b")` call, just parsed from a different object shape. This file covers
# that equi subset end to end. A join_by() with ANY non-equi condition or
# non-"none" filter (join_by_is_non_equi() == TRUE) is routed to the CPU
# fallback (gpu_fallback(), the same two-table shape nest_join()/set-ops use
# in R/fallback.R) rather than a hard stop() -- Phase 7 J7 replaces this
# with a native mixed_join/conditional_join lowering; the "non-equi forms"
# section below pins today's fallback behavior with names that will need
# updating (not just deleting) once J7 lands.

# =============================================================================
# Shared helper: run `join_fn` eagerly AND lazily on tbl_gpu, compare both
# against the dplyr-on-data.frame oracle. Mirrors test-join.R's own
# `compare_join_modes()` (this file is self-sufficient, per the project's
# testthat conventions, rather than reaching into another test file).
# =============================================================================

compare_join_by <- function(left_df, right_df, join_fn, by, ...,
                            arrange_by = NULL) {
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

  expect_equal(eager_df, expected_df)
  expect_equal(lazy_df, expected_df)
}

# =============================================================================
# 1. join_by() STRUCTURE (no GPU needed) -- pins the empirically-verified
#    field shapes this file's whole design rests on, so a future dplyr
#    upgrade that changes join_by()'s internals fails loudly here first.
# =============================================================================

test_that("join_by() field shapes match this package's assumptions", {
  jb_a <- dplyr::join_by(a)
  expect_identical(jb_a$x, "a")
  expect_identical(jb_a$y, "a")
  expect_identical(jb_a$condition, "==")
  expect_identical(jb_a$filter, "none")

  jb_multi <- dplyr::join_by(a == b, c >= d)
  expect_identical(jb_multi$x, c("a", "c"))
  expect_identical(jb_multi$y, c("b", "d"))
  expect_identical(jb_multi$condition, c("==", ">="))
  expect_identical(jb_multi$filter, c("none", "none"))

  jb_dollar <- dplyr::join_by(x$a == y$b)
  expect_identical(jb_dollar$x, "a")
  expect_identical(jb_dollar$y, "b")
  expect_identical(jb_dollar$condition, "==")

  jb_between <- dplyr::join_by(between(a, lo, hi))
  expect_identical(jb_between$condition, c(">=", "<="))
  expect_identical(jb_between$filter, c("none", "none"))

  jb_closest <- dplyr::join_by(closest(a >= b))
  expect_identical(jb_closest$condition, ">=")
  expect_identical(jb_closest$filter, "max")

  # filter is PARALLEL to condition/x/y, not a single scalar -- mixing
  # closest() with a plain equi condition in one call produces a length-2
  # filter vector, not a single "max/min for the whole join" value.
  jb_mixed <- dplyr::join_by(closest(a >= b), c == d)
  expect_identical(jb_mixed$filter, c("max", "none"))
})

test_that("join_by_is_non_equi() classifies every shape correctly", {
  expect_false(join_by_is_non_equi(NULL))
  expect_false(join_by_is_non_equi("a"))
  expect_false(join_by_is_non_equi(c(a = "b")))
  expect_false(join_by_is_non_equi(dplyr::join_by(a)))
  expect_false(join_by_is_non_equi(dplyr::join_by(a == b, c == d)))
  expect_false(join_by_is_non_equi(dplyr::join_by(x$a == y$b)))

  expect_true(join_by_is_non_equi(dplyr::join_by(a >= b)))
  expect_true(join_by_is_non_equi(dplyr::join_by(between(a, lo, hi))))
  expect_true(join_by_is_non_equi(dplyr::join_by(closest(a >= b))))
  expect_true(join_by_is_non_equi(dplyr::join_by(closest(a >= b), c == d)))
})

test_that("parse_join_by() maps an equi join_by() object onto the 4-vector spec", {
  x <- tbl_gpu(data.frame(a = 1, c = 1))
  y <- tbl_gpu(data.frame(b = 1, d = 1))

  spec <- parse_join_by(dplyr::join_by(a == b, c == d), x, y)
  expect_identical(spec$left, c("a", "c"))
  expect_identical(spec$right, c("b", "d"))
  expect_identical(spec$op, c("==", "=="))
  expect_identical(spec$filter, "none")

  spec_natural <- parse_join_by(dplyr::join_by(a), x, y)
  expect_identical(spec_natural$left, "a")
  expect_identical(spec_natural$right, "a")
})

# =============================================================================
# 2. Equi join_by() forms end to end, ALL join types, eager + lazy
# =============================================================================

mutating_join_fns <- list(
  left_join = dplyr::left_join,
  inner_join = dplyr::inner_join,
  full_join = dplyr::full_join,
  right_join = dplyr::right_join
)

test_that("join_by(a) (bare symbol, natural equi key) works for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, 4), v = c(10, 20, 30, 40))
  right_df <- data.frame(a = c(2, 3, 4, 5), w = c(200, 300, 400, 500))

  for (nm in names(mutating_join_fns)) {
    compare_join_by(left_df, right_df, mutating_join_fns[[nm]],
                    by = dplyr::join_by(a), arrange_by = "a")
  }

  compare_join_by(left_df, right_df, dplyr::semi_join, by = dplyr::join_by(a))
  compare_join_by(left_df, right_df, dplyr::anti_join, by = dplyr::join_by(a))
})

test_that("join_by(a == b) (renamed equi key) works for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, 4), v = c(10, 20, 30, 40))
  right_df <- data.frame(b = c(2, 3, 4, 5), w = c(200, 300, 400, 500))

  for (nm in names(mutating_join_fns)) {
    compare_join_by(left_df, right_df, mutating_join_fns[[nm]],
                    by = dplyr::join_by(a == b), arrange_by = "a")
  }

  compare_join_by(left_df, right_df, dplyr::semi_join, by = dplyr::join_by(a == b))
  compare_join_by(left_df, right_df, dplyr::anti_join, by = dplyr::join_by(a == b))
})

test_that("join_by(a == b, c == d) (multi-condition equi) works for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, 4), c = c(1, 1, 2, 2), v = c(10, 20, 30, 40))
  right_df <- data.frame(b = c(1, 2, 3, 4), d = c(1, 2, 1, 2), w = c(100, 200, 300, 400))

  for (nm in names(mutating_join_fns)) {
    compare_join_by(left_df, right_df, mutating_join_fns[[nm]],
                    by = dplyr::join_by(a == b, c == d), arrange_by = c("a", "c"))
  }

  compare_join_by(left_df, right_df, dplyr::semi_join, by = dplyr::join_by(a == b, c == d))
  compare_join_by(left_df, right_df, dplyr::anti_join, by = dplyr::join_by(a == b, c == d))
})

test_that("join_by(x$a == y$b) (explicit table-prefixed form) works", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), w = c(200, 300, 400))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(x$a == y$b), arrange_by = "a")
  compare_join_by(left_df, right_df, dplyr::inner_join,
                  by = dplyr::join_by(x$a == y$b), arrange_by = "a")
})

test_that("join_by() with keep = TRUE retains both key columns for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, 4), v = c(10, 20, 30, 40))
  right_df <- data.frame(a = c(2, 3, 4, 5), v = c(200, 300, 400, 500))

  for (nm in names(mutating_join_fns)) {
    compare_join_by(left_df, right_df, mutating_join_fns[[nm]],
                    by = dplyr::join_by(a), keep = TRUE, arrange_by = "v.x")
  }
})

test_that("join_by() works identically to an equivalent by = c(a = 'b') call", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), w = c(200, 300, 400))

  by_string <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = c(a = "b")) |>
    collect()
  by_join_by <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = dplyr::join_by(a == b)) |>
    collect()

  expect_equal(as.data.frame(by_string), as.data.frame(by_join_by))
})

# =============================================================================
# 3. Non-equi join_by() forms: NATIVE as of Phase 7 J7 (mixed_join/
#    conditional_join, see `gpu_cond_join()` src/ops_join.cpp and
#    `build_join_ast()` src/expr_eval.hpp) for every operator join_by()
#    itself can produce ("==", ">=", ">", "<=", "<"), on non-STRING columns,
#    with `filter == "none"` (no `closest()`/rolling). `join_by_natively_
#    supported()` (R/join.R) is what screens the still-unsupported subset
#    (closest()/rolling `filter=`, STRING non-equi columns) to the CPU
#    fallback -- see section 4 below, which pins THAT behavior instead.
#
#    An important, empirically-verified (dplyr 1.2.1) wrinkle pinned here:
#    `na_matches = "na"` (the default) makes a NA on BOTH sides a genuine
#    match for the INCLUSIVE operators ("==", ">=", "<="), not just plain
#    equi keys -- `inner_join(data.frame(a=NA), data.frame(b=NA), by =
#    join_by(a >= b))` returns ONE row, not zero. The STRICT operators (">",
#    "<") never match NA regardless of `na_matches`. `na_matches = "never"`
#    disables the inclusive-operator NA-match too (matches cudf's native
#    null-propagating AST comparison, which is what happens with no special
#    handling at all).
# =============================================================================

non_equi_join_fns <- list(
  left_join = dplyr::left_join,
  inner_join = dplyr::inner_join,
  full_join = dplyr::full_join,
  right_join = dplyr::right_join
)

#' Assert `expr` doesn't trigger a fallback notification
#'
#' NOT the same question as "no warnings at all": several of the cases
#' below legitimately trigger dplyr's OWN many-to-many cardinality warning
#' (a real, unrelated `dplyr_warning_join_relationship_many_to_many`,
#' `check_join_cardinality()`, R/join.R) on genuinely duplicated non-equi
#' matches -- that warning must NOT be misread as (or masked by checking
#' for) a fallback notification. Only the specific "fell back to CPU
#' evaluation" message (`cuplyr_fallback_notify()`, R/execute.R) counts.
#' @keywords internal
expect_no_fallback_notification <- function(expr) {
  withr::local_options(cuplyr.fallback = "warn")
  saw_fallback <- FALSE
  withCallingHandlers(
    force(expr),
    warning = function(w) {
      if (grepl("fell back to CPU evaluation", conditionMessage(w), fixed = TRUE)) {
        saw_fallback <<- TRUE
        invokeRestart("muffleWarning")
      }
    }
  )
  expect_false(saw_fallback)
}

test_that("join_by(a <op> b) is native (no fallback) for every inequality operator and join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 5, 10, 15), w = c("A", "B", "C", "D"))

  for (op in c(">=", ">", "<=", "<")) {
    by <- switch(op,
      ">=" = dplyr::join_by(a >= b), ">" = dplyr::join_by(a > b),
      "<=" = dplyr::join_by(a <= b), "<" = dplyr::join_by(a < b))

    for (nm in names(non_equi_join_fns)) {
      expect_no_fallback_notification(
        compare_join_by(left_df, right_df, non_equi_join_fns[[nm]],
                        by = by, arrange_by = c("v", "w"))
      )
    }
    expect_no_fallback_notification(
      compare_join_by(left_df, right_df, dplyr::semi_join, by = by)
    )
    expect_no_fallback_notification(
      compare_join_by(left_df, right_df, dplyr::anti_join, by = by)
    )
  }
})

test_that("join_by(between(...)/within(...)/overlaps(...)) is native for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(2, 12, 22, 40), v = c("p", "q", "r", "s"))
  right_df <- data.frame(id = c(1, 2, 3, 4), lo = c(0, 10, 20, 30),
                         hi = c(5, 15, 25, 35), w = c("A", "B", "C", "D"))

  for (nm in names(non_equi_join_fns)) {
    expect_no_fallback_notification(
      compare_join_by(left_df, right_df, non_equi_join_fns[[nm]],
                      by = dplyr::join_by(between(a, lo, hi)), arrange_by = "v")
    )
  }
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::semi_join,
                    by = dplyr::join_by(between(a, lo, hi)))
  )
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::anti_join,
                    by = dplyr::join_by(between(a, lo, hi)))
  )

  # within(): two range columns on the left, matched against a single range
  # on the right (a duplicated non-equi condition pair on the RIGHT side --
  # the mirror image of between()'s left-side duplication).
  left_df2 <- data.frame(lo = c(1, 11), hi = c(4, 14), v = c("p", "q"))
  right_df2 <- data.frame(lo2 = c(0, 10), hi2 = c(5, 15), w = c("A", "B"))
  expect_no_fallback_notification(
    compare_join_by(left_df2, right_df2, dplyr::inner_join,
                    by = dplyr::join_by(within(lo, hi, lo2, hi2)), arrange_by = "v")
  )

  # overlaps(): two range pairs, TRUE iff the ranges intersect.
  expect_no_fallback_notification(
    compare_join_by(left_df2, right_df2, dplyr::inner_join,
                    by = dplyr::join_by(overlaps(lo, hi, lo2, hi2)), arrange_by = "v")
  )
})

test_that("join_by() mixing an equi condition with a non-equi one is native", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), c = c(1, 1, 2), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 5, 10, 15), d = c(1, 1, 2, 2), w = c("A", "B", "C", "D"))

  for (nm in names(non_equi_join_fns)) {
    expect_no_fallback_notification(
      compare_join_by(left_df, right_df, non_equi_join_fns[[nm]],
                      by = dplyr::join_by(c == d, a >= b), arrange_by = c("v", "w"))
    )
  }
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::semi_join,
                    by = dplyr::join_by(c == d, a >= b))
  )
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::anti_join,
                    by = dplyr::join_by(c == d, a >= b))
  )

  # keep = TRUE forces uniform suffixing of EVERY column (including the
  # equi pair), unlike keep = NULL's per-condition drop-only-the-equi-pair
  # default pinned by the case above -- both must work.
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::left_join,
                    by = dplyr::join_by(c == d, a >= b), keep = TRUE,
                    arrange_by = c("v", "w"))
  )
})

test_that("join_by() with multiple non-equi conditions (no equi at all) is native", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 5, 9), c = c(2, 6, 10), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 4, 8), d = c(3, 7, 11), w = c("A", "B", "C"))

  for (nm in names(non_equi_join_fns)) {
    expect_no_fallback_notification(
      compare_join_by(left_df, right_df, non_equi_join_fns[[nm]],
                      by = dplyr::join_by(a >= b, c <= d), arrange_by = c("v", "w"))
    )
  }
})

test_that("a self-join with a non-equi condition binds columns by side, not by name", {
  skip_if_no_gpu()

  # Both tables share IDENTICAL column names ("a", "b") -- ir_bind_join()
  # must resolve each condition column against the correct side via the
  # spec's explicit side tag, not a cross-schema name lookup (which could
  # not disambiguate here at all).
  df <- data.frame(a = c(1, 5, 9), b = c(9, 5, 1))

  expect_no_fallback_notification(
    compare_join_by(df, df, dplyr::inner_join, by = dplyr::join_by(a >= b))
  )
  # keep is forced TRUE for a non-equi spec, so both sides' "a"/"b" columns
  # survive suffixed (a.x/b.x/a.y/b.y) -- there is no bare "a" to arrange
  # by; left_join()'s row order is already fully deterministic (left-row
  # order, right matches fanned out in right-index order) with no
  # arranging needed for a stable comparison.
  expect_no_fallback_notification(
    compare_join_by(df, df, dplyr::left_join, by = dplyr::join_by(a >= b))
  )
})

test_that("join_by() non-equi with mismatched key types (INT32 vs FLOAT64) is native and correct", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1L, 6L, 11L), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0.5, 5.5, 10.5, 15.5), w = c("A", "B", "C", "D"))

  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::left_join,
                    by = dplyr::join_by(a >= b), arrange_by = "v")
  )
  expect_no_fallback_notification(
    compare_join_by(left_df, right_df, dplyr::inner_join,
                    by = dplyr::join_by(a >= b), arrange_by = "v")
  )
})

test_that("join_by() non-equi produces a correct empty result", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2), v = c("p", "q"))
  right_df <- data.frame(b = c(100, 200), w = c("A", "B"))

  compare_join_by(left_df, right_df, dplyr::inner_join, by = dplyr::join_by(a >= b))
  compare_join_by(left_df, right_df, dplyr::semi_join, by = dplyr::join_by(a >= b))

  eager <- dplyr::inner_join(tbl_gpu(left_df), tbl_gpu(right_df),
                             by = dplyr::join_by(a >= b)) |> collect()
  expect_equal(nrow(eager), 0)
  expect_identical(names(eager), c("a", "v", "b", "w"))
})

test_that("join_by() non-equi NA-key matching matches dplyr's na_matches= semantics", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, NA, 3), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(1, NA, 2), w = c("A", "B", "C"))

  # Inclusive operators: na_matches = "na" (default) matches a NA-NA pair.
  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a >= b),
                  arrange_by = "v")
  compare_join_by(left_df, right_df, dplyr::inner_join, by = dplyr::join_by(a >= b),
                  arrange_by = "v")
  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a <= b),
                  arrange_by = "v")

  # na_matches = "never": NA never matches, on either side.
  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a >= b),
                  na_matches = "never", arrange_by = "v")

  # Strict operator: NA never matches regardless of na_matches.
  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a > b),
                  arrange_by = "v")
  compare_join_by(left_df, right_df, dplyr::inner_join, by = dplyr::join_by(a > b),
                  arrange_by = "v")
})

test_that("join_by() non-equi duplicate fan-out matches dplyr's row order", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 0, 5, 5, 10), w = c("A1", "A2", "B1", "B2", "C"))

  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a >= b),
                  arrange_by = c("v", "w"))
  compare_join_by(left_df, right_df, dplyr::inner_join, by = dplyr::join_by(a >= b),
                  arrange_by = c("v", "w"))
})

test_that("multiple=/relationship=/unmatched= work on a native non-equi join_by()", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 0, 5, 10, 15), w = c("A1", "A2", "B", "C", "D"))

  # multiple = "first": caps each left row to its first (right_map-order)
  # match, exactly like the equi path (test-dplyr-join-rows.R).
  compare_join_by(left_df, right_df, dplyr::left_join, by = dplyr::join_by(a >= b),
                  multiple = "first", arrange_by = "v")

  # relationship = "one-to-many": errors when a y row matches multiple x
  # rows -- verify dplyr allows this combination for a non-equi by= at all,
  # and that this package's error matches exactly.
  expected_err <- tryCatch(
    dplyr::left_join(left_df, right_df, by = dplyr::join_by(a >= b),
                     relationship = "one-to-many"),
    error = function(e) conditionMessage(e)
  )
  actual_err <- tryCatch(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = dplyr::join_by(a >= b),
                     relationship = "one-to-many") |> collect(),
    error = function(e) conditionMessage(e)
  )
  expect_identical(actual_err, expected_err)

  # unmatched = "error" (inner_join, x-side): every left row must match.
  left_df2 <- data.frame(a = c(1, 2), v = c("p", "q"))
  right_df2 <- data.frame(b = c(100), w = c("A"))
  expected_err2 <- tryCatch(
    dplyr::inner_join(left_df2, right_df2, by = dplyr::join_by(a >= b),
                      unmatched = "error"),
    error = function(e) conditionMessage(e)
  )
  actual_err2 <- tryCatch(
    dplyr::inner_join(tbl_gpu(left_df2), tbl_gpu(right_df2), by = dplyr::join_by(a >= b),
                      unmatched = "error") |> collect(),
    error = function(e) conditionMessage(e)
  )
  expect_identical(actual_err2, expected_err2)
})

test_that("keep = FALSE with a non-equi join_by() errors with dplyr's exact text (native path)", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), w = c(200, 300, 400))

  expected <- tryCatch(
    dplyr::left_join(left_df, right_df, by = dplyr::join_by(a >= b), keep = FALSE),
    error = function(e) e
  )
  actual <- tryCatch(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(a >= b), keep = FALSE),
    error = function(e) e
  )

  expect_true(inherits(expected, "error"))
  expect_identical(conditionMessage(actual), conditionMessage(expected))
})

# =============================================================================
# 4. Still-fallback non-equi forms (Phase 7 J8 pins these permanently):
#    closest()/rolling `filter != "none"`, and STRING non-equi columns.
#    join_by_natively_supported() (R/join.R) is what screens these before
#    parse_join_by() is ever called.
# =============================================================================

test_that("join_by(closest(...)) still works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 5, 10, 15), w = c("A", "B", "C", "D"))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(closest(a >= b)), arrange_by = "v")
})

test_that("join_by() mixing closest() with an equi condition still works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2), c = c(10, 20), v = c("p", "q"))
  right_df <- data.frame(b = c(0, 3), d = c(10, 20), w = c("A", "B"))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(closest(a >= b), c == d), arrange_by = "v")
})

test_that("join_by() with a STRING non-equi condition still works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c("apple", "banana", "cherry"), v = c(1, 2, 3))
  right_df <- data.frame(b = c("banana", "date"), w = c(10, 20))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(a >= b), arrange_by = "v")
})

test_that("a non-natively-supported join_by() triggers the fallback notification; a native non-equi one doesn't", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), w = c(200, 300, 400))

  withr::local_options(cuplyr.fallback = "warn")

  expect_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(closest(a >= b))) |> collect(),
    "fell back to CPU evaluation"
  )

  # Single-row-per-side data: an inequality join across multiple rows
  # almost always fans out (dplyr's own many-to-many cardinality warning is
  # a SEPARATE, legitimate concern already covered above) -- this check
  # must rule out ANY warning at all, so it needs data with no possible
  # duplicated match.
  single_left <- data.frame(a = 5, v = "p")
  single_right <- data.frame(b = 3, w = "A")

  expect_no_warning(
    dplyr::left_join(tbl_gpu(single_left), tbl_gpu(single_right),
                     by = dplyr::join_by(a >= b)) |> collect()
  )

  expect_no_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(a == b)) |> collect()
  )
})

test_that("join_by() with a STRING non-equi condition triggers the fallback notification", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c("apple", "banana", "cherry"), v = c(1, 2, 3))
  right_df <- data.frame(b = c("banana", "date"), w = c(10, 20))

  withr::local_options(cuplyr.fallback = "warn")

  expect_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(a >= b)) |> collect(),
    "fell back to CPU evaluation"
  )
})
