# Oracle parity tests for Phase 7 J6: suffix=/keep= edge cases.
#
# build_join_output_info() (R/join.R) is a direct port of
# `dplyr:::join_cols()`'s own column-naming algorithm (verified via
# `body(dplyr:::join_cols)`/`body(dplyr:::add_suffixes)` against dplyr
# 1.2.1), replacing a previous one-shot "does this name collide with a
# would-be join key" check. The key empirical findings this file pins:
#
#   1. `add_suffixes()` (the port of `dplyr:::add_suffixes()`) is an
#      ITERATE-UNTIL-UNIQUE loop, not a one-shot rename: if a column's
#      natural suffixed name is ALREADY TAKEN by another column on the
#      same side, it gets suffixed AGAIN (e.g. left has both `val` and a
#      literal `val.y` column; `val` collides with the right table's `val`
#      and would naturally become `val.y`, but that's already taken, so it
#      becomes `val.y.y` instead) -- dplyr does NOT error or silently
#      produce a duplicate column name here, it disambiguates.
#   2. `suffix = ""` (the literal empty string, for either element) is
#      dplyr's own documented escape hatch meaning "don't suffix this side
#      at all" -- `add_suffixes()` short-circuits to the input unchanged,
#      even when that leaves an actual name collision. dplyr resolves that
#      collision downstream via a name-keyed column OVERWRITE
#      (`join_mutate()`'s `out[names(y_out)] <- vec_slice(y_out, ...)`,
#      verified via `body(dplyr:::join_mutate)`): the right column's DATA
#      replaces the left column's data in place, and the total column
#      count drops by one. cuplyr's gather path has a fixed physical
#      column count per join and can't reproduce a count-changing overwrite
#      natively, so `join_output_would_collide()` detects this specific
#      degenerate case up front (computable from schemas alone) and routes
#      the whole join through the CPU fallback instead of emitting a wrong
#      or duplicate-named result -- see the "suffix = c('', '') degenerate
#      collision" tests below, which assert PARITY (not "this case is
#      unsupported"), just achieved via a different code path.
#   3. `suffix=`/keep=TRUE`/`by = c(x = y)` renamed-key naming all compose
#      correctly together -- covered by the matrix below.

# =============================================================================
# Shared helper (self-sufficient, mirrors test-join.R's own
# compare_join_modes()/test-dplyr-join-by.R's compare_join_by())
# =============================================================================

compare_join_cols <- function(left_df, right_df, join_fn, by, ...,
                              arrange_by = NULL) {
  expected <- join_fn(left_df, right_df, by = by, ...)

  eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
    collect()
  lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                  by = by, ...) |>
    collect()

  expect_identical(names(eager), names(expected))
  expect_identical(names(lazy), names(expected))

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
# 1. add_suffixes() unit tests (no GPU needed) -- pins the port itself
#    against dplyr's own body(dplyr:::add_suffixes) semantics directly.
# =============================================================================

test_that("add_suffixes() reproduces dplyr's own iterate-until-unique loop", {
  # Plain, non-colliding case ("val" vs "id"): no change needed.
  expect_identical(add_suffixes(c("val"), c("id"), ".x"), "val")

  # A genuine collision: "val" appears on both sides, so it gets suffixed.
  expect_identical(add_suffixes(c("val"), c("id", "val"), ".x"), "val.x")

  # "Already taken" collision: val -> val.y collides with an existing
  # literal val.y, so it becomes val.y.y instead.
  expect_identical(
    add_suffixes(c("val", "val.y"), c("id", "val"), ".x"),
    c("val.x", "val.y")
  )

  # suffix = "" short-circuits to the input unchanged, even mid-collision.
  expect_identical(add_suffixes(c("val"), c("id", "val"), ""), "val")

  # Zero-length input returns zero-length output (no recycling weirdness).
  expect_identical(add_suffixes(character(0), c("a", "b"), ".x"), character(0))
})

test_that("join_output_would_collide() only fires for the suffix = '' degenerate case", {
  left_schema <- list(names = c("id", "val"), types = c("FLOAT64", "FLOAT64"))
  right_schema <- list(names = c("id", "val"), types = c("FLOAT64", "FLOAT64"))
  spec <- list(left = "id", right = "id", op = "==", filter = "none")

  expect_false(join_output_would_collide(left_schema, right_schema, spec,
                                        c(".x", ".y"), FALSE))
  expect_false(join_output_would_collide(left_schema, right_schema, spec,
                                        c("", ".y"), FALSE))
  expect_true(join_output_would_collide(left_schema, right_schema, spec,
                                       c("", ""), FALSE))
})

# =============================================================================
# 2. keep=TRUE with same-named keys -> BOTH copies suffixed (a.x/a.y)
# =============================================================================

test_that("keep = TRUE suffixes the shared key on both sides for every join type", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, 4), val = c(10, 20, 30, 40))
  right_df <- data.frame(a = c(2, 3, 4, 5), val = c(200, 300, 400, 500))

  for (join_fn in list(dplyr::left_join, dplyr::inner_join,
                       dplyr::full_join, dplyr::right_join)) {
    compare_join_cols(left_df, right_df, join_fn, by = "a", keep = TRUE,
                      arrange_by = "val.x")
  }
})

test_that("keep = TRUE with join_by(a) (bare symbol) matches by = 'a'", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(a = c(2, 3, 4), val = c(200, 300, 400))

  by_string <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                                by = "a", keep = TRUE) |> collect()
  by_join_by <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                                 by = dplyr::join_by(a), keep = TRUE) |> collect()

  expect_identical(names(by_string), names(by_join_by))
  expect_equal(as.data.frame(by_string), as.data.frame(by_join_by))
})

# =============================================================================
# 3. by = c(a = "c") renamed-key naming (keep = FALSE default and keep = TRUE)
# =============================================================================

test_that("by = c(a = 'c') keeps the LEFT key's name (keep = FALSE default)", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(c = c(1, 2, 3), val = c(100, 200, 300))

  compare_join_cols(left_df, right_df, dplyr::left_join, by = c(a = "c"),
                    arrange_by = "a")
})

test_that("by = c(a = 'c') with keep = TRUE retains both differently-named keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(c = c(1, 2, 3), val = c(100, 200, 300))

  compare_join_cols(left_df, right_df, dplyr::left_join, by = c(a = "c"),
                    keep = TRUE, arrange_by = "a")
})

# =============================================================================
# 4. Suffix collisions producing an ALREADY-TAKEN name: dplyr disambiguates
#    (iterate-until-unique), it does not error or emit a literal duplicate
#    name -- see add_suffixes()'s own roxygen and the unit test above.
# =============================================================================

test_that("a suffix collision with an already-taken left-side name gets double-suffixed", {
  skip_if_no_gpu()

  # left already has a literal `val.y` column; `val` collides with right's
  # `val` and would naturally become `val.y`, but that name is taken.
  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30), val.y = c(100, 200, 300))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id")

  expect_identical(names(result), names(expected))
  expect_identical(names(result), c("id", "val.x", "val.y", "val.y.y"))
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("a suffix collision with an already-taken name on BOTH sides double-suffixes correctly", {
  skip_if_no_gpu()

  # left has val AND val.x (so val's natural rename collides with its own
  # sibling column); right just has val.
  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30), val.x = c(-1, -2, -3))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id")

  expect_identical(names(result), names(expected))
  expect_identical(names(result), c("id", "val.x", "val.x.x", "val.y"))
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("suffix collision disambiguation works for full_join()/right_join() too", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30), val.y = c(100, 200, 300))
  right_df <- data.frame(id = c(2, 3, 4), val = c(1, 2, 3))

  for (join_fn in list(dplyr::full_join, dplyr::right_join)) {
    result <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |> collect()
    expected <- join_fn(left_df, right_df, by = "id")
    expect_identical(names(result), names(expected))
    expect_identical(names(result), c("id", "val.x", "val.y", "val.y.y"))
  }
})

# =============================================================================
# 5. Zero-length suffix elements: suffix = "" for one side only. Legal, no
#    name clash results here (the OTHER side's normal suffix already
#    disambiguates), so the native path handles it -- no fallback needed.
# =============================================================================

test_that("suffix = c('', '.y') suffixes only the right side, no collision", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  compare_join_cols(left_df, right_df, dplyr::left_join, by = "id",
                    suffix = c("", ".y"), arrange_by = "val")
})

test_that("suffix = c('.x', '') suffixes only the left side, no collision", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  compare_join_cols(left_df, right_df, dplyr::left_join, by = "id",
                    suffix = c(".x", ""), arrange_by = "val.x")
})

# =============================================================================
# 6. suffix = c("", "") degenerate collision: BOTH sides skip suffixing,
#    the right column's data silently overwrites the left's (dplyr's own
#    behavior, verified empirically) -- cuplyr detects this and reproduces
#    it via the CPU fallback, not the native gather path.
# =============================================================================

test_that("suffix = c('', '') on a genuine name collision matches dplyr via fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id",
                             suffix = c("", "")) |> collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id", suffix = c("", ""))

  # dplyr's own contract here: the right table's column silently replaces
  # the left's under the shared name -- only ONE `val` column survives,
  # holding the RIGHT table's values.
  expect_identical(names(expected), c("id", "val"))
  expect_equal(expected$val, right_df$val)

  expect_identical(names(result), names(expected))
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("suffix = c('', '') triggers the fallback notification", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), val = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3), val = c(1, 2, 3))

  withr::local_options(cuplyr.fallback = "warn")

  expect_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id",
                     suffix = c("", "")) |> collect(),
    "fell back to CPU evaluation"
  )
})

test_that("suffix = c('', '') with NO actual collision does NOT fall back", {
  skip_if_no_gpu()

  # Disjoint non-key names: suffix = "" never gets a chance to matter.
  left_df <- data.frame(id = c(1, 2, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3), y = c(1, 2, 3))

  withr::local_options(cuplyr.fallback = "error")

  result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id",
                             suffix = c("", "")) |> collect()
  expected <- dplyr::left_join(left_df, right_df, by = "id", suffix = c("", ""))

  expect_identical(names(result), names(expected))
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("suffix = c('', '') collision fallback works for cross_join() too", {
  skip_if_no_gpu()

  left_df <- data.frame(v = c(1, 2))
  right_df <- data.frame(v = c(10, 20))

  result <- dplyr::cross_join(tbl_gpu(left_df), tbl_gpu(right_df),
                              suffix = c("", "")) |> collect()
  expected <- dplyr::cross_join(left_df, right_df, suffix = c("", ""))

  expect_identical(names(result), names(expected))
  expect_identical(names(expected), "v")
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

# =============================================================================
# 7. suffix = NA / invalid shapes -> error, matching dplyr's exact text
#    (standardise_join_suffix()'s own contract, verified empirically)
# =============================================================================

expect_same_suffix_error <- function(left_df, right_df, ...) {
  expected <- tryCatch(dplyr::left_join(left_df, right_df, ...), error = function(e) e)
  actual <- tryCatch(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), ...),
    error = function(e) e
  )
  expect_true(inherits(expected, "error"))
  expect_identical(conditionMessage(actual), conditionMessage(expected))
}

test_that("suffix = c(NA, '.y') errors with dplyr's exact text", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))
  expect_same_suffix_error(left_df, right_df, by = "id", suffix = c(NA, ".y"))
})

test_that("suffix = NA (bare, not a length-2 character vector) errors with dplyr's exact text", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))
  expect_same_suffix_error(left_df, right_df, by = "id", suffix = NA)
})

test_that("suffix = character(0) errors with dplyr's exact text", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))
  expect_same_suffix_error(left_df, right_df, by = "id", suffix = character(0))
})

test_that("suffix of length 1 errors with dplyr's exact text", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))
  expect_same_suffix_error(left_df, right_df, by = "id", suffix = c(".x"))
})

test_that("suffix of length 3 errors with dplyr's exact text", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))
  expect_same_suffix_error(left_df, right_df, by = "id", suffix = c(".x", ".y", ".z"))
})

test_that("suffix validation applies to inner_join()/full_join()/right_join()/cross_join() too", {
  skip_if_no_gpu()
  left_df <- data.frame(id = c(1, 2), val = c(10, 20))
  right_df <- data.frame(id = c(1, 2), val = c(1, 2))

  for (join_fn in list(dplyr::inner_join, dplyr::full_join, dplyr::right_join)) {
    expected <- tryCatch(join_fn(left_df, right_df, by = "id", suffix = NA),
                        error = function(e) e)
    actual <- tryCatch(join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = "id", suffix = NA),
                       error = function(e) e)
    expect_true(inherits(expected, "error"))
    expect_identical(conditionMessage(actual), conditionMessage(expected))
  }

  expected_cross <- tryCatch(dplyr::cross_join(left_df, right_df, suffix = NA),
                             error = function(e) e)
  actual_cross <- tryCatch(dplyr::cross_join(tbl_gpu(left_df), tbl_gpu(right_df), suffix = NA),
                           error = function(e) e)
  expect_true(inherits(expected_cross, "error"))
  expect_identical(conditionMessage(actual_cross), conditionMessage(expected_cross))
})
