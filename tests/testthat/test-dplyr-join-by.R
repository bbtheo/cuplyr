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
# 3. Non-equi join_by() forms: routed to the CPU fallback (Phase 7 J6
#    decision -- see R/join.R's `join_route_to_fallback()`/
#    `join_by_is_non_equi()`). Test names deliberately say "via fallback"
#    (not "unsupported"/"errors") so Phase 7 J7 -- which replaces this with
#    a native mixed_join/conditional_join lowering -- has a clear signal to
#    UPDATE these names/assertions (e.g. add a "no fallback notification"
#    check) rather than just deleting them.
# =============================================================================

test_that("join_by(a >= b) (inequality) works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 5, 10, 15), w = c("A", "B", "C", "D"))

  for (nm in names(mutating_join_fns)) {
    compare_join_by(left_df, right_df, mutating_join_fns[[nm]],
                    by = dplyr::join_by(a >= b), arrange_by = c("v", "w"))
  }

  compare_join_by(left_df, right_df, dplyr::semi_join, by = dplyr::join_by(a >= b))
  compare_join_by(left_df, right_df, dplyr::anti_join, by = dplyr::join_by(a >= b))
})

test_that("join_by(between(...)) (rolling/range join) works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(2, 12, 22, 40), v = c("p", "q", "r", "s"))
  right_df <- data.frame(id = c(1, 2, 3, 4), lo = c(0, 10, 20, 30),
                         hi = c(5, 15, 25, 35), w = c("A", "B", "C", "D"))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(between(a, lo, hi)), arrange_by = "v")
  compare_join_by(left_df, right_df, dplyr::inner_join,
                  by = dplyr::join_by(between(a, lo, hi)), arrange_by = "v")
})

test_that("join_by(closest(...)) works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 6, 11), v = c("p", "q", "r"))
  right_df <- data.frame(b = c(0, 5, 10, 15), w = c("A", "B", "C", "D"))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(closest(a >= b)), arrange_by = "v")
})

test_that("join_by() mixing closest() with an equi condition works via the CPU fallback", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2), c = c(10, 20), v = c("p", "q"))
  right_df <- data.frame(b = c(0, 3), d = c(10, 20), w = c("A", "B"))

  compare_join_by(left_df, right_df, dplyr::left_join,
                  by = dplyr::join_by(closest(a >= b), c == d), arrange_by = "v")
})

test_that("a non-equi join_by() triggers the fallback notification, an equi one doesn't", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(b = c(2, 3, 4), w = c(200, 300, 400))

  withr::local_options(cuplyr.fallback = "warn")

  expect_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(a >= b)) |> collect(),
    "fell back to CPU evaluation"
  )

  expect_no_warning(
    dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df),
                     by = dplyr::join_by(a == b)) |> collect()
  )
})

test_that("keep = FALSE with a non-equi join_by() errors with dplyr's exact text", {
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
