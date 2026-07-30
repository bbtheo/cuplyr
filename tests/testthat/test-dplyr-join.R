# Oracle parity tests for Phase 7 J1: the 4-vector join spec
# (list(left, right, op, filter)), na_matches = "never" threaded through
# left/inner/full/right joins via nulls_equal, and the new (currently
# dormant) multiple=/unmatched=/relationship= argument plumbing.
#
# NO arrange_by() omission here for row order: unlike test-join.R's
# full/right coverage, every comparison below sorts both sides by the join
# key before comparing (na_matches doesn't change dplyr's row-order
# contract, but NA-key placement across left/full/right joins isn't worth
# pinning exactly here -- that's test-join.R's job for the na_matches = "na"
# case already).

# =============================================================================
# na_matches = "never": empirically verified against dplyr 1.2.1
#
# Under na_matches = "never" (null_equality::UNEQUAL), NA keys never match
# ANYTHING, including another NA. This means:
#   - left_join: a left row with an NA key never matches any right row, so
#     its right-side columns come back NA (same shape as an ordinary
#     unmatched left row).
#   - full_join/right_join: an NA-keyed row from EITHER side is treated as
#     wholly unmatched -- a left row with NA key and a right row with NA key
#     each appear as SEPARATE output rows (never coalesced together), with
#     the other side's columns NA. The join-key coalescing fix (full_join's
#     right-only rows, src/ops_join.cpp build_join_result()) still fires
#     correctly here since it operates on cudf's already-computed match
#     result, independent of null_equality.
# =============================================================================

# Shared helper: run `join_fn` eagerly AND lazily on tbl_gpu, and compare
# both against the dplyr-on-data.frame oracle, after sorting all three by
# `arrange_by` (row order across NA-key rows is not part of the contract
# under test here).
compare_na_join <- function(left_df, right_df, join_fn, by, ..., arrange_by) {
  expected <- join_fn(left_df, right_df, by = by, ...)

  eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
    collect()
  lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                  by = by, ...) |>
    collect()

  ord <- function(df) do.call(order, c(as.list(df[arrange_by]), list(na.last = TRUE)))

  expected_df <- as.data.frame(expected)
  eager_df <- as.data.frame(eager)
  lazy_df <- as.data.frame(lazy)

  expected_df <- expected_df[ord(expected_df), , drop = FALSE]
  eager_df <- eager_df[ord(eager_df), , drop = FALSE]
  lazy_df <- lazy_df[ord(lazy_df), , drop = FALSE]
  rownames(expected_df) <- NULL
  rownames(eager_df) <- NULL
  rownames(lazy_df) <- NULL

  expect_equal(eager_df, expected_df)
  expect_equal(lazy_df, expected_df)
}

test_that("left_join() na_matches = 'never' matches dplyr with NA keys", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, NA, 3), v = c("a", "b", "c", "d"))
  right_df <- data.frame(id = c(1, NA, 3, 5), w = c("A", "B", "C", "D"))

  compare_na_join(left_df, right_df, dplyr::left_join, by = "id",
                  na_matches = "never", arrange_by = c("v", "w"))
  compare_na_join(left_df, right_df, dplyr::left_join, by = "id",
                  na_matches = "na", arrange_by = c("v", "w"))
})

test_that("inner_join() na_matches = 'never' matches dplyr with NA keys", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, NA, 3), v = c("a", "b", "c", "d"))
  right_df <- data.frame(id = c(1, NA, 3, 5), w = c("A", "B", "C", "D"))

  compare_na_join(left_df, right_df, dplyr::inner_join, by = "id",
                  na_matches = "never", arrange_by = c("v", "w"))
  compare_na_join(left_df, right_df, dplyr::inner_join, by = "id",
                  na_matches = "na", arrange_by = c("v", "w"))
})

test_that("full_join() na_matches = 'never' matches dplyr with NA keys", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, NA, 3), v = c("a", "b", "c", "d"))
  right_df <- data.frame(id = c(1, NA, 3, 5), w = c("A", "B", "C", "D"))

  compare_na_join(left_df, right_df, dplyr::full_join, by = "id",
                  na_matches = "never", arrange_by = c("v", "w"))
  compare_na_join(left_df, right_df, dplyr::full_join, by = "id",
                  na_matches = "na", arrange_by = c("v", "w"))
})

test_that("right_join() na_matches = 'never' matches dplyr with NA keys", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, NA, 3), v = c("a", "b", "c", "d"))
  right_df <- data.frame(id = c(1, NA, 3, 5), w = c("A", "B", "C", "D"))

  compare_na_join(left_df, right_df, dplyr::right_join, by = "id",
                  na_matches = "never", arrange_by = c("v", "w"))
  compare_na_join(left_df, right_df, dplyr::right_join, by = "id",
                  na_matches = "na", arrange_by = c("v", "w"))
})

test_that("na_matches = 'never' works with multi-key joins", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, NA, 2), k2 = c(10, 20, NA), x = c(5, 6, 7))
  right_df <- data.frame(k1 = c(1, NA, 2), k2 = c(10, 20, NA), y = c(50, 60, 70))

  compare_na_join(left_df, right_df, dplyr::left_join, by = c("k1", "k2"),
                  na_matches = "never", arrange_by = c("x", "y"))
  compare_na_join(left_df, right_df, dplyr::full_join, by = c("k1", "k2"),
                  na_matches = "never", arrange_by = c("x", "y"))
})

test_that("na_matches = 'never' works with renamed keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, NA, 3), x = c(10, 20, 30))
  right_df <- data.frame(b = c(1, NA, 4), y = c(100, 200, 400))

  compare_na_join(left_df, right_df, dplyr::left_join, by = c("a" = "b"),
                  na_matches = "never", arrange_by = c("x", "y"))
  compare_na_join(left_df, right_df, dplyr::full_join, by = c("a" = "b"),
                  na_matches = "never", arrange_by = c("x", "y"))
  compare_na_join(left_df, right_df, dplyr::right_join, by = c("a" = "b"),
                  na_matches = "never", arrange_by = c("x", "y"))
})

test_that("na_matches = 'never' works with keep = TRUE", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, NA, 3), x = c(10, 20, 30))
  right_df <- data.frame(id = c(1, NA, 4), y = c(100, 200, 400))

  compare_na_join(left_df, right_df, dplyr::left_join, by = "id",
                  na_matches = "never", keep = TRUE, arrange_by = c("x", "y"))
  compare_na_join(left_df, right_df, dplyr::full_join, by = "id",
                  na_matches = "never", keep = TRUE, arrange_by = c("x", "y"))
})

# =============================================================================
# Bad-value validation: multiple=/unmatched=/relationship=/na_matches=, with
# dplyr's own error text (verified empirically against dplyr 1.2.1).
# =============================================================================

test_that("na_matches= rejects bad values with dplyr's exact text", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::left_join(gx, gy, by = "id", na_matches = "bogus"),
    '`na_matches` must be one of "na" or "never", not "bogus".',
    fixed = TRUE
  )
  expect_error(
    dplyr::left_join(gx, gy, by = "id", na_matches = 1),
    "`na_matches` must be a string or character vector.",
    fixed = TRUE
  )
})

test_that("multiple= rejects bad values with dplyr's exact text", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::left_join(gx, gy, by = "id", multiple = "bogus"),
    '`multiple` must be one of "all", "any", "first", or "last".',
    fixed = TRUE
  )
  expect_error(
    dplyr::left_join(gx, gy, by = "id", multiple = NA),
    "`multiple` must be a string.",
    fixed = TRUE
  )
})

test_that("unmatched= rejects bad values with dplyr's exact text", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::left_join(gx, gy, by = "id", unmatched = "bogus"),
    '`unmatched` must be one of "drop" or "error", not "bogus".',
    fixed = TRUE
  )
  expect_error(
    dplyr::left_join(gx, gy, by = "id", unmatched = TRUE),
    "`unmatched` must be a character vector, not TRUE.",
    fixed = TRUE
  )
  expect_error(
    dplyr::left_join(gx, gy, by = "id", unmatched = c("drop", "drop")),
    "`unmatched` must be length 1, not 2.",
    fixed = TRUE
  )
  expect_error(
    dplyr::inner_join(gx, gy, by = "id", unmatched = c("drop", "drop", "drop")),
    "`unmatched` must be length 1 or 2, not 3.",
    fixed = TRUE
  )
})

test_that("relationship= rejects bad values with dplyr's exact text", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::inner_join(gx, gy, by = "id", relationship = "bogus"),
    paste0('`relationship` must be one of "one-to-one", "one-to-many", ',
           '"many-to-one", or "many-to-many", not "bogus".'),
    fixed = TRUE
  )
  expect_error(
    dplyr::inner_join(gx, gy, by = "id", relationship = 1),
    "`relationship` must be a string or character vector.",
    fixed = TRUE
  )
})

test_that("non-default multiple=/unmatched=/relationship= stop as not-yet-supported", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::left_join(gx, gy, by = "id", multiple = "first"),
    "not supported yet for tbl_gpu joins \\(Phase 7 J5\\)"
  )
  expect_error(
    dplyr::left_join(gx, gy, by = "id", unmatched = "error"),
    "not supported yet for tbl_gpu joins \\(Phase 7 J5\\)"
  )
  expect_error(
    dplyr::inner_join(gx, gy, by = "id", relationship = "one-to-one"),
    "not supported yet for tbl_gpu joins \\(Phase 7 J5\\)"
  )

  # Defaults never hit the not-yet-supported path.
  expect_no_error(dplyr::left_join(gx, gy, by = "id", multiple = "all"))
  expect_no_error(dplyr::left_join(gx, gy, by = "id", unmatched = "drop"))
  expect_no_error(dplyr::inner_join(gx, gy, by = "id", relationship = NULL))
})

test_that("full_join() has no unmatched= parameter (matches dplyr's own signature)", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  # `unmatched=` isn't a formal of full_join.tbl_gpu(), so it's swallowed by
  # `...` and never validated/errors -- matching dplyr's own full_join(),
  # which likewise has no `unmatched=` argument.
  expect_no_error(dplyr::full_join(gx, gy, by = "id", unmatched = "drop") |> collect())
})

# =============================================================================
# resolve_join_keep() unit tests (no GPU needed: pure list-spec logic)
# =============================================================================

test_that("resolve_join_keep() resolves NULL against equi/non-equi specs", {
  equi_spec <- list(left = "id", right = "id", op = "==", filter = "none")
  cond_spec <- list(left = "id", right = "id", op = ">=", filter = "none")

  expect_identical(cuplyr:::resolve_join_keep(NULL, equi_spec), FALSE)
  expect_identical(cuplyr:::resolve_join_keep(NULL, cond_spec), TRUE)
})

test_that("resolve_join_keep() passes through explicit TRUE/FALSE when equi", {
  equi_spec <- list(left = "id", right = "id", op = "==", filter = "none")

  expect_identical(cuplyr:::resolve_join_keep(TRUE, equi_spec), TRUE)
  expect_identical(cuplyr:::resolve_join_keep(FALSE, equi_spec), FALSE)
})

test_that("resolve_join_keep() errors on keep = FALSE with a non-equi spec", {
  cond_spec <- list(left = "id", right = "id", op = ">=", filter = "none")

  expect_error(
    cuplyr:::resolve_join_keep(FALSE, cond_spec),
    "Can't set `keep = FALSE` when using an inequality, rolling, or overlap join.",
    fixed = TRUE
  )

  # keep = TRUE is always fine, even for non-equi.
  expect_identical(cuplyr:::resolve_join_keep(TRUE, cond_spec), TRUE)
})

test_that("join_is_equi()/join_equi_spec()/join_cond_spec() partition a mixed spec", {
  spec <- list(left = c("a", "b"), right = c("a", "b"), op = c("==", ">="),
               filter = "none")

  expect_false(cuplyr:::join_is_equi(spec))
  expect_true(cuplyr:::join_is_equi(list(left = "a", right = "a", op = "==", filter = "none")))

  equi <- cuplyr:::join_equi_spec(spec)
  expect_identical(equi$left, "a")
  expect_identical(equi$right, "a")
  expect_identical(equi$op, "==")

  cond <- cuplyr:::join_cond_spec(spec)
  expect_identical(cond$left, "b")
  expect_identical(cond$right, "b")
  expect_identical(cond$op, ">=")
})

test_that("parse_join_by() returns the 4-vector spec for every existing by= form", {
  skip_if_no_gpu()

  x <- tbl_gpu(data.frame(a = 1, id = 1))
  y <- tbl_gpu(data.frame(b = 1, id = 1))

  natural <- cuplyr:::parse_join_by(NULL, x, y)
  expect_identical(natural$left, "id")
  expect_identical(natural$right, "id")
  expect_identical(natural$op, "==")
  expect_identical(natural$filter, "none")

  unnamed <- cuplyr:::parse_join_by("id", x, y)
  expect_identical(unnamed$op, "==")
  expect_identical(unnamed$filter, "none")

  named <- cuplyr:::parse_join_by(c(a = "b"), x, y)
  expect_identical(named$left, "a")
  expect_identical(named$right, "b")
  expect_identical(named$op, "==")
  expect_identical(named$filter, "none")
})

# =============================================================================
# Phase 7 J2: semi_join()/anti_join() native via cudf::filtered_join
#
# Verified empirically against dplyr 1.2.1 before implementation:
#   - Left row ORDER IS a real contract (unlike the mutating joins' NA-key
#     ordering, which this file's na_matches block above deliberately sorts
#     around) -- no arrange_by() anywhere in this section.
#   - No duplication: even when multiple right rows match one left key,
#     that left row appears at most once in the result.
#   - Duplicate left rows (same key value, different row) are each kept or
#     dropped independently -- semi_join()/anti_join() never merge rows.
#   - na_matches = "na" (default): an NA key on one side matches an NA key
#     on the other (semi keeps the row, anti drops it). For multi-key
#     joins this is a per-key-column comparison: only a row whose EVERY
#     key column matches (NA-to-NA counted as a match under "na") is a hit.
#   - na_matches = "never": NA keys never match anything, including
#     another NA (semi drops the row, anti keeps it).
#   - Grouping (`group_by()`) on x is PRESERVED by both verbs -- a real
#     divergence from the mutating joins, which always clear groups
#     (D3/`propagate_groups()`) -- `push_join()`'s `"semi"`/`"anti"`
#     special case (R/execute.R) carries `x$groups` forward instead.
# =============================================================================

# Shared helper: run `join_fn` (semi_join/anti_join) eagerly AND lazily on
# tbl_gpu, and compare row-for-row -- INCLUDING order, a real contract for
# these verbs -- against the dplyr-on-data.frame oracle.
compare_semi_anti <- function(left_df, right_df, join_fn, by, ...) {
  expected <- join_fn(left_df, right_df, by = by, ...)

  eager <- join_fn(tbl_gpu(left_df), tbl_gpu(right_df), by = by, ...) |>
    collect()
  lazy <- join_fn(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                  by = by, ...) |>
    collect()

  expected_df <- as.data.frame(expected)
  eager_df <- as.data.frame(eager)
  lazy_df <- as.data.frame(lazy)
  rownames(expected_df) <- NULL
  rownames(eager_df) <- NULL
  rownames(lazy_df) <- NULL

  expect_equal(eager_df, expected_df)
  expect_equal(lazy_df, expected_df)
}

test_that("semi_join()/anti_join() match dplyr: single key, default na_matches", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(3, 1, 2, 1, NA), v = 1:5)
  right_df <- data.frame(id = c(1, 2, 2, NA))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id")
})

test_that("semi_join()/anti_join() na_matches = 'never'", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(3, 1, 2, 1, NA), v = 1:5)
  right_df <- data.frame(id = c(1, 2, 2, NA))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id", na_matches = "never")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id", na_matches = "never")
})

test_that("semi_join()/anti_join() work with multi-key joins (incl. NA keys, both na_matches)", {
  skip_if_no_gpu()

  left_df <- data.frame(k1 = c(1, 2, 1, NA), k2 = c(10, 20, 10, 30), x = c(5, 6, 7, 8))
  right_df <- data.frame(k1 = c(1, NA), k2 = c(10, 30))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = c("k1", "k2"))
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = c("k1", "k2"))
  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = c("k1", "k2"), na_matches = "never")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = c("k1", "k2"), na_matches = "never")
})

test_that("semi_join()/anti_join() work with renamed keys", {
  skip_if_no_gpu()

  left_df <- data.frame(a = c(1, 2, 3, NA), x = c(10, 20, 30, 40))
  right_df <- data.frame(b = c(1, 3, NA), y = c(100, 300, 400))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = c("a" = "b"))
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = c("a" = "b"))
})

test_that("semi_join() does not duplicate a left row on multiple right matches", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), v = c("a", "b", "c"))
  right_df <- data.frame(id = c(1, 1, 1, 2, 2))  # id 1 matches 3x, id 2 matches 2x

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")

  result <- dplyr::semi_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |> collect()
  expect_equal(nrow(result), 2)
})

test_that("semi_join()/anti_join() keep duplicate left rows independently", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 1, 2, 1), v = c("a", "b", "c", "d"))
  right_df <- data.frame(id = c(1))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id")
})

test_that("semi_join()/anti_join() with an empty right table", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), v = 1:3)
  right_df <- data.frame(id = numeric(0))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id")
})

test_that("semi_join()/anti_join() with an empty (0-row) left table", {
  skip_if_no_gpu()

  left_df <- data.frame(id = numeric(0), v = character(0))
  right_df <- data.frame(id = c(1, 2))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id")
})

test_that("semi_join()/anti_join() with both sides empty (0-row)", {
  skip_if_no_gpu()

  left_df <- data.frame(id = numeric(0), v = character(0))
  right_df <- data.frame(id = numeric(0))

  compare_semi_anti(left_df, right_df, dplyr::semi_join, by = "id")
  compare_semi_anti(left_df, right_df, dplyr::anti_join, by = "id")
})

test_that("semi_join()/anti_join() preserve x's grouping (group_vars()), eager and lazy", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3, 1), grp = c("a", "a", "b", "b"), v = 1:4)
  right_df <- data.frame(id = c(1, 3))

  oracle_semi <- dplyr::semi_join(dplyr::group_by(left_df, grp), right_df, by = "id")
  oracle_anti <- dplyr::anti_join(dplyr::group_by(left_df, grp), right_df, by = "id")

  gpu_semi <- dplyr::semi_join(dplyr::group_by(tbl_gpu(left_df), grp), tbl_gpu(right_df), by = "id")
  gpu_anti <- dplyr::anti_join(dplyr::group_by(tbl_gpu(left_df), grp), tbl_gpu(right_df), by = "id")

  expect_identical(gpu_semi$groups, dplyr::group_vars(oracle_semi))
  expect_identical(gpu_anti$groups, dplyr::group_vars(oracle_anti))
  expect_equal(as.data.frame(collect(gpu_semi)), as.data.frame(dplyr::ungroup(oracle_semi)))
  expect_equal(as.data.frame(collect(gpu_anti)), as.data.frame(dplyr::ungroup(oracle_anti)))

  lazy_x <- as_lazy(dplyr::group_by(tbl_gpu(left_df), grp))
  gpu_semi_lazy <- dplyr::semi_join(lazy_x, tbl_gpu(right_df, lazy = TRUE), by = "id")
  expect_identical(gpu_semi_lazy$groups, dplyr::group_vars(oracle_semi))
  expect_equal(as.data.frame(collect(gpu_semi_lazy)), as.data.frame(dplyr::ungroup(oracle_semi)))
})

test_that("semi_join()/anti_join() reject bad na_matches values with dplyr's exact text", {
  skip_if_no_gpu()

  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_error(
    dplyr::semi_join(gx, gy, by = "id", na_matches = "bogus"),
    '`na_matches` must be one of "na" or "never", not "bogus".',
    fixed = TRUE
  )
  expect_error(
    dplyr::anti_join(gx, gy, by = "id", na_matches = "bogus"),
    '`na_matches` must be one of "na" or "never", not "bogus".',
    fixed = TRUE
  )
})
