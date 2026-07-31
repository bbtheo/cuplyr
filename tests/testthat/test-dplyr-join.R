# Oracle parity tests for Phase 7 J1: the 4-vector join spec
# (list(left, right, op, filter)), na_matches = "never" threaded through
# left/inner/full/right joins via nulls_equal, and the multiple=/unmatched=/
# relationship= argument plumbing (validation only here; the actual
# cardinality-changing behavior these arguments control is activated in
# Phase 7 J5 and covered by tests/testthat/test-dplyr-join-rows.R).
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

test_that("non-default multiple=/unmatched=/relationship= are honored (Phase 7 J5)", {
  skip_if_no_gpu()

  # `id` is a clean 1:1 key on both sides here (no duplicates, no unmatched
  # rows), so none of these non-default values have anything to filter/
  # complain about -- they should simply succeed (activated, not stubbed).
  gx <- tbl_gpu(data.frame(id = 1:3))
  gy <- tbl_gpu(data.frame(id = 1:3))

  expect_no_error(dplyr::left_join(gx, gy, by = "id", multiple = "first") |> collect())
  expect_no_error(dplyr::left_join(gx, gy, by = "id", unmatched = "error") |> collect())
  expect_no_error(dplyr::inner_join(gx, gy, by = "id", relationship = "one-to-one") |> collect())

  # Defaults still work.
  expect_no_error(dplyr::left_join(gx, gy, by = "id", multiple = "all") |> collect())
  expect_no_error(dplyr::left_join(gx, gy, by = "id", unmatched = "drop") |> collect())
  expect_no_error(dplyr::inner_join(gx, gy, by = "id", relationship = NULL) |> collect())
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
  # Phase 7 J7 finding (verified against dplyr 1.2.1's own
  # `dplyr:::join_cols()` source): `keep = NULL`'s real default resolution
  # is NOT a whole-join TRUE/FALSE decision -- it drops the right key
  # column PER CONDITION, only for genuinely equi ("==") entries, and keeps
  # both sides for a non-equi one. `resolve_join_keep()` itself now always
  # returns `FALSE` for `NULL` (both equi and non-equi specs): the
  # per-condition distinction lives entirely in
  # `build_join_output_info()`'s `keep != TRUE` branch (see its own
  # roxygen, R/join.R), which reads `join_spec$op` directly instead of a
  # pre-collapsed scalar.
  equi_spec <- list(left = "id", right = "id", op = "==", filter = "none")
  cond_spec <- list(left = "id", right = "id", op = ">=", filter = "none")

  expect_identical(cuplyr:::resolve_join_keep(NULL, equi_spec), FALSE)
  expect_identical(cuplyr:::resolve_join_keep(NULL, cond_spec), FALSE)
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

# =============================================================================
# Phase 7 J4: right_join() row-order contract (test-first pin)
#
# KNOWN LIVE DIVERGENCE (scratchpad/phase7_joins_design.md section 2, fixed by
# this task): the OLD right_join.tbl_gpu() implementation ran a swapped
# left_join(y, x) and reordered/renamed columns after the fact -- but never
# reordered ROWS, so the result came back in *y's own row order* (cudf's
# left_join, applied with y in the "left" role, preserves y's row order).
# dplyr's actual right_join() contract is different (verified empirically
# against dplyr 1.2.1, both here and independently via a standalone Rscript
# check against real dplyr, not the GPU code): x-MATCHED rows come first, in
# x's own row order, and unmatched-y rows are appended LAST. This is the same
# shape as full_join()'s left-order-then-right-only-tail contract, just
# restricted to right_join's smaller row set (x-unmatched rows are dropped
# entirely, not kept).
#
# This test deliberately shuffles y relative to x's match order AND places a
# y-only row EARLY in y, so the old (wrong) y-order implementation and the
# correct x-matched-order implementation produce genuinely different row
# orders -- not just genuinely different in theory, but empirically
# different for this exact fixture (checked against real dplyr above). NO
# arrange_by() here: row order is the entire point of this test.
# =============================================================================

test_that("right_join() row order matches dplyr: x-matched order then unmatched-y tail", {
  skip_if_no_gpu()

  # x's own row order is id = 3, 1, 2 (plus a 4th, id = 9, with no y match at
  # all -- dropped entirely by right_join, exercising that unmatched-LEFT
  # rows are correctly excluded, not just correctly ordered).
  left_df <- data.frame(id = c(3, 1, 2, 9), v = c(30, 10, 20, 90))
  # y is shuffled relative to x's match order, with a y-only row (id = 5)
  # placed FIRST -- the old swapped-left-join implementation would put id 5
  # first (y's own row order); dplyr puts it last.
  right_df <- data.frame(id = c(5, 2, 1, 3), w = c(500, 200, 100, 300))

  expected <- dplyr::right_join(left_df, right_df, by = "id")
  # Pin the oracle itself, not just the GPU result, so this test documents
  # dplyr's real contract independent of any GPU bug: x-matched rows (3, 1,
  # 2) in x's order, then the unmatched-y row (5) last.
  expect_equal(expected$id, c(3, 1, 2, 5))

  eager <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  lazy <- dplyr::right_join(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                            by = "id") |>
    collect()

  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

test_that("right_join() duplicate matches: right rows for one left row appear in right's own order", {
  skip_if_no_gpu()

  # id = 1 matches THREE right rows -- dplyr's contract (verified empirically
  # against dplyr 1.2.1) is that they appear in right_df's own row order for
  # that shared left row: 100, 200, 400 (not sorted, not reversed).
  left_df <- data.frame(id = c(1, 2), v = c("a", "b"))
  right_df <- data.frame(id = c(1, 1, 2, 1), w = c(100, 200, 300, 400))

  expected <- dplyr::right_join(left_df, right_df, by = "id")
  expect_equal(expected$w, c(100, 200, 400, 300))

  eager <- dplyr::right_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |>
    collect()
  lazy <- dplyr::right_join(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                            by = "id") |>
    collect()

  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

# =============================================================================
# Factor columns (Phase 11 L2, Bug 1): mutating joins used to drop
# `schema$factor_levels` entirely (build_join_schema() only ever returned
# `names`/`types`) -- collect() would then present a factor column as raw
# INT32 codes instead of a real factor. build_join_output_info() now
# propagates factor_levels keyed by each output column's POST-suffix name.
# =============================================================================

test_that("left_join() preserves factor levels on a non-key factor column", {
  skip_if_no_gpu()

  left_df <- data.frame(id = c(1, 2, 3), v = c(10, 20, 30))
  right_df <- data.frame(id = c(1, 2, 3),
                         grp = factor(c("m", "f", "m"), levels = c("f", "m")))

  expected <- dplyr::left_join(left_df, right_df, by = "id")

  eager <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "id") |> collect()
  lazy <- dplyr::left_join(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE),
                           by = "id") |> collect()

  expect_s3_class(eager$grp, "factor")
  expect_equal(levels(eager$grp), c("f", "m"))
  expect_equal(as.data.frame(eager), as.data.frame(expected))
  expect_s3_class(lazy$grp, "factor")
  expect_equal(as.data.frame(lazy), as.data.frame(expected))
})

test_that("left_join() preserves factor levels on an equi JOIN KEY (same levels both sides)", {
  skip_if_no_gpu()

  lvls <- c("a", "b", "c")
  left_df <- data.frame(grp = factor(c("a", "b", "c"), levels = lvls), v = 1:3)
  right_df <- data.frame(grp = factor(c("b", "c"), levels = lvls), w = c(20, 30))

  expected <- dplyr::left_join(left_df, right_df, by = "grp")

  result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp") |> collect()

  expect_s3_class(result$grp, "factor")
  expect_equal(levels(result$grp), lvls)
  expect_equal(as.character(result$grp), as.character(expected$grp))
  expect_equal(as.data.frame(result), as.data.frame(expected))
})

test_that("inner_join() with keep = TRUE preserves factor levels on BOTH suffixed key copies", {
  skip_if_no_gpu()

  lvls <- c("a", "b", "c")
  left_df <- data.frame(grp = factor(c("a", "b"), levels = lvls), v = 1:2)
  right_df <- data.frame(grp = factor(c("b", "a"), levels = lvls), w = c(20, 30))

  expected <- dplyr::inner_join(left_df, right_df, by = "grp", keep = TRUE)

  result <- dplyr::inner_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp", keep = TRUE) |>
    collect()

  expect_s3_class(result$grp.x, "factor")
  expect_s3_class(result$grp.y, "factor")
  expect_equal(levels(result$grp.x), lvls)
  expect_equal(levels(result$grp.y), lvls)

  ord <- order(result$v)
  expected_ord <- order(expected$v)
  expect_equal(as.character(result$grp.x[ord]), as.character(expected$grp.x[expected_ord]))
  expect_equal(as.character(result$grp.y[ord]), as.character(expected$grp.y[expected_ord]))
})

test_that("cross_join() preserves factor levels on both sides' columns", {
  skip_if_no_gpu()

  left_df <- data.frame(g = factor(c("a", "b")))
  right_df <- data.frame(g = factor(c("x", "y", "z")))

  expected <- dplyr::cross_join(left_df, right_df)
  result <- dplyr::cross_join(tbl_gpu(left_df), tbl_gpu(right_df)) |> collect()

  expect_s3_class(result$g.x, "factor")
  expect_s3_class(result$g.y, "factor")
  expect_equal(levels(result$g.x), levels(left_df$g))
  expect_equal(levels(result$g.y), levels(right_df$g))

  ord <- order(as.character(result$g.x), as.character(result$g.y))
  expected_ord <- order(as.character(expected$g.x), as.character(expected$g.y))
  expect_equal(as.character(result$g.x[ord]), as.character(expected$g.x[expected_ord]))
  expect_equal(as.character(result$g.y[ord]), as.character(expected$g.y[expected_ord]))
})

test_that("semi_join()/anti_join() preserve factor levels (x's own schema verbatim)", {
  skip_if_no_gpu()

  left_df <- data.frame(grp = factor(c("a", "b", "c"), levels = c("a", "b", "c")), v = 1:3)
  right_df <- data.frame(grp = factor(c("b", "c")))

  semi_expected <- dplyr::semi_join(left_df, right_df, by = "grp")
  anti_expected <- dplyr::anti_join(left_df, right_df, by = "grp")

  semi_result <- dplyr::semi_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp") |> collect()
  anti_result <- dplyr::anti_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp") |> collect()

  expect_s3_class(semi_result$grp, "factor")
  expect_equal(levels(semi_result$grp), c("a", "b", "c"))
  expect_equal(as.data.frame(semi_result), as.data.frame(semi_expected))

  expect_s3_class(anti_result$grp, "factor")
  expect_equal(as.data.frame(anti_result), as.data.frame(anti_expected))
})

test_that("join() falls back to CPU when an equi factor KEY has different levels on both sides", {
  skip_if_no_gpu()

  # Native joins compare equi keys by raw INT32 code -- two factor columns
  # with DIFFERENT level vectors would silently mismatch if compared that
  # way (e.g. "a" is code 1 on the left but code 2 on the right). Routing
  # the whole join to the CPU fallback (join_factor_key_levels_conflict())
  # avoids that; the *result* must still match dplyr exactly.
  left_df <- data.frame(grp = factor(c("a", "b", "c")), v = 1:3)
  right_df <- data.frame(grp = factor(c("c", "b", "a"), levels = c("c", "b", "a")), w = c(30, 20, 10))

  expected <- dplyr::left_join(left_df, right_df, by = "grp")

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::left_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp") |> collect(),
    "fell back to CPU evaluation"
  )

  ord <- order(as.character(result$grp))
  expected_ord <- order(as.character(expected$grp))
  expect_equal(as.character(result$grp[ord]), as.character(expected$grp[expected_ord]))
  expect_equal(result$w[ord], expected$w[expected_ord])
})

test_that("semi_join() falls back to CPU when an equi factor KEY has different levels on both sides", {
  skip_if_no_gpu()

  left_df <- data.frame(grp = factor(c("a", "b", "c")), v = 1:3)
  right_df <- data.frame(grp = factor(c("b", "c"), levels = c("c", "b", "a")))

  expected <- dplyr::semi_join(left_df, right_df, by = "grp")

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::semi_join(tbl_gpu(left_df), tbl_gpu(right_df), by = "grp") |> collect(),
    "fell back to CPU evaluation"
  )

  expect_equal(sort(as.character(result$grp)), sort(as.character(expected$grp)))
})

# =============================================================================
# Phase 12 optimizer audit: lazy joins whose left side is itself a pending
# barrier-type op (rename()/distinct()/slice_min()) used to have their
# required columns corrupted by push_down_projections()/prune_dead_columns()
# when a trailing select() narrowed the output -- these three passes'
# "join" case recurses directly into ast$left/ast$right, bypassing the
# normal optimizer-barrier machinery entirely (see R/optimizer.R's audit
# comments and tests/testthat/test-optimizer.R's AST-level unit tests for
# the mechanism). These are full GPU round-trip pins for the same bugs.
# =============================================================================

test_that("lazy join whose left side has a pending rename() survives projection pushdown", {
  skip_if_no_gpu()

  left_df <- data.frame(x = c(1, 2, 3, 4, 5), y = c(10, 11, 12, 13, 14))
  right_df <- data.frame(z = c(2, 4), w = c(200, 400))

  pipeline <- function(x, y) {
    x |>
      dplyr::rename(new_x = x) |>
      dplyr::inner_join(y, by = c("new_x" = "z")) |>
      dplyr::select(new_x, y, w)
  }

  expected <- pipeline(left_df, right_df)

  # A trailing select() narrows required_cols, forcing push_down_projections()
  # to actually recurse into the join's rename()-holding left side.
  result <- pipeline(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE)) |> collect()

  expect_equal(
    as.data.frame(result)[order(result$new_x), ],
    as.data.frame(expected)[order(expected$new_x), ],
    ignore_attr = TRUE
  )
})

test_that("lazy join whose left side has a pending distinct(.keep_all=TRUE) survives projection pushdown", {
  skip_if_no_gpu()

  left_df <- data.frame(k = c(1, 1, 2, 2, 3), extra = c(1, 2, 3, 4, 5))
  right_df <- data.frame(k2 = c(1, 2, 3))

  pipeline <- function(x, y) {
    x |>
      dplyr::distinct(k, extra, .keep_all = TRUE) |>
      dplyr::inner_join(y, by = c("k" = "k2")) |>
      dplyr::select(k)
  }

  expected <- pipeline(left_df, right_df)
  result <- pipeline(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE)) |> collect()

  expect_equal(sort(result$k), sort(expected$k))
})

test_that("lazy join whose left side has a pending slice_min() survives projection pushdown", {
  skip_if_no_gpu()

  left_df <- data.frame(k = c(1, 2, 3), rank_by = c(30, 10, 20))
  right_df <- data.frame(k2 = c(1, 2, 3))

  pipeline <- function(x, y) {
    x |>
      dplyr::slice_min(rank_by, n = 1) |>
      dplyr::inner_join(y, by = c("k" = "k2")) |>
      dplyr::select(k)
  }

  expected <- pipeline(left_df, right_df)
  result <- pipeline(tbl_gpu(left_df, lazy = TRUE), tbl_gpu(right_df, lazy = TRUE)) |> collect()

  expect_equal(sort(result$k), sort(expected$k))
})
