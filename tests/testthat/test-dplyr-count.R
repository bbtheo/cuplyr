# Mirrored dplyr-oracle tests for count()/tally()/add_count()/add_tally()
# semantics (Phase 3, task 5 -- the count/tally family)
#
# Native implementation: sugar over the existing group_by()/summarise()/
# mutate()/arrange() machinery -- no new AST node, no new C++ binding (see
# R/count.R, and grouped_aggregate_mutate()/match_grouped_aggregate_dot() in
# R/mutate.R for add_count()'s per-row broadcast). `add_tally()` is a plain
# (non-generic) dplyr function; it needs no tbl_gpu-specific code at all,
# only R/group-by.R's `tbl_vars.tbl_gpu()` plus the grouped-aggregate
# mutate() support (see the dedicated tests below).
#
# Semantics verified empirically against dplyr 1.2.1 before implementation
# (see R/count.R's own header comment for the full narrative):
#
#   - count(x, ...) restores x's ORIGINAL grouping in full (not a
#     "drop-last" rule) -- verified even when x had more than one grouping
#     column.
#   - tally(x) (no dots) drops only the LAST grouping level, matching
#     summarise()'s own default `.groups` behavior for a real grouped_df.
#   - Name collisions: count()/tally()'s default name ("n", bumped to "nn"
#     while colliding) checks against GROUP columns only; add_count()/
#     add_tally()'s check is against EVERY column -- a real, non-obvious
#     asymmetry (verified: an ordinary data column named "n" does not
#     affect count()'s naming, but does bump add_count()'s).
#   - wt = sum(wt) (NA-excluding, matching dplyr's sum(wt, na.rm = TRUE)).
#   - Along the way, this work found and fixed a real, previously
#     untested bug: cudf's groupby::groupby defaults to
#     null_policy::EXCLUDE, silently dropping every row whose GROUP KEY is
#     NA -- fixed in src/ops_groupby.cpp (null_policy::INCLUDE), see
#     test-summarise.R's "keeps an NA group key" test. Without that fix,
#     add_count()'s join-back broadcast would silently produce NA counts
#     for NA-keyed rows (verified as a real, reproducible mismatch during
#     development).
#
# A previously-discovered bug fixed as a side effect while enabling
# add_count()/add_tally(): mutate()'s "aggregates aren't yet supported"
# restriction now has one narrow, explicit exception (a single dot that's
# exactly `n()` or `sum(<expr>)`), see R/mutate.R's `match_grouped_aggregate_dot()`.
#
# Row order caveat (matches the rest of this codebase's own convention,
# e.g. test-summarise.R): our native groupby's row order is NOT guaranteed
# to match dplyr's (cudf's hash-based groupby vs dplyr's sorted-by-key
# convention) -- oracle comparisons below pass `arrange_by` whenever the
# pipeline's own row order isn't part of the contract being tested (e.g.
# sort = TRUE re-sorts a groupby result, whose PRE-sort order already
# differs, so ties among the `n` column may land in a different relative
# order than dplyr's -- verified empirically, see the dedicated "sort =
# TRUE with ties" test below, which checks VALUES only, not tie order).

count_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    s = c("a", "b", "a", "b", "a"),
    x = c(10, 20, 30, 40, 50)
  )
}

# =============================================================================
# count(): basic shapes
# =============================================================================

test_that("count(df, x) on an ungrouped table matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("count(df, x) on an ungrouped table matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("count(df, x, y) with multiple grouping columns matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g, s)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s"))
})

test_that("count(df, x, y) with multiple grouping columns matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g, s)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("g", "s"))
})

test_that("count(df) with no dots on an ungrouped table matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d)
  expect_same_as_dplyr(df, pipeline)
})

test_that("count(gdf) with no dots on a grouped table matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::count()
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

# =============================================================================
# count(): grouping restoration (the empirically-verified "restore in full"
# rule, not a drop-last rule)
# =============================================================================

test_that("count() on an ungrouped table returns an ungrouped result", {
  skip_if_no_gpu()
  gt <- tbl_gpu(count_df())
  result <- gt |> dplyr::count(g)
  expect_equal(result$groups, character(0))
})

test_that("count() restores the ORIGINAL grouping (single group column) matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, s) |> dplyr::count(g)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("s", "g"))
})

test_that("count() restores the FULL original grouping, not drop-last, matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    a = c(1, 1, 2, 2), b = c(1, 2, 1, 2), g = c(1, 1, 2, 2), x = 1:4
  )
  pipeline <- function(d) dplyr::group_by(d, a, b) |> dplyr::count(g)
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = c("a", "b", "g"))
  # Direct assertion this is genuinely "restore in full", not "drop last":
  # a naive drop-last rule would give groups = "a" here, not c("a", "b").
  gt <- tbl_gpu(df) |> dplyr::group_by(a, b)
  expect_equal(group_vars(gt |> dplyr::count(g)), c("a", "b"))
})

test_that("count() with no dots on a multiply-grouped table restores full grouping", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 1, 2, 2), b = c(1, 2, 1, 2), x = 1:4)
  pipeline <- function(d) dplyr::group_by(d, a, b) |> dplyr::count()
  expect_same_as_dplyr(df, pipeline, arrange_by = c("a", "b"))
})

# =============================================================================
# count(): wt =
# =============================================================================

test_that("count(df, x, wt = w) matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g, wt = x)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("count(df, x, wt = w) matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g, wt = x)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("count(wt = ) excludes NA weights (sum(wt, na.rm = TRUE)) matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), w = c(1, NA, 3))
  pipeline <- function(d) dplyr::count(d, g, wt = w)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

# =============================================================================
# count(): sort = TRUE
# =============================================================================

test_that("count(sort = TRUE) sorts descending by count, values match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c("a", "b", "b", "c", "c", "c"))
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::count(g, sort = TRUE) |> collect()
  oracle <- dplyr::count(df, g, sort = TRUE)

  # Row order itself matches here (no ties -- 3, 2, 1 are all distinct).
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("count(sort = TRUE) with tied counts: values match dplyr, n is non-increasing", {
  skip_if_no_gpu()
  # g = 1 and g = 4 both have count 2 -- dplyr's own tie-break order (which
  # of the two comes first) is an implementation detail of its sorted-by-key
  # groupby, NOT part of its documented contract; our own groupby's
  # PRE-sort order differs (cudf hash-based, not sorted-by-key -- see this
  # file's header comment), so we verify VALUES match (via a full arrange)
  # and that the sort itself is genuinely descending, without asserting on
  # which of the tied rows lands first.
  df <- tibble::tibble(g = c(3, 1, 1, 2, 2, 2, 4, 4))
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::count(g, sort = TRUE) |> collect()
  oracle <- dplyr::count(df, g, sort = TRUE)

  expect_true(all(diff(result$n) <= 0))
  expect_equal(
    dplyr::arrange(tibble::as_tibble(result), g),
    dplyr::arrange(oracle, g)
  )
})

# =============================================================================
# count(): name = / name collisions
# =============================================================================

test_that("count(name = ) uses the custom name matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::count(d, g, name = "total")
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("count() default name bumps to 'nn' only when 'n' collides with a GROUP column", {
  skip_if_no_gpu()
  # `n` is the group column itself here -- collides, bumps to "nn".
  df <- tibble::tibble(n = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) suppressMessages(dplyr::count(d, n))
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = "n")
  expect_identical(names(result$gpu), c("n", "nn"))
})

test_that("count() default name does NOT bump when 'n' is an ordinary (non-group) column", {
  skip_if_no_gpu()
  # `n` here is NOT part of the grouping (`g` is) -- count()'s own
  # check_n_name() only looks at group_vars(), so this does NOT bump,
  # unlike add_count()'s check (see the add_count() section below).
  df <- tibble::tibble(g = c(1, 1, 2), n = c(10, 20, 30))
  pipeline <- function(d) dplyr::count(d, g)
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_identical(names(result$gpu), c("g", "n"))
})

# =============================================================================
# count(): expression dots (mutate-then-count)
# =============================================================================

test_that("count() with a computed expression dot matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 15, 25, 35))
  pipeline <- function(d) dplyr::count(d, grp = x %/% 10)
  expect_same_as_dplyr(df, pipeline, arrange_by = "grp")
})

test_that("count() with a computed expression dot matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 15, 25, 35))
  pipeline <- function(d) dplyr::count(d, grp = x %/% 10)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "grp")
})

test_that("count() with an unnamed computed expression dot names itself like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 15, 25, 35))
  pipeline <- function(d) dplyr::count(d, x %/% 10)
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = "x%/%10")
  expect_identical(names(result$gpu), c("x%/%10", "n"))
})

# =============================================================================
# tally()
# =============================================================================

test_that("tally() on an ungrouped table matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::tally(d)
  expect_same_as_dplyr(df, pipeline)
})

test_that("tally() on a singly-grouped table matches dplyr and drops to ungrouped", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::tally()
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")

  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  expect_equal(group_vars(gt |> dplyr::tally()), character(0))
})

test_that("tally() on a multiply-grouped table drops only the LAST grouping level", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 1, 2, 2), b = c(1, 2, 1, 2), x = 1:4)
  pipeline <- function(d) dplyr::group_by(d, a, b) |> dplyr::tally()
  expect_same_as_dplyr(df, pipeline, arrange_by = c("a", "b"))

  gt <- tbl_gpu(df) |> dplyr::group_by(a, b)
  expect_equal(group_vars(gt |> dplyr::tally()), "a")
})

test_that("tally(wt = ) matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::tally(wt = x)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("tally(sort = TRUE) matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c("a", "b", "b", "c", "c", "c"))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::tally(sort = TRUE)
  result <- expect_same_as_dplyr(df, pipeline)
  expect_true(all(diff(result$gpu$n) <= 0))
})

test_that("tally(name = ) matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::tally(name = "total")
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("tally()'s default name does NOT bump for a non-group column named 'n'", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2), n = c(10, 20, 30))
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::tally()
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_identical(names(result$gpu), c("g", "n"))
})

# =============================================================================
# NA group keys (the bug this work found and fixed, src/ops_groupby.cpp)
# =============================================================================

test_that("count() keeps an NA group key as its own group, matching dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, NA, NA, 2, NA), x = 1:5)

  # arrange_by()/dplyr::arrange() don't sort by a column containing NA
  # reliably in a way both engines are guaranteed to agree on, so this
  # compares the NA-inclusive result directly: both sides must produce the
  # SAME number of groups (incl. one NA group), and the same row set once
  # NA is mapped to a sortable sentinel.
  gt <- tbl_gpu(df)
  result <- gt |> dplyr::count(g) |> collect()
  oracle <- dplyr::count(df, g)

  expect_equal(nrow(result), nrow(oracle))
  sort_key <- function(tbl) tbl[order(ifelse(is.na(tbl$g), Inf, tbl$g)), ]
  expect_equal(
    tibble::as_tibble(sort_key(as.data.frame(result))),
    tibble::as_tibble(sort_key(as.data.frame(oracle)))
  )
})

test_that("tally() keeps an NA group key as its own group, matching dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, NA, NA, 2, NA), x = 1:5)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  result <- gt |> dplyr::tally() |> collect()
  oracle <- dplyr::tally(dplyr::group_by(df, g))

  expect_equal(nrow(result), nrow(oracle))
  sort_key <- function(tbl) tbl[order(ifelse(is.na(tbl$g), Inf, tbl$g)), ]
  expect_equal(
    tibble::as_tibble(sort_key(as.data.frame(result))),
    tibble::as_tibble(sort_key(as.data.frame(oracle)))
  )
})

# =============================================================================
# add_count()
# =============================================================================

test_that("add_count(df, x) keeps all rows/columns and matches dplyr (eager)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::add_count(d, g)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("add_count(df, x) keeps all rows/columns and matches dplyr (lazy)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::add_count(d, g)
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("add_count() preserves row count and original row order (no arrange needed)", {
  skip_if_no_gpu()
  df <- count_df()
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::add_count(g) |> collect()
  oracle <- dplyr::add_count(df, g)

  expect_equal(nrow(result), nrow(df))
  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("add_count() with wt = matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::add_count(d, g, wt = x)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("add_count() restores the ORIGINAL grouping (not the temporary .add'd dots)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, s) |> dplyr::add_count(g)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("s", "g", "x"))

  gt <- tbl_gpu(df) |> dplyr::group_by(s)
  expect_equal(group_vars(gt |> dplyr::add_count(g)), "s")
})

test_that("add_count() name collision bumps to 'nn' for ANY existing column named 'n'", {
  skip_if_no_gpu()
  # Unlike count()'s narrower group_vars()-only check, add_count()'s check
  # (dplyr's add_tally() -> check_n_name(name, tbl_vars(x))) is against
  # EVERY column -- verified empirically to differ from count()'s own rule
  # (see the count() name-collision tests above).
  df <- tibble::tibble(g = c(1, 1, 2), n = c(10, 20, 30))
  pipeline <- function(d) suppressMessages(dplyr::add_count(d, g))
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "n"))
  expect_identical(names(result$gpu), c("g", "n", "nn"))
})

test_that("add_count() with a computed expression dot matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(5, 15, 25, 35))
  pipeline <- function(d) dplyr::add_count(d, grp = x %/% 10)
  expect_same_as_dplyr(df, pipeline, arrange_by = "x")
})

test_that("add_count() on an ungrouped table with no dots broadcasts total row count", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::add_count(d)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("add_count(.drop = ) errors like the deprecated dplyr argument", {
  skip_if_no_gpu()
  gt <- tbl_gpu(count_df())
  expect_error(dplyr::add_count(gt, g, .drop = TRUE), "deprecated")
})

# =============================================================================
# add_count(): NA-key join-back safety (write this test FIRST per the
# design mandate -- verifying the join-back broadcast matches NA keys
# together, exactly like dplyr's own grouped semantics, before trusting the
# native implementation). This is also what caught the null_policy bug in
# src/ops_groupby.cpp (see this file's header comment).
# =============================================================================

test_that("add_count() matches NA keys together (join-back is NA-safe)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, NA, NA, 2, NA), x = 1:5)
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::add_count(g) |> collect()
  oracle <- dplyr::add_count(df, g)

  expect_equal(nrow(result), nrow(df))
  expect_equal(
    dplyr::arrange(tibble::as_tibble(result), x),
    dplyr::arrange(oracle, x)
  )
  # All three NA rows must see n = 3 (matched together, not NA or 1 each).
  na_rows <- is.na(result$g)
  expect_equal(sum(na_rows), 3)
  expect_true(all(result$n[na_rows] == 3))
})

test_that("add_count() with multiple grouping columns incl. NA matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    a = c(1, 1, NA, NA, 2), b = c("x", "x", "y", "y", "x"), v = 1:5
  )
  pipeline <- function(d) dplyr::add_count(d, a, b)
  expect_same_as_dplyr(df, pipeline, arrange_by = "v")
})

# =============================================================================
# add_tally() -- a plain (non-generic) dplyr function, no tbl_gpu-specific
# code at all (see R/count.R's header comment and R/group-by.R's
# tbl_vars.tbl_gpu()). Tests here specifically exercise that transparency
# path (calling the REAL, unmodified dplyr::add_tally()).
# =============================================================================

test_that("dplyr::add_tally() works transparently on a grouped tbl_gpu (eager)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::add_tally()
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("dplyr::add_tally() works transparently on a grouped tbl_gpu (lazy)", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::add_tally()
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("dplyr::add_tally() works transparently on an ungrouped tbl_gpu", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::add_tally(d)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("dplyr::add_tally() preserves the table's existing grouping", {
  skip_if_no_gpu()
  gt <- tbl_gpu(count_df()) |> dplyr::group_by(g)
  result <- gt |> dplyr::add_tally()
  expect_equal(group_vars(result), "g")
})

test_that("dplyr::add_tally() matches NA keys together", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, NA, NA, 2, NA), x = 1:5)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::add_tally() |> collect()
  oracle <- dplyr::add_tally(dplyr::group_by(df, g))

  expect_equal(
    dplyr::arrange(tibble::as_tibble(result), x),
    dplyr::arrange(dplyr::ungroup(oracle), x)
  )
})

test_that("dplyr::add_tally(wt = ) works transparently", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::add_tally(wt = x)
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

# =============================================================================
# tbl_vars.tbl_gpu()
# =============================================================================

test_that("tbl_vars() returns every column name, matching dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  gt <- tbl_gpu(df)

  expect_equal(dplyr::tbl_vars(gt), dplyr::tbl_vars(df))
  expect_equal(as.character(dplyr::tbl_vars(gt)), names(df))
})

test_that("tbl_vars() on a grouped table matches dplyr (grouped attribute included)", {
  skip_if_no_gpu()
  df <- count_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  gdf <- dplyr::group_by(df, g)

  expect_equal(dplyr::tbl_vars(gt), dplyr::tbl_vars(gdf))
})

test_that("tbl_nongroup_vars() works transparently via tbl_vars()/group_vars()", {
  skip_if_no_gpu()
  df <- count_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  gdf <- dplyr::group_by(df, g)

  expect_equal(dplyr::tbl_nongroup_vars(gt), dplyr::tbl_nongroup_vars(gdf))
})

# =============================================================================
# Narrow grouped-aggregate mutate() support (added to enable add_count()/
# add_tally(); see match_grouped_aggregate_dot()/grouped_aggregate_mutate(),
# R/mutate.R)
# =============================================================================

test_that("mutate(gdf, name = n()) broadcasts the per-group count, matching dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::mutate(cnt = dplyr::n())
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("mutate(gdf, name = sum(x)) broadcasts the per-group sum, matching dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::mutate(total = sum(x))
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("mutate(df, name = n()) on an ungrouped table broadcasts the total row count", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::mutate(d, cnt = dplyr::n())
  expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s", "x"))
})

test_that("mutate(gdf, x = sum(x)) replacing an existing column matches dplyr", {
  skip_if_no_gpu()
  df <- count_df()
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::mutate(x = sum(x))
  result <- expect_same_as_dplyr(df, pipeline, arrange_by = c("g", "s"))
  expect_identical(names(result$gpu), names(count_df()))
})

test_that("mutate() combining an aggregate with another op still hard-errors (Phase 5 scope)", {
  skip_if_no_gpu()
  gt <- tbl_gpu(count_df()) |> dplyr::group_by(g)
  expect_error(dplyr::mutate(gt, y = dplyr::n() + 1))
  expect_error(dplyr::mutate(gt, y = mean(x)))
})
