# Oracle parity tests for Phase 10: across()/if_any()/if_all() and the
# tidy-eval context functions (cur_column()/cur_group_id()/cur_group()/
# cur_group_rows()), plus fallback pins for the deliberately-deferred
# shapes (pick()-as-data-frame, cur_group()/cur_group_rows()/cur_data()/
# cur_data_all()/c_across(), nested across()/if_any()/if_all(),
# `across(.unpack = TRUE)`).
#
# Mirrors the core semantics of dplyr's own installed test-across.R/
# test-context.R (~/R/*/dplyr/tests/testthat/, read directly) adapted to
# this package's tbl_gpu/oracle conventions -- see R/across.R for the
# implementation notes and scratchpad/workflow_state.md's Phase 10 entry
# for the full empirical-verification ledger this file's expectations are
# drawn from.

# =============================================================================
# across(): single fn (bare / formula / lambda), named/unnamed list,
# .names glue, everything()-excludes-groups, zero-match no-op
# =============================================================================

test_that("across() with a single bare function overwrites in place ({.col} default)", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), mean))

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("across() with a formula (~ .x)", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), ~ .x * 2))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with a `\\(x)` lambda", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), \(v) v + 1))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with a `function(x)` lambda", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), function(v) v * 10))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with a named list of functions: {.col}_{.name}", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y), list(avg = mean, sdv = sd)))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with an unnamed list of functions: {.col}_{position}", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), list(mean, sd)))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() `.names=` glue template", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y), mean, .names = "mean_{.col}"))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() `.names=` with both {.col} and {.fn}", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y), list(avg = mean), .names = "{.fn}_{.col}"))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with `.fns` omitted entirely is identity", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y)))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with the deprecated `...`-extra-args-to-a-bare-fn form", {
  skip_if_no_gpu()

  # `round(x, digits)` is a GPU-native two-arg mutate() shape (unlike
  # `mean(x, na.rm=)`, which mutate()'s IR doesn't accept at all -- that's
  # an existing IR limitation unrelated to across() itself, see
  # R/mutate.R's own "Supported expressions" docs), so this exercises the
  # deprecated `...`-forwarding mechanism without tripping over it.
  df <- tibble::tibble(x = c(1.23, 4.56, 7.89))
  pipeline <- function(d) suppressWarnings(dplyr::mutate(d, dplyr::across(x, round, 1)))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across(everything()) excludes grouping columns (grouped mutate)", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::mutate(dplyr::across(dplyr::everything(), mean))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("across(everything()) excludes grouping columns (grouped summarise)", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::summarise(dplyr::across(dplyr::everything(), mean))
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("across(everything()) includes every column when ungrouped", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(dplyr::everything(), ~ .x + 100))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() explicitly referencing a group column is a real tidyselect error", {
  skip_if_no_gpu()

  gt <- tbl_gpu(tibble::tibble(g = 1, x = 2)) |> dplyr::group_by(g)
  expect_error(dplyr::mutate(gt, dplyr::across(c(g, x), mean)))
})

test_that("across() with `where(is.numeric)`", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30), s = c("a", "b", "c"))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(dplyr::where(is.numeric), ~ .x + 1))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() matching zero columns is a silent no-op", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:3)
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(dplyr::starts_with("zzz"), mean))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() with cur_column() inside its function", {
  skip_if_no_gpu()

  # `paste()` isn't a mutate()-IR-supported shape at all (string
  # concatenation is out of scope, see R/mutate.R's own docs) -- this
  # exercises cur_column()'s own textual substitution (a whole-body
  # literal-string replacement) with a shape the IR DOES support (a plain
  # string-literal column), rather than conflating an IR limitation with
  # an across()/cur_column() one.
  df <- tibble::tibble(x = c(1, 2), y = c(10, 20))
  pipeline <- function(d) dplyr::mutate(d, dplyr::across(c(x, y), ~ dplyr::cur_column()))

  expect_same_as_dplyr(df, pipeline)
})

test_that("across() works inside summarise() with a named-list of fns", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::group_by(d, g) |>
      dplyr::summarise(dplyr::across(c(x, y), list(avg = mean, mx = max)))
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("across() a later dot in the same mutate() can reference an across()-generated column", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y), ~ .x + 1), total = x + y)
  }

  expect_same_as_dplyr(df, pipeline)
})

# =============================================================================
# across() disallowed in filter() (real dplyr 1.2.1 also rejects this,
# pointing at if_any()/if_all() instead -- verified empirically)
# =============================================================================

test_that("across() is rejected inside filter() with a message pointing at if_any()/if_all()", {
  skip_if_no_gpu()

  gt <- tbl_gpu(tibble::tibble(x = c(1, 2), y = c(3, 4)))
  expect_error(
    dplyr::filter(gt, dplyr::across(c(x, y), ~ .x > 1)),
    "if_any"
  )
})

# =============================================================================
# if_any() / if_all(): filter(), mutate(), NA propagation, group exclusion
# =============================================================================

test_that("if_any() in filter()", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 5, 10), y = c(2, 6, 20))
  pipeline <- function(d) dplyr::filter(d, dplyr::if_any(c(x, y), ~ .x > 5))

  expect_same_as_dplyr(df, pipeline)
})

test_that("if_all() in filter()", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, 5, 10), y = c(2, 6, 20))
  pipeline <- function(d) dplyr::filter(d, dplyr::if_all(c(x, y), ~ .x > 1))

  expect_same_as_dplyr(df, pipeline)
})

test_that("if_any()/if_all() in mutate() produce a logical column with NA propagation", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = c(1, NA, 3), y = c(NA, 5, 6))
  pipeline_any <- function(d) dplyr::mutate(d, flag = dplyr::if_any(c(x, y), ~ .x > 2))
  pipeline_all <- function(d) dplyr::mutate(d, flag = dplyr::if_all(c(x, y), ~ .x > 2))

  expect_same_as_dplyr(df, pipeline_any)
  expect_same_as_dplyr(df, pipeline_all)
})

test_that("if_any()/if_all() exclude grouping columns from `.cols`", {
  skip_if_no_gpu()

  gt <- tbl_gpu(tibble::tibble(g = 1, x = 2, y = 3)) |> dplyr::group_by(g)
  expect_error(dplyr::mutate(gt, z = dplyr::if_any(c(g, x), ~ .x > 1)))
})

test_that("if_any()/if_all() with everything() (grouped mutate)", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = c(1, 2, 3), y = c(4, 5, 6))
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::mutate(z = dplyr::if_any(dplyr::everything(), ~ .x > 1))
  }

  expect_same_as_dplyr(df, pipeline)
})

# =============================================================================
# cur_column(): only meaningful inside across() -- covered above; a
# standalone cur_column() call (outside across()) is a real dplyr error
# too (no per-dot "current column" context exists), so cuplyr's own
# "unrecognized expression" error for it is a legitimate (non-fallback)
# divergence in MESSAGE text only, not behavior -- not pinned here.
# =============================================================================

# =============================================================================
# cur_group_id(): GPU-native (mutate()/filter()), deferred (summarise())
# =============================================================================

test_that("cur_group_id() in mutate() on an ungrouped table is always 1", {
  skip_if_no_gpu()

  df <- tibble::tibble(x = 1:3)
  pipeline <- function(d) dplyr::mutate(d, gid = dplyr::cur_group_id())

  expect_same_as_dplyr(df, pipeline)
})

test_that("cur_group_id() in mutate() on a group_by()-grouped table", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2, 2, 3), x = 1:5)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::mutate(gid = dplyr::cur_group_id())

  expect_same_as_dplyr(df, pipeline)
})

test_that("cur_group_id() in mutate() with multi-key grouping", {
  skip_if_no_gpu()

  df <- tibble::tibble(g1 = c(1, 1, 2, 2), g2 = c("a", "b", "a", "b"), x = 1:4)
  pipeline <- function(d) {
    dplyr::group_by(d, g1, g2) |> dplyr::mutate(gid = dplyr::cur_group_id())
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("cur_group_id() in mutate() with `.by=`", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) dplyr::mutate(d, gid = dplyr::cur_group_id(), .by = g)

  expect_same_as_dplyr(df, pipeline)
})

test_that("cur_group_id() nested inside a comparison in filter()", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::filter(dplyr::cur_group_id() == 1)

  expect_same_as_dplyr(df, pipeline)
})

test_that("cur_group_id() in mutate() never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)) |> dplyr::group_by(g)
  expect_no_error(dplyr::mutate(gt, gid = dplyr::cur_group_id()))
})

test_that("cur_group_id() in filter() never triggers the CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)) |> dplyr::group_by(g)
  expect_no_error(dplyr::filter(gt, dplyr::cur_group_id() == 1))
})

test_that("cur_group_id() in summarise() falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  gt <- tbl_gpu(tibble::tibble(g = c(1, 1, 2), x = 1:3)) |> dplyr::group_by(g)
  expect_warning(dplyr::summarise(gt, gid = dplyr::cur_group_id()), "fell back to CPU evaluation")
})

test_that("cur_group_id()'s hidden helper column never leaks into the mutate() output", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  result <- dplyr::mutate(gt, gid = dplyr::cur_group_id()) |> collect()
  expect_false(any(grepl("cur_group_id", names(result), fixed = TRUE)))
})

test_that("cur_group_id()'s hidden helper column never leaks into the filter() output", {
  skip_if_no_gpu()

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  gt <- tbl_gpu(df) |> dplyr::group_by(g)
  result <- dplyr::filter(gt, dplyr::cur_group_id() == 1) |> collect()
  expect_equal(names(result), names(df))
})

# =============================================================================
# Deferred shapes: whole-call CPU fallback, with correct results and the
# expected notification (pick()-as-data-frame, cur_group()/
# cur_group_rows()/cur_data()/cur_data_all(), nested across()/if_any()/
# if_all(), across(.unpack = TRUE))
# =============================================================================

test_that("pick()-as-data-frame (rowSums(pick(...))) falls back with correct results", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, total = rowSums(dplyr::pick(x, y)))

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(result, pipeline(df))
})

test_that("cur_group() falls back with correct results", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::mutate(k = dplyr::cur_group()$g)
  }

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(as.data.frame(result), as.data.frame(dplyr::ungroup(pipeline(df))))
})

test_that("cur_group_rows() falls back with correct results", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) {
    dplyr::group_by(d, g) |> dplyr::mutate(k = length(dplyr::cur_group_rows()))
  }

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(as.data.frame(result), as.data.frame(dplyr::ungroup(pipeline(df))))
})

test_that("across() nested inside another call (not the whole dot) falls back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2, 3), y = c(10, 20, 30))
  pipeline <- function(d) dplyr::mutate(d, total = rowSums(dplyr::across(c(x, y))))

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(result, pipeline(df))
})

test_that("across(.unpack = TRUE) falls back rather than erroring or silently ignoring it", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2), y = c(3, 4))
  pipeline <- function(d) {
    dplyr::mutate(d, dplyr::across(c(x, y), ~ list(lo = min(.x), hi = max(.x)), .unpack = TRUE))
  }

  expect_warning(
    result <- tbl_gpu(df) |> pipeline() |> collect(),
    "fell back to CPU evaluation"
  )
  expect_equal(result, pipeline(df))
})

# =============================================================================
# n(): already GPU-native in summarise(); pinned here for mutate()/filter()
# too (Phase 10 brief: "extend to mutate/filter", already implemented via
# the window registry -- this is a coverage pin, not new behavior)
# =============================================================================

test_that("n() works (and never falls back) inside mutate()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::mutate(cnt = dplyr::n())

  expect_no_error(tbl_gpu(df) |> pipeline() |> collect())
  expect_same_as_dplyr(df, pipeline)
})

test_that("n() works (and never falls back) inside filter()", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")

  df <- tibble::tibble(g = c(1, 1, 2), x = 1:3)
  pipeline <- function(d) dplyr::group_by(d, g) |> dplyr::filter(dplyr::row_number() <= dplyr::n())

  expect_no_error(tbl_gpu(df) |> pipeline() |> collect())
  expect_same_as_dplyr(df, pipeline)
})

# =============================================================================
# c_across(): rowwise()'s own companion -- rowwise() is already a full
# CPU-fallback verb (R/fallback.R, Phase 2); pin that c_across() composes
# with it correctly rather than crashing.
# =============================================================================

test_that("rowwise() falls back to a real rowwise_df (not re-uploaded to GPU)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # A `rowwise_df` has no `tbl_gpu` representation at all (Phase 9); Phase
  # 10 found (and fixed, R/fallback.R's gpu_fallback()) that it used to be
  # silently re-uploaded to a plain (non-rowwise) tbl_gpu on the way out,
  # discarding its per-row grouping -- which meant `c_across()` (or ANY
  # other rowwise-dependent verb) chained immediately afterward silently
  # computed the wrong thing instead of erroring or working. Pinned here:
  # the result must be a genuine `rowwise_df`, so a chained `mutate()`
  # dispatches to REAL dplyr, not `mutate.tbl_gpu()`.
  gt <- tbl_gpu(tibble::tibble(x = c(1, 2), y = c(3, 4)))
  expect_warning(rt <- dplyr::rowwise(gt), "fell back to CPU evaluation")
  expect_true(inherits(rt, "rowwise_df"))
  expect_false(is_tbl_gpu(rt))
})

test_that("c_across() composes with the existing rowwise() CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  df <- tibble::tibble(x = c(1, 2), y = c(3, 4))
  gt <- tbl_gpu(df)

  expect_warning(rt <- dplyr::rowwise(gt), "fell back to CPU evaluation")
  result <- rt |> dplyr::mutate(total = sum(dplyr::c_across(c(x, y))))

  expect_equal(as.data.frame(dplyr::ungroup(result)), as.data.frame(dplyr::ungroup(
    dplyr::rowwise(df) |> dplyr::mutate(total = sum(dplyr::c_across(c(x, y))))
  )))
})
