# Tests for the Phase 2 transparent CPU fallback layer (R/fallback.R).
#
# `gpu_fallback()` is the shared primitive: materialize pending lazy ops,
# collect() to a tibble (restoring group_by() structure), run the real dplyr
# verb, and re-upload via tbl_gpu() (restoring groups/exec_mode/factor
# levels). This file:
#
#   1. Sweeps a representative pipeline for every registered fallback verb
#      that returns a transformed data frame, oracle-comparing eager tbl_gpu
#      against dplyr-on-tibble via expect_same_as_dplyr() (helper-oracle.R).
#   2. Dedicated tests for the verbs that don't fit that oracle shape (they
#      return metadata, a list, or a list-column data frame).
#   3. Dedicated tests for the cross-cutting behaviors: grouping restoration,
#      lazy materialization/exec_mode, non-data-frame returns, factor
#      round-trip, the cuplyr.fallback option, and the "no fallback for
#      GPU-native verbs" guarantee.

fallback_df <- function() {
  tibble::tibble(
    g = c(1, 1, 2, 2, 3),
    x = c(10, 20, 30, 40, 50),
    s = c("a", "b", "a", "b", "a")
  )
}

# =============================================================================
# 1. Sweep: one representative, deterministic, df-in/df-out pipeline per
#    verb whose result an oracle_compare() (expect_same_as_dplyr()) can
#    directly check.
# =============================================================================

fallback_sweep_pipelines <- function() {
  y_small <- tibble::tibble(g = c(1, 3))

  list(
    # transmute()/glimpse() are GPU-native now (Phase 3 task 4, see
    # R/mutate.R, R/glimpse.R, test-dplyr-transmute.R) -- no longer part of
    # this fallback sweep. reframe() takes over as this file's vehicle for
    # the cross-cutting fallback behaviors below (materialization/exec_mode,
    # factor round-trip, the cuplyr.fallback option).
    reframe = list(fn = function(d) dplyr::reframe(d, mx = max(x), .by = g), arrange_by = "g"),
    rowwise = list(fn = function(d) dplyr::rowwise(d)),
    # slice()/slice_head()/slice_tail() are GPU-native for ungrouped input
    # (Phase 3 task 2, see R/slice.R and test-dplyr-slice.R for that
    # coverage) -- grouped here so this sweep still exercises their
    # fallback path.
    slice = list(fn = function(d) dplyr::group_by(d, g) |> dplyr::slice(1), arrange_by = "g"),
    slice_head = list(fn = function(d) dplyr::group_by(d, g) |> dplyr::slice_head(n = 1), arrange_by = "g"),
    slice_tail = list(fn = function(d) dplyr::group_by(d, g) |> dplyr::slice_tail(n = 1), arrange_by = "g"),
    # slice_min()/slice_max() are GPU-native for a plain, ungrouped
    # order_by, but `by = g` forces the fallback even on ungrouped input.
    slice_min = list(fn = function(d) dplyr::slice_min(d, x, n = 1, by = g), arrange_by = "g"),
    slice_max = list(fn = function(d) dplyr::slice_max(d, x, n = 1, by = g), arrange_by = "g"),
    slice_sample = list(
      fn = function(d) {
        set.seed(42)
        dplyr::slice_sample(d, n = 3)
      },
      arrange_by = c("g", "x")
    ),
    # rename()/rename_with()/relocate()/pull() are GPU-native now (Phase 3
    # task 3, see R/rename.R, R/relocate.R, R/pull.R, test-dplyr-rename.R,
    # test-dplyr-relocate.R, test-dplyr-pull.R) -- no longer part of this
    # fallback sweep. See the "never triggers a fallback notification"
    # block below instead.
    # count()/tally()/add_count() are GPU-native now too (Phase 3 task 5,
    # see R/count.R, test-dplyr-count.R) -- no longer part of this fallback
    # sweep either. See the "never triggers a fallback notification" block
    # below instead.
    semi_join = list(fn = function(d) dplyr::semi_join(d, y_small, by = "g")),
    anti_join = list(fn = function(d) dplyr::anti_join(d, y_small, by = "g")),
    cross_join = list(
      fn = function(d) {
        dplyr::cross_join(dplyr::distinct(dplyr::select(d, g)), tibble::tibble(k = 1:2))
      },
      arrange_by = c("g", "k")
    ),
    union = list(
      fn = function(d) dplyr::union(dplyr::select(d, g), tibble::tibble(g = 9)),
      arrange_by = "g"
    ),
    union_all = list(
      fn = function(d) dplyr::union_all(dplyr::select(d, g), tibble::tibble(g = 9)),
      arrange_by = "g"
    ),
    intersect = list(
      fn = function(d) dplyr::intersect(dplyr::select(d, g), tibble::tibble(g = c(1, 9))),
      arrange_by = "g"
    ),
    setdiff = list(
      fn = function(d) dplyr::setdiff(dplyr::select(d, g), tibble::tibble(g = 1)),
      arrange_by = "g"
    ),
    symdiff = list(
      fn = function(d) dplyr::symdiff(dplyr::select(d, g), tibble::tibble(g = c(1, 9))),
      arrange_by = "g"
    ),
    group_modify = list(
      fn = function(d) dplyr::group_modify(dplyr::group_by(d, g), ~ dplyr::summarise(.x, s = sum(x))),
      arrange_by = "g"
    ),
    group_trim = list(
      fn = function(d) dplyr::group_trim(dplyr::filter(dplyr::group_by(d, g), g != 2))
    ),
    rows_update = list(
      fn = function(d) dplyr::rows_update(d, tibble::tibble(g = 1, s = "a", x = 999), by = c("g", "s"))
    ),
    rows_patch = list(
      fn = function(d) dplyr::rows_patch(d, tibble::tibble(g = 1, s = "a", x = 999), by = c("g", "s"))
    )
  )
}

fallback_pipelines <- fallback_sweep_pipelines()
for (.nm in names(fallback_pipelines)) {
  local({
    nm <- .nm
    spec <- fallback_pipelines[[nm]]
    test_that(sprintf("%s() fallback matches dplyr oracle", nm), {
      skip_if_no_gpu()
      expect_same_as_dplyr(fallback_df(), spec$fn, arrange_by = spec$arrange_by)
    })
  })
}

# =============================================================================
# 2. Dedicated tests for verbs that don't return a plain transformed data
#    frame (metadata queries, list results, list-column results, superseded
#    sampling verbs) -- one per remaining registered fallback verb.
# =============================================================================

test_that("nest_join() returns a plain data frame with a list-column, not a tbl_gpu", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::select(g) |> dplyr::distinct()
  y <- tibble::tibble(g = c(1, 2), extra = c("A", "B"))

  result <- gt |> dplyr::nest_join(y, by = "g")
  oracle <- dplyr::nest_join(dplyr::distinct(dplyr::select(df, g)), y, by = "g")

  expect_false(is_tbl_gpu(result))
  expect_true(any(vapply(result, is.list, logical(1))))
  expect_equal(nrow(result), nrow(oracle))
})

test_that("setequal() returns a logical, matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::select(g)

  expect_true(gt |> dplyr::setequal(tibble::tibble(g = c(1, 1, 2, 2, 3))))
  expect_false(gt |> dplyr::setequal(tibble::tibble(g = c(1, 2))))
})

test_that("rows_insert() adds new rows matching dplyr", {
  skip_if_no_gpu()
  base_df <- tibble::tibble(id = 1:3, v = c("a", "b", "c"))
  gt <- tbl_gpu(base_df)

  result <- gt |> dplyr::rows_insert(tibble::tibble(id = 4, v = "d"), by = "id") |> collect()
  oracle <- dplyr::rows_insert(base_df, tibble::tibble(id = 4, v = "d"), by = "id")

  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("rows_append() adds rows matching dplyr", {
  skip_if_no_gpu()
  base_df <- tibble::tibble(id = 1:3, v = c("a", "b", "c"))
  gt <- tbl_gpu(base_df)

  result <- gt |> dplyr::rows_append(tibble::tibble(id = 4, v = "d")) |> collect()
  oracle <- dplyr::rows_append(base_df, tibble::tibble(id = 4, v = "d"))

  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("rows_upsert() updates and inserts rows matching dplyr", {
  skip_if_no_gpu()
  base_df <- tibble::tibble(id = 1:3, v = c("a", "b", "c"))
  gt <- tbl_gpu(base_df)
  y <- tibble::tibble(id = c(1, 4), v = c("Z", "d"))

  result <- gt |> dplyr::rows_upsert(y, by = "id") |> collect()
  oracle <- dplyr::rows_upsert(base_df, y, by = "id")

  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("rows_delete() removes rows matching dplyr", {
  skip_if_no_gpu()
  base_df <- tibble::tibble(id = 1:3, v = c("a", "b", "c"))
  gt <- tbl_gpu(base_df)

  result <- gt |> dplyr::rows_delete(tibble::tibble(id = 1), by = "id") |> collect()
  oracle <- dplyr::rows_delete(base_df, tibble::tibble(id = 1), by = "id")

  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("rows_*(in_place = TRUE) is rejected for tbl_gpu", {
  skip_if_no_gpu()
  gt <- tbl_gpu(tibble::tibble(id = 1:3, v = c("a", "b", "c")))

  expect_error(
    gt |> dplyr::rows_update(tibble::tibble(id = 1, v = "Z"), by = "id", in_place = TRUE),
    "in_place"
  )
})

test_that("group_data() returns grouping metadata matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_data()
  oracle <- dplyr::group_data(dplyr::group_by(df, g))

  expect_false(is_tbl_gpu(result))
  expect_equal(tibble::as_tibble(result)$g, tibble::as_tibble(oracle)$g)
})

test_that("group_keys() returns key metadata matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_keys()
  oracle <- dplyr::group_keys(dplyr::group_by(df, g))

  expect_false(is_tbl_gpu(result))
  expect_equal(result, oracle)
})

test_that("group_indices() returns per-row group index matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_indices()
  oracle <- dplyr::group_indices(dplyr::group_by(df, g))

  expect_equal(result, oracle)
})

test_that("group_size() returns per-group sizes matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  expect_equal(dplyr::group_size(gt), dplyr::group_size(dplyr::group_by(df, g)))
})

test_that("n_groups() returns the group count matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  expect_equal(dplyr::n_groups(gt), dplyr::n_groups(dplyr::group_by(df, g)))
})

test_that("group_split() returns a plain list of tibbles matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_split()
  oracle <- dplyr::group_split(dplyr::group_by(df, g))

  expect_type(result, "list")
  expect_false(is_tbl_gpu(result))
  expect_equal(length(result), length(oracle))
  for (i in seq_along(result)) {
    expect_equal(
      tibble::as_tibble(dplyr::ungroup(result[[i]])),
      tibble::as_tibble(dplyr::ungroup(oracle[[i]]))
    )
  }
})

test_that("group_map() returns a plain list matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_map(~ sum(.x$x))
  oracle <- dplyr::group_map(dplyr::group_by(df, g), ~ sum(.x$x))

  expect_type(result, "list")
  expect_false(is_tbl_gpu(result))
  expect_equal(result, oracle)
})

test_that("group_nest() returns a data frame with a list-column, not a tbl_gpu", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df) |> dplyr::group_by(g)

  result <- gt |> dplyr::group_nest()

  expect_false(is_tbl_gpu(result))
  expect_true(any(vapply(result, is.list, logical(1))))
  expect_equal(nrow(result), dplyr::n_groups(dplyr::group_by(df, g)))
})

test_that("nest_by() returns a rowwise data frame with a list-column, not a tbl_gpu", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df)

  result <- gt |> dplyr::nest_by(g)

  expect_false(is_tbl_gpu(result))
  expect_true(any(vapply(result, is.list, logical(1))))
  expect_equal(nrow(result), length(unique(df$g)))
})

test_that("sample_n() draws a seeded sample matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df)

  set.seed(123)
  oracle <- suppressWarnings(dplyr::sample_n(df, 3))
  set.seed(123)
  result <- suppressWarnings(gt |> dplyr::sample_n(3) |> collect())

  expect_equal(tibble::as_tibble(result), oracle)
})

test_that("sample_frac() draws a seeded sample matching dplyr", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df)

  set.seed(123)
  oracle <- suppressWarnings(dplyr::sample_frac(df, 0.6))
  set.seed(123)
  result <- suppressWarnings(gt |> dplyr::sample_frac(0.6) |> collect())

  expect_equal(tibble::as_tibble(result), oracle)
})

# =============================================================================
# 3. Cross-cutting behaviors
# =============================================================================

test_that("gpu_fallback() restores pre-existing group_by() grouping for count()", {
  skip_if_no_gpu()
  df <- fallback_df()
  pipeline <- function(d) d |> dplyr::group_by(s) |> dplyr::count()
  expect_same_as_dplyr(df, pipeline, arrange_by = "s")
})

test_that("gpu_fallback() restores pre-existing group_by() grouping for slice_min()", {
  skip_if_no_gpu()
  df <- fallback_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::slice_min(x, n = 1)
  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("fallback verbs materialize pending lazy ops and restore lazy exec_mode", {
  skip_if_no_gpu()
  df <- fallback_df()

  lazy_gt <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 10)
  expect_true(is_lazy(lazy_gt))
  expect_true(has_pending_ops(lazy_gt))

  # transmute() is GPU-native now (Phase 3 task 4, R/mutate.R) -- reframe()
  # takes over as this test's fallback vehicle.
  result <- lazy_gt |> dplyr::reframe(z = max(x), .by = g)

  expect_true(is_tbl_gpu(result))
  expect_true(is_lazy(result))
  expect_false(has_pending_ops(result))

  oracle <- dplyr::reframe(dplyr::filter(df, x > 10), z = max(x), .by = g)
  expect_equal(
    dplyr::arrange(collect(result), z),
    dplyr::arrange(oracle, z)
  )
})

test_that("eager fallback input stays eager after re-upload", {
  skip_if_no_gpu()
  df <- fallback_df()
  gt <- tbl_gpu(df, lazy = FALSE)

  result <- gt |> dplyr::reframe(z = max(x), .by = g)

  expect_true(is_tbl_gpu(result))
  expect_false(is_lazy(result))
})

test_that("factor columns round-trip through a fallback verb", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(1, 1, 2),
    f = factor(c("lo", "hi", "lo"), levels = c("lo", "hi"))
  )
  gt <- tbl_gpu(df)

  # rename() is GPU-native now (Phase 3 task 3, R/rename.R) -- its own
  # factor round-trip is covered directly in test-dplyr-rename.R.
  # transmute() is GPU-native now too (Phase 3 task 4, R/mutate.R). Use
  # reframe() here instead, which is still a genuine CPU fallback verb, to
  # keep this test actually exercising gpu_fallback()'s factor restoration
  # path.
  result <- gt |> dplyr::reframe(f, grp = g) |> collect()

  expect_true(is.factor(result$f))
  expect_equal(levels(result$f), c("lo", "hi"))
  expect_equal(as.character(result$f), as.character(df$f))
})

test_that("options(cuplyr.fallback = 'warn') warns for a fallback verb", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(fallback_df())

  expect_warning(
    gt |> dplyr::reframe(z = max(x), .by = g),
    "reframe.*fell back to CPU evaluation"
  )
})

test_that("options(cuplyr.fallback = 'error') stops for a fallback verb", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(fallback_df())

  expect_error(
    gt |> dplyr::reframe(z = max(x), .by = g),
    "reframe.*fell back to CPU evaluation"
  )
})

test_that("fallback verbs are silent by default", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL)
  gt <- tbl_gpu(fallback_df())

  expect_no_warning(gt |> dplyr::reframe(z = max(x), .by = g))
})

test_that("GPU-native verbs never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")
  gt <- tbl_gpu(fallback_df())

  expect_no_warning({
    gt |>
      dplyr::filter(x > 10) |>
      dplyr::mutate(y = x * 2) |>
      dplyr::select(g, y) |>
      dplyr::arrange(g) |>
      dplyr::group_by(g) |>
      dplyr::summarise(total = sum(y), .groups = "drop") |>
      dplyr::ungroup() |>
      collect()
  })
})

test_that("distinct() is GPU-native and never triggers a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(fallback_df())

  expect_no_error(gt |> dplyr::distinct())
  expect_no_error(gt |> dplyr::distinct(g))
  expect_no_error(gt |> dplyr::distinct(g, .keep_all = TRUE))
  expect_no_error(gt |> dplyr::distinct(z = x + 1))
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::distinct(x))
})

test_that("rename()/rename_with()/relocate() are GPU-native and never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(fallback_df())

  expect_no_error(gt |> dplyr::rename(y2 = x))
  expect_no_error(gt |> dplyr::rename_with(toupper))
  expect_no_error(gt |> dplyr::relocate(s))
  expect_no_error(gt |> dplyr::select(renamed = x, g))
  expect_no_error(gt |> dplyr::pull(x))
})

test_that("transmute()/glimpse() are GPU-native and never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(fallback_df())

  expect_no_error(gt |> dplyr::transmute(z = x * 2))
  expect_no_error(gt |> dplyr::transmute(z = x * 2, g))
  expect_no_error(gt |> dplyr::mutate(z = x * 2, .keep = "used"))
  expect_no_error(gt |> dplyr::mutate(z = x * 2, .after = g))
  expect_no_error(capture.output(dplyr::glimpse(gt)))
})

test_that("count()/tally()/add_count()/add_tally() are GPU-native and never trigger a fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(fallback_df())

  expect_no_error(gt |> dplyr::count(g))
  expect_no_error(gt |> dplyr::count())
  expect_no_error(gt |> dplyr::count(g, wt = x, sort = TRUE))
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::tally())
  expect_no_error(gt |> dplyr::add_count(g))
  # add_tally() is a plain (non-generic) dplyr function -- it works
  # transparently via tbl_vars.tbl_gpu()/group_vars.tbl_gpu()/mutate()/
  # arrange(), with no tbl_gpu-specific code of its own (R/count.R).
  expect_no_error(gt |> dplyr::group_by(g) |> dplyr::add_tally())
})
