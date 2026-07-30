# Performance-regression benchmarks
#
# These guard against silent performance regressions as core-execution code
# (filter/mutate/arrange/summarise/join, the AST optimizer, transfer layer)
# is rewritten. They complement the correctness oracles (helper-oracle.R):
# this file never checks *values*, only *wall-clock time* relative to a
# recorded baseline (see helper-perf.R for the full harness contract).
#
# Opt-in and machine-specific: every benchmark is skipped unless
# CUPLYR_PERF=1 is set (see expect_no_perf_regression()). To (re)record
# baselines on this machine:
#
#   CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 pixi run Rscript -e \
#     'pkgload::load_all(quiet=TRUE); testthat::test_file("tests/testthat/test-perf-regression.R")'
#
# and to check against the recorded baseline:
#
#   CUPLYR_PERF=1 pixi run Rscript -e \
#     'pkgload::load_all(quiet=TRUE); testthat::test_file("tests/testthat/test-perf-regression.R")'

# =============================================================================
# GPU warm-up (file start)
# =============================================================================
#
# Force GPU clocks out of idle (P8) before any benchmark in this file is
# timed -- see perf_warm_gpu() in helper-perf.R for why. Idempotent per
# process; expect_no_perf_regression() also calls this defensively so
# benchmarks are protected even if this file-level call is skipped (e.g.
# test_file() run without going through the whole suite).
if (identical(Sys.getenv("CUPLYR_PERF"), "1") && has_gpu()) {
  perf_warm_gpu()
}

# =============================================================================
# Shared test data (built once per test-file run, not per benchmark)
# =============================================================================

perf_big_df <- local({
  set.seed(20260726)
  n <- 1e6
  data.frame(
    id = sample.int(5000L, n, replace = TRUE),
    x = runif(n),
    y = runif(n),
    z = rnorm(n),
    grp = sample(paste0("g", seq_len(20)), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
})

perf_join_probe_df <- local({
  set.seed(20260727)
  n <- 5e4
  data.frame(
    id = sample.int(5000L, n, replace = TRUE),
    w = runif(n)
  )
})

# =============================================================================
# filter()
# =============================================================================

test_that("perf: filter_chain_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)

  expect_no_perf_regression("filter_chain_eager", function() {
    gpu_df |>
      dplyr::filter(x > 0.25, y <= 0.9, z > 0.1) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: filter_chain_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("filter_chain_lazy", function() {
    gpu_df |>
      dplyr::filter(x > 0.25, y <= 0.9, z > 0.1) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# mutate()
# =============================================================================

test_that("perf: mutate_chain_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)

  expect_no_perf_regression("mutate_chain_eager", function() {
    gpu_df |>
      dplyr::mutate(a = x + y, b = a * 2, c = b - z) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: mutate_chain_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("mutate_chain_lazy", function() {
    gpu_df |>
      dplyr::mutate(a = x + y, b = a * 2, c = b - z) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# select()
# =============================================================================

test_that("perf: select_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)

  expect_no_perf_regression("select_eager", function() {
    gpu_df |>
      dplyr::select(id, x, grp) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: select_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("select_lazy", function() {
    gpu_df |>
      dplyr::select(id, x, grp) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# arrange() - multi-key with desc() and ties
# =============================================================================

test_that("perf: arrange_multikey_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)

  expect_no_perf_regression("arrange_multikey_eager", function() {
    gpu_df |>
      dplyr::arrange(grp, dplyr::desc(id), x) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: arrange_multikey_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("arrange_multikey_lazy", function() {
    gpu_df |>
      dplyr::arrange(grp, dplyr::desc(id), x) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# group_by() |> summarise()
# =============================================================================

test_that("perf: summarise_grouped_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)

  expect_no_perf_regression("summarise_grouped_eager", function() {
    gpu_df |>
      dplyr::group_by(grp) |>
      dplyr::summarise(m = mean(x), s = sum(y), n = n(), .groups = "drop") |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: summarise_grouped_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("summarise_grouped_lazy", function() {
    gpu_df |>
      dplyr::group_by(grp) |>
      dplyr::summarise(m = mean(x), s = sum(y), n = n(), .groups = "drop") |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# Window functions (Phase 5) - grouped mutate()
#
# Calibration class: "kernel" -- like the other grouped-mutate/summarise
# benchmarks above, these are compute-heavy (a groupby-scan or groupby-
# aggregate pass over 1M rows, ~200 rows/group across 20 groups) and
# transfer-light (result stays the same 1M-row shape as the input, no
# fan-out like the join benchmarks below). Missing baseline entries for
# these 3 new ids skip cleanly (see helper-perf.R) until first recorded --
# added in Phase 5 task W9 per the design doc's perf policy.
# =============================================================================

test_that("perf: window_scan_grouped_lazy", {
  skip_if_no_gpu()

  expect_no_perf_regression("window_scan_grouped_lazy", function() {
    tbl_gpu(perf_big_df, lazy = TRUE) |>
      dplyr::group_by(grp) |>
      dplyr::mutate(c = cumsum(x), r = dplyr::row_number()) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: window_agg_broadcast_lazy", {
  skip_if_no_gpu()

  expect_no_perf_regression("window_agg_broadcast_lazy", function() {
    tbl_gpu(perf_big_df, lazy = TRUE) |>
      dplyr::group_by(id) |>
      dplyr::mutate(d = x - mean(x)) |>
      collect()
  }, calibration = "kernel")
})

test_that("perf: window_rank_grouped_lazy", {
  skip_if_no_gpu()

  expect_no_perf_regression("window_rank_grouped_lazy", function() {
    tbl_gpu(perf_big_df, lazy = TRUE) |>
      dplyr::group_by(grp) |>
      dplyr::mutate(r = dplyr::min_rank(x)) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# left_join()
#
# Calibration class: "transfer". perf_big_df's `id` is sampled uniformly
# from 1:5000 over 1e6 rows (~200 rows/id); perf_join_probe_df's `id` is
# sampled uniformly from the same 1:5000 range over 5e4 rows (~10 rows/id).
# A left join therefore fans out to roughly 200*10 = 2000 matches per id
# across 5000 ids -- on the order of 1e7 output rows, ~10x the row count of
# transfer_roundtrip's collect() (1e6 rows). At recorded baseline times
# (~0.44s for the join vs ~0.01-0.03s for filter/mutate/select/arrange on
# the same-sized 1e6-row input, and ~0.056s for transfer_roundtrip's plain
# 1e6-row round trip), the join's cost scales with the size of the
# *collected* result, not the input -- i.e. it is dominated by the
# device->host transfer of a ~10x-larger result, not by kernel work. That
# makes it a transfer-bound benchmark despite doing a hash join internally.
#
# Phase 7 J5: this fixture's ~200 rows/id x ~10 rows/id key distribution is
# a genuine many-to-many pattern, so `left_join()`'s default `relationship =
# NULL`/`multiple = "all"` now raises dplyr's own "unexpected many-to-many
# relationship" advisory warning (check_join_cardinality(), R/join.R) on
# every iteration -- expected and harmless (confirmed: still 17/17 on the
# perf gate: the warning is a handful of string-formatting/condition-
# dispatch calls, utterly dwarfed by this benchmark's multi-MB device->host
# transfer), but suppressWarnings() keeps it from cluttering every perf run
# with a duplicate warning per timed iteration.
# =============================================================================

test_that("perf: join_left_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)
  gpu_probe <- tbl_gpu(perf_join_probe_df, lazy = FALSE)

  expect_no_perf_regression("join_left_eager", function() {
    suppressWarnings(
      gpu_df |>
        dplyr::left_join(gpu_probe, by = "id") |>
        collect()
    )
  }, calibration = "transfer")
})

test_that("perf: join_left_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)
  gpu_probe <- tbl_gpu(perf_join_probe_df, lazy = TRUE)

  expect_no_perf_regression("join_left_lazy", function() {
    suppressWarnings(
      gpu_df |>
        dplyr::left_join(gpu_probe, by = "id") |>
        collect()
    )
  }, calibration = "transfer")
})

# =============================================================================
# Fused lazy pipeline - exists specifically to catch optimizer regressions
# (lost fusion/pushdown across filter/mutate/select).
# =============================================================================

test_that("perf: fused_pipeline_lazy", {
  skip_if_no_gpu()

  expect_no_perf_regression("fused_pipeline_lazy", function() {
    tbl_gpu(perf_big_df) |>
      as_lazy() |>
      dplyr::filter(x > 0.1) |>
      dplyr::mutate(a = x + y, b = a * z) |>
      dplyr::filter(b > 0) |>
      dplyr::select(id, grp, a, b) |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# Transfer layer (guards df -> GPU -> R round trip in isolation)
#
# Calibration class: "transfer" -- this benchmark *is* the transfer probe's
# shape (tbl_gpu() construction + collect(), no compute in between), so it
# is the canonical transfer-bound case.
# =============================================================================

test_that("perf: transfer_roundtrip", {
  skip_if_no_gpu()

  expect_no_perf_regression("transfer_roundtrip", function() {
    tbl_gpu(perf_big_df) |>
      collect()
  }, calibration = "transfer")
})
