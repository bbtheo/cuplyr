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
  })
})

test_that("perf: filter_chain_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("filter_chain_lazy", function() {
    gpu_df |>
      dplyr::filter(x > 0.25, y <= 0.9, z > 0.1) |>
      collect()
  })
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
  })
})

test_that("perf: mutate_chain_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("mutate_chain_lazy", function() {
    gpu_df |>
      dplyr::mutate(a = x + y, b = a * 2, c = b - z) |>
      collect()
  })
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
  })
})

test_that("perf: select_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("select_lazy", function() {
    gpu_df |>
      dplyr::select(id, x, grp) |>
      collect()
  })
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
  })
})

test_that("perf: arrange_multikey_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("arrange_multikey_lazy", function() {
    gpu_df |>
      dplyr::arrange(grp, dplyr::desc(id), x) |>
      collect()
  })
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
  })
})

test_that("perf: summarise_grouped_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)

  expect_no_perf_regression("summarise_grouped_lazy", function() {
    gpu_df |>
      dplyr::group_by(grp) |>
      dplyr::summarise(m = mean(x), s = sum(y), n = n(), .groups = "drop") |>
      collect()
  })
})

# =============================================================================
# left_join()
# =============================================================================

test_that("perf: join_left_eager", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = FALSE)
  gpu_probe <- tbl_gpu(perf_join_probe_df, lazy = FALSE)

  expect_no_perf_regression("join_left_eager", function() {
    gpu_df |>
      dplyr::left_join(gpu_probe, by = "id") |>
      collect()
  })
})

test_that("perf: join_left_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)
  gpu_probe <- tbl_gpu(perf_join_probe_df, lazy = TRUE)

  expect_no_perf_regression("join_left_lazy", function() {
    gpu_df |>
      dplyr::left_join(gpu_probe, by = "id") |>
      collect()
  })
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
  })
})

# =============================================================================
# Transfer layer (guards df -> GPU -> R round trip in isolation)
# =============================================================================

test_that("perf: transfer_roundtrip", {
  skip_if_no_gpu()

  expect_no_perf_regression("transfer_roundtrip", function() {
    tbl_gpu(perf_big_df) |>
      collect()
  })
})
