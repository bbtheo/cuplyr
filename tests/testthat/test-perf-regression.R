# Performance-regression benchmarks
#
# These guard against silent performance regressions as core-execution code
# (filter/mutate/arrange/summarise/join, the AST optimizer, transfer layer)
# is rewritten. They complement the correctness oracles (helper-oracle.R):
# this file never checks *values*, only *wall-clock time* relative to a
# recorded baseline (see helper-perf.R for the full harness contract).
#
# Opt-in and machine-specific: every benchmark is skipped unless
# CUPLYR_PERF=1 is set (see expect_no_perf_regression()). The package must be
# loaded from an optimized (-O2) build: pkgload::load_all() defaults to
# debug = TRUE, which recompiles any edited .cpp at -O0 and makes timings
# meaningless (expect_no_perf_regression() refuses such a build). So build and
# load like this, from the worktree root. To (re)record baselines on this
# machine:
#
#   CUPLYR_PERF=1 CUPLYR_PERF_RECORD=1 pixi run Rscript -e \
#     'source("tools/ensure-optimized.R"); cuplyr_ensure_optimized_dll("."); pkgload::load_all(quiet=TRUE, debug=FALSE); testthat::test_file("tests/testthat/test-perf-regression.R")'
#
# and to check against the recorded baseline:
#
#   CUPLYR_PERF=1 pixi run Rscript -e \
#     'source("tools/ensure-optimized.R"); cuplyr_ensure_optimized_dll("."); pkgload::load_all(quiet=TRUE, debug=FALSE); testthat::test_file("tests/testthat/test-perf-regression.R")'
#
# The baseline is adaptive: after a fully passing merge gate, tools/merge-queue
# lowers an entry to the slower of its last two observations and adds new
# ids (tools/perf-ratchet; the gate records them via CUPLYR_PERF_OBSERVATIONS,
# see helper-perf.R). Re-recording is Theo's, as is raising an entry
# (tools/perf-baseline-set). A re-record takes no merge.lock, so don't start
# one while tools/merge-queue is running: it could overwrite or lose a ratchet.
#
# The calibration probes are measured once per process, at the first
# benchmark that uses each. The kernel probe reads ~0.7x its baseline in this
# file's process and ~1.2x in a bare session or on a repeat measurement (see
# perf_calibration_time() in helper-perf.R). That no longer matters: it never
# widens a kernel threshold and only errors above 1.5x. The transfer probe
# does widen transfer thresholds, so run the whole file the way the gate
# does when comparing numbers with gate logs.

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

# Phase 7 J9: a small, deliberately-narrow-range probe for the non-equi
# (join_by(id >= id2)) benchmark below. perf_big_df's `id` is uniform over
# 1:5000 across 1e6 rows; this probe's 100 rows have `id2` uniform over the
# narrow high sub-range 4901:5000 (one row per value, no duplicates). Only
# left rows with id >= 4901 (~2% of 1e6 = ~20000 rows) can match anything at
# all, and each of those matches at most 100 right rows (fewer for smaller
# id) -- average ~50 matches per qualifying left row, for a total output on
# the order of ~1e6 rows, the same order of magnitude as this file's other
# "kernel"-calibrated (compute-bound, not fan-out-dominated) benchmarks. A
# uniform-range probe over the FULL 1:5000 span (like perf_join_probe_df)
# would instead fan out closer to a near-cartesian ~1e8 rows for `>=`, which
# is deliberately avoided here.
perf_join_nonequi_probe_df <- local({
  data.frame(
    id2 = 4901:5000,
    w2 = runif(100)
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

# F74: summarise() on a `.drop = FALSE` table with fixed group keys (a lazy
# semi_join() empties half of the 2e4 character-key groups). The cost over a
# plain grouped summarise is the observed-key download/match and one gather;
# before F74 this whole call ran on the CPU. Calibration "kernel": compute
# bound, a 2e4-row result.
perf_fixed_keys_df <- local({
  set.seed(74)
  n <- 1e6
  data.frame(
    k = sprintf("k%05d", sample.int(2e4, n, replace = TRUE)),
    x = runif(n),
    stringsAsFactors = FALSE
  )
})
perf_fixed_keys_keep <- data.frame(k = sprintf("k%05d", seq(1, 2e4, by = 2)),
                                   stringsAsFactors = FALSE)

test_that("perf: summarise_fixed_keys_lazy", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_fixed_keys_df, lazy = TRUE)
  gpu_keep <- tbl_gpu(perf_fixed_keys_keep, lazy = TRUE)

  expect_no_perf_regression("summarise_fixed_keys_lazy", function() {
    gpu_df |>
      dplyr::group_by(k, .drop = FALSE) |>
      dplyr::semi_join(gpu_keep, by = "k") |>
      dplyr::summarise(n = n(), s = sum(x), m = mean(x)) |>
      collect()
  }, calibration = "kernel")
})

# F33/F74: summarise() on a `group_by(.drop = FALSE)` factor key (no fixed
# keys) with 5e4 levels, half of them never observed. Everything this path
# adds over a plain grouped summarise is per-group R work on the keys (the
# level expansion, the observed-key match); F74 round 3 found it had doubled
# through a per-element `lengths()` on the `.rows` list_of. Calibration
# "kernel", like the fixed-keys benchmark above.
perf_factor_drop_df <- local({
  set.seed(33)
  n <- 1e6
  lv <- sprintf("f%05d", seq_len(5e4))
  data.frame(
    f = factor(lv[sample(seq(1, 5e4, by = 2), n, replace = TRUE)], levels = lv),
    x = runif(n)
  )
})

test_that("perf: summarise_factor_drop_false_lazy", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_factor_drop_df, lazy = TRUE)

  expect_no_perf_regression("summarise_factor_drop_false_lazy", function() {
    gpu_df |>
      dplyr::group_by(f, .drop = FALSE) |>
      dplyr::summarise(n = n(), s = sum(x), m = mean(x)) |>
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
# semi_join() (Phase 7 J9)
#
# Calibration class: "kernel" -- unlike left_join()'s fan-out-bound benchmark
# above, semi_join() never duplicates rows: the result is a subset of
# perf_big_df's own 1e6 rows (bounded above by 1e6, same order of magnitude
# as filter_chain/mutate_chain/etc.), so this benchmark is compute-bound
# (the filtered_join build+probe pass) like the other native-verb kernel
# benchmarks, not transfer-bound like the fan-out mutating joins.
# perf_join_probe_df's 5e4 rows sampled uniformly over 1:5000 almost
# certainly cover every distinct `id` value perf_big_df has, so this
# benchmark exercises the real filtered_join machinery against a
# realistically-sized build table without collapsing to a trivial no-op.
# =============================================================================

test_that("perf: join_semi_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)
  gpu_probe <- tbl_gpu(perf_join_probe_df, lazy = TRUE)

  expect_no_perf_regression("join_semi_lazy", function() {
    gpu_df |>
      dplyr::semi_join(gpu_probe, by = "id") |>
      collect()
  }, calibration = "kernel")
})

# =============================================================================
# Non-equi join_by() (Phase 7 J9)
#
# Calibration class: "kernel" -- a join_by(id >= id2) against
# perf_join_nonequi_probe_df's deliberately narrow-range 100-row probe (see
# that fixture's own comment above for the fan-out derivation): output is on
# the order of ~1e6 rows, the same order of magnitude as perf_big_df itself,
# so this benchmark is dominated by gpu_cond_join()'s conditional_join()
# compute pass (no equi component in this condition), not by transferring a
# huge fan-out result.
# =============================================================================

test_that("perf: join_nonequi_lazy", {
  skip_if_no_gpu()
  gpu_df <- tbl_gpu(perf_big_df, lazy = TRUE)
  gpu_probe <- tbl_gpu(perf_join_nonequi_probe_df, lazy = TRUE)

  expect_no_perf_regression("join_nonequi_lazy", function() {
    suppressWarnings(
      gpu_df |>
        dplyr::left_join(gpu_probe, by = dplyr::join_by(id >= id2)) |>
        collect()
    )
  }, calibration = "kernel")
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

# =============================================================================
# Checked integer arithmetic and rolling windows (F61)
#
# Calibration class: "kernel". These calls take 1-3 ms at 1e6 rows, so a
# collect() of the result would drown them in the device->host copy. Each body
# runs the verb and then fences with gpu_sync() (every verb returns when its
# work is queued): an eager mutate() executes inside the call, a lazy plan
# through compute(). perf_time() batches the calls into >= 50 ms samples (F52).
# =============================================================================

perf_int_df <- local({
  set.seed(20261023)
  n <- 1e6
  data.frame(
    a = sample.int(1000L, n, replace = TRUE),
    b = sample.int(1000L, n, replace = TRUE),
    c = sample.int(1000L, n, replace = TRUE)
  )
})

test_that("perf: int_arith_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_int_df, lazy = FALSE)

  expect_no_perf_regression("int_arith_eager", function() {
    out <- dplyr::mutate(gpu_df, s = a + b, p = a * c, q = (a + b) * 2L - c)
    gpu_sync()
    out
  }, calibration = "kernel")
})

test_that("perf: int_arith_lazy", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_int_df, lazy = TRUE)

  expect_no_perf_regression("int_arith_lazy", function() {
    out <- gpu_df |>
      dplyr::mutate(s = a + b, p = a * c, q = (a + b) * 2L - c) |>
      compute()
    gpu_sync()
    out
  }, calibration = "kernel")
})

# Operands that sum to ~2e9 row by row, but whose column ranges do not: the
# interval proof of the fast path fails (a's max + b's max is ~4e9), so this
# times the exact two-pass check. No row overflows, so nothing warns.
perf_int_bounds_df <- local({
  set.seed(20261024)
  n <- 1e6
  a <- sample.int(2000000000L, n, replace = TRUE)
  data.frame(
    a = a,
    b = 2000000000L - a + sample.int(1000L, n, replace = TRUE),
    c = sample.int(1000L, n, replace = TRUE)
  )
})

test_that("perf: int_arith_bounds_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_int_bounds_df, lazy = FALSE)

  expect_no_perf_regression("int_arith_bounds_eager", function() {
    out <- dplyr::mutate(gpu_df, s = a + b, q = (a + b) - c)
    gpu_sync()
    out
  }, calibration = "kernel")
})

test_that("perf: roll_mean_w11_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_big_df[, "x", drop = FALSE], lazy = FALSE)

  expect_no_perf_regression("roll_mean_w11_eager", function() {
    out <- dplyr::mutate(gpu_df, r = roll_mean(x, 11))
    gpu_sync()
    out
  }, calibration = "kernel")
})

test_that("perf: roll_mean_w1001_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_big_df[, "x", drop = FALSE], lazy = FALSE)

  expect_no_perf_regression("roll_mean_w1001_eager", function() {
    out <- dplyr::mutate(gpu_df, r = roll_mean(x, 1001))
    gpu_sync()
    out
  }, calibration = "kernel")
})

test_that("perf: roll_sd_w11_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_big_df[, "x", drop = FALSE], lazy = FALSE)

  expect_no_perf_regression("roll_sd_w11_eager", function() {
    out <- dplyr::mutate(gpu_df, r = roll_sd(x, 11))
    gpu_sync()
    out
  }, calibration = "kernel")
})

test_that("perf: roll_sd_w1001_eager", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  gpu_df <- tbl_gpu(perf_big_df[, "x", drop = FALSE], lazy = FALSE)

  expect_no_perf_regression("roll_sd_w1001_eager", function() {
    out <- dplyr::mutate(gpu_df, r = roll_sd(x, 1001))
    gpu_sync()
    out
  }, calibration = "kernel")
})
