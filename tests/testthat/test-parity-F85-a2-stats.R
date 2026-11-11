# F85 attempt 2: what is left to push (INT32, INT64, Date, POSIXct) against the
# footers that lost rows on other types in attempt 1: deprecated min/max fields,
# missing or one-sided statistics, no null count, no statistics, timestamp unit
# and isAdjustedToUTC. Strings are not pushed any more (B13).
#
# Every fixture holds a known frame. Those named *_stats with patched footers
# were hand-written from an arrow file (the generator is
# scratchpad/parity/features/F85-a2/measure/make-audit-fixtures.R); each test
# compares against dplyr on that frame, with pushdown on, off, and eagerly.

stats_ids <- function(path, q, push, lazy = TRUE) {
  withr::with_options(list(cuplyr.parquet_pushdown = push),
                      collect(filter(read_parquet_gpu(path, lazy = lazy), !!q))$id)
}

# The row set of `q` by every route. Returns what the pushing route read.
expect_stats_rows <- function(path, oracle, q, info) {
  want <- dplyr::filter(oracle, !!q)$id
  parquet_last_read(reset = TRUE)
  on <- stats_ids(path, q, TRUE)
  read <- parquet_last_read()
  expect_identical(on, want, info = paste(info, "pushdown on"))
  expect_identical(stats_ids(path, q, FALSE), want, info = paste(info, "pushdown off"))
  expect_identical(stats_ids(path, q, TRUE, lazy = FALSE), want, info = paste(info, "eager"))
  read
}

# The frame behind the fixed-width audit files: three row groups of three, with
# nulls in the first and a third group that is all null but for `id`.
stats_audit_df <- function() {
  t0 <- as.POSIXct("2021-03-04 05:06:00", tz = "UTC")
  tibble::tibble(
    id = 1:9,
    i = c(1L, NA, 3L, 10L, 11L, 12L, NA, NA, NA),
    k = bit64::as.integer64(c(1, NA, 3, 10, 11, 12, NA, NA, NA)) + bit64::as.integer64("1099511627776"),
    dt = as.Date("2020-01-01") + c(0, NA, 2, 9, 10, 11, NA, NA, NA),
    ts = t0 + c(0, NA, 2, 9, 10, 11, NA, NA, NA)
  )
}

stats_audit_preds <- function() {
  k <- function(v) bit64::as.integer64("1099511627776") + bit64::as.integer64(v)
  dt <- function(v) as.Date("2020-01-01") + v
  ts <- function(v) as.POSIXct("2021-03-04 05:06:00", tz = "UTC") + v
  list(
    int_eq_low = rlang::expr(i == 3L), int_eq_high = rlang::expr(i == 11L), int_eq_absent = rlang::expr(i == 7L),
    int_lt = rlang::expr(i < 3L), int_le = rlang::expr(i <= 3L), int_gt = rlang::expr(i > 3L),
    int_ge = rlang::expr(i >= 10L), int_ne = rlang::expr(i != 1L), int_flip = rlang::expr(5L > i),
    int_not_na = rlang::expr(!is.na(i)),
    i64_eq = rlang::expr(k == !!k(11)), i64_lt = rlang::expr(k < !!k(3)), i64_ge = rlang::expr(k >= !!k(10)),
    i64_ne = rlang::expr(k != !!k(1)), i64_not_na = rlang::expr(!is.na(k)),
    date_eq = rlang::expr(dt == !!dt(10)), date_lt = rlang::expr(dt < !!dt(2)), date_ge = rlang::expr(dt >= !!dt(9)),
    date_ne = rlang::expr(dt != !!dt(0)),
    ts_eq = rlang::expr(ts == !!ts(10)), ts_lt = rlang::expr(ts < !!ts(2)), ts_ge = rlang::expr(ts >= !!ts(9)),
    ts_ne = rlang::expr(ts != !!ts(0))
  )
}

test_that("a string comparison is never decided by the reader", {
  skip_if_no_gpu()
  path <- pq_fixture("deprecated_string_stats.parquet")
  oracle <- tibble::tibble(id = 1:6, s = c("a", "é", "c", "k", "m", "z"))
  preds <- list(
    rlang::expr(s == "a"), rlang::expr(s == "é"), rlang::expr(s == "c"),
    rlang::expr(s == "k"), rlang::expr(s == "z"), rlang::expr(s == "b"),
    rlang::expr(s != "a"), rlang::expr("a" == s)
  )
  for (q in preds) {
    read <- expect_stats_rows(path, oracle, q, rlang::expr_text(q))
    expect_length(read$predicates, 0L)
    expect_true(is.na(read$row_groups_after_stats))
  }
})

test_that("integer, Date and POSIXct filters keep their rows on footers with only the deprecated min and max", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  path <- pq_fixture("deprecated_fixed_stats.parquet")
  oracle <- stats_audit_df()
  skipped <- integer()
  for (nm in names(stats_audit_preds())) {
    read <- expect_stats_rows(path, oracle, stats_audit_preds()[[nm]], nm)
    expect_gt(length(read$predicates), 0L)
    if (isTRUE(read$row_groups_after_stats < read$row_groups_total)) {
      skipped <- c(skipped, sub("_.*", "", nm))
    }
  }
  # the reader really used these statistics on every type, so the row sets above mean something
  expect_setequal(skipped, c("int", "i64", "date", "ts"))
})

test_that("integer, Date and POSIXct filters keep their rows on footers with no null count", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  path <- pq_fixture("no_null_count_stats.parquet")
  oracle <- stats_audit_df()
  skipped <- integer()
  for (nm in names(stats_audit_preds())) {
    read <- expect_stats_rows(path, oracle, stats_audit_preds()[[nm]], nm)
    expect_gt(length(read$predicates), 0L)
    if (isTRUE(read$row_groups_after_stats < read$row_groups_total)) {
      skipped <- c(skipped, sub("_.*", "", nm))
    }
  }
  expect_setequal(skipped, c("int", "i64", "date", "ts"))
})

test_that("integer, Date and POSIXct filters keep their rows when a row group has only one bound or none", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  path <- pq_fixture("partial_bound_stats.parquet")
  oracle <- stats_audit_df()
  for (nm in names(stats_audit_preds())) {
    read <- expect_stats_rows(path, oracle, stats_audit_preds()[[nm]], nm)
    expect_gt(length(read$predicates), 0L)
  }
})

test_that("integer, Date and POSIXct filters keep their rows on a file with no statistics", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  path <- pq_fixture("no_stats.parquet")
  oracle <- stats_audit_df()
  for (nm in names(stats_audit_preds())) {
    read <- expect_stats_rows(path, oracle, stats_audit_preds()[[nm]], nm)
    expect_gt(length(read$predicates), 0L)
    expect_identical(as.integer(read$row_groups_after_stats), as.integer(read$row_groups_total), info = nm)
  }
})

test_that("a timestamp with isAdjustedToUTC = false compares by its stored value", {
  skip_if_no_gpu()
  path <- pq_fixture("ts_unadjusted_stats.parquet")
  t0 <- as.POSIXct("2021-03-04 05:06:00", tz = "UTC")
  oracle <- tibble::tibble(id = 1:6, ts = t0 + c(0, 1, 2, 10, 11, 12))
  for (q in list(rlang::expr(ts >= !!(t0 + 10)), rlang::expr(ts == !!(t0 + 11)),
                 rlang::expr(ts < !!(t0 + 2)), rlang::expr(ts != !!(t0 + 1)))) {
    read <- expect_stats_rows(path, oracle, q, rlang::expr_text(q))
    expect_length(read$predicates, 1L)
  }
  read <- expect_stats_rows(path, oracle, rlang::expr(ts >= !!(t0 + 10)), "skip")
  expect_lt(read$row_groups_after_stats, read$row_groups_total)
})

test_that("columns read through a cast keep their rows and are not given to the reader", {
  skip_if_no_gpu()
  t0 <- as.POSIXct("2021-03-04 05:06:00", tz = "UTC")
  ts_frame <- tibble::tibble(id = 1:6, ts = t0 + c(0, 1, 2, 10, 11, 12))
  ts_preds <- list(rlang::expr(ts >= !!(t0 + 10)), rlang::expr(ts == !!(t0 + 11)), rlang::expr(ts < !!(t0 + 2)))
  for (file in c("ts_int96.parquet", "ts_millis_stats.parquet")) {
    for (q in ts_preds) {
      read <- expect_stats_rows(pq_fixture(file), ts_frame, q, paste(file, rlang::expr_text(q)))
      expect_length(read$predicates, 0L)
    }
  }
  # unsigned statistics cross the signed range: 3e9 reads negative as an INT32
  u_frame <- tibble::tibble(id = 1:6, u = c(1, 2, 3e9, 3.1e9, 3.2e9, 5))
  for (q in list(rlang::expr(u > 2e9), rlang::expr(u == 3e9), rlang::expr(u < 2e9), rlang::expr(u >= 3.1e9))) {
    read <- expect_stats_rows(pq_fixture("uint32_stats.parquet"), u_frame, q, rlang::expr_text(q))
    expect_length(read$predicates, 0L)
  }
})
