# Phase 11 L3: timestamp literals in filter()/mutate(), TIMESTAMP_* <->
# TIMESTAMP_* casts (gpu_cast_column), and as.Date()/as.POSIXct() applied to
# an existing TIMESTAMP_* column. L6 (datetime accessors: year()/month()/
# etc.) is covered in its own section near the bottom of this file.
#
# Pre-fix confirmation (see scratchpad/workflow_state.md for the full
# narrative): before this task, `filter(d < as.Date("2010-01-05"))` on a
# Date column hard-errored with "Unsupported GPU type string: TIMESTAMP_DAYS"
# (from `resolve_static_type()` -> `expr_type_from_str()` in
# src/expr_eval.hpp, reached before `build_scalar()`'s own "Unsupported
# literal type" branch would ever have fired) -- confirmed via a targeted
# `git stash` + rebuild of only the L3 files, per the repo's STRICT
# test-first-bugfix mandate.

test_that("filter() with a Date column vs. an as.Date() literal works natively", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-10", "2010-01-05")), id = 1:3)

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, d < as.Date("2010-01-05"))
  })
  expect_same_as_dplyr_lazy(df, function(tbl) {
    dplyr::filter(tbl, d < as.Date("2010-01-05"))
  })
})

test_that("filter() with every comparison operator against a Date literal matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-05", "2010-01-10", NA)))

  for (op in list(`<`, `<=`, `>`, `>=`, `==`, `!=`)) {
    pipeline <- (function(op) {
      force(op)
      function(tbl) dplyr::filter(tbl, op(d, as.Date("2010-01-05")))
    })(op)
    expect_same_as_dplyr(df, pipeline)
  }
})

test_that("filter() with a POSIXct column vs. an as.POSIXct() literal works natively", {
  skip_if_no_gpu()

  df <- data.frame(dt = as.POSIXct(
    c("2020-01-01 10:00:00", "2020-01-02 10:00:00", "2020-01-02 23:00:00"),
    tz = "UTC"
  ))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, dt < as.POSIXct("2020-01-02 00:00:00", tz = "UTC"))
  })
  expect_same_as_dplyr_lazy(df, function(tbl) {
    dplyr::filter(tbl, dt >= as.POSIXct("2020-01-02 00:00:00", tz = "UTC"))
  })
})

test_that("filter() with NA in an as.Date() literal comparison matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2010-01-01", NA, "2010-01-10")))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, d < as.Date(NA))
  })
})

test_that("filter() combining a Date comparison with a plain numeric predicate stays fused", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-10", "2010-01-05")), x = c(1, 2, 3))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, d < as.Date("2010-01-08"), x > 1)
  })
})

test_that("filter() Date column vs. numeric literal errors with a clear message", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-10")))
  g <- tbl_gpu(df)

  expect_error(
    dplyr::filter(g, d < 5),
    "Cannot compare column 'd' \\(TIMESTAMP_DAYS\\) with a numeric literal"
  )
})

test_that("filter() POSIXct column vs. logical literal errors with a clear message", {
  skip_if_no_gpu()

  df <- data.frame(dt = as.POSIXct("2020-01-01", tz = "UTC"))
  g <- tbl_gpu(df)

  expect_error(
    dplyr::filter(g, dt == TRUE),
    "Cannot compare column 'dt' \\(TIMESTAMP_MICROSECONDS\\) with a numeric literal"
  )
})

test_that("filter() Date column vs. a different-granularity timestamp literal errors clearly (documented boundary)", {
  skip_if_no_gpu()

  # A column-vs-column mismatch already promotes correctly via cudf::cast()
  # (see the bind_rows()/union() timestamp-mix tests) -- this is narrower:
  # a MISMATCHED LITERAL is rebuilt directly via build_scalar() at the
  # column's own hint type, which has no way to rescale an already-evaluated
  # R value between day-count and fractional-seconds conventions. Rejected
  # explicitly rather than risking silent wrongness (see
  # check_filter_comparison_types()'s own doc comment, R/filter.R).
  # NOTE: the literal must be a single-arg `as.POSIXct(x)` call -- a 2-arg
  # call (e.g. adding `tz=`) doesn't match the native registry entry's
  # `arity = 1L` at all, so it takes the (also-safe) L1 CPU-fallback path
  # instead of ever reaching this native rejection (see the next test).
  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-10")))
  g <- tbl_gpu(df)

  expect_error(
    dplyr::filter(g, d < as.POSIXct("2010-01-05")),
    "different timestamp precision"
  )
})

test_that("filter() with as.POSIXct(x, tz=) (extra arg) falls back to CPU and matches dplyr's own warning-based behavior", {
  skip_if_no_gpu()

  # arity mismatch (2 args, not 1) means the native "as.POSIXct" registry
  # entry never dispatches -- the whole dot falls back to L1's CPU path,
  # which reproduces real dplyr's own behavior exactly, including its own
  # warning for a Date/POSIXct `Ops` mismatch (real dplyr does NOT treat
  # this as zero rows -- `Ops.Date`'s default method still runs after
  # warning, comparing the two sides' raw numeric representations).
  df <- data.frame(d = as.Date(c("2010-01-01", "2010-01-10")))
  g <- tbl_gpu(df)
  oracle <- suppressWarnings(
    dplyr::filter(df, d < as.POSIXct("2020-01-02 00:00:00", tz = "UTC"))
  )

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::filter(g, d < as.POSIXct("2020-01-02 00:00:00", tz = "UTC")),
    "fell back to CPU evaluation"
  )
  expect_equal(as.data.frame(collect(result)), oracle)
})

# -----------------------------------------------------------------------------
# as.Date()/as.POSIXct() casts (mutate())
# -----------------------------------------------------------------------------

test_that("as.Date() on a POSIXct column casts natively and matches dplyr (UTC truncation)", {
  skip_if_no_gpu()

  # Empirical finding (verified via plain Rscript, no GPU needed):
  # as.Date.POSIXct()'s default `tz` argument is the literal string "UTC"
  # (NOT `attr(x, "tzone")` or `Sys.timezone()`) -- so as.Date() truncates to
  # the UTC calendar day of the underlying instant regardless of the
  # POSIXct's own display timezone. cuplyr's own POSIXct storage is already
  # UTC-normalized (R's POSIXct is always epoch-seconds internally), so a
  # plain cudf::cast() reproduces this exactly with no tz adjustment needed.
  # Compared on the `d` output column's VALUES only (not the whole frame):
  # cuplyr's own POSIXct round-trip always normalizes `tzone` to "UTC" on
  # `collect()` (`src/transfer_io.cpp`'s `gpu_collect()`) regardless of the
  # column's original display timezone -- a real, pre-existing, UNRELATED
  # divergence (the underlying instants are identical; only the display
  # attribute differs) that would otherwise fail a whole-frame comparison
  # here for a reason that has nothing to do with `as.Date()` itself.
  df <- data.frame(dt = as.POSIXct(
    c("2020-01-01 23:30:00", "2020-01-02 04:30:00", NA),
    tz = "America/New_York"
  ))

  result <- tbl_gpu(df) |> dplyr::mutate(d = as.Date(dt)) |> collect()
  oracle <- dplyr::mutate(df, d = as.Date(dt))
  expect_equal(result$d, oracle$d)
})

test_that("as.POSIXct() on a Date column casts natively and matches dplyr", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2020-01-01", "2020-06-15", NA)))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate(tbl, dt = as.POSIXct(d))
  })
})

test_that("as.Date() on an already-Date column is a native no-op cast", {
  skip_if_no_gpu()

  df <- data.frame(d = as.Date(c("2020-01-01", "2020-06-15")))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::mutate(tbl, d2 = as.Date(d))
  })
})

test_that("as.Date()/as.POSIXct() on a STRING column falls back to CPU (not yet native)", {
  skip_if_no_gpu()

  df <- tibble::tibble(s = c("2020-01-01", "2020-06-15"))

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(tbl_gpu(df), d = as.Date(s)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, d = as.Date(s)))
})

test_that("as.Date() on a literal string constant still constant-folds (no fallback)", {
  skip_if_no_gpu()

  # Registering "as.Date" in ir_call_registry intercepts EVERY call
  # (including a purely literal one) before the old "not in registry ->
  # constant-fold" path ever gets a chance -- the "lit" branch of its parse
  # hook must reproduce that exact behavior itself.
  df <- tibble::tibble(x = 1:3)

  withr::local_options(cuplyr.fallback = "error")
  result <- dplyr::mutate(tbl_gpu(df), d = as.Date("2020-01-01"))
  oracle <- dplyr::mutate(df, d = as.Date("2020-01-01"))
  expect_equal(collect(result), oracle)
})

# -----------------------------------------------------------------------------
# Literal round-trip precision (build_scalar()/emplace_literal())
# -----------------------------------------------------------------------------

test_that("POSIXct literal comparisons preserve sub-second precision", {
  skip_if_no_gpu()

  df <- data.frame(dt = as.POSIXct(c("2020-01-01 00:00:00.5", "2020-01-01 00:00:01.5"),
                                    tz = "UTC"))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, dt < as.POSIXct("2020-01-01 00:00:01.0", tz = "UTC"))
  })
})

# -----------------------------------------------------------------------------
# Phase 11 L6: lubridate datetime accessors (year()/month()/day()/hour()/
# minute()/second()/quarter()/yday()/wday()) native in filter()/mutate().
#
# Every test in this section skips outright when lubridate isn't installed
# (it's a Suggests-only soft dependency, per DESCRIPTION -- the namespace
# guard, ir_require_lubridate_fn() in R/ir.R, makes every one of these
# functions fall back to L1's CPU path when lubridate is unavailable, which
# would itself reproduce dplyr's own "could not find function" error since
# none of these are base R -- not exercised here, since there is nothing
# GPU-native to pin without lubridate actually present).
# -----------------------------------------------------------------------------

skip_if_no_lubridate <- function() {
  testthat::skip_if_not_installed("lubridate")
}

test_that("year()/month()/day()/quarter()/yday() match lubridate on a Date column", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-02", "2024-06-15",
                                      "2024-12-31", NA)))

  pipeline <- function(tbl) {
    dplyr::mutate(
      tbl,
      y = lubridate::year(d),
      m = lubridate::month(d),
      da = lubridate::day(d),
      q = lubridate::quarter(d),
      yd = lubridate::yday(d)
    )
  }
  expect_same_as_dplyr(df, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = TRUE)
})

test_that("hour()/minute()/second() match lubridate on a POSIXct column", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(dt = as.POSIXct(
    c("2024-01-01 13:45:30", "2024-06-15 23:59:59", "2024-12-31 00:00:01", NA),
    tz = "UTC"
  ))

  pipeline <- function(tbl) {
    dplyr::mutate(tbl, h = lubridate::hour(dt), mi = lubridate::minute(dt),
                  se = lubridate::second(dt))
  }
  expect_same_as_dplyr(df, pipeline, ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, pipeline, ignore_col_types = TRUE)
})

test_that("wday() matches lubridate's default (Sun=1..Sat=7) on both Date and POSIXct", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  # Verified empirically first (plain Rscript, no GPU): with
  # lubridate.week.start at its own default (7, Sunday),
  # lubridate::wday(as.Date(c("2024-01-01", ..., "2024-01-07"))) (a
  # Mon..Sun run) is 2,3,4,5,6,7,1 -- i.e. Sun=1, Mon=2, ..., Sat=7. cuDF's
  # own WEEKDAY component is ISO Mon=1..Sun=7; the C++ handler's
  # `(w %% 7) + 1` remap (src/expr_eval.hpp) is what's under test here.
  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-02", "2024-01-03",
                                      "2024-01-04", "2024-01-05", "2024-01-06",
                                      "2024-01-07")))
  oracle_wday <- lubridate::wday(df$d)
  expect_equal(oracle_wday, c(2, 3, 4, 5, 6, 7, 1))

  expect_same_as_dplyr(df, function(tbl) dplyr::mutate(tbl, w = lubridate::wday(d)),
                        ignore_col_types = TRUE)
  expect_same_as_dplyr_lazy(df, function(tbl) dplyr::mutate(tbl, w = lubridate::wday(d)),
                             ignore_col_types = TRUE)
})

test_that("wday() with a non-default week_start falls back to CPU and matches dplyr", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-02", "2024-01-03")))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, w = lubridate::wday(d, week_start = 1)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, w = lubridate::wday(d, week_start = 1)))
})

test_that("wday() falls back when the global lubridate.week.start option is non-default", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  withr::local_options(lubridate.week.start = 1)
  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-01-02", "2024-01-03")))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, w = lubridate::wday(d)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, w = lubridate::wday(d)))
})

test_that("month(label = TRUE) and quarter(with_year = TRUE) fall back to CPU and match dplyr", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(d = as.Date(c("2024-01-15", "2024-06-15")))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, m = lubridate::month(d, label = TRUE)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, m = lubridate::month(d, label = TRUE)))

  expect_warning(
    result2 <- dplyr::mutate(g, q = lubridate::quarter(d, with_year = TRUE)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result2), dplyr::mutate(df, q = lubridate::quarter(d, with_year = TRUE)))
})

test_that("datetime accessors on a STRING/numeric column fall back to CPU and match dplyr", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(s = c("2024-01-01", "2024-06-15"))
  g <- tbl_gpu(df)

  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    result <- dplyr::mutate(g, y = lubridate::year(s)),
    "fell back to CPU evaluation"
  )
  expect_equal(collect(result), dplyr::mutate(df, y = lubridate::year(s)))
})

test_that("a user-defined function shadowing a lubridate accessor name is honored, not misinterpreted", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  # Namespace guard (ir_require_lubridate_fn(), R/ir.R): `year` here resolves
  # to this LOCAL function, not lubridate::year -- must fall back to L1's
  # CPU path (which correctly calls the user's own function), never silently
  # treat this as a native datetime accessor call.
  year <- function(x) "shadowed"
  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-06-15")))
  g <- tbl_gpu(df)

  result <- collect(dplyr::mutate(g, y2 = year(d)))
  oracle <- dplyr::mutate(df, y2 = year(d))
  expect_equal(result, oracle)
  expect_equal(unique(oracle$y2), "shadowed")
})

test_that("year()/month()/etc. nest correctly inside a larger filter() AST", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-03-15", "2024-06-15", "2024-12-31")))

  expect_same_as_dplyr(df, function(tbl) {
    dplyr::filter(tbl, lubridate::year(d) == 2024 & lubridate::month(d) > 3)
  })
})

test_that("second() truncates fractional seconds natively (documented divergence)", {
  skip_if_no_gpu()
  skip_if_no_lubridate()

  # KNOWN, DOCUMENTED DIVERGENCE (see ir_parse_lubridate_accessor()'s own
  # docs, R/ir.R): lubridate::second() returns a FRACTIONAL value for a
  # sub-second-precision POSIXct (verified empirically: second() on
  # 13:45:30.75 is 30.75, not 30). cuDF's SECOND component is always a whole
  # integer. This is a real, accepted divergence for sub-second input,
  # scoped out of native support this wave -- pinned explicitly here rather
  # than silently passing or silently failing.
  dt <- as.POSIXct("2024-01-01 13:45:30.75", tz = "UTC")
  expect_equal(lubridate::second(dt), 30.75)

  df <- tibble::tibble(dt = dt)
  result <- collect(tbl_gpu(df) |> dplyr::mutate(se = lubridate::second(dt)))
  expect_equal(result$se, 30L)
  expect_false(isTRUE(all.equal(result$se, lubridate::second(dt))))
})
