# Mirrored dplyr-oracle tests for summarise()'s NA semantics (Phase 6, task
# 6.1, Fix A).
#
# Real dplyr's mean()/sum()/min()/max()/sd()/var() all default to
# na.rm = FALSE, so ANY NA in a group makes the WHOLE group's aggregation
# result NA (verified empirically against dplyr 1.2.1 -- see the file-level
# comment in src/ops_window.cpp, which already implements this correctly for
# the window-aggregate path, W4). cudf's own groupby MEAN/SUM/MIN/MAX/STD/
# VARIANCE aggregations always exclude nulls unconditionally, so
# src/ops_groupby.cpp's gpu_summarise() previously returned the NA-excluding
# result unconditionally for every one of these -- a live, silent correctness
# divergence from dplyr for any group containing an NA, in BOTH the grouped
# and ungrouped (whole-table) forms. na.rm = TRUE was also not parseable at
# all before this fix (parse_aggregations() only recognizes a bare
# `fn(col)` shape; `fn(col, na.rm = TRUE)` fell through to its "Invalid
# aggregation expression" error).
#
# n() is unaffected (it counts rows regardless of nulls, matching dplyr
# already).

# =============================================================================
# na.rm = FALSE (default): whole group goes NA when the group has any NA
# =============================================================================

oracle_summarise_na_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 3, 3),
    x = c(1, NA, 3, 4, NA, NA)
  )
}

test_that("summarise() mean/sum/min/max/sd/var honor na.rm=FALSE (grouped, eager)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          m = mean(x), s = sum(x), mn = min(x), mx = max(x),
          sd = sd(x), v = var(x), cnt = dplyr::n(),
          .groups = "drop"
        )
    },
    arrange_by = "g"
  )
})

test_that("summarise() mean/sum/min/max/sd/var honor na.rm=FALSE (grouped, lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()

  expect_same_as_dplyr_lazy(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          m = mean(x), s = sum(x), mn = min(x), mx = max(x),
          sd = sd(x), v = var(x), cnt = dplyr::n(),
          .groups = "drop"
        )
    },
    arrange_by = "g"
  )
})

test_that("summarise() mean/sum/min/max/sd/var honor na.rm=FALSE (ungrouped, eager)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::summarise(
          m = mean(x), s = sum(x), mn = min(x), mx = max(x),
          sd = sd(x), v = var(x), cnt = dplyr::n()
        )
    }
  )
})

test_that("summarise() mean/sum/min/max/sd/var honor na.rm=FALSE (ungrouped, lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()

  expect_same_as_dplyr_lazy(
    df,
    function(d) {
      d |>
        dplyr::summarise(
          m = mean(x), s = sum(x), mn = min(x), mx = max(x),
          sd = sd(x), v = var(x), cnt = dplyr::n()
        )
    }
  )
})

test_that("summarise() with no NAs in any group is unaffected by na.rm=FALSE handling", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 2, 2), x = c(1, 2, 3, 4))

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(m = mean(x), s = sum(x), mn = min(x), mx = max(x), .groups = "drop")
    },
    arrange_by = "g"
  )
})

# =============================================================================
# na.rm = TRUE: excludes NAs, matching cudf's native (and dplyr's opt-in) behavior
# =============================================================================

# Deliberately NOT using oracle_summarise_na_df() here: its group 3 is
# entirely NA, which surfaces a SEPARATE, deeper divergence under
# na.rm = TRUE that this task does not fix (see the "KNOWN GAP" test at the
# bottom of this file) -- every group here has at least one non-NA value,
# isolating the na.rm=TRUE exclusion behavior this task DOES fix.
oracle_summarise_na_rm_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 3, 3),
    x = c(1, NA, 3, 4, 5, NA)
  )
}

test_that("summarise() mean/sum/min/max/sd/var with na.rm=TRUE excludes NAs (grouped, eager)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_rm_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          m = mean(x, na.rm = TRUE), s = sum(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
          .groups = "drop"
        )
    },
    arrange_by = "g"
  )
})

test_that("summarise() mean/sum/min/max/sd/var with na.rm=TRUE excludes NAs (grouped, lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_rm_df()

  expect_same_as_dplyr_lazy(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          m = mean(x, na.rm = TRUE), s = sum(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
          .groups = "drop"
        )
    },
    arrange_by = "g"
  )
})

test_that("summarise() mean/sum/min/max/sd/var with na.rm=TRUE excludes NAs (ungrouped)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_rm_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::summarise(
          m = mean(x, na.rm = TRUE), s = sum(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE)
        )
    }
  )
})

test_that("summarise() na.rm=FALSE is accepted explicitly (no-op vs default)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(m = mean(x, na.rm = FALSE), s = sum(x, na.rm = FALSE), .groups = "drop")
    },
    arrange_by = "g"
  )
})

test_that("na.rm=TRUE and na.rm=FALSE aggregations on the SAME column in one summarise() call don't cross-contaminate", {
  skip_if_no_gpu()

  # Regression test for a real bug caught by this exact scenario while
  # implementing Fix A: gpu_summarise() (src/ops_groupby.cpp) interleaves an
  # extra is_null-indicator request into the SAME `requests` vector for
  # every na.rm=FALSE aggregation that needs whole-group-null propagation,
  # which shifts every LATER aggregation's own main request to a higher
  # request-vector index than its user-facing position `i` -- the first
  # implementation indexed `result_aggs[i]` directly (assuming no shift),
  # so `s_drop`/`m_keep` below silently read a DIFFERENT aggregation's (or
  # the null-indicator's own) result column. Confirmed failing before the
  # `main_request_idx` fix, passing after.
  df <- oracle_summarise_na_rm_df()

  expect_same_as_dplyr(
    df,
    function(d) {
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          s_keep = sum(x), s_drop = sum(x, na.rm = TRUE),
          m_keep = mean(x), m_drop = mean(x, na.rm = TRUE),
          .groups = "drop"
        )
    },
    arrange_by = "g"
  )
})

# =============================================================================
# KNOWN GAP (out of scope for this task, not fixed here): na.rm=TRUE on a
# group that is ENTIRELY NA
# =============================================================================

test_that("KNOWN GAP: na.rm=TRUE sum/mean/min/max on an all-NA group return NA, not base R's 0/NaN/Inf/-Inf", {
  skip_if_no_gpu()

  # Real R's own sum(numeric(0)) is 0, mean(numeric(0)) is NaN, and
  # min(numeric(0))/max(numeric(0)) are Inf/-Inf (with a warning) -- dplyr's
  # summarise(fn(x, na.rm = TRUE)) inherits this exact empty-vector
  # convention when a group's `x` is entirely NA (so na.rm=TRUE reduces
  # over zero values). cudf's own null-excluding groupby aggregations
  # (MEAN/SUM/MIN/MAX with EXCLUDE null policy, which na.rm=TRUE routes
  # straight through to) return NULL for a wholly-excluded group instead --
  # a genuine, separate divergence from a DIFFERENT root cause than this
  # task's whole-group-null (na.rm=FALSE) propagation rule, only reachable
  # now that na.rm=TRUE parsing exists at all. Deliberately left unfixed
  # here (would need an additional per-group "ALL values null" check, with
  # a function-specific identity/Inf patch, for every na.rm=TRUE
  # aggregation) -- tracked in scratchpad/workflow_state.md's "Parked /
  # discovered" section for a future Phase 6 task. This test PINS the
  # current (divergent) cuplyr behavior so a future fix has a clear,
  # deliberately-failing regression test to flip.
  df <- data.frame(g = c(1, 1), x = c(NA_real_, NA_real_))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::group_by(g) |>
    dplyr::summarise(
      s = sum(x, na.rm = TRUE), m = mean(x, na.rm = TRUE),
      mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
      .groups = "drop"
    ) |>
    collect()

  # Current (divergent) cuplyr behavior: NA for all four, not 0/NaN/Inf/-Inf.
  expect_true(is.na(result$s))
  expect_true(is.na(result$m))
  expect_true(is.na(result$mn))
  expect_true(is.na(result$mx))
})
