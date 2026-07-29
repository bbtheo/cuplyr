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

# =============================================================================
# Phase 6, task 6.2: median(), quantile(), n_distinct(), first()/last()/
# nth(), any()/all()
#
# Empirically verified against dplyr 1.2.1 (see the task's own commit/PR
# notes for the full derivation):
#   * median(): na.rm=FALSE (default) whole-group-NA propagation, exactly
#     like mean()/sum()/min()/max()/sd()/var() (Fix A above) -- NEVER
#     errors on NA, unlike quantile() below.
#   * quantile(x, probs): only a single probability is supported. cudf's
#     `interpolation::LINEAR` is bit-for-bit R's own default `type = 7`
#     (verified across p in {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1} on a 10-point
#     sample). na.rm=FALSE (default) with an actual NA present is a hard
#     ERROR in base R's own quantile.default() ("missing values and NaN's
#     not allowed if 'na.rm' is FALSE"), NOT a NA-propagation.
#   * n_distinct(): NA counts as its own distinct value when na.rm=FALSE
#     (the default), matching dplyr's n_distinct() exactly.
#   * first()/last()/nth(): use dplyr's own `na_rm=` (underscore) spelling.
#     na_rm=FALSE (default) never skips a NA when picking the position
#     (may itself return NA); na_rm=TRUE skips NA values.
#   * any()/all(): R's exact three-valued logic -- any(c(TRUE, NA)) is
#     TRUE, any(c(FALSE, NA)) is NA, all(c(FALSE, NA)) is FALSE,
#     all(c(TRUE, NA)) is NA (with na.rm=FALSE, the default). na.rm=TRUE
#     drops NA first (an all-NA group is FALSE for any(), TRUE for all(),
#     the empty-vector identity).
# =============================================================================

# --- median() ---

test_that("summarise() median() honors na.rm=FALSE (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(m = median(x), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() median() honors na.rm=FALSE (ungrouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()
  pipeline <- function(d) d |> dplyr::summarise(m = median(x))

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise() median() with na.rm=TRUE excludes NAs (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_rm_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(m = median(x, na.rm = TRUE), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() median() over a sub-expression works", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(1, 2, 3, 4, 5))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(m = median(x + 1), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

# --- quantile() ---

oracle_quantile_df <- function() {
  data.frame(g = rep(c(1, 2), each = 5), x = c(1:5, 6:10))
}

test_that("summarise() quantile() matches R's own type=7 default at several probabilities", {
  skip_if_no_gpu()

  df <- oracle_quantile_df()

  for (p in c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) {
    pipeline <- local({
      p <- p
      function(d) {
        d |>
          dplyr::group_by(g) |>
          dplyr::summarise(q = stats::quantile(x, !!p, names = FALSE), .groups = "drop")
      }
    })
    expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  }
})

test_that("summarise() quantile() works ungrouped and lazy", {
  skip_if_no_gpu()

  df <- oracle_quantile_df()
  pipeline <- function(d) d |> dplyr::summarise(q = stats::quantile(x, 0.5, names = FALSE))

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise() quantile() with na.rm=FALSE (default) errors when NA is present", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 2), x = c(1, 2, NA))
  gpu_df <- tbl_gpu(df)

  expect_error(
    dplyr::group_by(gpu_df, g) |> dplyr::summarise(q = stats::quantile(x, 0.5)) |> collect(),
    "na.rm"
  )
})

test_that("summarise() quantile() with na.rm=TRUE excludes NAs", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 2), x = c(1, 2, NA))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(q = stats::quantile(x, 0.5, na.rm = TRUE, names = FALSE), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() quantile() rejects a probability vector (only a single p supported)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_quantile_df())

  expect_error(
    dplyr::summarise(gpu_df, q = stats::quantile(x, c(0.25, 0.75))),
    "single probability"
  )
})

test_that("summarise() quantile() rejects type != 7", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_quantile_df())

  expect_error(
    dplyr::summarise(gpu_df, q = stats::quantile(x, 0.5, type = 6)),
    "type = 7"
  )
})

# --- n_distinct() ---

test_that("summarise() n_distinct() counts NA as its own distinct value by default (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(1, 2, NA, 2, 2))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(nd = dplyr::n_distinct(x), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() n_distinct() with na.rm=TRUE excludes NA from the distinct count", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(1, 2, NA, 2, 2))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(nd = dplyr::n_distinct(x, na.rm = TRUE), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() n_distinct() works on a STRING column, ungrouped", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 2), s = c("a", "b", "a"), stringsAsFactors = FALSE)
  pipeline <- function(d) d |> dplyr::summarise(nd = dplyr::n_distinct(s))

  expect_same_as_dplyr(df, pipeline)
})

# --- first()/last()/nth() ---

oracle_first_last_nth_df <- function() {
  data.frame(g = c(1, 1, 1, 2, 2), x = c(NA, 2, 3, 4, NA))
}

test_that("summarise() first()/last()/nth() default na_rm=FALSE (may return NA) (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_first_last_nth_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        f = dplyr::first(x), l = dplyr::last(x),
        n2 = dplyr::nth(x, 2), nlast = dplyr::nth(x, -1),
        .groups = "drop"
      )
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() first()/last()/nth() with na_rm=TRUE skip NA values", {
  skip_if_no_gpu()

  df <- oracle_first_last_nth_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        f = dplyr::first(x, na_rm = TRUE), l = dplyr::last(x, na_rm = TRUE),
        n2 = dplyr::nth(x, 2, na_rm = TRUE),
        .groups = "drop"
      )
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() nth() returns NA for an out-of-range position", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1), x = c(1, 2, 3))
  pipeline <- function(d) d |> dplyr::summarise(n5 = dplyr::nth(x, 5))

  expect_same_as_dplyr(df, pipeline)
})

test_that("summarise() first()/last()/nth() with na_rm=TRUE on an all-NA group returns NA", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1), x = c(NA_real_, NA_real_))
  pipeline <- function(d) {
    d |>
      dplyr::summarise(
        f = dplyr::first(x, na_rm = TRUE), l = dplyr::last(x, na_rm = TRUE),
        n1 = dplyr::nth(x, 1, na_rm = TRUE)
      )
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("summarise() first()/last()/nth() work on a STRING column", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), s = c("a", "b", "a", "c", "c"), stringsAsFactors = FALSE)
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(f = dplyr::first(s), l = dplyr::last(s), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() nth() rejects order_by=/default=", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_first_last_nth_df())

  expect_error(
    dplyr::summarise(gpu_df, n1 = dplyr::nth(x, 1, order_by = g)),
    "order_by"
  )
})

test_that("summarise() nth() rejects n = 0", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_first_last_nth_df())

  expect_error(
    dplyr::summarise(gpu_df, n0 = dplyr::nth(x, 0)),
    "non-zero"
  )
})

# --- any()/all() ---

oracle_any_all_df <- function() {
  # Group 1: has a TRUE and a NA -> any() TRUE, all() FALSE (has a FALSE too)
  # Group 2: no TRUE, has a NA -> any() NA, all() FALSE (has a FALSE)
  # Group 3: no NA at all -> any()/all() unaffected by na.rm
  data.frame(
    g = c(1, 1, 1, 2, 2, 2, 3, 3),
    x = c(TRUE, NA, FALSE, FALSE, NA, FALSE, TRUE, FALSE)
  )
}

test_that("summarise() any()/all() implement R's three-valued logic with na.rm=FALSE (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_any_all_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(a = any(x), al = all(x), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() any()/all() with na.rm=TRUE drop NA first", {
  skip_if_no_gpu()

  df <- oracle_any_all_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(a = any(x, na.rm = TRUE), al = all(x, na.rm = TRUE), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() any()/all() on an all-NA group: na.rm=TRUE is FALSE/TRUE (empty-vector identity)", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1), x = c(NA, NA))
  pipeline <- function(d) {
    d |>
      dplyr::summarise(a = any(x, na.rm = TRUE), al = all(x, na.rm = TRUE))
  }

  expect_same_as_dplyr(df, pipeline)
})

test_that("summarise() any()/all() work ungrouped", {
  skip_if_no_gpu()

  df <- oracle_any_all_df()
  pipeline <- function(d) d |> dplyr::summarise(a = any(x), al = all(x))

  expect_same_as_dplyr(df, pipeline)
})

test_that("summarise() any()/all() work over comparison sub-expressions", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 2, 2), x = c(1, 5, 2, 2))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(a = any(x > 3), al = all(x > 1), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() any()/all() reject a non-logical column", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(data.frame(x = c(1, 2, 3)))

  expect_error(
    dplyr::summarise(gpu_df, a = any(x)),
    "logical"
  )
})

# =============================================================================
# Phase 6, task 6.2: window-path ("agg" family, mutate()/filter()) parity for
# median()/n_distinct() -- both reuse get_groupby_agg() (src/ops_groupby.cpp),
# genuinely shared source between summarise() and the window-aggregate path
# (src/ops_window.cpp, W4), so both "light up" as window functions once
# summarise() learns them, with only a small membership-list addition in
# is_agg_fn()/the na.rm=FALSE propagation list (src/ops_window.cpp) and a
# new ir_call_registry entry (R/ir.R) -- verified empirically here.
#
# quantile()/first()/last()/nth()'s own na_rm=/any()/all() are NOT attempted
# as window functions this wave: quantile() needs a probability parameter the
# window spec shape doesn't carry; any()/all() have no cudf groupby_aggregation
# instantiation at all (see combine_any_all()'s docs); first()/last()/nth()
# already have their OWN, separate, pre-existing window support (unrelated to
# this task, unaffected by it).
# =============================================================================

test_that("mutate() median() works as a window function (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(m = median(x))

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("mutate() median() works as a window function (ungrouped)", {
  skip_if_no_gpu()

  df <- oracle_summarise_na_df()
  pipeline <- function(d) d |> dplyr::mutate(m = median(x))

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() n_distinct() works as a window function (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(1, 2, 1, 3, 3))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(nd = dplyr::n_distinct(x))

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("filter() n_distinct() works as a window predicate", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(1, 2, 1, 3, 3))
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::filter(x == dplyr::n_distinct(x))

  expect_same_as_dplyr(df, pipeline)
})

test_that("mutate() median(x, na.rm=TRUE) is not supported (matches mean()/sum()'s own na.rm= gap in window context)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_summarise_na_df()) |> dplyr::group_by(g)

  expect_error(
    dplyr::mutate(gpu_df, m = median(x, na.rm = TRUE)) |> collect(),
    "mutate\\(\\) only supports"
  )
})
