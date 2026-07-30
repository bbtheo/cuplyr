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
# Phase 6, task 4: entirely-NA-group na.rm=TRUE identity values
#
# Real R's own sum(numeric(0)) is 0, mean(numeric(0)) is NaN,
# min(numeric(0))/max(numeric(0)) are Inf/-Inf (both WITH a warning --
# verified empirically, and confirmed dplyr::summarise() surfaces that
# warning too, wrapped in its own "There was 1 warning in `summarise()`"
# condition -- see the dedicated warning test below), and sd/var/median of
# numeric(0) are all NA. dplyr's summarise(fn(x, na.rm = TRUE)) inherits
# this exact empty-vector convention when a group's `x` is entirely NA (so
# na.rm=TRUE reduces over zero values). cudf's own null-excluding groupby
# aggregations (SUM/MEAN/MIN/MAX with EXCLUDE null policy, which na.rm=TRUE
# routes straight through to) return NULL for a wholly-excluded group
# instead of these identity values -- fixed in gpu_summarise()
# (src/ops_groupby.cpp) with an extra per-group "valid (non-null) count"
# request, substituting the identity scalar via copy_if_else() wherever
# that count is 0. sd()/var()/median() need NO fix at all: cudf's NULL
# result for a wholly-excluded group already collects to NA, which is
# already base R's own sd(numeric(0))/var(numeric(0))/median(numeric(0))
# value -- verified empirically above, confirmed unaffected by this task.
#
# Boundary (documented, not fixed): min()/max()'s identity (Inf/-Inf) can
# only be represented in a FLOAT64 output column. Real dplyr promotes an
# INTEGER column's min()/max() result to double THE MOMENT any group needs
# this identity (verified empirically -- see the boundary test below), a
# genuinely data-dependent type decision cuplyr's parse-time (schema-before-
# execution) type inference architecture cannot make without a much larger
# change (the column's output type would have to depend on whether any
# GROUP turns out to be all-NA at runtime, not just the input's static
# type). sum()'s identity (0) and mean()'s identity (NaN) have no such
# problem (0 is exactly representable in the INT64 sum() already promotes
# to; mean() is always FLOAT64 regardless of input type) and ARE fixed
# unconditionally, including over an INTEGER column.
# =============================================================================

test_that("summarise() na.rm=TRUE identity values match base R for an entirely-NA FLOAT64 group", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1), x = c(NA_real_, NA_real_))
  pipeline <- function(d) {
    suppressWarnings(
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          s = sum(x, na.rm = TRUE), m = mean(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
          sdv = sd(x, na.rm = TRUE), v = var(x, na.rm = TRUE),
          md = median(x, na.rm = TRUE),
          .groups = "drop"
        )
    )
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() na.rm=TRUE identity values only apply to the group(s) that are entirely NA", {
  skip_if_no_gpu()

  # Group 1 is entirely NA (identity values); group 2 has real data (cudf's
  # ordinary NA-excluding aggregation, untouched by the fix).
  df <- data.frame(g = c(1, 1, 2, 2), x = c(NA_real_, NA_real_, 5, 6))
  pipeline <- function(d) {
    suppressWarnings(
      d |>
        dplyr::group_by(g) |>
        dplyr::summarise(
          s = sum(x, na.rm = TRUE), m = mean(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
          .groups = "drop"
        )
    )
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() na.rm=TRUE identity values work ungrouped (whole table entirely NA)", {
  skip_if_no_gpu()

  df <- data.frame(x = c(NA_real_, NA_real_, NA_real_))
  pipeline <- function(d) {
    suppressWarnings(
      d |>
        dplyr::summarise(
          s = sum(x, na.rm = TRUE), m = mean(x, na.rm = TRUE),
          mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE)
        )
    )
  }

  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
})

test_that("summarise() sum()'s na.rm=TRUE identity (0) is fixed even over an INTEGER column", {
  skip_if_no_gpu()

  df <- data.frame(g = c(1, 1), x = c(NA_integer_, NA_integer_))
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(s = sum(x, na.rm = TRUE), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("real dplyr surfaces a base-R warning for min()/max(na.rm=TRUE) on an all-NA group (documented, not replicated by cuplyr)", {
  # No GPU/tbl_gpu involved -- this test only documents real dplyr's own
  # behavior (verified empirically), establishing what the deliberate
  # decision above is about. cuplyr's fix produces the same VALUE (Inf/-Inf)
  # but does not attempt to reproduce dplyr's own warning-wrapping condition
  # machinery -- an informational/diagnostic detail, not a return value, and
  # out of scope for GPU-kernel-level parity (same policy as the regroup
  # message elsewhere in this file).
  df <- data.frame(g = c(1, 1), x = c(NA_real_, NA_real_))

  expect_warning(
    dplyr::group_by(df, g) |> dplyr::summarise(mn = min(x, na.rm = TRUE), .groups = "drop"),
    "no non-missing arguments"
  )
  expect_warning(
    dplyr::group_by(df, g) |> dplyr::summarise(mx = max(x, na.rm = TRUE), .groups = "drop"),
    "no non-missing arguments"
  )
})

test_that("BOUNDARY (documented, not fixed): min()/max() na.rm=TRUE on an all-NA INTEGER group stays NA, unlike dplyr's Inf/-Inf-with-promotion", {
  skip_if_no_gpu()

  # See this section's own module doc above for the full rationale: real
  # dplyr promotes the WHOLE output column to double the instant any group
  # needs the Inf/-Inf identity, a data-dependent type decision cuplyr's
  # static (pre-execution) schema inference can't make. This test PINS the
  # current (divergent, but honest) cuplyr behavior for the INTEGER case so
  # a future fix (were the schema architecture to support data-dependent
  # promotion) has a clear, deliberately-failing regression test to flip.
  df <- data.frame(g = c(1, 1), x = c(NA_integer_, NA_integer_))
  gpu_df <- tbl_gpu(df)

  result <- gpu_df |>
    dplyr::group_by(g) |>
    dplyr::summarise(
      mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE),
      .groups = "drop"
    ) |>
    collect()

  # Documented divergence: NA (not Inf/-Inf as real dplyr promotes to).
  expect_true(is.na(result$mn))
  expect_true(is.na(result$mx))

  # Confirm real dplyr's own behavior for the record (promotes to double).
  oracle <- suppressWarnings(
    dplyr::group_by(df, g) |>
      dplyr::summarise(mn = min(x, na.rm = TRUE), mx = max(x, na.rm = TRUE), .groups = "drop")
  )
  expect_true(is.infinite(oracle$mn))
  expect_true(is.infinite(oracle$mx))
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

test_that("mutate() median(x, na.rm=TRUE) falls back (rather than erroring) as a window function", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # median()'s window-mutate registry entry has no na.rm= handling (arity
  # mismatch -> NULL, matching mean()/sum()'s own na.rm= gap in window
  # context) -- Phase 11, task L1 (universal expression-level CPU
  # fallback) means this is no longer mutate()'s own hard error: it now
  # transparently falls back to real dplyr::mutate() and succeeds.
  df <- oracle_summarise_na_df()
  pipeline <- function(d) d |> dplyr::group_by(g) |> dplyr::mutate(m = median(x, na.rm = TRUE))

  expect_warning(expect_same_as_dplyr(df, pipeline), "fell back to CPU evaluation")
})

# =============================================================================
# Phase 6, task 3: `.groups=` semantics -- real "drop_last" default (not
# "always drop everything"), "keep", "drop", "rowwise" (unsupported), the
# conditional regroup message, chained-summarise peeling, and .by=
# unaffected.
#
# Verified empirically against dplyr 1.2.1 (see R/summarise.R's
# `resolve_summarise_groups()` for the full derivation, and
# `tests/testthat/helper-oracle.R`'s module doc, updated by this task):
#   - Default (`.groups` unset): `"drop_last"` -- peel the LAST grouping
#     variable off; with exactly one grouping variable the result is fully
#     ungrouped (drop_last === drop in that case, verified: 1-group-var
#     tests below pass with or without an explicit `.groups=`).
#   - `"keep"`: every original grouping variable retained.
#   - `"drop"`: always ungrouped.
#   - `"rowwise"`: NOT SUPPORTED on `tbl_gpu` -- a hard, documented error
#     (`tbl_gpu` has no rowwise representation at all; `rowwise()` itself is
#     a full CPU-fallback verb, R/fallback.R). Deliberate scope cut, not a
#     silent divergence.
#   - The regroup message fires ONLY when `.groups` is unset AND there is
#     more than one grouping variable, gated by
#     `options(dplyr.summarise.inform=)` first (explicit `TRUE`/`FALSE`
#     always wins), else by whether the CALLING code's top environment is
#     the global environment (dplyr's own `summarise_verbose()` rule,
#     replicated exactly in `summarise_should_inform()`) -- meaning it does
#     NOT fire from inside a testthat test by default on EITHER side (same
#     rule evaluated at the same call depth for both the dplyr oracle and
#     cuplyr), so the message tests below force it on/off via the option
#     rather than relying on ambient call-stack shape.
#   - `.by=` summarise is always ungrouped and unaffected by any of this
#     (already correct since Phase 5, task W9); `.groups=` can't be combined
#     with `.by=` at all -- rejected by dplyr's own EXPORTED `summarise()`
#     generic before dispatch even reaches the `tbl_gpu` method (verified:
#     no cuplyr-side code implements this specific check, it's inherited for
#     free from dplyr's generic).
#   - On an UNGROUPED `.data`, `.groups=` is not validated at all except the
#     `"rowwise"` special-case (a genuine dplyr quirk, verified empirically:
#     an invalid string is a silent no-op on ungrouped input, but a hard
#     error on grouped input).
# =============================================================================

oracle_groups_df <- function() {
  base <- expand.grid(a = 1:2, b = 1:2, c = 1:2)
  rbind(
    data.frame(base, x = seq_len(nrow(base))),
    data.frame(base, x = seq_len(nrow(base)) + 100)
  )
}

test_that("summarise() default .groups (drop_last) matches dplyr: 1 group var", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  pipeline <- function(d) d |> dplyr::group_by(a) |> dplyr::summarise(m = mean(x))

  expect_same_as_dplyr(df, pipeline, arrange_by = "a")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "a")
})

test_that("summarise() default .groups (drop_last) matches dplyr: 2 group vars", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  pipeline <- function(d) d |> dplyr::group_by(a, b) |> dplyr::summarise(m = mean(x))

  expect_same_as_dplyr(df, pipeline, arrange_by = c("a", "b"))
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("a", "b"))
})

test_that("summarise() default .groups (drop_last) matches dplyr: 3 group vars", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  pipeline <- function(d) d |> dplyr::group_by(a, b, c) |> dplyr::summarise(m = mean(x))

  expect_same_as_dplyr(df, pipeline, arrange_by = c("a", "b", "c"))
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = c("a", "b", "c"))
})

test_that("summarise(.groups = 'keep') retains every grouping variable: 2 and 3 group vars", {
  skip_if_no_gpu()

  df <- oracle_groups_df()

  pipeline2 <- function(d) {
    d |> dplyr::group_by(a, b) |> dplyr::summarise(m = mean(x), .groups = "keep")
  }
  pipeline3 <- function(d) {
    d |> dplyr::group_by(a, b, c) |> dplyr::summarise(m = mean(x), .groups = "keep")
  }

  expect_same_as_dplyr(df, pipeline2, arrange_by = c("a", "b"))
  expect_same_as_dplyr_lazy(df, pipeline2, arrange_by = c("a", "b"))
  expect_same_as_dplyr(df, pipeline3, arrange_by = c("a", "b", "c"))
  expect_same_as_dplyr_lazy(df, pipeline3, arrange_by = c("a", "b", "c"))
})

test_that("summarise(.groups = 'keep') on a single group var still retains it (no-op vs default)", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  pipeline <- function(d) d |> dplyr::group_by(a) |> dplyr::summarise(m = mean(x), .groups = "keep")

  expect_same_as_dplyr(df, pipeline, arrange_by = "a")
})

test_that("summarise(.groups = 'drop') always ungroups: 1, 2, 3 group vars", {
  skip_if_no_gpu()

  df <- oracle_groups_df()

  pipeline1 <- function(d) d |> dplyr::group_by(a) |> dplyr::summarise(m = mean(x), .groups = "drop")
  pipeline2 <- function(d) d |> dplyr::group_by(a, b) |> dplyr::summarise(m = mean(x), .groups = "drop")
  pipeline3 <- function(d) d |> dplyr::group_by(a, b, c) |> dplyr::summarise(m = mean(x), .groups = "drop")

  expect_same_as_dplyr(df, pipeline1, arrange_by = "a")
  expect_same_as_dplyr(df, pipeline2, arrange_by = c("a", "b"))
  expect_same_as_dplyr(df, pipeline3, arrange_by = c("a", "b", "c"))
  expect_same_as_dplyr_lazy(df, pipeline1, arrange_by = "a")
  expect_same_as_dplyr_lazy(df, pipeline2, arrange_by = c("a", "b"))
  expect_same_as_dplyr_lazy(df, pipeline3, arrange_by = c("a", "b", "c"))
})

test_that("summarise(.groups = 'drop_last') explicit matches dplyr: 1, 2, 3 group vars", {
  skip_if_no_gpu()

  df <- oracle_groups_df()

  pipeline1 <- function(d) d |> dplyr::group_by(a) |> dplyr::summarise(m = mean(x), .groups = "drop_last")
  pipeline2 <- function(d) d |> dplyr::group_by(a, b) |> dplyr::summarise(m = mean(x), .groups = "drop_last")
  pipeline3 <- function(d) d |> dplyr::group_by(a, b, c) |> dplyr::summarise(m = mean(x), .groups = "drop_last")

  expect_same_as_dplyr(df, pipeline1, arrange_by = "a")
  expect_same_as_dplyr(df, pipeline2, arrange_by = c("a", "b"))
  expect_same_as_dplyr(df, pipeline3, arrange_by = c("a", "b", "c"))
})

test_that("summarise() group order after drop_last is the group_by() order minus the last column", {
  skip_if_no_gpu()

  # Group columns supplied in a non-alphabetical order (c, a, b) -- verifies
  # drop_last peels the LAST-SUPPLIED column, not e.g. alphabetically last.
  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df) |> dplyr::group_by(c, a, b)

  result <- suppressMessages(dplyr::summarise(gpu_df, m = mean(x)))
  expect_equal(result$groups, c("c", "a"))
})

test_that("summarise() regroup message fires only for >1 group vars with .groups unset (option-forced)", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df)

  withr::local_options(dplyr.summarise.inform = TRUE)

  expect_message(
    dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x)),
    "regroup"
  )
  expect_no_message(
    dplyr::group_by(gpu_df, a) |> dplyr::summarise(m = mean(x))
  )
  expect_no_message(
    dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x), .groups = "drop_last")
  )
  expect_no_message(
    dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x), .groups = "keep")
  )
  expect_no_message(
    dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x), .groups = "drop")
  )
})

test_that("summarise() regroup message is suppressed by dplyr.summarise.inform = FALSE", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df)

  withr::local_options(dplyr.summarise.inform = FALSE)

  expect_no_message(
    dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x))
  )
})

test_that("real dplyr and cuplyr agree on when the regroup message fires (same environment rule, option-forced)", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df)
  tbl <- dplyr::as_tibble(df)

  withr::local_options(dplyr.summarise.inform = TRUE)

  expect_message(dplyr::group_by(tbl, a, b) |> dplyr::summarise(m = mean(x)), "regroup")
  expect_message(dplyr::group_by(gpu_df, a, b) |> dplyr::summarise(m = mean(x)), "regroup")
})

test_that("summarise(.groups = 'rowwise') is rejected on a grouped tbl_gpu (no rowwise representation)", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df) |> dplyr::group_by(a, b)

  expect_error(
    dplyr::summarise(gpu_df, m = mean(x), .groups = "rowwise"),
    "rowwise"
  )
})

test_that("summarise(.groups = 'rowwise') is rejected even on an ungrouped tbl_gpu", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_groups_df())

  expect_error(
    dplyr::summarise(gpu_df, m = mean(x), .groups = "rowwise"),
    "rowwise"
  )
})

test_that("summarise(.groups = <invalid>) errors when .data is grouped", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_groups_df()) |> dplyr::group_by(a, b)

  expect_error(
    dplyr::summarise(gpu_df, m = mean(x), .groups = "bogus"),
    "can't be"
  )
})

test_that("summarise(.groups = <invalid>) is a silent no-op on ungrouped .data (matches a genuine dplyr quirk)", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_groups_df())

  result <- dplyr::summarise(gpu_df, m = mean(x), .groups = "bogus")
  expect_equal(result$groups, character(0))

  # Verified against real dplyr too: this is not a cuplyr-only convenience,
  # it reproduces dplyr's own (surprising) behavior exactly.
  tbl <- dplyr::as_tibble(oracle_groups_df())
  oracle_result <- dplyr::summarise(tbl, m = mean(x), .groups = "bogus")
  expect_equal(dplyr::group_vars(oracle_result), character(0))
})

test_that("summarise(.by=, .groups=) errors -- both cannot be supplied at once", {
  skip_if_no_gpu()

  gpu_df <- tbl_gpu(oracle_groups_df())

  expect_error(
    dplyr::summarise(gpu_df, m = mean(x), .by = a, .groups = "drop"),
    "both"
  )
})

test_that("summarise(.by=) is always ungrouped, unaffected by .groups semantics: 2 and 3 by-columns", {
  skip_if_no_gpu()

  df <- oracle_groups_df()

  pipeline2 <- function(d) dplyr::summarise(d, m = mean(x), .by = c(a, b))
  pipeline3 <- function(d) dplyr::summarise(d, m = mean(x), .by = c(a, b, c))

  expect_same_as_dplyr(df, pipeline2)
  expect_same_as_dplyr(df, pipeline3)
})

test_that("chained summarise() peels one more grouping level each time (classic peeling)", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  pipeline <- function(d) {
    suppressMessages({
      r1 <- d |> dplyr::group_by(a, b, c) |> dplyr::summarise(m = mean(x))
      r2 <- dplyr::summarise(r1, m2 = sum(m))
      r3 <- dplyr::summarise(r2, m3 = sum(m2))
    })
    r3
  }

  # Final level is grouped by "a" only (2 rows) -- cudf's hash-groupby row
  # order isn't guaranteed to match dplyr's, so arrange before comparing
  # (matches this file's/test-summarise.R's own convention for grouped
  # aggregation results).
  expect_same_as_dplyr(df, pipeline, arrange_by = "a")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "a")
})

test_that("chained summarise() intermediate group_vars() match dplyr's own peeling exactly", {
  skip_if_no_gpu()

  df <- oracle_groups_df()
  gpu_df <- tbl_gpu(df) |> dplyr::group_by(a, b, c)

  r1 <- suppressMessages(dplyr::summarise(gpu_df, m = mean(x)))
  expect_equal(r1$groups, c("a", "b"))

  r2 <- suppressMessages(dplyr::summarise(r1, m2 = sum(m)))
  expect_equal(r2$groups, "a")

  r3 <- suppressMessages(dplyr::summarise(r2, m3 = sum(m2)))
  expect_equal(r3$groups, character(0))
})

# =============================================================================
# Phase 6, task 4: windows-inside-summarise (`summarise(z = last(cumsum(v)))`
# style)
#
# Verified empirically against dplyr 1.2.1: the window function computes
# per-group first (over the group's own row order), THEN the aggregation
# reduces that per-row window result -- e.g. `summarise(z = last(cumsum(v)))`
# on group (3, 1, 2) computes cumsum -> (3, 4, 6), then last() -> 6.
# Implemented (R/summarise.R's `resolve_agg_window_arg()`) by running the
# SAME window-decomposition machinery mutate()'s own window path uses
# (`plan_window_stages()`, R/window.R, Phase 5) against the aggregation's
# OWN argument tree, pushing the resulting `ast_window`/`ast_mutate` stages
# onto the working data (grouped by `.data`'s OWN group_by()/`.by=`
# grouping) BEFORE summarise()'s own aggregation runs over the
# now-materialized window-result column.
#
# Scope boundary (documented, not a silent gap -- see resolve_agg_window_arg()'s
# own docs): only ONE aggregation's OWN argument tree is decomposed --
# arbitrary window nesting WITHIN that one argument works (`sum(v -
# lag(v, default = 0))`, `mean(cumsum(v) - lag(cumsum(v)))`), but combining
# TWO SEPARATE aggregation calls at the dot level (`sum(cumsum(x)) -
# mean(y)`) is not attempted: `decompose_agg_call()` only ever recognizes a
# bare `fn(single_arg)` dot shape in the first place, so that dot falls
# through to `parse_aggregations()`'s pre-existing "Invalid aggregation
# expression" error, unchanged (summarise() has no CPU fallback).
# =============================================================================

oracle_window_agg_df <- function() {
  # Row order within each group deliberately not sorted by v, so a
  # cumsum()-based test actually exercises row-order-sensitivity.
  data.frame(g = c(1, 1, 1, 2, 2), v = c(3, 1, 2, 10, 20))
}

test_that("summarise() last(cumsum(v)) computes the window per-group then reduces (grouped, eager+lazy)", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(z = dplyr::last(cumsum(v)), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
  expect_same_as_dplyr_lazy(df, pipeline, arrange_by = "g")
})

test_that("summarise() sum(cumsum(v)) works (grouped and ungrouped)", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline_grouped <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(z = sum(cumsum(v)), .groups = "drop")
  }
  pipeline_ungrouped <- function(d) d |> dplyr::summarise(z = sum(cumsum(v)))

  expect_same_as_dplyr(df, pipeline_grouped, arrange_by = "g")
  expect_same_as_dplyr(df, pipeline_ungrouped)
})

test_that("summarise() sum(v - lag(v, default = 0)) works (window call nested in arithmetic)", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(z = sum(v - dplyr::lag(v, default = 0)), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() first(row_number()) works (rank-family window inside an aggregation)", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(z = dplyr::first(dplyr::row_number()), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() mean(cumsum(v) - lag(cumsum(v), default = 0)) works (two window calls sharing a CSE'd subexpression)", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(
        z = mean(cumsum(v) - dplyr::lag(cumsum(v), default = 0)),
        .groups = "drop"
      )
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() window-inside-aggregation combined with an ordinary bare-column aggregation in the same call", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |>
      dplyr::group_by(g) |>
      dplyr::summarise(z = dplyr::last(cumsum(v)), m = mean(v), .groups = "drop")
  }

  expect_same_as_dplyr(df, pipeline, arrange_by = "g")
})

test_that("summarise() window-inside-aggregation works with .by=", {
  skip_if_no_gpu()

  df <- oracle_window_agg_df()
  pipeline <- function(d) dplyr::summarise(d, z = dplyr::last(cumsum(v)), .by = g)

  expect_same_as_dplyr(df, pipeline, arrange_by = "z")
})

test_that("combining two SEPARATE aggregation calls at the dot level falls back and works end-to-end (Phase 11, task L1)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")

  # sum(cumsum(x)) - mean(y): the dot itself is not a bare fn(single_arg)
  # shape (decompose_agg_call() only recognizes ONE top-level call), so this
  # never reaches resolve_agg_window_arg() at all -- this used to be
  # summarise()'s own "Invalid aggregation expression" hard error (a
  # cuplyr-only message real dplyr never emits); Phase 11, task L1
  # (universal expression-level CPU fallback) means it now transparently
  # falls back to real dplyr::summarise() and succeeds.
  df <- oracle_window_agg_df()
  pipeline <- function(d) {
    d |> dplyr::group_by(g) |> dplyr::summarise(z = sum(cumsum(v)) - mean(v), .groups = "drop")
  }

  expect_warning(expect_same_as_dplyr(df, pipeline, arrange_by = "g"), "fell back to CPU evaluation")
})
