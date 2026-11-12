# F22 acceptance tests: native reframe() for the common multi-row shapes.
#
# dplyr 1.2.1 contract (verified with real dplyr on plain tibbles):
# - reframe() returns any number of rows per group: the group keys repeated,
#   then each expression's values, size-1 results recycled to the group's
#   common size; rows within a group in the expression's own order.
# - The result is always ungrouped. group_by() groups come out sorted;
#   `.by` groups in first-appearance order. A bare data.frame stays one and
#   data frame attributes are dropped.
# - quantile(x, probs) gives length(probs) rows; range(x) gives 2 (min, max).
#   quantile() with NA and na.rm = FALSE errors; range() gives NA, NA.
# - An empty group (`.drop = FALSE`) evaluates the expressions on an empty
#   slice: quantile -> NA rows, mean -> NaN, range -> Inf/-Inf (+ warnings).
# - A 0-row ungrouped input is one group: quantile -> NA rows, range ->
#   c(Inf, -Inf) with warnings. A 0-row GROUPED input (no groups) errors with
#   vctrs' "Can't recycle `q` (size 2) to size 0." when any result has size
#   > 1, and gives 0 rows when every result has size 1.
# - Mismatched sizes (2 vs 3) error ("Can't recycle").
#
# One documented divergence: dplyr keeps quantile()'s element names on the
# result column ("10%", "90%", ...). A tbl_gpu column can't carry per-element
# names (the CPU fallback's re-upload already drops them), so the oracle side
# is compared with the column names stripped (f22_unname()).
#
# The native shapes run with options(cuplyr.fallback = "error"), in eager and
# lazy mode, so a CPU fallback fails the test. Shapes outside the native
# allowlist must still give dplyr's result through the fallback.

f22_df <- function() {
  data.frame(
    g = c(2, 1, 2, 1, 3, 2, 1, 3),
    h = c("b", "a", "b", "b", "a", "a", "a", "a"),
    x = c(1.5, NA, 3.25, 4, 8, -2, 6.5, 0),
    i = c(7L, 3L, 1L, 9L, 4L, 2L, 5L, 8L),
    s = c("q", "b", "z", "a", "m", "c", "y", "d"),
    dt = as.Date("2024-01-01") + c(5, 2, 9, 0, 3, 7, 1, 4),
    stringsAsFactors = FALSE
  )
}

# Drop per-element names from every column (see the header).
f22_unname <- function(df) {
  for (nm in names(df)) {
    df[[nm]] <- unname(df[[nm]])
  }
  df
}

# Run `expr`, returning its value and whether it raised any warning.
f22_with_warnings <- function(expr) {
  warned <- FALSE
  value <- withCallingHandlers(expr, warning = function(w) {
    warned <<- TRUE
    invokeRestart("muffleWarning")
  })
  list(value = value, warned = warned)
}

# Compare `pipeline` on a tbl_gpu (eager and lazy) against real dplyr on a
# tibble. `native = TRUE` forbids the CPU fallback (also during collect()).
# Checks: values/types/names, ungrouped result, lazy stays lazy with pending
# ops (native only, unless `deferred = FALSE`: a `.drop = FALSE` summarise
# computes its input up front), and that the GPU warns iff dplyr warns.
f22_check <- function(df, pipeline, native = TRUE, deferred = TRUE) {
  oracle <- f22_with_warnings(pipeline(tibble::as_tibble(df)))
  expected <- f22_unname(tibble::as_tibble(oracle$value))

  for (lazy in c(FALSE, TRUE)) {
    mode <- if (native) "error" else "silent"
    got <- withr::with_options(list(cuplyr.fallback = mode), {
      f22_with_warnings({
        res <- pipeline(tbl_gpu(df, lazy = lazy))
        expect_true(is_tbl_gpu(res))
        expect_identical(dplyr::group_vars(res), character(0))
        expect_identical(is_lazy(res), lazy)
        if (native && lazy && deferred) expect_true(has_pending_ops(res))
        collect(res)
      })
    })
    label <- if (lazy) "lazy" else "eager"
    expect_equal(tibble::as_tibble(got$value), expected,
                 info = sprintf("reframe result differs from dplyr (%s)", label))
    expect_identical(got$warned, oracle$warned,
                     info = sprintf("warning presence differs from dplyr (%s)", label))
  }
}

# --- quantile() -------------------------------------------------------------

test_that("quantile() with a literal probs vector is native: ungrouped, group_by, .by", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) reframe(d, q = quantile(x, c(0.1, 0.5, 0.9), na.rm = TRUE)))
  f22_check(df, function(d) reframe(group_by(d, g), q = quantile(i, c(0.25, 0.75))))
  # .by keeps first-appearance group order (2, 1, 3)
  f22_check(df, function(d) reframe(d, q = quantile(i, c(0.1, 0.5, 0.9)), .by = g))
  f22_check(df, function(d) reframe(d, q = quantile(x, c(0, 1), na.rm = TRUE), .by = g))
})

test_that("quantile() argument forms: default probs, named args, env probs, stats::", {
  skip_if_no_gpu()
  df <- f22_df()
  probs <- c(0.2, 0.8)
  f22_check(df, function(d) reframe(group_by(d, g), q = quantile(i)))
  f22_check(df, function(d) reframe(d, q = quantile(i, probs = c(0.3, 0.6), na.rm = FALSE), .by = h))
  f22_check(df, function(d) reframe(d, q = quantile(i, probs), .by = g))
  f22_check(df, function(d) reframe(d, q = stats::quantile(i, c(0.5, 0.95)), .by = g))
  f22_check(df, function(d) reframe(d, q = quantile(i, c(0.1, 0.9), type = 7, names = FALSE)))
  f22_check(df, function(d) reframe(d, q = quantile(i, c(0.5, 0.5)), .by = g))
})

test_that("quantile() keeps R's type-7 values at +-Inf", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), x = c(1, 2, Inf, -Inf, 3, 5))
  f22_check(df, function(d) reframe(d, q = quantile(x, c(0, 0.25, 0.5, 0.75, 1)), .by = g))
  f22_check(df, function(d) reframe(group_by(d, g), q = quantile(x)))
})

test_that("arguments that read no column, and masked function names, match dplyr", {
  skip_if_no_gpu()
  df <- f22_df()
  v <- c(1, 5, 9, 2)
  f22_check(df, function(d) reframe(d, q = quantile(v, c(0.1, 0.9)), .by = g), native = FALSE)
  f22_check(df, function(d) reframe(d, r = range(v), m = mean(i), .by = g), native = FALSE)
  f22_check(df, function(d) reframe(d, q = quantile(i, c(0.1, 0.9)), m = mean(v), .by = g),
            native = FALSE)
  range <- function(x, ...) max(x) - min(x)
  f22_check(df, function(d) reframe(d, r = range(i), .by = g), native = FALSE)
})

test_that("many probabilities of a computed argument give dplyr's result", {
  skip_if_no_gpu()
  set.seed(22)
  df <- data.frame(g = sample(50, 2e4, TRUE), x = runif(2e4))
  f22_check(df, function(d) reframe(group_by(d, g), q = quantile(x * 2, seq(0, 1, 0.01))),
            native = FALSE)
})

test_that("median() keeps R's values at +-Inf", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2, 3, 3, 3), x = c(1, Inf, Inf, -Inf, -Inf, 3, 2, 5, 9))
  f22_check(df, function(d) reframe(d, m = median(x), q = quantile(x, c(0.5, 1)), .by = g))
  f22_check(df, function(d) reframe(group_by(d, g), m = median(x), r = range(x)))
})

test_that("NaN computed on the GPU counts as missing", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2), x = c(0, 1, 2, 3, 4), y = c(0, 1, 1, 1, 1))
  f22_check(df, function(d) reframe(d, q = quantile(x / y, c(0, 1), na.rm = TRUE), .by = g),
            native = FALSE)
  f22_check(df, function(d) {
    reframe(d, m = mean(x / y, na.rm = TRUE), q = quantile(x, c(0.5, 1)), .by = g)
  }, native = FALSE)
  f22_check(df, function(d) {
    d |> mutate(z = x / y) |> group_by(g) |> reframe(q = quantile(z, c(0, 1), na.rm = TRUE))
  }, native = FALSE)
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(reframe(tbl_gpu(df, lazy = lazy), q = quantile(x / y, c(0, 1)), .by = g)),
      "missing values and NaN's not allowed"
    )
    # dplyr gives NaN, NaN for group 1; a missing value (NA) is the accepted
    # NaN-as-NA divergence, a number is not.
    out <- collect(reframe(tbl_gpu(df, lazy = lazy), r = range(x / y), .by = g))
    expect_true(all(is.na(out$r[1:2])))
    expect_equal(out$r[3:4], c(3, 4))
  }
})

test_that("string range() follows the session collation", {
  skip_if_no_gpu()
  withr::local_collate("en_US.UTF-8")
  df <- data.frame(g = c(1, 1, 2, 2), s = c("B", "a", "c", "D"), i = 1:4)
  f22_check(df, function(d) reframe(d, r = range(s), .by = g), native = FALSE)
  f22_check(df, function(d) {
    reframe(group_by(d, g), lo = min(s), hi = max(s), q = quantile(i, c(0.5, 1)))
  }, native = FALSE)
})

test_that("a window-function argument with many probabilities gives dplyr's result", {
  skip_if_no_gpu()
  set.seed(23)
  df <- data.frame(g = sample(100, 2e5, TRUE), x = runif(2e5))
  f22_check(df, function(d) reframe(group_by(d, g), q = quantile(x - mean(x), seq(0, 1, 0.01))),
            native = FALSE)
})

test_that("integer/logical median(), string mean(), string comparisons, window NaN match dplyr", {
  skip_if_no_gpu()
  # median() of an integer keeps the integer type when every group is odd-sized
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2, 2, 2), i = c(5L, 1L, 3L, 2L, 9L, 4L, 4L, 7L),
                   b = c(TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE), s = letters[1:8])
  fb <- function(data, pipeline) f22_check(data, pipeline, native = FALSE)
  fb(df, function(d) reframe(d, m = median(i), q = quantile(i, c(0, 1)), .by = g))
  fb(df[1:7, ], function(d) reframe(d, m = median(i), r = range(i)))
  fb(df, function(d) reframe(d, m = median(b), r = range(i), .by = g))
  fb(df, function(d) reframe(d, m = mean(s), q = quantile(i, c(0, 1)), .by = g))
  # string comparisons follow the session collation
  withr::with_collate("en_US.UTF-8", {
    ds <- data.frame(g = c(1, 1, 2, 2), s = c("a", "c", "B", "D"), i = 1:4)
    fb(ds, function(d) reframe(d, m = sum(s < "B"), q = quantile(i, c(0, 1)), .by = g))
    fb(ds, function(d) reframe(group_by(d, g), r = range(dplyr::if_else(s < "B", i, 0L))))
  })
  # a NaN computed on the GPU inside an order-sensitive window argument
  dn <- data.frame(g = c(1, 1, 1, 2, 2, 2), x = c(1, 0, 3, 0, 2, 1), y = c(1, 0, 1, 0, 1, 1))
  fb(dn, function(d) reframe(d, q = quantile(cummax(x / y), c(0, 1), na.rm = TRUE), .by = g))
})

test_that("quantile() of a computed expression is native", {
  skip_if_no_gpu()
  f22_check(f22_df(), function(d) reframe(d, q = quantile(i * 2, c(0.1, 0.9)), .by = g))
})

test_that("a literal companion column lines up with the quantiles", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) {
    reframe(d, prob = c(0.25, 0.5, 0.75), q = quantile(i, c(0.25, 0.5, 0.75)), .by = g)
  })
  f22_check(df, function(d) {
    reframe(group_by(d, h), stat = c("lo", "hi"), v = range(i), k = 1L)
  })
})

# --- range() ----------------------------------------------------------------

test_that("range() is native for integer, double (NA), string and Date columns", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) reframe(d, r = range(i), .by = g))
  f22_check(df, function(d) reframe(group_by(d, g), r = range(x)))
  f22_check(df, function(d) reframe(group_by(d, g), r = range(x, na.rm = TRUE)))
  f22_check(df, function(d) reframe(d, r = range(s), .by = h))
  f22_check(df, function(d) reframe(d, r = range(dt)))
})

# --- recycling with size-1 expressions ----------------------------------------

test_that("size-1 aggregations are recycled next to multi-row ones", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) {
    reframe(group_by(d, g), n = n(), m = mean(i), q = quantile(i, c(0.1, 0.9)),
            mx = max(x, na.rm = TRUE))
  })
  f22_check(df, function(d) {
    reframe(d, r = range(i), q = quantile(i, c(0.25, 0.75)), s = sum(i), .by = c(g, h))
  })
})

test_that("reframe() of size-1 aggregations only is native (one row per group)", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) reframe(d, x = mean(x, na.rm = TRUE)))
  f22_check(df, function(d) reframe(group_by(d, h, g), m = mean(i), n = n()))
  f22_check(df, function(d) reframe(d, m = median(x, na.rm = TRUE), .by = g))
})

test_that("factor and multiple group keys are repeated with their types", {
  skip_if_no_gpu()
  df <- f22_df()
  df$f <- factor(df$h, levels = c("b", "a"))
  f22_check(df, function(d) reframe(group_by(d, f, g), q = quantile(i, c(0.5, 0.9))))
})

# --- edge cases ---------------------------------------------------------------

test_that("empty groups under .drop = FALSE evaluate on an empty slice", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("c", "a", "c", "a"), levels = c("a", "b", "c")),
    i = c(1L, 2L, 3L, 6L)
  )
  f22_check(df, function(d) {
    reframe(group_by(d, f, .drop = FALSE), q = quantile(i, c(0.1, 0.9)), m = mean(i))
  }, deferred = FALSE)
  f22_check(df, function(d) reframe(group_by(d, f, .drop = FALSE), r = range(i)),
            deferred = FALSE)
  f22_check(df, function(d) reframe(group_by(d, f, .drop = FALSE), m = mean(i)),
            deferred = FALSE)
})

test_that("a 0-row ungrouped input is one group", {
  skip_if_no_gpu()
  df <- f22_df()[0, ]
  f22_check(df, function(d) reframe(d, q = quantile(i, c(0.1, 0.9))))
  f22_check(df, function(d) reframe(d, r = range(i)))
})

test_that("a 0-row grouped input gives 0 rows when every result has size 1", {
  skip_if_no_gpu()
  df <- f22_df()[0, ]
  f22_check(df, function(d) reframe(group_by(d, g), n = n(), m = mean(i)))
})

test_that("a 0-row grouped input with a multi-row result errors like dplyr", {
  skip_if_no_gpu()
  df <- f22_df()[0, ]
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(reframe(group_by(tbl_gpu(df, lazy = lazy), g), q = quantile(i, c(0.1, 0.9)))),
      class = "vctrs_error_incompatible_size"
    )
    # (range() of the empty evaluation warns first, in dplyr too)
    suppressWarnings(expect_error(
      collect(reframe(tbl_gpu(df, lazy = lazy), n = n(), r = range(i), .by = g)),
      "`r` \\(size 2\\) to size 0"
    ))
  }
})

test_that("the result drops attributes and keeps a bare data.frame bare", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f22_df()
  attr(df, "foo") <- "bar"
  for (lazy in c(FALSE, TRUE)) {
    out <- collect(reframe(tbl_gpu(df, lazy = lazy), q = quantile(i, c(0.5, 0.9)), .by = g))
    expect_s3_class(out, "data.frame", exact = TRUE)
    expect_null(attr(out, "foo"))
    out <- collect(reframe(group_by(tbl_gpu(tibble::as_tibble(df), lazy = lazy), g),
                           r = range(i)))
    expect_s3_class(out, class(tibble::tibble()), exact = TRUE)
    expect_null(attr(out, "foo"))
  }
})

# --- errors -------------------------------------------------------------------

test_that("quantile() with NA and na.rm = FALSE errors", {
  skip_if_no_gpu()
  df <- f22_df()
  expect_error(reframe(tibble::as_tibble(df), q = quantile(x, c(0.1, 0.9)), .by = g),
               "missing values and NaN's not allowed")
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(reframe(tbl_gpu(df, lazy = lazy), q = quantile(x, c(0.1, 0.9)), .by = g)),
      "missing values and NaN's not allowed"
    )
  }
})

test_that("mismatched result sizes error like dplyr", {
  skip_if_no_gpu()
  df <- f22_df()
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(reframe(group_by(tbl_gpu(df, lazy = lazy), g),
                      r = range(i), q = quantile(i, c(0.1, 0.5, 0.9)))),
      "Can't recycle"
    )
    expect_error(
      collect(reframe(tbl_gpu(df, lazy = lazy), p = c(0.1, 0.9), q = quantile(i, c(0.1, 0.5, 0.9)))),
      "Can't recycle"
    )
  }
})

test_that("probs outside [0, 1] error like dplyr", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    expect_error(
      collect(reframe(tbl_gpu(f22_df(), lazy = lazy), q = quantile(i, c(0.1, 1.5)))),
      "outside \\[0,1\\]"
    )
  }
})

# --- lazy pipelines -------------------------------------------------------------

test_that("native reframe() composes with later and earlier lazy verbs", {
  skip_if_no_gpu()
  df <- f22_df()
  f22_check(df, function(d) {
    d |>
      mutate(y = i * 3L) |>
      filter(i > 1L) |>
      group_by(g) |>
      reframe(q = quantile(y, c(0.25, 0.75)), r = range(s)) |>
      filter(q > 5) |>
      select(r, q)
  })
  f22_check(df, function(d) {
    d |>
      reframe(r = range(i), .by = h) |>
      mutate(r2 = r * 2L) |>
      arrange(desc(r2))
  })
})

# --- shapes that stay on the CPU fallback ------------------------------------------

test_that("shapes outside the native allowlist still give dplyr's result", {
  skip_if_no_gpu()
  df <- f22_df()
  fb <- function(pipeline) f22_check(df, pipeline, native = FALSE)
  fb(function(d) reframe(d, u = unique(h), .by = g))
  fb(function(d) reframe(group_by(d, g), w = which(i > 3)))
  fb(function(d) reframe(d, q = quantile(i, c(0.1, 0.9), type = 1), .by = g))
  fb(function(d) reframe(d, r = range(i, x, na.rm = TRUE)))
  fb(function(d) reframe(group_by(d, g), r = range(i), m = mean(r)))
  fb(function(d) reframe(d, quantile(i, c(0.1, 0.9))))
  fb(function(d) reframe(d, q = quantile(i, c(0.1, NA))))
  fb(function(d) reframe(group_by(d, g), g = range(i)))
  fb(function(d) reframe(d, id = dplyr::cur_group_id(), r = range(i), .by = g))
})

# --- Attempt 2: the round-4 blocking findings ---------------------------------

test_that("integer %/% and %% by a fractional double literal give R's values", {
  skip_if_no_gpu()
  # R corrects the floored quotient for 0.1's binary value: 1L %/% 0.1 is 9,
  # 2L %/% 0.1 is 19, 3L %/% 0.1 is 29, 1L %/% 0.2 is 4.
  df <- data.frame(g = c(1, 1, 2), i = c(1L, 2L, 3L))
  f22_check(df, function(d) reframe(d, s = sum(i %/% 0.1), q = quantile(i, c(0, 1)), .by = g),
            native = FALSE)
  f22_check(df, function(d) reframe(d, s = sum(i %/% 0.2), m = sum(i %% 0.2), .by = g),
            native = FALSE)
  f22_check(df, function(d) reframe(group_by(d, g), r = range(i %/% 0.1)), native = FALSE)
  # a whole double divisor is exact and stays native
  f22_check(df, function(d) {
    reframe(d, s = sum(i %/% 2), m = sum(i %% 2), q = quantile(i %/% 2, c(0, 1)), .by = g)
  })
})

test_that("quantile() at a whole type-7 index next to +-Inf takes that element", {
  skip_if_no_gpu()
  # seq(0, 1, 0.1)[8] is 0.7000000000000001: R's index 1 + 10 * p rounds to
  # exactly 8, so R returns x[8] = 7, not an interpolation towards Inf.
  df <- data.frame(g = rep(1:2, each = 11), x = c(0:7, Inf, Inf, Inf, 0:10))
  f22_check(df, function(d) reframe(d, q = quantile(x, seq(0, 1, 0.1)), .by = g))
  f22_check(df, function(d) {
    reframe(group_by(d, g), p = seq(0, 1, 0.1), q = quantile(x, seq(0, 1, 0.1)))
  })
  # the whole index counts only the non-missing values under na.rm = TRUE
  df_na <- data.frame(g = rep(1:2, each = 12), x = c(NA, 0:7, Inf, Inf, Inf, 0:10, NA))
  f22_check(df_na, function(d) {
    reframe(d, q = quantile(x, seq(0, 1, 0.1), na.rm = TRUE), .by = g)
  })
  # 0.99999999999999989 is 1 - 2^-53: (n - 1) * p sits just below 1, R's index
  # rounds up to 2, so R returns the upper element (5, not -Inf)
  df_hi <- data.frame(g = c(1, 1, 2, 2, 2), x = c(-Inf, 5, -Inf, 1, 2))
  f22_check(df_hi, function(d) reframe(d, q = quantile(x, c(0.99999999999999989, 0.5)), .by = g))
})

test_that("summarise() quantile() at a whole type-7 index next to Inf takes that element", {
  skip_if_no_gpu()
  df <- data.frame(g = rep(1:2, each = 11), x = c(0:7, Inf, Inf, Inf, 0:10))
  p <- seq(0, 1, 0.1)[8]
  # dplyr's summarise() keeps quantile()'s "70%" names too; f22_check() strips
  # them (the documented divergence, see the header)
  f22_check(df, function(d) summarise(d, q = quantile(x, p), m = median(x), .by = g))
})

test_that("lazy reframe() with an integer sum() past the integer range gives a double", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), i = c(2147483647L, 1L, 3L))
  # the lazy result is the GPU's, without a CPU fallback
  f22_check(df, function(d) reframe(d, s = sum(i), q = quantile(i, c(0, 1)), .by = g),
            deferred = FALSE)
  f22_check(df, function(d) reframe(d, s = sum(i), q = quantile(i, c(0, 1))), deferred = FALSE)
  f22_check(df, function(d) reframe(group_by(d, g), s = sum(i), n = n(), r = range(i)),
            deferred = FALSE)
  # with pending lazy ops before it
  f22_check(df, function(d) {
    d |> mutate(j = i) |> reframe(s = sum(j), q = quantile(j, c(0, 1)), .by = g)
  }, native = FALSE)
})

test_that("quantile() at a whole type-7 index takes that element next to a far neighbour", {
  skip_if_no_gpu()
  # Same rounding as above without Inf: blending x[8] with a far x[9] by
  # cudf's 1e-16 fraction moves the result off R's exact x[8] (8, 7).
  df <- data.frame(
    g = rep(1:2, each = 11),
    x = c(1:8, 1e9, 2e9, 3e9, 0:7, 1e300, 1e300, 1e300),
    i = c(1:8, 1000000000L, 2000000000L, 2100000000L, 0:7, 2000000000L, 2000000000L, 2000000000L)
  )
  f22_check(df, function(d) {
    reframe(d, q = quantile(x, seq(0, 1, 0.1)), qi = quantile(i, seq(0, 1, 0.1)), .by = g)
  })
  p <- seq(0, 1, 0.1)[8]
  f22_check(df, function(d) summarise(d, q = quantile(x, p), qi = quantile(i, p), .by = g))
})

test_that("lazy reframe() with a computed integer sum() past the integer range gives a double", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), i = c(1500000000L, 1000000000L, 3L))
  f22_check(df, function(d) reframe(d, s = sum(i + 1L), r = range(i), .by = g), deferred = FALSE)
})
