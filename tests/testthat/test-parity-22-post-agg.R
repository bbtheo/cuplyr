# #22: arithmetic around aggregates in summarise() runs on the GPU.
#
# A dot such as `mean(x) + 1`, `sum(x) / n()` or
# `mean(y) - qnorm(.975) * sd(y) / sqrt(n())` combines reducer calls with
# elementwise arithmetic. dplyr evaluates it once per group; cuplyr computes
# the reducers as aggregates and the arithmetic on the one-row-per-group
# result. These tests hold the GPU result to dplyr's values, types, NA/NaN,
# signed zeros, grouping and warnings, eager and lazy, and check that the
# supported shapes run with no CPU fallback (`f124_check(native = TRUE)`, see
# helper-parity-F124.R: values, collected and planned types, group vars,
# warning messages and classes, no fallback). Shapes dplyr rejects or that
# have no GPU form must still give dplyr's result or error.

p22_df <- tibble::tibble(
  g = c(1L, 1L, 2L, 2L, 3L, 3L, 3L),
  x = c(1L, 2L, 3L, NA, 5L, 6L, -7L),
  y = c(1.5, -2, 0, 4, NA, 2.5, 3),
  s = c("a", "b", "a", "c", "c", "c", "d"),
  d = as.Date("2024-01-01") + c(0L, 4L, 10L, 3L, 1L, 7L, 2L)
)

# A frame with no NA in x/y, so the na.rm-free reducers have values everywhere.
p22_full <- tibble::tibble(
  g = c(2L, 1L, 2L, 1L, 3L, 2L),
  x = c(4L, 1L, -3L, 8L, 5L, 2L),
  y = c(0.5, -1.25, 3, 2, 7.5, -0.5)
)

# Exact comparison of two collected results: names, row count, typeof/class,
# NA vs NaN, the sign of a zero, and values (doubles within a relative 1e-9:
# the GPU's mean/sd need not round like R's).
p22_diff <- function(got, want) {
  got <- tibble::as_tibble(dplyr::ungroup(got))
  want <- tibble::as_tibble(dplyr::ungroup(want))
  if (!identical(names(got), names(want))) {
    return(paste("names differ:", toString(names(got)), "vs", toString(names(want))))
  }
  if (nrow(got) != nrow(want)) {
    return(sprintf("row counts differ: %d vs %d", nrow(got), nrow(want)))
  }
  out <- character()
  for (nm in names(want)) {
    a <- got[[nm]]
    b <- want[[nm]]
    if (!identical(typeof(a), typeof(b)) || !identical(class(a), class(b))) {
      out <- c(out, sprintf("%s: type %s/%s vs %s/%s", nm, typeof(a), toString(class(a)),
                            typeof(b), toString(class(b))))
      next
    }
    a <- unclass(a)
    b <- unclass(b)
    if (is.double(a)) {
      both <- !is.na(a) & !is.na(b)
      # (the tolerance only between finite values: an infinity must match exactly)
      close <- both & (a == b | (is.finite(a) & is.finite(b) & abs(a - b) <= 1e-9 * pmax(1, abs(b))))
      close[both & a == 0 & b == 0] <- ((1 / a > 0) == (1 / b > 0))[both & a == 0 & b == 0]
      same <- (is.na(a) & is.na(b) & is.nan(a) == is.nan(b)) | close
    } else {
      same <- (is.na(a) & is.na(b)) | (!is.na(a) & !is.na(b) & a == b)
    }
    same[is.na(same)] <- FALSE
    bad <- which(!same)
    if (length(bad) > 0) {
      k <- utils::head(bad, 5)
      out <- c(out, sprintf("%s: %d of %d rows differ, e.g. row %s: GPU %s, R %s", nm,
                            length(bad), length(a), toString(k),
                            toString(format(a[k], digits = 17)),
                            toString(format(b[k], digits = 17))))
    }
  }
  paste(out, collapse = "; ")
}

# `pipeline` on the tibble (dplyr) and on a tbl_gpu, eager and lazy, compared
# exactly (p22_diff), group vars included. `native = TRUE`: no CPU fallback.
p22_exact <- function(df, pipeline, native = TRUE) {
  want <- pipeline(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    res <- withr::with_options(
      list(cuplyr.fallback = if (native) "error" else "silent"),
      {
        r <- pipeline(tbl_gpu(df, lazy = lazy))
        list(groups = dplyr::group_vars(r), value = collect(r))
      }
    )
    expect_identical(res$groups, dplyr::group_vars(want), info = sprintf("group vars (lazy = %s)", lazy))
    expect_identical(p22_diff(res$value, want), "", info = sprintf("exact result (lazy = %s)", lazy))
  }
}

# dplyr errors: the GPU side gives an error too, eager and lazy, with dplyr's
# message (matched by `regexp`) and never a raw cuDF failure.
p22_same_error <- function(df, pipeline, regexp) {
  expect_error(pipeline(tibble::as_tibble(df)), regexp)
  for (lazy in c(FALSE, TRUE)) {
    err <- tryCatch(collect(pipeline(tbl_gpu(df, lazy = lazy))), error = function(e) e)
    expect_true(inherits(err, "error"), info = sprintf("GPU side errors (lazy = %s)", lazy))
    if (inherits(err, "error")) {
      msg <- cli::ansi_strip(conditionMessage(err))
      expect_false(grepl("CUDF failure", msg, fixed = TRUE), info = msg)
      expect_match(msg, regexp, info = sprintf("error message (lazy = %s)", lazy))
    }
  }
}

# ---------------------------------------------------------------------------
# The issue's examples
# ---------------------------------------------------------------------------

test_that("#22: the issue's three examples run on the GPU, grouped", {
  skip_if_no_gpu()
  f124_check(p22_df, function(d) summarise(group_by(d, g), m = mean(x) + 1))
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()))
  })
  f124_check(p22_df, function(d) summarise(group_by(d, g), avg = sum(x) / n()))
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), m = mean(x) + 1,
              lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()), avg = sum(x) / n())
  })
})

test_that("#22: the issue's three examples run on the GPU, ungrouped and with .by", {
  skip_if_no_gpu()
  f124_check(p22_df, function(d) {
    summarise(d, m = mean(x) + 1, lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()),
              avg = sum(x) / n())
  })
  f124_check(p22_full, function(d) {
    summarise(d, m = mean(x) + 1, lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()),
              avg = sum(x) / n())
  })
  # .by keeps the groups in order of first appearance
  f124_check(p22_full, function(d) {
    summarise(d, m = mean(x) + 1, lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()),
              avg = sum(x) / n(), .by = g)
  })
})

test_that("#22: a lazy composite summarise stays in the plan when no aggregate needs the data", {
  skip_if_no_gpu()
  # double-valued reducers only: nothing to compute at the verb, no plan break
  f124_check(p22_full, function(d) {
    summarise(group_by(d, g), m = mean(y) + 1, lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()),
              r = n() / 2)
  }, lazy_warns_at = "collect")
  lz <- summarise(group_by(tbl_gpu(p22_full, lazy = TRUE), g), m = mean(y) * 2, k = n() + 1L)
  expect_true(has_pending_ops(lz))
  expect_no_error(utils::capture.output(show_query(lz)))
})

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

test_that("#22: result types follow R's arithmetic on the aggregates", {
  skip_if_no_gpu()
  # integer stays integer: sum(int) + 1L, %/%, %%, n() * 2L
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), a = sum(x, na.rm = TRUE) + 1L, b = sum(x) %/% n(),
              c = sum(x) %% 2L, e = n() * 2L, f = n() + n_distinct(s))
  })
  # integer and double mix to double: mean(int) + 1L, sum(int) / n(), n()^2, median()
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), a = mean(x) + 1L, b = sum(x) / n(), c = n()^2,
              e = median(y, na.rm = TRUE) + 1L, f = sqrt(n()))
  })
  # median() of an integer is an integer for an odd count and a double for an
  # even one (data-dependent, so it may stay with dplyr): only the result is held
  p22_exact(p22_df, function(d) summarise(group_by(d, g), e = median(x, na.rm = TRUE) + 1L),
            native = FALSE)
  # logical results and logical reducers
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), b = any(x > 3) & all(y > 0), p = mean(x > 2) * 100,
              q = max(y, na.rm = TRUE) > min(y, na.rm = TRUE))
  })
  # a string result, rounding and absolute values
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), lab = dplyr::if_else(mean(y) > 0, "pos", "neg"),
              r = round(mean(y) * 3, 1), a = abs(min(y)))
  })
  # a Date aggregate plus days is a Date
  f124_check(p22_df, function(d) summarise(group_by(d, g), nx = max(d) + 1, first_d = min(d) - 2L))
})

test_that("#22: na.rm, namespaced reducers and R-evaluable constants", {
  skip_if_no_gpu()
  k <- 2
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), a = mean(x, na.rm = TRUE) + 1, b = sum(y, na.rm = TRUE) / dplyr::n(),
              c = stats::sd(y, na.rm = TRUE) / sqrt(n()), e = base::mean(y) * k,
              f = mean(y) * !!k, h = mean(y) * .env$k, z = max(x, na.rm = TRUE) - qnorm(0.5))
  })
})

test_that("#22: a sum(int) that leaves the integer range is a double, before the arithmetic", {
  skip_if_no_gpu()
  big <- tibble::tibble(g = c(1L, 1L, 2L), x = c(.Machine$integer.max, 1L, 5L))
  f124_check(big, function(d) summarise(group_by(d, g), s = sum(x) + 1L, avg = sum(x) / n()))
  f124_check(big, function(d) summarise(d, s = sum(x) + 1L))
})

test_that("#22: integer arithmetic that overflows is NA with R's warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = 1:2, x = c(.Machine$integer.max, 1L))
  pipeline <- function(d) summarise(group_by(d, g), s = max(x) + 1L, t = n() * 3L)
  expect_warning(pipeline(df), "NAs produced by integer overflow")
  for (lazy in c(FALSE, TRUE)) {
    withr::local_options(cuplyr.fallback = "error")
    expect_warning(res <- collect(pipeline(tbl_gpu(df, lazy = lazy))),
                   "NAs produced by integer overflow", info = sprintf("lazy = %s", lazy))
    want <- suppressWarnings(pipeline(df))
    expect_identical(p22_diff(res, want), "", info = sprintf("lazy = %s", lazy))
  }
})

# ---------------------------------------------------------------------------
# Later dots that read earlier results
# ---------------------------------------------------------------------------

test_that("#22: a later dot can read an earlier result outside a reducer", {
  skip_if_no_gpu()
  f124_check(p22_df, function(d) summarise(group_by(d, g), s = sum(x, na.rm = TRUE), m = s / n()))
  f124_check(p22_df, function(d) summarise(group_by(d, g), n = n(), m = n * 2L))
  # an arithmetic result read by a later arithmetic dot
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), a = mean(y) + 1, b = a * 2, c = b - mean(y))
  })
  # through the .data pronoun, and with .by
  f124_check(p22_full, function(d) {
    summarise(d, tot = sum(y), share = .data$tot / n(), .by = g)
  })
  # an output named like an input column: the later dot reads the result
  f124_check(p22_full, function(d) summarise(group_by(d, g), y = mean(y), z = y + 1))
})

test_that("#22: a name that a later dot assigns is still the caller's variable", {
  skip_if_no_gpu()
  k <- 10
  n <- 100
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), y = c(4, 5, 6))
  # when `m` is evaluated, `k` and `n` are not results yet: dplyr reads the variables
  f124_check(df, function(d) summarise(group_by(d, g), m = mean(x) + k, k = sum(y)), native = FALSE)
  f124_check(df, function(d) summarise(group_by(d, g), share = sum(x) / n, n = n()), native = FALSE)
  f124_check(df, function(d) summarise(d, share = sum(x) / n, n = n(), r = n / 2), native = FALSE)
})

test_that("#22: a reducer passed in with {{ }} or !! keeps its own environment", {
  skip_if_no_gpu()
  k <- 2
  df <- tibble::tibble(x = c(1, 2, 3), g = c(1L, 2L, 1L))
  f <- function(d, stat) {
    k <- 100
    summarise(d, m = {{ stat }} + 1)
  }
  h <- function(d, stat) summarise(group_by(d, g), m = {{ stat }} * 2, s = sum(x) + {{ stat }})
  w_caller <- function(d) {
    w <- 3
    h(d, mean(x * w))
  }
  f124_check(df, function(d) f(d, mean(x * k)), native = FALSE)
  f124_check(df, function(d) w_caller(d), native = FALSE)
  q <- rlang::quo(mean(x * k))
  f124_check(df, function(d) {
    k <- 50
    summarise(d, m = !!q / 2, n = n())
  }, native = FALSE)
})

test_that("#22: a timestamp result read by a later dot keeps its time zone", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L),
                       t = as.POSIXct(c("2024-01-01 10:00:00", "2024-01-02 11:30:00",
                                        "2024-03-01 00:00:00"), tz = "America/New_York"),
                       y = c(1, 2, 3))
  f124_check(df, function(d) summarise(group_by(d, g), mx = max(t), m2 = mx, m = mean(y) + 1))
})

test_that("#22: a factor reducer next to a composite dot gives dplyr's result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), f = factor(c("b", "a", "c")), y = c(1, 2, 3))
  f124_check(df, function(d) summarise(group_by(d, g), f1 = dplyr::first(f), m = mean(y) + 1), native = FALSE)
})

test_that("#22: an earlier result read inside a reducer, or redefined, keeps dplyr's meaning", {
  skip_if_no_gpu()
  # dplyr reduces the one-value earlier result, not the input column
  p22_exact(p22_full, function(d) summarise(group_by(d, g), x = sum(x), t = sum(x) + 1L), native = FALSE)
  p22_exact(p22_full, function(d) summarise(group_by(d, g), y = mean(y) + 1, z = mean(y)), native = FALSE)
  # a name assigned twice
  p22_exact(p22_df, function(d) summarise(group_by(d, g), s = sum(x), s = s + 1L), native = FALSE)
})

# ---------------------------------------------------------------------------
# NA, NaN, empty input and empty groups
# ---------------------------------------------------------------------------

test_that("#22: NA, NaN and NA group keys", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(NA, 1L, NA, 2L, 2L), y = c(1, 2, 3, NaN, 1), x = c(NA, NA, 1L, 2L, 3L))
  f124_check(df, function(d) {
    summarise(group_by(d, g), m = sum(y) / n(), a = mean(y) + 1, b = sum(x) - 1L,
              c = mean(x, na.rm = TRUE) * 2)
  })
})

test_that("#22: zero-row input", {
  skip_if_no_gpu()
  empty <- p22_df[0, ]
  # ungrouped: one row of empty-input values
  f124_check(empty, function(d) {
    summarise(d, avg = sum(x) / n(), m = mean(y) + 1, lo = mean(y) - qnorm(.975) * sd(y) / sqrt(n()),
              s = sum(x) + 1L)
  })
  # grouped: no rows, the types kept
  f124_check(empty, function(d) summarise(group_by(d, g), avg = sum(x) / n(), s = sum(x) + 1L))
})

test_that("#22: empty groups of a .drop = FALSE grouping", {
  skip_if_no_gpu()
  fdf <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")),
                        x = c(1L, 2L, 3L), y = c(1, 2, 3))
  f124_check(fdf, function(d) {
    summarise(group_by(d, f, .drop = FALSE), avg = sum(x) / n(), m = mean(y) + 1,
              sd1 = sd(y) * 2, q = sum(x) %/% n(), r = sum(x) %% n())
  })
  # an empty group's max() is -Inf with dplyr's warning, which names the dot as written
  f124_check(fdf, function(d) summarise(group_by(d, f, .drop = FALSE), mx = max(x) + 1L))
})

test_that("#22: a power next to a reducer follows R's NA rules", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(NA, 1, 2))
  # R: NA^0 and 1^NA are 1
  f124_check(df, function(d) {
    summarise(group_by(d, g), p = mean(y)^0, q = 1^mean(y), .groups = "drop")
  }, native = FALSE)
  k <- 0
  f124_check(df, function(d) summarise(group_by(d, g), p = mean(y)^k, r = sum(y)^NA), native = FALSE)
  # a literal exponent that is neither 0 nor NA keeps R's value on the GPU
  f124_check(df, function(d) summarise(group_by(d, g), p = mean(y)^2, n2 = n()^2, h = sd(y)^0.5))
})

test_that("#22: a power of a -Inf or -0 aggregate follows R's rules", {
  skip_if_no_gpu()
  # max() of an all-NA group (na.rm = TRUE), of an empty group or of zero rows is
  # -Inf: R gives NaN for a non-integer power of it
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(4, 9, NA))
  f124_check(df, function(d) {
    summarise(group_by(d, g), r = max(y, na.rm = TRUE)^0.5, s = max(y, na.rm = TRUE)^-0.5,
              t = max(y, na.rm = TRUE)^(1 / 3), u = max(y, na.rm = TRUE)^-3, .groups = "drop")
  }, native = FALSE)
  fdf <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")), x = c(1, 4, 9))
  f124_check(fdf, function(d) summarise(group_by(d, f, .drop = FALSE), r = max(x)^0.5, .groups = "drop"),
             native = FALSE)
  f124_check(tibble::tibble(y = numeric(0)), function(d) summarise(d, r = max(y)^0.5), native = FALSE)
  # -mean(y) is -0 for a zero mean: R gives +Inf for a negative odd power and +0 for a
  # positive odd one (read through 1 / v)
  dz <- tibble::tibble(g = c(1L, 1L, 2L, 2L), y = c(-1, 1, -0, 2))
  p22_exact(dz, function(d) {
    summarise(group_by(d, g), a = (-mean(y))^-1, b = min(y)^-1, c3 = (-mean(y))^3,
              inv_c3 = 1 / (-mean(y))^3, .groups = "drop")
  }, native = FALSE)
  # an even power is exact for every base, and a count's or a spread's power too
  f124_check(df, function(d) {
    summarise(group_by(d, g), e = max(y, na.rm = TRUE)^2, n3 = n()^0.5, v = var(y)^1.5,
              .groups = "drop")
  })
})

test_that("#22: .groups = \"rowwise\" and fixed group keys with an arithmetic dot", {
  skip_if_no_gpu()
  f124_check(p22_full, function(d) summarise(group_by(d, g), m = mean(y) + 1, .groups = "rowwise"))
  # filter(.preserve = TRUE) keeps the group it empties (g = 1): its row has the empty-group values
  f124_check(p22_full, function(d) {
    d |> group_by(g) |> filter(y > 2.5, .preserve = TRUE) |> summarise(avg = sum(x) / n(), m = mean(y) * 2)
  })
})

test_that("#22: the sign of a zero survives the arithmetic", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = 1:3, y = c(0, -0.5, 2))
  p22_exact(df, function(d) {
    summarise(group_by(d, g), z = mean(y) * 0, neg = -sum(y), m = -mean(y) + 0, w = 0 - max(y))
  })
})

# ---------------------------------------------------------------------------
# Grouping and names
# ---------------------------------------------------------------------------

test_that("#22: .groups, the regroup message and unnamed dots match dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 1, 2), b = c(1, 2, 2), x = 1:3)
  # default drop_last, with dplyr's message on both sides
  # dplyr only informs when the caller's top environment is the global one
  # (summarise_verbose()); the check evaluates the tests in the package
  # namespace's child, so ask for the message
  withr::with_options(list(dplyr.summarise.inform = TRUE), {
    expect_message(summarise(group_by(df, a, b), m = mean(x) + 1), "regrouped")
    expect_message(summarise(group_by(tbl_gpu(df, lazy = FALSE), a, b), m = mean(x) + 1), "regrouped")
  })
  suppressMessages(f124_check(df, function(d) summarise(group_by(d, a, b), m = mean(x) + 1)))
  f124_check(df, function(d) summarise(group_by(d, a, b), m = mean(x) + 1, .groups = "keep"))
  f124_check(df, function(d) summarise(group_by(d, a, b), m = mean(x) + 1, .groups = "drop"))
  # an unnamed dot is named after its expression
  f124_check(df, function(d) summarise(group_by(d, a), mean(x) + 1, sum(x) / n()))
  # composite dots mixed with plain ones keep the dots' order
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), n = n(), m = mean(y) + 1, mx = max(y), avg = sum(x) / n())
  })
})

test_that("#22: a dot that reads no column, next to an aggregate", {
  skip_if_no_gpu()
  f124_check(p22_df, function(d) {
    summarise(group_by(d, g), m = mean(y), k = 1, q = qnorm(.975), lab = "a", i = 2L)
  })
})

test_that("#22: across() of a composite lambda", {
  skip_if_no_gpu()
  f124_check(p22_full, function(d) {
    summarise(group_by(d, g), across(c(x, y), ~ mean(.x) * 2 + 1))
  })
  f124_check(p22_full, function(d) {
    summarise(group_by(d, g), across(c(x, y), list(lo = ~ mean(.x) - sd(.x), hi = ~ mean(.x) + sd(.x))))
  })
})

# ---------------------------------------------------------------------------
# Later verbs and pending ops
# ---------------------------------------------------------------------------

test_that("#22: verbs before and after a composite summarise", {
  skip_if_no_gpu()
  f124_check(p22_full, function(d) {
    d |>
      filter(y > -1) |>
      mutate(w = y * 2) |>
      group_by(g) |>
      summarise(m = mean(w) + 1, avg = sum(x) / n()) |>
      filter(m > 0) |>
      arrange(desc(m))
  })
  # the plan reads one output only: the arithmetic's inputs still compute
  f124_check(p22_full, function(d) {
    d |> group_by(g) |> summarise(m = mean(y) + 1, other = sd(y) * 2) |> select(m)
  })
  f124_check(p22_full, function(d) {
    d |> group_by(g) |> summarise(m = mean(y) + 1, s = sum(y)) |> mutate(r = m / s) |> ungroup()
  })
})

# ---------------------------------------------------------------------------
# Shapes that stay with dplyr, or that dplyr rejects
# ---------------------------------------------------------------------------

test_that("#22: a function of the group's rows around an aggregate keeps dplyr's per-group meaning", {
  skip_if_no_gpu()
  # lag()/cumsum() of the one-value aggregate, within each group: NA and the value
  p22_exact(p22_full, function(d) {
    summarise(group_by(d, g), l = dplyr::lag(mean(y)), c = cumsum(mean(y)), .groups = "drop")
  }, native = FALSE)
  p22_exact(p22_full, function(d) summarise(d, l = dplyr::lag(sum(y)), c = cumsum(n())), native = FALSE)
})

test_that("#22: shapes with no GPU form still give dplyr's result", {
  skip_if_no_gpu()
  # a difference of two Dates is a difftime (here made a number of days)
  p22_exact(p22_df, function(d) summarise(group_by(d, g), span = as.numeric(max(d) - min(d))),
            native = FALSE)
  # a reducer of a one-value argument
  p22_exact(p22_df, function(d) summarise(group_by(d, g), m = sum(x) / sum(n())), native = FALSE)
  # rowwise: each row is its own group
  p22_exact(p22_df, function(d) summarise(rowwise(d), m = mean(y) + 1), native = FALSE)
})

test_that("#22: dplyr's errors for composite dots", {
  skip_if_no_gpu()
  # a bare column (even a group column) beside an aggregate is a group-sized value
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), m = mean(x) + x), "must be size 1")
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), m = mean(x) + g), "must be size 1")
  # a vector constant recycles to two values
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), m = mean(y) + c(1, 2)), "must be size 1")
  # R's own error from the arithmetic
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), m = mean(x) + "a"),
                 "non-numeric argument to binary operator")
  # `&`/`|` of a string aggregate
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), r = dplyr::first(s) & TRUE),
                 "operations are possible only for numeric, logical or complex types")
  p22_same_error(p22_df, function(d) summarise(group_by(d, g), r = dplyr::first(s) | FALSE),
                 "operations are possible only for numeric, logical or complex types")
})
