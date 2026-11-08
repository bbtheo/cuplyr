# F74 acceptance tests: summarise()/count()/tally() on a grouped table whose
# group keys are fixed (`$fixed_keys`: a `.drop = FALSE` table after
# semi_join()/anti_join()/filter()/slice*(), or `.preserve = TRUE` slices).
#
# dplyr 1.2.1 (R/summarise.R, summarise_build()/summarise_cols()): summarise()
# evaluates every expression once per group of group_data(), empty groups
# included, so an emptied group gets one row with the expression's value on a
# zero-length slice: n() 0L, sum() 0 (0L for integers/logicals), mean() NaN,
# min()/max() Inf/-Inf (double, with base R's warning), median()/sd()/var()/
# quantile() NA, n_distinct() 0L, first()/last()/nth() NA of the column's
# type, any() FALSE, all() TRUE. Rows come in group_data() order. The result
# is regrouped with `.drop = group_by_drop_default(.data)`.
#
# The contract checked here: identical values (classes, NaN positions), row
# order, group_vars, .drop and group_data() of the result, eager and lazy, and
# (for the common aggregates) that the GPU computes it: no CPU fallback under
# options(cuplyr.fallback = "error").

f74_df <- data.frame(
  g = c("b", "a", "b", "c", NA, "c", "B", "d", "a", "e"),
  h = c(1L, 2L, 1L, 2L, 1L, NA, 2L, 1L, 1L, 2L),
  f = factor(c("u", "v", "u", NA, "u", "v", "u", "v", "w", "u"),
             levels = c("u", "v", "w", "z")),
  x = c(1.5, NA, 3, 4, 5, 6, -2, 0.5, 7, 8),
  i = c(1L, 2L, 3L, NA, 5L, 6L, 7L, 8L, 9L, 10L),
  l = c(TRUE, FALSE, NA, TRUE, FALSE, TRUE, TRUE, FALSE, NA, TRUE),
  s = c("p", "q", "r", "s", "t", "u", "v", "w", "x", "y"),
  d = as.Date("2020-01-01") + c(0:9),
  stringsAsFactors = FALSE
)
f74_y <- data.frame(g = c("a", "c", "zz"), stringsAsFactors = FALSE)

# y as the same kind of table as d.
f74_as <- function(d, ydf) {
  if (is_tbl_gpu(d)) tbl_gpu(ydf) else tibble::as_tibble(ydf)
}

f74_rows_of <- function(gd) {
  keys <- tibble::as_tibble(gd[setdiff(names(gd), ".rows")])
  attr(keys, ".drop") <- NULL
  list(keys = keys, rows = lapply(gd$.rows, as.integer))
}

# Run `f()` and return its value plus the messages of every warning it raised
# (muffled).
f74_capture <- function(f) {
  msgs <- character()
  value <- withCallingHandlers(f(), warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = msgs)
}

# Eager and lazy against dplyr: values in exact row order, column classes,
# NaN positions, group_vars, .drop, group_data(). Warnings: when `warns` is
# empty neither side may warn; otherwise both must warn and some GPU message
# must contain one of the `warns` strings (dplyr wraps base R's messages in
# "There were N warnings ... The first warning was:", so only the first is
# visible in its text). With `native = TRUE` the GPU pipeline (verb and
# collect) must not fall back to the CPU. Names on result columns (quantile()'s
# "25%") are not compared: no tbl_gpu column carries names.
f74_strict <- function(pipeline, df = f74_df, warns = character(), native = FALSE,
                       modes = c("eager", "lazy")) {
  exp_cap <- f74_capture(function() pipeline(tibble::as_tibble(df)))
  expected <- exp_cap$value
  if (length(warns) == 0L) {
    expect_identical(exp_cap$warnings, character(), info = "dplyr warned unexpectedly")
  } else {
    expect_true(length(exp_cap$warnings) > 0L, info = "dplyr did not warn")
  }
  for (mode in modes) {
    lazy <- identical(mode, "lazy")
    run <- function() {
      res <- pipeline(tbl_gpu(df, lazy = lazy))
      list(res = res, collected = collect(res))
    }
    cap <- if (native) {
      withr::with_options(list(cuplyr.fallback = "error"), f74_capture(run))
    } else {
      f74_capture(run)
    }
    got_tbl <- cap$value$res
    got <- cap$value$collected
    if (length(warns) == 0L) {
      expect_identical(cap$warnings, character(), info = paste(mode, "GPU warned"))
    } else {
      hit <- vapply(warns, function(p) any(grepl(p, cap$warnings, fixed = TRUE)), logical(1))
      expect_true(any(hit), info = paste(mode, "missing warning:", paste(warns, collapse = " | ")))
    }
    expect_identical(dplyr::group_vars(got_tbl), dplyr::group_vars(expected),
                     info = paste(mode, "group_vars"))
    expect_identical(dplyr::group_vars(got), dplyr::group_vars(expected),
                     info = paste(mode, "collected group_vars"))
    if (dplyr::is_grouped_df(expected)) {
      expect_identical(dplyr::group_by_drop_default(got),
                       dplyr::group_by_drop_default(expected),
                       info = paste(mode, ".drop"))
      expect_equal(f74_rows_of(dplyr::group_data(got)),
                   f74_rows_of(dplyr::group_data(expected)),
                   info = paste(mode, "group_data"))
      expect_equal(f74_rows_of(dplyr::group_data(got_tbl)),
                   f74_rows_of(dplyr::group_data(expected)),
                   info = paste(mode, "group_data of the tbl_gpu"))
    }
    g <- dplyr::ungroup(tibble::as_tibble(got))
    e <- dplyr::ungroup(tibble::as_tibble(expected))
    e[] <- lapply(e, function(v) if (is.atomic(v)) unname(v) else v)
    expect_identical(names(g), names(e), info = paste(mode, "names"))
    expect_identical(lapply(g, class), lapply(e, class),
                     info = paste(mode, "column classes"))
    expect_equal(g, e, info = paste(mode, "values"))
    for (nm in intersect(names(e), names(g))) {
      if (is.double(e[[nm]]) && is.double(g[[nm]])) {
        expect_identical(is.nan(g[[nm]]), is.nan(e[[nm]]),
                         info = paste(mode, "NaN positions in", nm))
      }
    }
  }
  invisible(expected)
}

# Table with fixed keys: semi_join() empties groups "b", "B", "d", "e", NA.
f74_semi <- function(d, ...) {
  semi_join(group_by(d, ..., .drop = FALSE), f74_as(d, f74_y), by = "g")
}

# ---------------------------------------------------------------------------
# A. The tables under test really carry fixed keys and empty groups
# ---------------------------------------------------------------------------

test_that("the semi_join() prefix keeps empty groups (precondition)", {
  skip_if_no_gpu()
  expected <- f74_semi(tibble::as_tibble(f74_df), g)
  expect_true(any(dplyr::group_size(expected) == 0L))
  for (lazy in c(FALSE, TRUE)) {
    got <- f74_semi(tbl_gpu(f74_df, lazy = lazy), g)
    expect_equal(f74_rows_of(group_data(got)), f74_rows_of(group_data(expected)))
  }
})

# ---------------------------------------------------------------------------
# B. Common aggregates: exact and native
# ---------------------------------------------------------------------------

test_that("n()/sum()/mean() fill empty groups natively (character key, NA key)", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(n = n(), s = sum(x), m = mean(x),
                                 sn = sum(x, na.rm = TRUE), mn = mean(x, na.rm = TRUE))
  }, native = TRUE)
})

test_that("integer and logical sums, integer mean() on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(si = sum(i), sl = sum(l), mi = mean(i),
                                 si2 = sum(i, na.rm = TRUE))
  }, native = TRUE)
})

test_that("median()/sd()/var()/quantile() are NA on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(md = median(x), sd = sd(x), v = var(x),
                                 vi = var(i), sdn = sd(x, na.rm = TRUE),
                                 q = quantile(x, 0.25, na.rm = TRUE))
  }, native = TRUE)
})

test_that("n_distinct() is 0L on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(nd = n_distinct(x), ni = n_distinct(i, na.rm = TRUE),
                                 ns = n_distinct(s))
  }, native = TRUE)
})

test_that("first()/last()/nth() are a typed NA on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(fx = dplyr::first(x), li = dplyr::last(i), ns = dplyr::nth(s, 1),
                                 fl = dplyr::first(l), n2 = dplyr::nth(x, 2),
                                 fr = dplyr::first(x, na_rm = TRUE))
  }, native = TRUE)
})

test_that("first()/last() of factor and Date columns on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(ff = dplyr::first(f), ld = dplyr::last(d))
  })
})

test_that("any()/all() are FALSE/TRUE on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(a = any(l), b = all(l), a2 = any(l, na.rm = TRUE),
                                 b2 = all(l, na.rm = TRUE))
  }, native = TRUE)
})

test_that("min()/max() give Inf/-Inf doubles on empty groups and warn", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(lo = min(x), hi = max(x, na.rm = TRUE))
  }, warns = c("no non-missing arguments to min", "no non-missing arguments to max"),
  native = TRUE)
})

test_that("integer min()/max() turn double when a group is empty", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(lo = min(i), hi = max(i, na.rm = TRUE), n = n())
  }, warns = c("no non-missing arguments to min", "no non-missing arguments to max"),
  native = TRUE)
})

test_that("expressions inside aggregates and across() on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(a = sum(x * 2), b = mean(i > 3L), c = sum(is.na(x)))
  }, native = TRUE)
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(across(c(x, i), mean))
  }, native = TRUE)
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(across(c(x, i), list(s = sum, m = mean)), n = n())
  }, native = TRUE)
})

test_that("cur_group_id() numbers the empty groups too", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(id = dplyr::cur_group_id(), n = n())
  })
})

test_that("zero-dot summarise() returns every fixed key", {
  skip_if_no_gpu()
  f74_strict(function(d) summarise(f74_semi(d, g)), native = TRUE)
  f74_strict(function(d) summarise(f74_semi(d, g, h), .groups = "keep"), native = TRUE)
})

# ---------------------------------------------------------------------------
# C. count() / tally()
# ---------------------------------------------------------------------------

test_that("count() and tally() give zero counts for the emptied groups", {
  skip_if_no_gpu()
  f74_strict(function(d) count(f74_semi(d, g)), native = TRUE)
  f74_strict(function(d) tally(f74_semi(d, g)), native = TRUE)
  f74_strict(function(d) count(f74_semi(d, g), wt = i), native = TRUE)
  f74_strict(function(d) tally(f74_semi(d, g), wt = x), native = TRUE)
  f74_strict(function(d) count(f74_semi(d, g), sort = TRUE), native = TRUE)
  f74_strict(function(d) tally(f74_semi(d, g, h), sort = TRUE), native = TRUE)
  f74_strict(function(d) count(f74_semi(d, g, h), name = "cnt"), native = TRUE)
})

test_that("count() with extra keys regroups like dplyr", {
  skip_if_no_gpu()
  f74_strict(function(d) count(f74_semi(d, g), h))
  f74_strict(function(d) count(f74_semi(d, g), f, .drop = FALSE))
})

test_that("add_count()/add_tally() on a fixed-keys table are unchanged", {
  skip_if_no_gpu()
  f74_strict(function(d) add_count(f74_semi(d, g)))
  f74_strict(function(d) add_tally(f74_semi(d, g), wt = x))
})

# ---------------------------------------------------------------------------
# D. Grouping of the result: .groups and .drop
# ---------------------------------------------------------------------------

test_that("two keys: every .groups value, and the result's .drop", {
  skip_if_no_gpu()
  for (groups in list(NULL, "drop_last", "keep", "drop")) {
    f74_strict(function(d) {
      f74_semi(d, g, h) |> summarise(n = n(), s = sum(x, na.rm = TRUE), .groups = groups)
    }, native = TRUE)
    f74_strict(function(d) {
      f74_semi(d, h, g) |> summarise(n = n(), m = mean(x), .groups = groups)
    }, native = TRUE)
  }
})

test_that("a summarise result can be summarised again", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g, h) |> summarise(n = n(), s = sum(i, na.rm = TRUE)) |>
      summarise(n = sum(n), k = n(), s = max(s))
  }, native = TRUE)
})

test_that("factor and character keys mixed, factor key with NA", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g, f) |> summarise(n = n(), m = mean(x), .groups = "keep")
  }, native = TRUE)
  f74_strict(function(d) {
    group_by(d, f, .drop = FALSE) |> filter(x > 2) |>
      summarise(n = n(), s = sum(i, na.rm = TRUE))
  }, native = TRUE)
})

test_that(".preserve = TRUE on a .drop = TRUE table: rows for the kept empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    group_by(d, g) |> filter(x > 3.5, .preserve = TRUE) |>
      summarise(n = n(), s = sum(x), .groups = "keep")
  }, native = TRUE)
  f74_strict(function(d) {
    group_by(d, g, h) |> slice(1, .preserve = TRUE) |> filter(i > 4L, .preserve = TRUE) |>
      summarise(n = n(), m = mean(x))
  })
  f74_strict(function(d) {
    group_by(d, g) |> filter(x > 3.5, .preserve = TRUE) |> count()
  }, native = TRUE)
})

# ---------------------------------------------------------------------------
# E. Other verbs that fix keys
# ---------------------------------------------------------------------------

test_that("anti_join(), filter(), filter_out() and the slice family", {
  skip_if_no_gpu()
  agg <- function(t) summarise(t, n = n(), s = sum(x), m = mean(i, na.rm = TRUE))
  f74_strict(function(d) {
    agg(anti_join(group_by(d, g, .drop = FALSE), f74_as(d, f74_y), by = "g"))
  }, native = TRUE)
  f74_strict(function(d) agg(filter(group_by(d, g, .drop = FALSE), i > 4L)), native = TRUE)
  f74_strict(function(d) agg(filter_out(group_by(d, g, .drop = FALSE), i > 4L)), native = TRUE)
  f74_strict(function(d) agg(slice_head(group_by(d, g, .drop = FALSE), n = 0)), native = TRUE)
  f74_strict(function(d) agg(slice_tail(group_by(d, g, h, .drop = FALSE), n = 1)), native = TRUE)
  f74_strict(function(d) agg(slice_min(group_by(d, g, .drop = FALSE), x, n = 1)))
  f74_strict(function(d) agg(slice_max(group_by(d, g, .drop = FALSE), i, n = 1)))
  f74_strict(function(d) agg(slice(group_by(d, g, .drop = FALSE), 2)))
})

test_that("keys carried through mutate()/select()/arrange()/distinct()", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    f74_semi(d, g) |> mutate(z = x * 2) |> arrange(x) |>
      summarise(n = n(), z = sum(z))
  }, native = TRUE)
  f74_strict(function(d) {
    f74_semi(d, g) |> select(g, i) |> distinct() |> summarise(n = n(), s = sum(i))
  }, native = TRUE)
})

test_that("every group emptied: one row per fixed key", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    filter(group_by(d, g, .drop = FALSE), x > 100) |>
      summarise(n = n(), s = sum(x), m = mean(x), si = sum(i))
  }, native = TRUE)
  f74_strict(function(d) {
    filter(group_by(d, g, .drop = FALSE), x > 100) |> count()
  }, native = TRUE)
})

test_that("lazy key capture with no group lost stays native and exact", {
  skip_if_no_gpu()
  y_all <- data.frame(g = unique(f74_df$g), stringsAsFactors = FALSE)
  f74_strict(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f74_as(d, y_all), by = "g") |>
      summarise(n = n(), lo = min(i), m = mean(x))
  }, native = TRUE)
})

test_that("a lazy pipeline with pending ops before and after the key capture", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    d |> mutate(x2 = x + 1) |> group_by(g, .drop = FALSE) |> filter(i > 6L) |>
      mutate(x3 = x2 * i) |> summarise(n = n(), s = sum(x3), m = max(x2, na.rm = TRUE))
  }, warns = "no non-missing arguments to max", native = TRUE)
})

# ---------------------------------------------------------------------------
# F. Shapes that need dplyr: still exact
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# F2. Timestamps (round 1 review: B1, N3, N4)
# ---------------------------------------------------------------------------
# dplyr: min()/max() of an empty Date/POSIXct is Inf/-Inf of that class (plus
# the warning), mean() is a NaN Date. A GPU timestamp column can hold neither
# (every route re-uploads the result), so cuplyr gives NA there: a documented
# divergence, but never a garbage date (INT_MIN days, LLONG_MIN micros).
# median()/first()/last()/nth() of an empty Date/POSIXct are a typed NA in
# dplyr, which the GPU gives exactly.

f74_ts_df <- data.frame(
  g = c("a", "a", "a", "b", "c", "c", "d"),
  f = factor(c("u", "u", "u", "w", "w", "w", "u"), levels = c("u", "v", "w")),
  d = as.Date("2020-01-01") + c(0, 5, 9, 3, 2, 8, 7),
  t = as.POSIXct("2020-01-01 10:00:00", tz = "America/New_York") + c(0, 60, 3600, 5, 7, 9, 11),
  x = c(1, 2, 3, 4, 5, 6, 7),
  stringsAsFactors = FALSE
)
f74_ts_y <- data.frame(g = c("a", "c"), stringsAsFactors = FALSE)

# `pipeline` on f74_ts_df: rows whose key is in `empty_keys` (a column name and
# its values) must be NA in `ts_cols`; everything else, classes and zones
# included, equals dplyr. Native, eager and lazy, warns.
f74_ts_minmax <- function(pipeline, key, empty_keys, ts_cols) {
  expected <- suppressWarnings(pipeline(tibble::as_tibble(f74_ts_df)))
  e <- dplyr::ungroup(tibble::as_tibble(expected))
  empty <- as.character(e[[key]]) %in% empty_keys
  expect_true(any(empty))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    cap <- withr::with_options(list(cuplyr.fallback = "error"), f74_capture(function() {
      collect(pipeline(tbl_gpu(f74_ts_df, lazy = lazy)))
    }))
    g <- dplyr::ungroup(tibble::as_tibble(cap$value))
    expect_true(any(grepl("no non-missing arguments", cap$warnings, fixed = TRUE)), info = label)
    expect_identical(names(g), names(e), info = label)
    expect_identical(lapply(g, class), lapply(e, class), info = paste(label, "classes"))
    for (nm in ts_cols) {
      expect_identical(attr(g[[nm]], "tzone"), attr(e[[nm]], "tzone"), info = paste(label, nm, "tzone"))
      expect_true(all(is.na(unclass(g[[nm]])[empty])), info = paste(label, nm, "empty groups NA"))
      expect_equal(g[[nm]][!empty], e[[nm]][!empty], info = paste(label, nm, "observed groups"))
    }
    other <- setdiff(names(e), ts_cols)
    expect_equal(g[other], e[other], info = paste(label, "other columns"))
  }
}

test_that("Date/POSIXct min()/max() of an empty group are NA, never a garbage date", {
  skip_if_no_gpu()
  # fixed keys (character key after semi_join())
  f74_ts_minmax(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f74_as(d, f74_ts_y), by = "g") |>
      summarise(lo = min(d), hi = max(t), n = n())
  }, key = "g", empty_keys = c("b", "d"), ts_cols = c("lo", "hi"))
  # F33 factor path (no fixed keys): unchanged from before F74
  f74_ts_minmax(function(d) {
    group_by(d, f, .drop = FALSE) |> summarise(lo = min(d), hi = max(t), n = n())
  }, key = "f", empty_keys = "v", ts_cols = c("lo", "hi"))
})

test_that("an uploaded non-finite Date/POSIXct is NA on the GPU", {
  skip_if_no_gpu()
  df <- data.frame(
    d = structure(c(Inf, 18262, -Inf, NA), class = "Date"),
    t = structure(c(-Inf, 1577872800, Inf, NA), class = c("POSIXct", "POSIXt"), tzone = "UTC")
  )
  expect_warning(g <- tbl_gpu(df), class = "cuplyr_warning_infinite_timestamp")
  got <- collect(g)
  expect_identical(is.na(unclass(got$d)), c(TRUE, FALSE, TRUE, TRUE))
  expect_identical(unclass(got$d)[2], 18262)
  expect_identical(is.na(unclass(got$t)), c(TRUE, FALSE, TRUE, TRUE))
  expect_equal(unclass(got$t)[2], 1577872800, ignore_attr = TRUE)
})

test_that("Date/POSIXct median()/first()/last()/n_distinct() on empty groups natively", {
  skip_if_no_gpu()
  f74_strict(function(d) {
    semi_join(group_by(d, g, .drop = FALSE), f74_as(d, f74_ts_y), by = "g") |>
      summarise(md = median(d), mt = median(t), fd = dplyr::first(d), lt = dplyr::last(t),
                nt = dplyr::nth(t, 2), nd = n_distinct(d), nt2 = n_distinct(t))
  }, df = f74_ts_df, native = TRUE)
  f74_strict(function(d) {
    group_by(d, f, .drop = FALSE) |>
      summarise(md = median(d), fd = dplyr::first(d), lt = dplyr::last(t))
  }, df = f74_ts_df, native = TRUE)
})

test_that("min() of a character column on an empty factor level (F33 path)", {
  skip_if_no_gpu()
  f74_strict(function(d) group_by(d, f, .drop = FALSE) |> summarise(lo = min(g), hi = max(g)),
             df = f74_ts_df, warns = "no non-missing arguments")
})

test_that("list() and multi-row results on empty groups", {
  skip_if_no_gpu()
  f74_strict(function(d) f74_semi(d, g) |> summarise(v = list(x), n = n()))
  f74_strict(function(d) f74_semi(d, g) |> reframe(v = range(i)),
             warns = "no non-missing arguments")
})

test_that("an aggregate named like a key", {
  skip_if_no_gpu()
  f74_strict(function(d) f74_semi(d, g, h) |> summarise(h = sum(i), n = n()))
})

test_that("min() of a character column is NA on empty groups, with R's warning", {
  skip_if_no_gpu()
  # dplyr 1.2.1 / R 4.6: min(character()) is NA_character_ with the warning
  # "no non-missing arguments, returning NA".
  f74_strict(function(d) f74_semi(d, g) |> summarise(lo = min(s), hi = max(s)),
             warns = "no non-missing arguments")
})

test_that("more native shapes: rowwise output, double and POSIXct keys, window arguments", {
  skip_if_no_gpu()
  f74_strict(function(d) f74_semi(d, g) |> summarise(n = n(), s = sum(x), .groups = "rowwise"),
             native = TRUE)
  kd <- data.frame(k = c(1.5, NaN, NA, 1.5, 2), x = c(1, 2, 3, 4, 5))
  f74_strict(function(d) {
    filter(group_by(d, k, .drop = FALSE), x > 3) |> summarise(n = n(), s = sum(x))
  }, df = kd, native = TRUE)
  f74_strict(function(d) {
    filter(group_by(d, t, .drop = FALSE), x > 4) |> summarise(n = n(), s = sum(x))
  }, df = f74_ts_df, native = TRUE)
  f74_strict(function(d) {
    f74_semi(d, g) |> summarise(a = sum(x - dplyr::lag(x, default = 0), na.rm = TRUE),
                                 b = dplyr::last(cumsum(i)), c = mean(x - mean(x)),
                                 e = max(dplyr::row_number()))
  }, warns = "no non-missing arguments to max", native = TRUE)
})

# ---------------------------------------------------------------------------
# G. Larger data: GPU sort order of many keys, C-locale strings
# ---------------------------------------------------------------------------

test_that("many groups, half emptied, match dplyr natively", {
  skip_if_no_gpu()
  set.seed(74)
  n <- 20000
  k <- sample(c(sprintf("k%04d", 1:1500), sprintf("K%04d", 1:500), NA), n, replace = TRUE)
  big <- data.frame(k = k, x = round(runif(n), 3), i = sample.int(50L, n, replace = TRUE),
                    stringsAsFactors = FALSE)
  yk <- data.frame(k = sprintf("k%04d", seq(1, 1500, by = 2)), stringsAsFactors = FALSE)
  f74_strict(function(d) {
    semi_join(group_by(d, k, .drop = FALSE), f74_as(d, yk), by = "k") |>
      summarise(n = n(), s = sum(i), m = mean(x), nd = n_distinct(i))
  }, df = big, native = TRUE)
  f74_strict(function(d) {
    semi_join(group_by(d, k, .drop = FALSE), f74_as(d, yk), by = "k") |> count()
  }, df = big, native = TRUE)
})
