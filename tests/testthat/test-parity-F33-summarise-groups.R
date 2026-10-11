# F33 acceptance tests (tier A): grouped summarise() output order, empty
# groups under .drop = FALSE, and 0-row grouped inputs.
#
# Behaviour pinned with tools/r against dplyr 1.2.1
# (scratchpad/parity/features/F33/probe/p1-p3.R):
# - group_by() + summarise() returns one row per group in group_by()'s sorted
#   order: ascending, NA last, strings in C locale ("B" < "a"), logical
#   FALSE < TRUE < NA, factors in level order. count()/tally() too.
# - summarise(.by = ) keeps FIRST-APPEARANCE order.
# - On a .drop = FALSE grouping with factor keys, every empty group gets a row
#   (recursive expansion, as group_data()), with each expression evaluated on
#   an empty vector: n() = 0L, sum = 0 (0L for integers), mean = NaN,
#   min/max = Inf/-Inf (double, plus a warning), median/var/sd/quantile = NA,
#   n_distinct = 0L, first/last = NA, any = FALSE, all = TRUE. The result
#   keeps .drop = FALSE.
# - A 0-row grouped input gives 0 rows that keep the key columns and types.
#
# Grouped summarise results are compared WITHOUT arrange_by on purpose.
# expect_equal() treats NaN == NA and 1L == 1, so f33_strict() also checks
# column classes and NaN positions.

f33_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

f33_rows_of <- function(gd) {
  keys <- tibble::as_tibble(gd[setdiff(names(gd), ".rows")])
  list(keys = keys, rows = lapply(gd$.rows, as.integer))
}

# Eager and lazy, compared with dplyr: values (exact row order), column
# classes, NaN positions, group_vars, .drop and group_data().
f33_strict <- function(df, pipeline, warnings_ok = FALSE) {
  run <- function(x) {
    if (warnings_ok) suppressWarnings(pipeline(x)) else pipeline(x)
  }
  expected <- run(tibble::as_tibble(df))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    got_tbl <- run(tbl_gpu(df, lazy = lazy))
    got <- collect(got_tbl)
    expect_identical(dplyr::group_vars(got), dplyr::group_vars(expected),
                     info = paste(label, "group_vars"))
    expect_identical(dplyr::group_by_drop_default(got),
                     dplyr::group_by_drop_default(expected),
                     info = paste(label, ".drop"))
    g <- dplyr::ungroup(tibble::as_tibble(got))
    e <- dplyr::ungroup(tibble::as_tibble(expected))
    expect_identical(names(g), names(e), info = paste(label, "names"))
    expect_identical(lapply(g, class), lapply(e, class),
                     info = paste(label, "column classes"))
    expect_equal(g, e, info = paste(label, "values"))
    for (nm in intersect(names(e), names(g))) {
      if (is.double(e[[nm]]) && is.double(g[[nm]])) {
        expect_identical(is.nan(g[[nm]]), is.nan(e[[nm]]),
                         info = paste(label, "NaN positions in", nm))
      }
    }
    if (length(dplyr::group_vars(expected)) > 0) {
      expect_equal(f33_rows_of(dplyr::group_data(got)),
                   f33_rows_of(dplyr::group_data(expected)),
                   info = paste(label, "group_data"))
    }
  }
}

f33_df <- data.frame(
  g = c("c", "b", "a", "b", NA, "B", "a", "c"),
  h = c(2L, 1L, NA, 1L, 3L, 2L, 1L, 1L),
  x = c(1, 2, 3, 4, 5, 6, 7, 8),
  i = c(5L, 3L, 9L, 1L, 2L, 8L, 4L, 6L),
  stringsAsFactors = FALSE
)

# --- A1. sorted group order -------------------------------------------------

test_that("character keys come back in C-locale order with NA last", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(n = n(), m = mean(x), s = sum(i))
  })
})

test_that("factor keys come back in level order, NA last", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("z", "y", "x", "y", NA, "z"), levels = c("z", "x", "y")),
    x = c(1, 2, 3, 4, 5, 6)
  )
  f33_strict(df, function(d) d |> group_by(f) |> summarise(n = n(), m = mean(x)))

  fx <- data.frame(f = factor(c("b", "a", "b"), levels = c("a", "b", "c")), x = 1:3)
  f33_strict(fx, function(d) d |> group_by(f) |> summarise(n = n()))
})

test_that("logical keys come back FALSE, TRUE, NA", {
  skip_if_no_gpu()
  df <- data.frame(l = c(TRUE, NA, FALSE, TRUE, NA), x = c(1, 2, 3, 4, 5))
  f33_strict(df, function(d) d |> group_by(l) |> summarise(n = n(), s = sum(x)))
})

test_that("numeric and date-time keys come back ascending (conformance: local group_by preserves variable types)", {
  skip_if_no_gpu()
  df <- data.frame(
    i = c(3L, -1L, 2L, NA, -1L),
    num = c(2.5, -0.5, 10, 1.5, NA),
    d = as.Date("2024-01-10") + c(3, -2, 0, NA, 3),
    t = as.POSIXct("2024-01-10 12:00:00", tz = "UTC") + c(60, -60, 0, 3600, NA),
    x = c(1, 2, 3, 4, 5)
  )
  for (var in c("i", "num", "d", "t")) {
    f33_strict(df, function(d) {
      d |> group_by(!!rlang::sym(var)) |> summarise(n = n(), m = mean(x))
    })
  }

  df_var <- data.frame(
    l = c(TRUE, FALSE),
    i = 2:1,
    d = as.Date("2024-01-10") + 2:1,
    f = factor(c("b", "a")),
    num = 2:1 + 0.5,
    c = c("b", "a")
  )
  for (var in names(df_var)) {
    f33_strict(df_var, function(d) d |> group_by(!!rlang::sym(var)) |> summarise(n = n()))
  }
})

test_that("multi-key groups are sorted key by key under every .groups", {
  skip_if_no_gpu()
  for (groups in list("drop_last", "keep", "drop")) {
    f33_strict(f33_df, function(d) {
      d |> group_by(g, h) |> summarise(n = n(), s = sum(x), .groups = groups)
    })
  }
  f33_strict(f33_df, function(d) {
    d |> group_by(h, g) |> summarise(m = mean(x), .groups = "drop_last")
  })
})

test_that("many groups come back sorted (integer and character keys)", {
  skip_if_no_gpu()
  set.seed(33)
  n <- 5000
  df <- data.frame(
    k = sample(c(-300:300, NA), n, replace = TRUE),
    s = sample(c(paste0("k", 1:400), NA), n, replace = TRUE),
    x = runif(n),
    stringsAsFactors = FALSE
  )
  f33_both(df, function(d) d |> group_by(k) |> summarise(n = n(), m = mean(x)))
  f33_both(df, function(d) d |> group_by(s) |> summarise(n = n(), mx = max(x)))
  f33_both(df, function(d) {
    d |> group_by(s, k) |> summarise(n = n(), .groups = "drop")
  })
})

test_that("sort-based and hash-based aggregations stay aligned with the sorted keys", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(
      md = median(x), nd = n_distinct(i), fi = dplyr::first(x), la = dplyr::last(i),
      m = mean(x), n = n(), v = var(x)
    )
  })
})

test_that("integer min()/max(), temp-column and window arguments are sorted too", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(mn = min(i), mx = max(i))
  })
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(s = sum(x > 2), m = mean(x * 2))
  })
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(cs = dplyr::last(cumsum(x)))
  })
})

test_that("the sorted grouped summarise runs on the GPU (eager)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(f33_df, function(d) {
    d |> group_by(g, h) |> summarise(n = n(), m = mean(x), s = sum(i), .groups = "drop")
  })
})

test_that("a redefined key keeps the original group order", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) {
    d |> group_by(g) |> summarise(n = n(), g = max(x))
  })
})

test_that("row-order-dependent verbs after a grouped summarise see the sorted order", {
  skip_if_no_gpu()
  f33_both(f33_df, function(d) {
    d |> group_by(g) |> summarise(n = n(), m = mean(x)) |> slice_head(n = 2)
  })
  f33_both(f33_df, function(d) {
    d |> group_by(g) |> summarise(m = mean(x)) |> mutate(r = dplyr::row_number())
  })
})

test_that("summarise(.by = ) keeps first-appearance order", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) d |> summarise(n = n(), m = mean(x), .by = g))
  f33_strict(f33_df, function(d) d |> summarise(s = sum(i), .by = c(h, g)))
})

test_that("count() and tally() come back sorted", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) d |> count(g))
  f33_strict(f33_df, function(d) d |> count(g, h))
  f33_strict(f33_df, function(d) d |> count(g, wt = x))
  # sort = TRUE: ties keep the group order
  f33_strict(f33_df, function(d) d |> count(g, sort = TRUE))
  f33_strict(f33_df, function(d) d |> group_by(g, h) |> tally())
  f33_strict(f33_df, function(d) d |> group_by(h) |> count(g))
})

# --- A2. .drop = FALSE empty groups ----------------------------------------

f33_fx <- data.frame(
  f = factor(c("b", "a", "b", "a", "b"), levels = c("c", "a", "b", "d")),
  g = c("y", "x", "x", "y", "y"),
  i = c(1L, 2L, 3L, 4L, NA),
  x = c(1.5, 2, NA, 4, 5),
  l = c(TRUE, FALSE, NA, TRUE, FALSE),
  stringsAsFactors = FALSE
)

test_that("summarise returns a row for each empty factor level (#341)", {
  skip_if_no_gpu()
  df <- data.frame(x = factor(rep(1:3, each = 10), levels = 1:4))
  f33_strict(df, function(d) d |> group_by(x, .drop = FALSE) |> summarise(n = n()))

  df2 <- data.frame(
    e = 1,
    f = factor(c(1, 1, 2, 2), levels = 1:3),
    g = c(1, 1, 2, 2),
    x = c(1, 2, 1, 4)
  )
  f33_strict(df2, function(d) d |> group_by(e, f, g, .drop = FALSE) |> summarise(z = n()))
})

test_that("empty groups get each aggregate's empty-vector value", {
  skip_if_no_gpu()
  f33_strict(f33_fx, function(d) {
    d |> group_by(f, .drop = FALSE) |> summarise(
      n = n(), si = sum(i, na.rm = TRUE), sx = sum(x), m = mean(x, na.rm = TRUE),
      md = median(x), v = var(x), sdv = sd(x), nd = n_distinct(i),
      fi = dplyr::first(x), la = dplyr::last(i), a = any(l), al = all(l)
    )
  })
  # quantile() separately: dplyr keeps its "50%" names, which cuplyr has never
  # carried (not an F33 contract), so compare unnamed values.
  expected <- f33_fx |> group_by(f, .drop = FALSE) |>
    summarise(q = quantile(x, 0.5, na.rm = TRUE))
  for (lazy in c(FALSE, TRUE)) {
    got <- tbl_gpu(f33_fx, lazy = lazy) |> group_by(f, .drop = FALSE) |>
      summarise(q = quantile(x, 0.5, na.rm = TRUE)) |> collect()
    expect_identical(as.character(got$f), as.character(expected$f))
    expect_identical(unname(got$q), unname(expected$q))
  }
})

test_that("empty groups are filled on the GPU (eager): NaN survives", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  res <- tbl_gpu(f33_fx) |>
    group_by(f, .drop = FALSE) |>
    summarise(n = n(), m = mean(x, na.rm = TRUE), s = sum(i, na.rm = TRUE)) |>
    collect()
  expect_identical(as.character(res$f), c("c", "a", "b", "d"))
  expect_identical(res$n, c(0L, 2L, 3L, 0L))
  expect_identical(is.nan(res$m), c(TRUE, FALSE, FALSE, TRUE))
  expect_identical(res$s, c(0L, 6L, 4L, 0L))
})

test_that("min()/max() on empty groups give Inf/-Inf doubles and warn", {
  skip_if_no_gpu()
  pipeline <- function(d) {
    d |> group_by(f, .drop = FALSE) |>
      summarise(mn = min(i, na.rm = TRUE), mx = max(x, na.rm = TRUE), n = n())
  }
  f33_strict(f33_fx, pipeline, warnings_ok = TRUE)
  for (lazy in c(FALSE, TRUE)) {
    msgs <- character()
    withCallingHandlers(
      collect(pipeline(tbl_gpu(f33_fx, lazy = lazy))),
      warning = function(w) {
        msgs <<- c(msgs, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    expect_true(any(grepl("no non-missing arguments to min", msgs)))
    expect_true(any(grepl("no non-missing arguments to max", msgs)))
  }
})

test_that("no empty group: integer min() stays integer and nothing warns", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("b", "a", "b"), levels = c("a", "b")), i = c(3L, 1L, 2L))
  pipeline <- function(d) d |> group_by(f, .drop = FALSE) |> summarise(mn = min(i), n = n())
  f33_strict(df, pipeline)
  expect_no_warning(collect(pipeline(tbl_gpu(df))))
})

test_that("multi-key .drop = FALSE expands recursively, every .groups", {
  skip_if_no_gpu()
  for (groups in list(NULL, "keep", "drop_last", "drop")) {
    f33_strict(f33_fx, function(d) {
      d |> group_by(f, g, .drop = FALSE) |>
        summarise(n = n(), s = sum(x, na.rm = TRUE), .groups = groups)
    })
    f33_strict(f33_fx, function(d) {
      d |> group_by(g, f, .drop = FALSE) |>
        summarise(n = n(), m = mean(x, na.rm = TRUE), .groups = groups)
    })
  }
})

test_that("an NA factor key sorts after the levels under .drop = FALSE", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", NA, "a"), levels = c("a", "b")), x = c(1, 2, 3))
  f33_strict(df, function(d) d |> group_by(f, .drop = FALSE) |> summarise(n = n(), m = mean(x)))
})

test_that("zero-dots summarise() on .drop = FALSE groups returns every group", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "a"), levels = c("a", "b")), x = 1:2)
  f33_strict(df, function(d) d |> group_by(f, .drop = FALSE) |> summarise())
  for (groups in list(NULL, "keep", "drop")) {
    f33_strict(f33_fx, function(d) {
      d |> group_by(g, f, .drop = FALSE) |> summarise(.groups = groups)
    })
  }
  f33_strict(f33_fx[0, ], function(d) d |> group_by(f, .drop = FALSE) |> summarise())
})

test_that("a 0-row factor input with .drop = FALSE gives one row per level", {
  skip_if_no_gpu()
  f33_strict(f33_fx[0, ], function(d) {
    d |> group_by(f, .drop = FALSE) |> summarise(n = n(), m = mean(x), s = sum(i))
  })
})

test_that(".drop = FALSE summarise composes with pending and later lazy verbs", {
  skip_if_no_gpu()
  f33_strict(f33_fx, function(d) {
    d |> filter(i > 1L) |> group_by(f, .drop = FALSE) |>
      summarise(n = n(), s = sum(x, na.rm = TRUE)) |>
      mutate(k = n + 1L)
  })
  f33_strict(f33_fx, function(d) {
    d |> group_by(f, g, .drop = FALSE) |>
      summarise(n = n(), .groups = "drop_last") |>
      summarise(t = sum(n), k = n())
  })
})

test_that(".drop = FALSE without a factor key changes nothing", {
  skip_if_no_gpu()
  f33_strict(f33_df, function(d) {
    d |> group_by(g, .drop = FALSE) |> summarise(n = n(), mn = min(i))
  })
})

test_that("count() and tally() on .drop = FALSE groups", {
  skip_if_no_gpu()
  f33_strict(f33_fx, function(d) d |> count(f, .drop = FALSE))
  f33_strict(f33_fx, function(d) d |> count(f, g, wt = x, .drop = FALSE))
  f33_strict(f33_fx, function(d) d |> count(f, sort = TRUE, .drop = FALSE))
  f33_strict(f33_fx, function(d) d |> group_by(g, f, .drop = FALSE) |> tally())
})

# --- A3. 0-row grouped input keeps the key columns ---------------------------

test_that("0-row grouped summarise keeps the key columns and their types", {
  skip_if_no_gpu()
  df0 <- data.frame(
    g = character(0),
    f = factor(character(0), levels = c("a", "b")),
    d = as.Date(character(0)),
    x = numeric(0),
    i = integer(0),
    stringsAsFactors = FALSE
  )
  f33_strict(df0, function(d) d |> group_by(g) |> summarise(n = n(), m = mean(x), s = sum(i)))
  f33_strict(df0, function(d) d |> group_by(f) |> summarise(n = n()))
  f33_strict(df0, function(d) d |> group_by(d) |> summarise(n = n(), m = mean(x)))
  f33_strict(df0, function(d) {
    d |> group_by(g, f) |> summarise(n = n(), .groups = "keep")
  })
  f33_strict(df0, function(d) d |> group_by(g, .drop = FALSE) |> summarise(n = n()))
  f33_strict(f33_df, function(d) {
    d |> filter(x > 100) |> group_by(g, h) |> summarise(n = n(), m = mean(x))
  })
})
