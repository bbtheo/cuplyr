# F101: a lazy summarise() with expression arguments stays in the plan. The temp
# column is a plan node below the summarise node, so the filters and the
# pruning apply and no plan break fires; the reasons to run at the verb
# (a result type that depends on the data) stay.

# (dplyr is not attached in the test environment)
lag <- dplyr::lag
first <- dplyr::first
last <- dplyr::last
nth <- dplyr::nth
if_else <- dplyr::if_else

f101_df <- function() {
  tibble::tibble(
    g = c(2L, 1L, 1L, 3L, 2L, 3L, 1L, 2L),
    x = c(1.5, NA, 3, 4.25, -1, 0.5, 2, 6),
    y = c(2, 1, NA, 0.5, 3, -2, 4, 1),
    i = c(1L, 2L, 3L, NA, 5L, 6L, 7L, 8L),
    j = c(2L, 3L, 1L, 4L, 2L, NA, 1L, 2L),
    b = c(TRUE, FALSE, NA, TRUE, TRUE, FALSE, FALSE, TRUE),
    f = factor(c("a", "b", "a", "c", "b", "a", "c", "b"),
               levels = c("c", "a", "b", "unused")),
    extra1 = as.numeric(1:8),
    extra2 = as.numeric(8:1)
  )
}

f101_lazy <- function(df = f101_df()) tbl_gpu(df, lazy = TRUE)

# The node types of a plan, top first (no joins in this file)
f101_plan <- function(x, opt = FALSE) {
  node <- .subset2(x, "lazy_ops")
  if (opt) node <- optimize_ast(node)
  types <- character()
  while (!is.null(node)) {
    types <- c(types, .subset2(node, "type"))
    node <- .subset2(node, "input")
  }
  types
}

# The mutate expressions of the whole plan
f101_mutate_outputs <- function(x) {
  node <- .subset2(x, "lazy_ops")
  out <- character()
  while (!is.null(node)) {
    if (identical(.subset2(node, "type"), "mutate")) {
      out <- c(out, vapply(.subset2(node, "expressions"), `[[`, character(1), "output_col"))
    }
    node <- .subset2(node, "input")
  }
  out
}

# Run `expr`; its value and the notices it signalled
f101_capture <- function(expr) {
  breaks <- list()
  cpu <- list()
  warned <- character()
  value <- withCallingHandlers(
    expr,
    cuplyr_plan_break = function(c) breaks[[length(breaks) + 1L]] <<- c,
    cuplyr_cpu_fallback = function(c) cpu[[length(cpu) + 1L]] <<- c,
    warning = function(w) {
      warned <<- c(warned, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, breaks = breaks, cpu = cpu, warned = warned)
}

# The optimized plan, walked to the columns its source is read through
f101_source_cols <- function(x) {
  node <- optimize_ast(.subset2(x, "lazy_ops"))
  cols <- NULL
  while (!is.null(node)) {
    if (identical(.subset2(node, "type"), "select") &&
        identical(.subset2(.subset2(node, "input"), "type"), "source")) {
      cols <- .subset2(node, "columns")
    }
    node <- .subset2(node, "input")
  }
  cols
}

# ---------------------------------------------------------------------------
# The expression case: one plan, no early compute

test_that("lazy filter |> summarise(mean(x * y)) is one plan (ungrouped)", {
  skip_if_no_gpu()
  got <- f101_capture(f101_lazy() |> filter(x > 0) |> summarise(m = mean(x * y)))
  expect_length(got$breaks, 0L)
  expect_length(got$cpu, 0L)
  expect_true(has_pending_ops(got$value))
  expect_identical(f101_plan(got$value), c("summarise", "mutate", "filter", "source"))
  # the filter applies below the temp column and only the columns read are read
  opt <- f101_plan(got$value, opt = TRUE)
  expect_lt(match("mutate", opt), match("filter", opt))
  expect_setequal(f101_source_cols(got$value), c("x", "y"))
  expect_same_as_dplyr_lazy(f101_df(), function(d) d |> filter(x > 0) |> summarise(m = mean(x * y)))
})

test_that("lazy filter |> group_by |> summarise(mean(x * y)) is one plan", {
  skip_if_no_gpu()
  got <- f101_capture(f101_lazy() |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * y)))
  expect_length(got$breaks, 0L)
  expect_length(got$cpu, 0L)
  expect_true(has_pending_ops(got$value))
  expect_identical(f101_plan(got$value), c("summarise", "mutate", "filter", "source"))
  expect_setequal(f101_source_cols(got$value), c("g", "x", "y"))
  expect_same_as_dplyr_lazy(f101_df(), function(d) {
    d |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * y))
  })
})

test_that("a lazy table with no pending ops keeps a plan too", {
  skip_if_no_gpu()
  got <- f101_capture(f101_lazy() |> group_by(g) |> summarise(m = mean(x * y)))
  expect_true(has_pending_ops(got$value))
  expect_length(got$breaks, 0L)
  expect_same_as_dplyr_lazy(f101_df(), function(d) d |> group_by(g) |> summarise(m = mean(x * y)))
})

test_that("expressions over earlier steps, arrange and later steps stay equal to dplyr", {
  skip_if_no_gpu()
  df <- f101_df()
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> mutate(z = x + 1) |> filter(g != 3L) |> group_by(g) |>
      summarise(a = mean(z * y), b = sd(z / 2), c = max(z - x), d = median(y + 1))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> arrange(desc(extra1)) |> group_by(g) |> summarise(a = first(x + y), b = last(x * y))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(m = mean(x * y, na.rm = TRUE)) |> filter(m > 0) |>
      mutate(m2 = m * 2)
  })
  # (a filter above the summarise reads the summarise's result)
  got <- f101_lazy() |> group_by(g) |> summarise(m = mean(x * y, na.rm = TRUE)) |> filter(m > 0)
  expect_identical(f101_plan(got), c("filter", "summarise", "mutate", "source"))
})

test_that("NA, empty inputs and na.rm follow dplyr through the plan temp column", {
  skip_if_no_gpu()
  df <- f101_df()
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(a = mean(x * y), b = mean(x * y, na.rm = TRUE),
                                  c = sum(x * y, na.rm = TRUE))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(g > 99L) |> summarise(a = mean(x * y), b = sum(x * y))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(g > 99L) |> group_by(g) |> summarise(a = mean(x * y))
  })
})

test_that("output schema, order, groups and types are unchanged; temp columns never leak", {
  skip_if_no_gpu()
  df <- f101_df()
  q <- f101_lazy() |> group_by(g, b) |> summarise(a = mean(x * y), n = n(), c = max(x + 1),
                                                  .groups = "keep")
  expect_identical(tbl_vars(q) |> as.character(), c("g", "b", "a", "n", "c"))
  expect_false(any(grepl("temp_agg", current_schema(q)$names)))
  expect_identical(group_vars(q), c("g", "b"))
  r <- collect(q)
  expect_false(any(grepl("temp_agg", names(r))))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g, b) |> summarise(a = mean(x * y), n = n(), c = max(x + 1), .groups = "keep")
  })
  # an aggregate named like a key replaces it in place
  expect_same_as_dplyr_lazy(df, function(d) d |> group_by(g, b) |> summarise(g = mean(x * y)))
  # repeated collect() gives the same answer
  expect_identical(tibble::as_tibble(collect(q)), tibble::as_tibble(r))
})

test_that("a user column named like a temp column is not touched", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3), .temp_agg_1 = c(10, 20, 30))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(a = mean(x * 2), b = mean(.temp_agg_1))
  })
})

test_that(".by, count(wt =) and reframe() with an expression stay in the plan", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(x > 0) |> summarise(m = mean(x * y), .by = g))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(x > 0) |> summarise(m = mean(x * y), .by = g))

  got <- f101_capture(f101_lazy() |> filter(x > 0) |> count(g, wt = x * y))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(x > 0) |> count(g, wt = x * y))

  got <- f101_capture(f101_lazy() |> filter(x > 0) |> group_by(g) |>
                        reframe(q = quantile(x * y, c(0.25, 0.75), na.rm = TRUE)))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  # (dplyr's column keeps quantile()'s names, the GPU column has none on dev too)
  oracle <- df |> filter(x > 0) |> group_by(g) |> reframe(q = quantile(x * y, c(0.25, 0.75), na.rm = TRUE))
  r <- collect(got$value)
  expect_identical(names(r), names(oracle))
  expect_equal(r$g, oracle$g)
  expect_equal(unname(r$q), unname(oracle$q))
})

# ---------------------------------------------------------------------------
# Window arguments

test_that("a window argument stays in the plan (grouped and ungrouped)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(x > 0) |> group_by(g) |>
                        summarise(v = sum(lag(x), na.rm = TRUE)))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  plan <- f101_plan(got$value)
  expect_identical(plan[[1L]], "summarise")
  expect_true("window" %in% plan)
  expect_lt(match("window", plan), match("filter", plan))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(x > 0) |> group_by(g) |> summarise(v = sum(lag(x), na.rm = TRUE))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(x > 0) |> summarise(v = sum(lag(x), na.rm = TRUE))
  })
})

test_that("window shapes through the plan equal dplyr, row order included", {
  skip_if_no_gpu()
  df <- f101_df()
  shapes <- list(
    function(d) d |> group_by(g) |> summarise(a = last(cumsum(extra1))),
    function(d) d |> group_by(g) |> summarise(a = mean(cumsum(extra1) - lag(cumsum(extra1)), na.rm = TRUE)),
    function(d) d |> group_by(g) |> summarise(a = sum(extra1 - lag(extra1, default = 0))),
    function(d) d |> filter(g != 2L) |> group_by(g) |> summarise(a = max(lag(extra2), na.rm = TRUE)),
    function(d) d |> arrange(desc(extra2)) |> group_by(g) |> summarise(a = last(cumsum(extra1))),
    function(d) d |> group_by(g) |> summarise(a = mean(rank(extra2) * 2))
  )
  for (p in shapes) expect_same_as_dplyr_lazy(df, p)
  # several window dots at once, sharing a window temp column
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(a = sum(lag(extra1), na.rm = TRUE),
                                  b = min(lag(extra1), na.rm = TRUE),
                                  c = last(cumsum(extra2)))
  })
  expect_no_error(q <- f101_lazy() |> group_by(g) |> summarise(a = last(cumsum(extra1))))
  expect_true(has_pending_ops(q))
})

test_that("window arguments keep their temp columns out of the result", {
  skip_if_no_gpu()
  q <- f101_lazy() |> group_by(g) |> summarise(a = mean(cumsum(extra1) * 2), b = sum(lag(extra2), na.rm = TRUE))
  expect_identical(as.character(tbl_vars(q)), c("g", "a", "b"))
  expect_identical(names(collect(q)), c("g", "a", "b"))
})

test_that("an integer window sum in the argument still runs at the verb (F42)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(i > 0L) |> group_by(g) |> summarise(a = mean(i * sum(i))))
  # the stages ran at the verb: only the summarise itself is still pending
  expect_false("window" %in% f101_plan(got$value))
  expect_length(got$breaks, 1L)
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(i > 0L) |> group_by(g) |> summarise(a = mean(i * sum(i)))
  })
  # overflow in one group: dplyr's per-group double, through the CPU rerun
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(2147483647L, 1L, 1L))
  r <- suppressWarnings(collect(summarise(group_by(tbl_gpu(big, lazy = TRUE), g), a = mean(i * sum(i)))))
  o <- suppressWarnings(summarise(group_by(big, g), a = mean(i * sum(i))))
  expect_equal(as.numeric(r$a), as.numeric(o$a))
})

# ---------------------------------------------------------------------------
# Reasons to run at the verb that stay

test_that("an integer sum() of an expression still takes the verb-time route (F80)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(i > 0L) |> group_by(g) |> summarise(s = sum(i + j)))
  expect_false(has_pending_ops(got$value))
  expect_length(got$breaks, 1L)
  expect_match(got$breaks[[1L]]$reason, "integer", fixed = TRUE)
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(i > 0L) |> group_by(g) |> summarise(s = sum(i + j))
  })
  # it turns into a double where dplyr's does
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(2147483647L, 2147483647L, 1L), j = c(0L, 1L, 2L))
  q <- suppressWarnings(summarise(group_by(tbl_gpu(big, lazy = TRUE), g), s = sum(i + 0L)))
  expect_false(has_pending_ops(q))
  o <- summarise(group_by(big, g), s = sum(i + 0L))
  expect_identical(typeof(collect(q)$s), typeof(o$s))
  expect_equal(as.numeric(collect(q)$s), as.numeric(o$s))
})

test_that("integer min()/max() of an expression still runs at the verb (F07)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(i > 0L) |> group_by(g) |> summarise(m = min(i + 1L)))
  expect_false(has_pending_ops(got$value))
  expect_length(got$breaks, 1L)
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(i > 0L) |> group_by(g) |> summarise(m = min(i + 1L)))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(i > 0L) |> group_by(g) |> summarise(m = max(i * 2L, na.rm = TRUE)))
})

test_that("a sum() of a logical expression stays in the plan and stays integer", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(extra1 > 1) |> group_by(g) |> summarise(s = sum(x > 1)))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(extra1 > 1) |> group_by(g) |> summarise(s = sum(x > 1)))
  expect_same_as_dplyr_lazy(df, function(d) d |> group_by(g) |> summarise(s = sum(x > 1, na.rm = TRUE)))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(g > 99L) |> summarise(s = sum(x > 1)))
  r <- collect(summarise(group_by(f101_lazy(), g), s = sum(x > 1, na.rm = TRUE)))
  expect_type(r$s, "integer")
})

test_that("first/last/nth of a logical expression and any/all stay logical in the plan (F76)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(extra1 > 0) |> group_by(g) |>
                        summarise(a = first(x > 1), b = last(x > 1), c = nth(x > 1, 2L),
                                  d = any(x > 1), e = all(x > 1, na.rm = TRUE)))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  r <- collect(got$value)
  for (nm in c("a", "b", "c", "d", "e")) expect_type(r[[nm]], "logical")
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(extra1 > 0) |> group_by(g) |>
      summarise(a = first(x > 1), b = last(x > 1), c = nth(x > 1, 2L),
                d = any(x > 1), e = all(x > 1, na.rm = TRUE))
  })
  # a sum of the same expression next to a first() of it gets its own type
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(a = first(x > 1), s = sum(x > 1, na.rm = TRUE))
  })
})

test_that("reducers of one expression share one temp column in the plan (F22)", {
  skip_if_no_gpu()
  df <- f101_df()
  q <- f101_lazy() |> group_by(g) |> summarise(a = mean(x * y), b = max(x * y), c = sd(x * y))
  expect_length(f101_mutate_outputs(q), 1L)
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(g) |> summarise(a = mean(x * y), b = max(x * y), c = sd(x * y))
  })
})

# ---------------------------------------------------------------------------
# Warnings follow the rows the op sees

test_that("a temp column's integer overflow warns at collect() and after the filter's rows", {
  skip_if_no_gpu()
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(2147483647L, 1L, 1L), j = c(2L, 1L, 1L))
  got <- f101_capture(summarise(group_by(tbl_gpu(big, lazy = TRUE), g), m = mean(i * j)))
  expect_length(got$warned, 0L)
  res <- f101_capture(collect(got$value))
  expect_true(any(grepl("NAs produced by integer overflow", res$warned, fixed = TRUE)))
  o <- suppressWarnings(summarise(group_by(big, g), m = mean(i * j)))
  expect_equal(as.numeric(res$value$m), as.numeric(o$m))

  # the filter removes the overflowing row, so nothing overflows (dplyr: none either)
  flt <- function(d) d |> filter(i < 5L) |> group_by(g) |> summarise(m = mean(i * j))
  expect_no_warning(oracle <- flt(big))
  expect_no_warning(r <- collect(flt(tbl_gpu(big, lazy = TRUE))))
  expect_equal(as.numeric(r$m), as.numeric(oracle$m))
})

test_that("a temp column's integer overflow on the verb-time route warns at the verb", {
  skip_if_no_gpu()
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(2147483647L, 1L, 1L), j = c(2L, 1L, 1L))
  got <- f101_capture(summarise(group_by(tbl_gpu(big, lazy = TRUE), g), m = sum(i * j)))
  expect_true(any(grepl("NAs produced by integer overflow", got$warned, fixed = TRUE)))
  expect_false(has_pending_ops(got$value))
})

# ---------------------------------------------------------------------------
# Regressions

test_that("factor keys go through plan temp columns (F34)", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(f101_lazy() |> filter(x > 0) |> group_by(f) |> summarise(m = mean(x * y)))
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(got$value))
  expect_identical(current_schema(got$value)$factor_levels$f, levels(df$f))
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(x > 0) |> group_by(f) |> summarise(m = mean(x * y)))
  expect_same_as_dplyr_lazy(df, function(d) d |> group_by(f, g) |> summarise(m = sum(x * y, na.rm = TRUE)))
  expect_same_as_dplyr_lazy(df, function(d) d |> group_by(f) |> summarise(a = n_distinct(.data$f), m = max(x + 1)))
})

test_that(".drop = FALSE empty groups keep working with an expression (F71/F74)", {
  skip_if_no_gpu()
  df <- f101_df()
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(f, .drop = FALSE) |> summarise(m = mean(x * y), s = sum(x > 1, na.rm = TRUE))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(f, .drop = FALSE) |> filter(x > 100) |> summarise(m = mean(x * y))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> group_by(f, .drop = FALSE) |> filter(x > 0) |> summarise(v = sum(lag(extra1), na.rm = TRUE))
  })
})

test_that("an impure column-free call inside the argument is dplyr's (F75)", {
  skip_if_no_gpu()
  df <- f101_df()
  counter <- function() {
    calls <- 0L
    function() {
      calls <<- calls + 1L
      calls
    }
  }
  g_oracle <- counter()
  oracle <- df |> group_by(g) |> summarise(m = mean(x + g_oracle(), na.rm = TRUE))
  g_gpu <- counter()
  got <- f101_capture(f101_lazy() |> group_by(g) |> summarise(m = mean(x + g_gpu(), na.rm = TRUE)))
  expect_length(got$cpu, 1L)
  expect_equal(environment(g_gpu)$calls, environment(g_oracle)$calls)
  expect_equal(tibble::as_tibble(collect(got$value))$m, oracle$m)
})

test_that("POSIXct and Date temp columns keep their time zone and class in the plan", {
  skip_if_no_gpu()
  tk <- as.POSIXct(c("2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 12:00:00",
                     "2024-01-04 00:00:00"), tz = "Asia/Tokyo")
  u <- tk + 86400
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), t = tk, u = u, d = as.Date(tk), cc = c(TRUE, FALSE, TRUE, TRUE))
  p <- function(d) {
    d |> group_by(g) |>
      summarise(a = first(if_else(cc, t, u)), b = min(lag(t), na.rm = TRUE),
                c = max(if_else(cc, d, d + 1L)))
  }
  r <- f101_capture(p(tbl_gpu(df, lazy = TRUE)))
  expect_length(r$breaks, 0L)
  expect_true(has_pending_ops(r$value))
  expect_same_as_dplyr_lazy(df, p)
  got <- collect(r$value)
  expect_identical(attr(got$a, "tzone"), "Asia/Tokyo")
  expect_identical(attr(got$b, "tzone"), "Asia/Tokyo")
})

test_that("an expression on a rowwise table is unchanged", {
  skip_if_no_gpu()
  df <- f101_df() |> select(g, x, y)
  expect_same_as_dplyr_lazy(df, function(d) d |> rowwise() |> summarise(m = sum(x * y)))
})

test_that("eager mode is unchanged: no plan, no notice, warning at the verb", {
  skip_if_no_gpu()
  df <- f101_df()
  got <- f101_capture(tbl_gpu(df, lazy = FALSE) |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * y)))
  expect_false(has_pending_ops(got$value))
  expect_length(got$breaks, 0L)
  expect_same_as_dplyr(df, function(d) d |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * y)))
  big <- tibble::tibble(g = c(1L, 2L), i = c(2147483647L, 1L), j = c(2L, 1L))
  q <- f101_capture(summarise(group_by(tbl_gpu(big, lazy = FALSE), g), m = mean(i * j)))
  expect_true(any(grepl("NAs produced by integer overflow", q$warned, fixed = TRUE)))
  expect_false(has_pending_ops(q$value))
})

# ---------------------------------------------------------------------------
# Notice counts, joins and collisions (round 1 review)

test_that("one summarise() with several verb-time reasons fires one notice for the user's steps", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), i = 1:4, j = 4:1, x = c(1, 2, 3, 4))
  calls <- list(
    function(t) summarise(t, a = mean(i * sum(i)), b = mean(j * sum(j))),
    function(t) summarise(t, a = mean(i * sum(i)), m = max(j)),
    function(t) summarise(t, m = max(j), a = mean(i * sum(i))),
    function(t) summarise(t, s = sum(i + 1L), a = mean(j * sum(j))),
    function(t) summarise(t, a = mean(i * sum(i)), s = sum(j + 1L), m = mean(x * 2))
  )
  for (cl in calls) {
    got <- f101_capture(cl(group_by(filter(tbl_gpu(df, lazy = TRUE), i > 0L), g)))
    expect_length(got$breaks, 1L)
    expect_equal(got$breaks[[1L]]$steps, 1)
    oracle <- cl(group_by(filter(df, i > 0L), g))
    expect_equal(tibble::as_tibble(dplyr::ungroup(collect(got$value))), tibble::as_tibble(dplyr::ungroup(oracle)))
  }
  # no pending user step: no notice, whatever runs at the verb
  got <- f101_capture(summarise(group_by(tbl_gpu(df, lazy = TRUE), g), a = mean(i * sum(i)), b = sum(j + 1L)))
  expect_length(got$breaks, 0L)
})

test_that("a join below the plan stays in it", {
  skip_if_no_gpu()
  df <- f101_df()
  right <- tibble::tibble(g = c(1L, 2L), k = c(10, 20))
  q <- f101_lazy() |> dplyr::inner_join(tbl_gpu(right, lazy = TRUE), by = "g") |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * k))
  got <- f101_capture(q)
  expect_length(got$breaks, 0L)
  expect_true(has_pending_ops(q))
  o <- df |> dplyr::inner_join(right, by = "g") |> filter(x > 0) |> group_by(g) |> summarise(m = mean(x * k))
  expect_equal(tibble::as_tibble(collect(q)), o)
})

test_that("user columns named like window temp columns are not touched", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), .temp_agg_win_1 = c(10, 20, 30, 40), `..win1..` = c(5, 6, 7, 8))
  p <- function(d) d |> group_by(g) |> summarise(a = mean(lag(x) * 2, na.rm = TRUE), b = mean(.temp_agg_win_1), c = mean(`..win1..`))
  expect_same_as_dplyr_lazy(df, p)
})

test_that("a window argument's integer overflow warns at collect()", {
  skip_if_no_gpu()
  big <- tibble::tibble(g = c(1L, 1L, 2L), i = c(2147483647L, 1L, 1L), j = c(2L, 1L, 1L))
  got <- f101_capture(summarise(group_by(tbl_gpu(big, lazy = TRUE), g), m = mean(lag(i * j), na.rm = TRUE)))
  expect_length(got$warned, 0L)
  res <- f101_capture(collect(got$value))
  expect_true(any(grepl("NAs produced by integer overflow", res$warned, fixed = TRUE)))
  oracle <- suppressWarnings(summarise(group_by(big, g), m = mean(lag(i * j), na.rm = TRUE)))
  expect_equal(as.numeric(res$value$m), as.numeric(oracle$m))
})

test_that("plan depth does not grow with the number of dots (wide lazy summarise)", {
  skip_if_no_gpu()
  wide <- function(n) {
    m <- matrix(as.numeric(seq_len(n * 3L)), nrow = 3L)
    colnames(m) <- paste0("v", seq_len(n))
    tibble::as_tibble(as.data.frame(m))
  }
  d <- wide(1000L)
  q <- tbl_gpu(d, lazy = TRUE) |> summarise(across(everything(), ~ mean(.x * 2)))
  expect_identical(f101_plan(q), c("summarise", "mutate", "source"))
  expect_equal(unlist(collect(q)), colMeans(d) * 2)

  d2 <- wide(1300L)
  q2 <- tbl_gpu(d2, lazy = TRUE) |>
    summarise(across(everything(), ~ mean(lag(.x) * 2, na.rm = TRUE)))
  # one window stage, not one per dot
  expect_lte(length(f101_plan(q2)), 5L)
  oracle <- d2 |> summarise(across(everything(), ~ mean(lag(.x) * 2, na.rm = TRUE)))
  expect_equal(unlist(collect(q2)), unlist(oracle))

  # plain temps and window temps of one call share the same single mutate
  d3 <- wide(4L)
  q3 <- tbl_gpu(d3, lazy = TRUE) |>
    summarise(a = mean(v1 * 2), b = sum(lag(v2) * 3, na.rm = TRUE), c = sd(v3 + 1),
              d = max(lag(v4) + 1, na.rm = TRUE))
  mutates <- sum(f101_plan(q3) == "mutate")
  expect_lte(mutates, 2L)  # a window stage's own pre-mutate plus the temp mutate
  expect_length(f101_mutate_outputs(q3)[grepl("temp_agg", f101_mutate_outputs(q3))], 4L)
  oracle3 <- d3 |> summarise(a = mean(v1 * 2), b = sum(lag(v2) * 3, na.rm = TRUE), c = sd(v3 + 1),
                             d = max(lag(v4) + 1, na.rm = TRUE))
  expect_equal(tibble::as_tibble(collect(q3)), oracle3)
})

test_that("`.by` with several verb-time reasons fires one notice too", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), i = 1:4, j = 4:1)
  got <- f101_capture(summarise(filter(tbl_gpu(df, lazy = TRUE), i > 0L), a = mean(i * sum(i)), b = sum(j + 1L), .by = g))
  expect_length(got$breaks, 1L)
  expect_equal(got$breaks[[1L]]$steps, 1)
  oracle <- summarise(filter(df, i > 0L), a = mean(i * sum(i)), b = sum(j + 1L), .by = g)
  expect_equal(tibble::as_tibble(collect(got$value)), oracle)
})
