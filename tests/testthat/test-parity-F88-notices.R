# F88 acceptance tests: CPU fallbacks and plan breaks are two kinds of notice.
#
# - `cuplyr_cpu_fallback`: data went to the host and (part of) the verb ran in
#   R. Governed by `options(cuplyr.fallback = "silent" | "warn" | "error")`
#   ("warning" is an alias of "warn"). Fields: verb, expr, reason, rows, cols,
#   bytes. The warning is `cuplyr_warning_cpu_fallback`, the error
#   `cuplyr_error_fallback`.
# - `cuplyr_plan_break`: a lazy plan computed early, on the GPU. Governed by
#   `options(cuplyr.plan_break = "silent" | "warn")` ("warning" alias), never
#   an error. Fields: verb, expr, reason, steps. Warning class
#   `cuplyr_warning_plan_break`.
# - An invalid option value warns once per session and value
#   (`cuplyr_warning_invalid_option`) and the default is used.
# - The default stays silent for both.

f88_df <- function(n = 1000L) {
  i <- seq_len(n)
  tibble::tibble(x = i / n, y = rev(i) / n, z = (i %% 7) * 1.5)
}

f88_small <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5, 6),
    y = c(2, 3, 4, 5, 6, 7),
    i = c(1L, 2L, 3L, 4L, 5L, 6L),
    g = c("a", "a", "b", "b", "c", "c")
  )
}

# Run `expr`; return its value, the conditions signalled by class, and every
# warning (muffled).
f88_capture <- function(expr) {
  cpu <- list()
  pb <- list()
  warns <- list()
  msgs <- list()
  value <- withCallingHandlers(
    expr,
    cuplyr_cpu_fallback = function(c) cpu[[length(cpu) + 1L]] <<- c,
    cuplyr_plan_break = function(c) pb[[length(pb) + 1L]] <<- c,
    warning = function(w) {
      warns[[length(warns) + 1L]] <<- w
      invokeRestart("muffleWarning")
    },
    message = function(m) {
      msgs[[length(msgs) + 1L]] <<- m
      invokeRestart("muffleMessage")
    }
  )
  list(value = value, cpu = cpu, pb = pb, warns = warns, msgs = msgs)
}

f88_of_class <- function(conds, class) {
  Filter(function(c) inherits(c, class), conds)
}

f88_quiet_options <- function(.env = parent.frame()) {
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL, .local_envir = .env)
}

# Two pending lazy steps (filter + mutate) on the small table
f88_lazy2 <- function(df = f88_small()) {
  tbl_gpu(df, lazy = TRUE) |> filter(x > 1) |> mutate(w = x + 1)
}

# A valid CPU-fallback condition (signal, warning or error)
f88_expect_cpu_fields <- function(c) {
  expect_true(is.character(c$verb) && length(c$verb) == 1L && nzchar(c$verb))
  expect_true(is.character(c$reason) && length(c$reason) == 1L && !is.na(c$reason) &&
                nzchar(c$reason))
  expect_false(identical(c$reason, paste0(c$verb, "()")))
  for (f in c("rows", "cols", "bytes")) {
    expect_true(is.numeric(c[[f]]) && length(c[[f]]) == 1L && is.finite(c[[f]]) && c[[f]] >= 0,
                info = f)
  }
  expect_true(is.character(c$expr) && length(c$expr) == 1L)
}

# ---------------------------------------------------------------------------
# (1) a plan break is not a CPU fallback

test_that("F88: cuplyr.fallback = 'error' never fires for a lazy summarise that stays on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = NULL)
  df <- f88_small()
  expect_same_as_dplyr_lazy(df, function(d) d |> filter(x > 1) |> summarise(m = mean(x * y)))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> filter(x > 1) |> group_by(g) |> summarise(m = mean(x * y), s = sum(cumsum(x)))
  })

  got <- f88_capture(f88_lazy2() |> summarise(m = mean(x * y)) |> collect())
  expect_length(got$cpu, 0L)
  expect_length(got$pb, 1L)
  expect_length(got$warns, 0L)
  pb <- got$pb[[1L]]
  expect_identical(pb$verb, "summarise")
  expect_equal(pb$steps, 2)
  expect_true(is.character(pb$reason) && nzchar(pb$reason))
  expect_match(conditionMessage(pb), "mean(x * y)", fixed = TRUE)
  expect_match(conditionMessage(pb), "summarise()", fixed = TRUE)
  expect_match(conditionMessage(pb), "2 pending step")
  expect_no_match(conditionMessage(pb), "fell back|CPU evaluation")
})

test_that("F88: cuplyr.fallback = 'warn' reports no plan break as a CPU fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = NULL)
  df <- f88_small()
  pipelines <- list(
    temp_col = function(t) t |> summarise(m = mean(x * y)),
    window_arg = function(t) t |> summarise(s = sum(cumsum(x))),
    int_max = function(t) t |> summarise(m = max(i)),
    grouped_temp = function(t) t |> group_by(g) |> summarise(m = mean(x * y)),
    no_dots = function(t) t |> summarise(),
    zero_select = function(t) t |> select(-everything()),
    zero_transmute = function(t) t |> transmute(),
    reframe_int_sum = function(t) t |> reframe(s = sum(i), q = quantile(x, c(0.25, 0.75)))
  )
  for (nm in names(pipelines)) {
    got <- f88_capture(pipelines[[nm]](f88_lazy2(df)) |> collect())
    expect_length(got$warns, 0L)
    expect_length(got$cpu, 0L)
    expect_gte(length(got$pb), 1L)
    for (pb in got$pb) {
      expect_equal(pb$steps, 2, info = nm)
      expect_true(is.character(pb$verb) && nzchar(pb$verb), info = nm)
      expect_true(is.character(pb$reason) && nzchar(pb$reason), info = nm)
    }
  }
})

test_that("F88: every plan-break site warns under cuplyr.plan_break = 'warn', with the verb", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = "warn")
  df <- f88_small()
  cases <- list(
    list(f = function(t) t |> summarise(m = mean(x * y)), verb = "summarise", text = "mean(x * y)"),
    list(f = function(t) t |> summarise(s = sum(cumsum(x))), verb = "summarise", text = "sum(cumsum(x))"),
    list(f = function(t) t |> summarise(m = max(i)), verb = "summarise", text = "max"),
    list(f = function(t) t |> summarise(), verb = "summarise", text = NULL),
    list(f = function(t) t |> select(-everything()), verb = "select", text = NULL),
    list(f = function(t) t |> transmute(), verb = "transmute", text = NULL),
    list(f = function(t) t |> reframe(s = sum(i), q = quantile(x, c(0.25, 0.75))),
         verb = "reframe", text = "sum")
  )
  for (k in seq_along(cases)) {
    cs <- cases[[k]]
    got <- f88_capture(cs$f(f88_lazy2(df)) |> collect())
    w <- f88_of_class(got$warns, "cuplyr_warning_plan_break")
    expect_length(w, 1L)
    expect_length(got$warns, 1L)
    expect_false(inherits(w[[1L]], "cuplyr_cpu_fallback"))
    expect_identical(w[[1L]]$verb, cs$verb, info = k)
    expect_equal(w[[1L]]$steps, 2, info = k)
    expect_match(conditionMessage(w[[1L]]), paste0(cs$verb, "()"), fixed = TRUE, info = k)
    expect_match(conditionMessage(w[[1L]]), "2 pending step", info = k)
    if (!is.null(cs$text)) {
      expect_match(conditionMessage(w[[1L]]), cs$text, fixed = TRUE, info = k)
    }
    expect_no_match(conditionMessage(w[[1L]]), "fell back|CPU evaluation", info = k)
  }
})

test_that("F88: an eager table has no plan break and no fallback on those calls", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = "warn")
  df <- f88_small()
  got <- f88_capture({
    t <- tbl_gpu(df)
    list(
      summarise(t, m = mean(x * y)) |> collect(),
      summarise(t, s = sum(cumsum(x))) |> collect(),
      summarise(t, m = max(i)) |> collect(),
      select(t, -everything()) |> collect()
    )
  })
  expect_length(got$pb, 0L)
  expect_length(got$cpu, 0L)
  expect_length(got$warns, 0L)
})

test_that("F88: a plan break inside summarise(cur_group_id()) keeps the call on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = NULL)
  df <- f88_small()
  pipeline <- function(d) {
    d |> filter(x > 1) |> group_by(g) |> summarise(id = dplyr::cur_group_id(), m = mean(x * y))
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
  got <- f88_capture(pipeline(tbl_gpu(df, lazy = TRUE)) |> collect())
  expect_length(got$cpu, 0L)
})

# ---------------------------------------------------------------------------
# (3) CPU-fallback content: verb, expression, reason, size

test_that("F88: an eager CPU fallback names the expression, the reason and the size", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = NULL)
  df <- f88_df(1000L)
  got <- f88_capture(tbl_gpu(df) |> mutate(a = x + 1, w = qnorm(x / 10)) |> collect())
  expect_equal(got$value, dplyr::mutate(df, a = x + 1, w = qnorm(x / 10)))

  expect_length(got$cpu, 1L)
  w <- f88_of_class(got$warns, "cuplyr_warning_cpu_fallback")
  expect_length(w, 1L)
  expect_length(got$warns, 1L)
  expect_false(inherits(w[[1L]], "cuplyr_cpu_fallback"))
  for (c in list(got$cpu[[1L]], w[[1L]])) {
    f88_expect_cpu_fields(c)
    expect_identical(c$verb, "mutate")
    expect_identical(c$expr, "qnorm(x / 10)")
    expect_match(c$reason, "qnorm()", fixed = TRUE)
    expect_equal(c$rows, 1000)
    expect_equal(c$cols, 3)
    expect_gte(c$bytes, 24000)
    expect_lte(c$bytes, 30000)
  }
  msg <- conditionMessage(w[[1L]])
  expect_match(msg, "cuplyr: mutate() fell back to CPU evaluation for: qnorm(x / 10)", fixed = TRUE)
  expect_match(msg, "qnorm()", fixed = TRUE)
  expect_match(msg, "1,000 rows x 3 cols (about ", fixed = TRUE)
  expect_match(msg, "\\(about [0-9.]+ [kMG]?B\\) to run on the CPU")
})

test_that("F88: a lazy CPU fallback reports the size of the computed table", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = "warn")
  df <- f88_df(1000L)
  got <- f88_capture(
    tbl_gpu(df, lazy = TRUE) |> filter(x > 0.5) |> mutate(w = qnorm(x / 10)) |> collect()
  )
  expect_equal(got$value, df |> dplyr::filter(x > 0.5) |> dplyr::mutate(w = qnorm(x / 10)))
  expect_length(got$cpu, 1L)
  c <- got$cpu[[1L]]
  f88_expect_cpu_fields(c)
  expect_identical(c$verb, "mutate")
  expect_equal(c$rows, 500)
  expect_equal(c$cols, 3)
  expect_gte(c$bytes, 12000)
  expect_lte(c$bytes, 15000)
  w <- f88_of_class(got$warns, "cuplyr_warning_cpu_fallback")
  expect_length(w, 1L)
  expect_match(conditionMessage(w[[1L]]), "500 rows x 3 cols", fixed = TRUE)
})

test_that("F88: filter(), summarise() and arrange() name the unsupported function", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = NULL)
  df <- f88_df(200L)
  cases <- list(
    list(f = function(t) filter(t, qnorm(x / 10) < -1.5), verb = "filter",
         expr = "qnorm(x / 10) < -1.5"),
    list(f = function(t) summarise(t, s = sum(qnorm(x / 10))), verb = "summarise",
         expr = "sum(qnorm(x / 10))"),
    list(f = function(t) arrange(t, qnorm(z / 20)), verb = "arrange",
         expr = "qnorm(z / 20)")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(cs$f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_length(got$cpu, 1L)
      c <- got$cpu[[1L]]
      f88_expect_cpu_fields(c)
      expect_identical(c$verb, cs$verb)
      expect_identical(c$expr, cs$expr)
      expect_match(c$reason, "qnorm()", fixed = TRUE)
      expect_equal(c$rows, 200)
      expect_equal(c$cols, 3)
      w <- f88_of_class(got$warns, "cuplyr_warning_cpu_fallback")
      expect_length(w, 1L)
      expect_match(conditionMessage(w[[1L]]),
                   paste0(cs$verb, "() fell back to CPU evaluation for: ", cs$expr), fixed = TRUE)
    }
  }
  withr::local_options(cuplyr.fallback = NULL)
  expect_same_as_dplyr(df, function(d) filter(d, qnorm(x / 10) < -1.5))
  expect_same_as_dplyr_lazy(df, function(d) summarise(d, s = sum(qnorm(x / 10))))
})

test_that("F88: the reason names rowwise, a user's shadowing function and an unsupported type", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = NULL)
  df <- f88_small()

  got <- f88_capture(tbl_gpu(df) |> rowwise() |> mutate(m = qnorm(x / 10)) |> collect())
  expect_length(got$cpu, 1L)
  f88_expect_cpu_fields(got$cpu[[1L]])
  expect_match(got$cpu[[1L]]$reason, "rowwise")

  mean <- function(x) 42
  got <- f88_capture(tbl_gpu(df) |> summarise(m = mean(x)) |> collect())
  expect_equal(got$value$m, 42)
  expect_length(got$cpu, 1L)
  f88_expect_cpu_fields(got$cpu[[1L]])
  expect_identical(got$cpu[[1L]]$expr, "mean(x)")
  expect_match(got$cpu[[1L]]$reason, "`mean`", fixed = TRUE)

  dd <- tibble::tibble(d = as.Date("2020-01-01") + 0:5, v = 1:6)
  got <- f88_capture(tbl_gpu(dd) |> filter(d < 18265) |> collect())
  expect_equal(got$value, dplyr::filter(dd, d < 18265))
  expect_length(got$cpu, 1L)
  f88_expect_cpu_fields(got$cpu[[1L]])
  expect_identical(got$cpu[[1L]]$expr, "d < 18265")
  expect_match(got$cpu[[1L]]$reason, "(?i)type|date", perl = TRUE)
  expect_no_match(got$cpu[[1L]]$reason, "no GPU implementation")
})

test_that("F88: a selection that reads values is a CPU fallback with the table's size", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = "warn")
  df <- f88_df(1000L)
  got <- f88_capture(
    tbl_gpu(df, lazy = TRUE) |> filter(x > 0.5) |> select(where(~ mean(.x) > 0.7)) |> collect()
  )
  expect_equal(got$value, df |> dplyr::filter(x > 0.5) |> dplyr::select(where(~ mean(.x) > 0.7)))
  expect_length(got$cpu, 1L)
  expect_length(got$pb, 0L)
  c <- got$cpu[[1L]]
  f88_expect_cpu_fields(c)
  expect_identical(c$verb, "select")
  expect_match(c$reason, "selection")
  expect_equal(c$rows, 500)
  expect_equal(c$cols, 3)

  # .by = <value predicate> names the user's verb
  got <- f88_capture(
    tbl_gpu(df) |> mutate(n = dplyr::n(), .by = where(~ is.numeric(.x) && max(.x) > 9)) |> collect()
  )
  expect_length(got$cpu, 1L)
  expect_identical(got$cpu[[1L]]$verb, "mutate")
})

test_that("F88: a two-table fallback reports both tables' sizes", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.plan_break = NULL)
  a <- tibble::tibble(x = c(1, 2, 3, 4), y = c(2, 3, 4, 5))
  b <- tibble::tibble(x = c(1, 3, 5), y = c(1, 4, 6), v = c(10, 20, 30))
  got <- f88_capture(
    left_join(tbl_gpu(a), tbl_gpu(b), by = dplyr::join_by(closest(x >= x), closest(y >= y))) |> collect()
  )
  expect_length(got$cpu, 1L)
  c <- got$cpu[[1L]]
  f88_expect_cpu_fields(c)
  expect_identical(c$verb, "left_join")
  expect_match(c$reason, "closest")
  expect_equal(c$rows, 7)
  expect_equal(c$cols, 5)
  expect_gte(c$bytes, 8 * (4 * 2 + 3 * 3))
})

test_that("F88: every CPU fallback carries a real reason and a size", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- f88_small()
  shadow_abs <- function(t) {
    abs <- function(x) x
    t |> mutate(a = abs(x))
  }
  pipelines <- list(
    function(t) t |> mutate(m = qnorm(x / 10)),
    function(t) t |> transmute(m = qnorm(x / 10)),
    function(t) t |> group_by(g) |> mutate(m = qnorm(x / 10)),
    function(t) t |> filter(qnorm(x / 10) > -2),
    function(t) t |> summarise(m = sum(qnorm(x / 10))),
    function(t) t |> group_by(g) |> summarise(m = sum(qnorm(x / 10))),
    function(t) t |> arrange(qnorm(x / 10)),
    function(t) t |> count(q = qnorm(x / 10)),
    function(t) t |> distinct(q = qnorm(x / 10)),
    function(t) t |> slice_min(qnorm(x / 10), n = 2),
    function(t) t |> reframe(q = range(qnorm(x / 10))),
    function(t) t |> rowwise() |> summarise(m = median(c(x, y))),
    shadow_abs
  )
  for (k in seq_along(pipelines)) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(pipelines[[k]](tbl_gpu(df, lazy = lazy)) |> collect())
      expect_gte(length(got$cpu), 1L)
      for (c in got$cpu) {
        f88_expect_cpu_fields(c)
        expect_equal(c$rows, 6, info = paste(k, lazy))
        expect_equal(c$cols, 4, info = paste(k, lazy))
      }
      expect_length(got$warns, 0L)
      expect_length(got$msgs, 0L)
    }
  }
})

test_that("F88: cuplyr.fallback = 'error' raises a classed error with the fields", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error", cuplyr.plan_break = NULL)
  df <- f88_df(100L)
  err <- tryCatch(tbl_gpu(df) |> mutate(w = qnorm(x / 10)), error = function(e) e)
  expect_s3_class(err, "cuplyr_error_fallback")
  f88_expect_cpu_fields(err)
  expect_identical(err$verb, "mutate")
  expect_identical(err$expr, "qnorm(x / 10)")
  expect_match(err$reason, "qnorm()", fixed = TRUE)
  expect_equal(err$rows, 100)
  expect_match(conditionMessage(err), "fell back to CPU evaluation for: qnorm(x / 10)", fixed = TRUE)
  expect_match(conditionMessage(err), "qnorm()", fixed = TRUE)
  expect_match(conditionMessage(err), "100 rows x 3 cols", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# (4) option values

test_that("F88: cuplyr.fallback = 'warning' is an alias of 'warn'", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warning", cuplyr.plan_break = NULL)
  got <- f88_capture(tbl_gpu(f88_small()) |> mutate(w = qnorm(x / 10)) |> collect())
  expect_length(f88_of_class(got$warns, "cuplyr_warning_cpu_fallback"), 1L)
  expect_length(f88_of_class(got$warns, "cuplyr_warning_invalid_option"), 0L)
  expect_length(got$warns, 1L)
})

test_that("F88: cuplyr.plan_break = 'warning' is an alias of 'warn'", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = "warning")
  got <- f88_capture(f88_lazy2() |> summarise(m = mean(x * y)) |> collect())
  expect_length(f88_of_class(got$warns, "cuplyr_warning_plan_break"), 1L)
  expect_length(got$warns, 1L)
})

test_that("F88: an invalid cuplyr.fallback value warns once and is treated as silent", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "bogus-f88-a", cuplyr.plan_break = NULL)
  df <- f88_small()
  got <- f88_capture(tbl_gpu(df) |> mutate(w = qnorm(x / 10)) |> collect())
  inv <- f88_of_class(got$warns, "cuplyr_warning_invalid_option")
  expect_length(inv, 1L)
  expect_length(got$warns, 1L)
  expect_identical(inv[[1L]]$option, "cuplyr.fallback")
  msg <- conditionMessage(inv[[1L]])
  expect_match(msg, "cuplyr.fallback", fixed = TRUE)
  expect_match(msg, "bogus-f88-a", fixed = TRUE)
  expect_match(msg, "\"silent\"", fixed = TRUE)
  expect_match(msg, "\"warn\"", fixed = TRUE)
  expect_match(msg, "\"error\"", fixed = TRUE)
  expect_length(got$cpu, 1L)

  # the same value again: no second warning
  got <- f88_capture(tbl_gpu(df) |> filter(qnorm(x / 10) > -2) |> collect())
  expect_length(got$warns, 0L)
  expect_length(got$cpu, 1L)

  # another invalid value warns again
  withr::local_options(cuplyr.fallback = "bogus-f88-b")
  got <- f88_capture(tbl_gpu(df) |> mutate(w = qnorm(x / 10)) |> collect())
  expect_length(f88_of_class(got$warns, "cuplyr_warning_invalid_option"), 1L)
  expect_length(got$warns, 1L)
})

test_that("F88: an invalid cuplyr.plan_break value ('error' included) warns once, never errors", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = "error")
  got <- f88_capture(f88_lazy2() |> summarise(m = mean(x * y)) |> collect())
  inv <- f88_of_class(got$warns, "cuplyr_warning_invalid_option")
  expect_length(inv, 1L)
  expect_length(got$warns, 1L)
  expect_identical(inv[[1L]]$option, "cuplyr.plan_break")
  expect_match(conditionMessage(inv[[1L]]), "cuplyr.plan_break", fixed = TRUE)
  expect_length(got$pb, 1L)
  expect_equal(nrow(got$value), 1L)

  got <- f88_capture(f88_lazy2() |> summarise(m = mean(x * y)) |> collect())
  expect_length(got$warns, 0L)

  withr::local_options(cuplyr.plan_break = TRUE)
  got <- f88_capture(f88_lazy2() |> summarise(m = mean(x * y)) |> collect())
  expect_length(f88_of_class(got$warns, "cuplyr_warning_invalid_option"), 1L)
  expect_length(got$warns, 1L)
})

test_that("F88: the default stays silent for fallbacks and plan breaks, but both are signalled", {
  skip_if_no_gpu()
  f88_quiet_options()
  got <- f88_capture({
    a <- f88_lazy2() |> summarise(m = mean(x * y)) |> collect()
    b <- tbl_gpu(f88_small()) |> mutate(w = qnorm(x / 10)) |> collect()
    list(a, b)
  })
  expect_length(got$warns, 0L)
  expect_length(got$msgs, 0L)
  expect_length(got$cpu, 1L)
  expect_length(got$pb, 1L)
})

# ---------------------------------------------------------------------------
# (6) docs

test_that("F88: the fallback options have a user-facing help page, vignette text and NEWS", {
  root <- testthat::test_path("..", "..")
  skip_if_not(file.exists(file.path(root, "DESCRIPTION")))
  rd_file <- file.path(root, "man", "cuplyr-fallback.Rd")
  expect_true(file.exists(rd_file))
  rd <- if (file.exists(rd_file)) paste(readLines(rd_file), collapse = "\n") else ""
  expect_no_match(rd, "\\keyword{internal}", fixed = TRUE)
  for (s in c("cuplyr.fallback", "cuplyr.plan_break", "cuplyr_cpu_fallback",
              "cuplyr_plan_break", "\"warning\"")) {
    expect_match(rd, s, fixed = TRUE, info = s)
  }
  vig <- paste(readLines(file.path(root, "vignettes", "performance.Rmd")), collapse = "\n")
  expect_match(vig, "cuplyr.plan_break", fixed = TRUE)
  news <- paste(readLines(file.path(root, "NEWS.md")), collapse = "\n")
  expect_match(news, "cuplyr.plan_break", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# Added after round 1 review (B1-B4): the reason blames the dot that fell
# back, not a native reducer; overflow reruns say so; a reason never comes
# from an earlier call; rowwise two-table fallbacks count both tables.

test_that("F88: summarise()/reframe() blame the dot that fell back, not a native reducer", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), g = c("a", "a", "b", "b", "c", "c"))
  cases <- list(
    list(f = function(t) reframe(t, q = range(qnorm(x / 10))), expr = "range(qnorm(x / 10))"),
    list(f = function(t) summarise(t, q = quantile(x, 0.9), s = sum(qnorm(x / 10))),
         expr = "sum(qnorm(x / 10))"),
    list(f = function(t) summarise(t, a = any(x > 3), s = sd(qnorm(x / 10))),
         expr = "sd(qnorm(x / 10))")
  )
  for (cs in cases) {
    got <- f88_capture(cs$f(tbl_gpu(df)) |> collect())
    expect_length(got$cpu, 1L)
    c <- got$cpu[[1L]]
    expect_identical(c$expr, cs$expr)
    expect_match(c$reason, "qnorm()", fixed = TRUE)
    expect_no_match(c$reason, "`(range|quantile|any|sum|sd)\\(\\)`")
  }
})

test_that("F88: an eager mutate() rerun after an integer sum() overflow says so", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  d <- tibble::tibble(i = c(.Machine$integer.max, 1L))
  got <- f88_capture(tbl_gpu(d) |> mutate(s = sum(i)) |> collect())
  expect_equal(got$value, dplyr::mutate(d, s = sum(i)))
  expect_length(got$cpu, 1L)
  f88_expect_cpu_fields(got$cpu[[1L]])
  expect_match(got$cpu[[1L]]$reason, "overflow")
})

test_that("F88: a fallback's reason never comes from an earlier call", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  dd <- tibble::tibble(d = as.Date("2020-01-01") + 0:5)
  nd <- tibble::tibble(d = c(18000, 18300), s = c("a", "b"))
  run <- function(f) {
    got <- f88_capture(f() |> collect())
    expect_length(got$cpu, 1L)
    got$cpu[[1L]]
  }
  c0 <- run(function() tbl_gpu(nd) |> filter(d < 18265, s < 5))
  c1 <- run(function() tbl_gpu(dd) |> filter(d < 18265))
  c2 <- run(function() tbl_gpu(nd) |> filter(d < 18265, s < 5))
  expect_identical(c2$expr, c0$expr)
  expect_identical(c2$reason, c0$reason)
  expect_identical(c0$expr, "s < 5")
  expect_no_match(c2$reason, "TIMESTAMP|Date")
  expect_identical(c1$expr, "d < 18265")
})

test_that("F88: a rowwise two-table fallback reports both tables' sizes", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  a <- tibble::tibble(x = c(1, 2, 3, 4), y = c(2, 3, 4, 5))
  b <- tibble::tibble(x = c(1, 3, 5), y = c(1, 4, 6))
  got <- f88_capture(union(rowwise(tbl_gpu(a)), tbl_gpu(b)) |> collect())
  expect_gte(length(got$cpu), 1L)
  c <- got$cpu[[length(got$cpu)]]
  f88_expect_cpu_fields(c)
  expect_equal(c$rows, 7)
  expect_equal(c$cols, 4)
  got <- f88_capture(rows_update(rowwise(tbl_gpu(a)), tbl_gpu(b[1:2, ]), by = "x") |> collect())
  expect_gte(length(got$cpu), 1L)
  c <- got$cpu[[length(got$cpu)]]
  f88_expect_cpu_fields(c)
  expect_equal(c$rows, 6)
  expect_equal(c$cols, 4)
})

# ---------------------------------------------------------------------------
# Added after round 2 review (B3 again, B5-B7): reasons are scoped to the
# call that falls back; a dot the GPU runs natively is never blamed; nothing
# a notice records outlives the call.

test_that("F88: a reason never comes from an earlier call, even when an environment value changed", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  nd <- tibble::tibble(d = c(18000, 18300), s = c("a", "b"))
  # (every call is written in this one environment, as at top level, so the
  # quosures share it)
  one <- function(x) {
    got <- f88_capture(collect(x))
    expect_length(got$cpu, 1L)
    got$cpu[[1L]]
  }
  v <- 18265
  a <- one(tbl_gpu(nd) |> filter(d < v, s < 5))
  b <- one(tbl_gpu(nd) |> mutate(a = d < v, b = s < 5))
  v <- as.Date("2020-01-01")
  c <- one(tbl_gpu(nd) |> filter(d < v))
  expect_identical(c$expr, "d < v")
  v <- 18265
  d <- one(tbl_gpu(nd) |> filter(d < v, s < 5))
  e <- one(tbl_gpu(nd) |> mutate(a = d < v, b = s < 5))
  expect_identical(a$expr, "s < 5")
  expect_identical(b$expr, "s < 5")
  expect_identical(d$expr, a$expr)
  expect_identical(d$reason, a$reason)
  expect_identical(e$expr, b$expr)
  expect_identical(e$reason, b$reason)
  expect_no_match(d$reason, "TIMESTAMP|Date")
  expect_no_match(e$reason, "TIMESTAMP|Date")
})

test_that("F88: a notice keeps nothing of the caller's environment alive", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  freed <- new.env()
  freed$done <- FALSE
  f <- function() {
    marker <- new.env()
    reg.finalizer(marker, function(e) freed$done <- TRUE)
    g <- tbl_gpu(tibble::tibble(d = as.Date("2020-01-01") + 0:9, s = letters[1:10]))
    a <- nrow(collect(filter(g, d < 18265)))
    b <- nrow(collect(filter(g, s < 5)))
    c <- nrow(collect(summarise(g, m = sum(qnorm(as.numeric(d) / 1e5)))))
    a + b + c
  }
  f88_capture(f())
  gc()
  gc()
  expect_true(freed$done)
})

test_that("F88: a rowwise fallback blames the dot that fell back, not a head the rowwise rewrite runs natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7))
  cases <- list(
    list(f = function(t) mutate(t, p = prod(c(x, y)), z = qnorm(x / 10)), expr = "qnorm(x / 10)"),
    list(f = function(t) mutate(t, l = length(c(x, y)), z = qnorm(x / 10)), expr = "qnorm(x / 10)"),
    list(f = function(t) filter(t, prod(c(x, y)) > 10, qnorm(x / 10) > -2),
         expr = "qnorm(x / 10) > -2"),
    list(f = function(t) summarise(t, p = prod(c(x, y)), z = qnorm(x / 10)), expr = "qnorm(x / 10)")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(cs$f(rowwise(tbl_gpu(df, lazy = lazy))) |> collect())
      expect_length(got$cpu, 1L)
      c <- got$cpu[[1L]]
      expect_identical(c$expr, cs$expr)
      expect_match(c$reason, "rowwise")
      expect_match(c$reason, "qnorm()", fixed = TRUE)
      expect_no_match(c$reason, "`(prod|length)\\(\\)`")
    }
  }
  controls <- list(
    function(t) mutate(t, p = prod(c(x, y))),
    function(t) mutate(t, l = length(c(x, y))),
    function(t) filter(t, prod(c(x, y)) > 10)
  )
  for (f in controls) {
    got <- f88_capture(f(rowwise(tbl_gpu(df))) |> collect())
    expect_length(got$cpu, 0L)
  }
})

test_that("F88: a column-free call folded inside a formula is not blamed", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7))
  k <- 1
  cases <- list(
    list(f = function(t) mutate(t, a = dplyr::case_when(x > 2 ~ log1p(k), .default = 0), b = qnorm(x / 10)),
         expr = "qnorm(x / 10)"),
    list(f = function(t) mutate(t, across(c(x, y), ~ .x * log1p(k)), b = qnorm(x / 10)),
         expr = "qnorm(x / 10)"),
    list(f = function(t) {
      filter(t, dplyr::case_when(x > 2 ~ log1p(k) > 0, .default = FALSE), qnorm(x / 10) > -2)
    }, expr = "qnorm(x / 10) > -2")
  )
  for (cs in cases) {
    got <- f88_capture(cs$f(tbl_gpu(df)) |> collect())
    expect_length(got$cpu, 1L)
    c <- got$cpu[[1L]]
    expect_identical(c$expr, cs$expr)
    expect_match(c$reason, "qnorm()", fixed = TRUE)
    expect_no_match(c$reason, "log1p")
  }
  controls <- list(
    function(t) mutate(t, a = dplyr::case_when(x > 2 ~ log1p(k), .default = 0)),
    function(t) mutate(t, across(c(x, y), ~ .x * log1p(k))),
    function(t) filter(t, dplyr::case_when(x > 2 ~ log1p(k) > 0, .default = FALSE))
  )
  for (f in controls) {
    got <- f88_capture(f(tbl_gpu(df)) |> collect())
    expect_length(got$cpu, 0L)
  }
})

# ---------------------------------------------------------------------------
# Added after round 3 review (B8-B11): "`f()` has no GPU implementation" is
# said only of a function cuplyr has no GPU handler for; a natively run dot is
# never blamed; one dot's cause never becomes another dot's reason.

# Function names a "`f()` has no GPU implementation" reason names
f88_no_gpu_names <- function(reason) {
  m <- regmatches(reason, gregexpr("`[^`]+\\(\\)` has no GPU implementation", reason))[[1L]]
  sub("^`([^`]+)\\(\\)`.*$", "\\1", m)
}

# Heads cuplyr runs on the GPU somewhere (IR registry, summarise reducers,
# syntax and sort helpers): never "no GPU implementation"
f88_gpu_heads <- function() {
  c(names(ir_call_registry), agg_functions, "range", "desc", "$", "[[", "(", "~",
    "c", "cur_group_id", "if_any", "if_all", "across", "pick")
}

test_that("F88: summarise() never says a function it has a GPU handler for has no GPU implementation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7))
  cases <- list(
    list(f = function(t) summarise(t, r = max(x) - min(x)), expr = "max(x) - min(x)"),
    list(f = function(t) summarise(t, m = sqrt(mean(x))), expr = "sqrt(mean(x))"),
    list(f = function(t) summarise(t, m = round(mean(x), 1)), expr = "round(mean(x), 1)")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(cs$f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_equal(got$value, cs$f(df))
      expect_length(got$cpu, 1L)
      c <- got$cpu[[1L]]
      f88_expect_cpu_fields(c)
      expect_identical(c$expr, cs$expr)
      expect_length(intersect(f88_no_gpu_names(c$reason), f88_gpu_heads()), 0L)
    }
  }
  controls <- list(
    function(t) mutate(t, r = x - y, s = sqrt(x), u = round(x, 1)),
    function(t) summarise(t, a = max(x), b = min(x), m = mean(x))
  )
  for (f in controls) {
    got <- f88_capture(f(tbl_gpu(df)) |> collect())
    expect_length(got$cpu, 0L)
  }
})

test_that("F88: a grouped summarise() with cur_group_id() blames the dot that fell back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), g = c("a", "a", "b", "b", "c", "c"))
  for (lazy in c(FALSE, TRUE)) {
    got <- f88_capture(
      tbl_gpu(df, lazy = lazy) |> group_by(g) |>
        summarise(id = dplyr::cur_group_id(), m = mean(qnorm(x / 10))) |> collect()
    )
    expect_gte(length(got$cpu), 1L)
    c <- got$cpu[[length(got$cpu)]]
    f88_expect_cpu_fields(c)
    expect_identical(c$expr, "mean(qnorm(x / 10))")
    expect_match(c$reason, "qnorm()", fixed = TRUE)
    expect_no_match(c$reason, "cur_group_id")
  }
  got <- f88_capture(
    tbl_gpu(df) |> group_by(g) |> summarise(id = dplyr::cur_group_id(), m = mean(x / 10)) |> collect()
  )
  expect_length(got$cpu, 0L)
})

test_that("F88: arrange() never blames desc(), a pronoun or another sort-key helper", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(3, 1, 2, 6, 5, 4), y = c(2, 3, 4, 5, 6, 7))
  pipelines <- list(
    function(t) arrange(t, desc(x + y)),
    function(t) arrange(t, .data$x + 1),
    function(t) arrange(t, desc(-x)),
    function(t) arrange(t, .data[["x"]] * 2)
  )
  for (f in pipelines) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(f(tbl_gpu(df, lazy = lazy)) |> collect())
      expect_equal(got$value, f(df))
      for (c in got$cpu) {
        f88_expect_cpu_fields(c)
        expect_length(intersect(f88_no_gpu_names(c$reason), f88_gpu_heads()), 0L)
      }
    }
  }
  got <- f88_capture(tbl_gpu(df) |> arrange(desc(x), y) |> collect())
  expect_length(got$cpu, 0L)
})

test_that("F88: one dot's cause never becomes another dot's reason", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7))
  for (lazy in c(FALSE, TRUE)) {
    alone <- f88_capture(
      tbl_gpu(df, lazy = lazy) |> mutate(p = as.difftime(1, units = "days"), p = NULL) |> collect()
    )
    both <- f88_capture(
      tbl_gpu(df, lazy = lazy) |>
        mutate(p = as.difftime(1, units = "days"),
               a = dplyr::if_any(c(x, y), ~ qnorm(.x / 10) > -2), p = NULL) |>
        collect()
    )
    expect_length(alone$cpu, 1L)
    expect_length(both$cpu, 1L)
    a <- alone$cpu[[1L]]
    b <- both$cpu[[1L]]
    if (identical(b$expr, a$expr)) {
      expect_identical(b$reason, a$reason)
    }
    if (!is.na(b$expr) && !grepl("qnorm", b$expr, fixed = TRUE)) {
      expect_no_match(b$reason, "qnorm", fixed = TRUE)
    }
  }
})

test_that("F88: no reason says a function with a GPU handler has no GPU implementation", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  df <- tibble::tibble(x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7),
                       g = c("a", "a", "b", "b", "c", "c"))
  pipelines <- list(
    function(t) summarise(t, r = max(x) / sum(y)),
    function(t) summarise(t, r = abs(mean(x) - median(y))),
    function(t) group_by(t, g) |> summarise(r = log(sum(x)), q = qnorm(mean(x) / 10)),
    function(t) mutate(t, a = round(x / 3, 1), b = qnorm(x / 10)),
    function(t) mutate(t, a = dplyr::if_else(x > 2, sqrt(x), 0), b = pnorm(x)),
    function(t) filter(t, abs(x - y) < 2, qnorm(x / 10) > -2),
    function(t) arrange(t, abs(x - 3)),
    function(t) arrange(t, desc(round(x / 2))),
    function(t) transmute(t, a = cumsum(x), b = qnorm(x / 10)),
    function(t) reframe(t, r = range(x) * 2),
    function(t) rowwise(t) |> mutate(m = max(c(x, y)) + qnorm(x / 10)),
    function(t) group_by(t, g) |> mutate(m = mean(x), z = qnorm(m / 10))
  )
  for (k in seq_along(pipelines)) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88_capture(pipelines[[k]](tbl_gpu(df, lazy = lazy)) |> collect())
      for (c in got$cpu) {
        f88_expect_cpu_fields(c)
        expect_length(intersect(f88_no_gpu_names(c$reason), f88_gpu_heads()), 0L)
      }
    }
  }
})
