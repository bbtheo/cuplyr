# F88: the reason and expr of a CPU-fallback notice name the cause. These pin
# the reason helper on shapes the acceptance file does not (round 2 of F88).

f88r_df <- function() {
  tibble::tibble(
    x = c(1, 2, 3, 4, 5, 6), y = c(2, 3, 4, 5, 6, 7), g = c("a", "a", "b", "b", "c", "c")
  )
}

# The CPU-fallback conditions `expr` signals (warnings muffled)
f88r_cpu <- function(expr) {
  got <- list()
  withCallingHandlers(
    expr,
    cuplyr_cpu_fallback = function(c) got[[length(got) + 1L]] <<- c,
    warning = function(w) invokeRestart("muffleWarning")
  )
  got
}

f88r_run <- function(f, lazy = FALSE) {
  f88r_cpu(f(tbl_gpu(f88r_df(), lazy = lazy)) |> collect())
}

test_that("F88: a native dot before the failing one, or a native reducer around it, is not blamed", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  cases <- list(
    list(f = function(t) mutate(t, a = x + 1, b = qnorm(x / 10)), expr = "qnorm(x / 10)"),
    list(f = function(t) mutate(t, a = abs(x), b = abs(qnorm(x / 10))), expr = "abs(qnorm(x / 10))"),
    list(f = function(t) transmute(t, a = x + 1, b = qnorm(x / 10)), expr = "qnorm(x / 10)"),
    list(f = function(t) filter(t, x > 1, qnorm(x / 10) > -2), expr = "qnorm(x / 10) > -2"),
    list(f = function(t) summarise(t, a = mean(x), s = sum(qnorm(x / 10)), b = max(y)),
         expr = "sum(qnorm(x / 10))"),
    list(f = function(t) group_by(t, g) |> summarise(a = mean(x), s = sd(qnorm(x / 10))),
         expr = "sd(qnorm(x / 10))"),
    list(f = function(t) reframe(t, a = range(x), b = quantile(qnorm(x / 10), c(0.25, 0.5))),
         expr = "quantile(qnorm(x / 10), c(0.25, 0.5))"),
    list(f = function(t) arrange(t, y, qnorm(x / 10)), expr = "qnorm(x / 10)"),
    list(f = function(t) mutate(t, z = cumsum(qnorm(x / 10))), expr = "cumsum(qnorm(x / 10))")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88r_run(cs$f, lazy)
      expect_length(got, 1L)
      expect_identical(got[[1L]]$expr, cs$expr)
      expect_match(got[[1L]]$reason, "`qnorm()` has no GPU implementation.", fixed = TRUE)
    }
  }
})

test_that("F88: a native reducer in a form the GPU does not take is not called unimplemented", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_run(function(t) summarise(t, q = quantile(x, 0.5, type = 3)))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "quantile(x, 0.5, type = 3)")
  expect_no_match(got[[1L]]$reason, "no GPU implementation")
  got <- f88r_run(function(t) reframe(t, q = quantile(x, c(0.25, 0.5), type = 3)))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "quantile(x, c(0.25, 0.5), type = 3)")
  expect_no_match(got[[1L]]$reason, "no GPU implementation")
  # a reducer that is not native is named, and only its own dot is blamed
  got <- f88r_run(function(t) summarise(t, a = mean(x), r = cor(x, y), m = max(y)))
  expect_identical(got[[1L]]$expr, "cor(x, y)")
  expect_match(got[[1L]]$reason, "`cor()` has no GPU implementation.", fixed = TRUE)
  # a type the reducer does not take on the GPU names the type
  d <- tibble::tibble(a = 1:3, d = as.Date("2020-01-01") + 0:2)
  got <- f88r_cpu(tbl_gpu(d) |> summarise(m = mean(a), s = sd(d)) |> collect())
  expect_identical(got[[1L]]$expr, "sd(d)")
  expect_match(got[[1L]]$reason, "TIMESTAMP_DAYS")
})

test_that("F88: across() names the function in .fns that has no GPU form", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  cases <- list(
    function(t) mutate(t, across(c(x, y), qnorm)),
    function(t) mutate(t, across(c(x, y), ~ qnorm(.x / 10))),
    function(t) mutate(t, across(c(x, y), list(a = abs, b = \(v) qnorm(v / 10)))),
    function(t) summarise(t, across(c(x, y), ~ sum(qnorm(.x / 10))))
  )
  for (f in cases) {
    got <- f88r_run(f)
    expect_length(got, 1L)
    expect_match(got[[1L]]$reason, "`qnorm()` has no GPU implementation.", fixed = TRUE)
  }
})

test_that("F88: a reason is recorded by the call that falls back, never read from an earlier one", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  nd <- tibble::tibble(d = c(18000, 18300), s = c("a", "b"))
  # The same expression in the same environment, first with a Date `v` (a type
  # the GPU hands to the CPU), then with a number (native): the later call's
  # notice names its own failing dot. A memo keyed on the expression, its
  # environment and the column types would hand the first call's reason over.
  v <- as.Date("2020-01-01")
  first <- f88r_cpu(tbl_gpu(nd) |> mutate(p = d < v) |> collect())
  expect_length(first, 1L)
  expect_identical(first[[1L]]$expr, "d < v")
  expect_match(first[[1L]]$reason, "TIMESTAMP_DAYS")
  v <- 18265
  for (f in list(
    function(t) mutate(t, p = d < v, q = qnorm(d / 1e5)),
    function(t) filter(t, d < v, qnorm(d / 1e5) > -10),
    function(t) summarise(t, p = sum(d < v), q = sum(qnorm(d / 1e5)))
  )) {
    got <- f88r_cpu(f(tbl_gpu(nd)) |> collect())
    expect_length(got, 1L)
    expect_match(got[[1L]]$expr, "qnorm(d / 1e+05)", fixed = TRUE)
    expect_match(got[[1L]]$reason, "`qnorm()` has no GPU implementation.", fixed = TRUE)
  }
  # nothing is left once the verb returns
  expect_null(fallback_reasons$cur)
})

test_that("F88: summarise() names the type of a computed argument, and of a bare one next to it", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  dd <- tibble::tibble(a = c(1, 2, 3), d = as.Date("2020-01-01") + 0:2)
  got <- f88r_cpu(tbl_gpu(dd) |> summarise(n = sum(d < 18265)) |> collect())
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "sum(d < 18265)")
  expect_match(got[[1L]]$reason, "TIMESTAMP_DAYS")
  # a computed dot first does not hide the bare dot's own reason
  got <- f88r_cpu(tbl_gpu(dd) |> summarise(m = mean(a + 1), s = sd(d)) |> collect())
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "sd(d)")
  expect_match(got[[1L]]$reason, "`sd()` of a TIMESTAMP_DAYS column", fixed = TRUE)
})

test_that("F88: a dot reading a column an earlier dot made names its own function", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  for (lazy in c(FALSE, TRUE)) {
    got <- f88r_run(function(t) mutate(t, a = x + 1, b = qnorm(a)), lazy)
    expect_length(got, 1L)
    expect_identical(got[[1L]]$expr, "qnorm(a)")
    expect_match(got[[1L]]$reason, "`qnorm()` has no GPU implementation.", fixed = TRUE)
  }
})

test_that("F88: an error inside the native rowwise attempt stays visible in the notice", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  local_mocked_bindings(mutate_run = function(...) stop("native rowwise boom"))
  got <- f88r_cpu(rowwise(tbl_gpu(f88r_df())) |> mutate(z = x + y) |> collect())
  expect_length(got, 1L)
  expect_match(got[[1L]]$reason, "rowwise table: the native rowwise attempt failed: native rowwise boom",
               fixed = TRUE)
})

test_that("F88: one dot with no cause found keeps the caller's own reason", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_run(function(t) arrange(t, x + y))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "x + y")
  expect_no_match(got[[1L]]$reason, "cannot translate")
})

test_that("F88: the reason of a notice has no attributes left over", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_run(function(t) mutate(t, b = qnorm(x / 10)))
  expect_null(attributes(got[[1L]]$reason))
  got <- f88r_run(function(t) summarise(t, s = sum(qnorm(x / 10))))
  expect_null(attributes(got[[1L]]$reason))
})

test_that("F88: a rowwise fallback names the function with no GPU form", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  cases <- list(
    function(t) rowwise(t) |> mutate(m = qnorm(x / 10)),
    function(t) rowwise(t) |> transmute(m = qnorm(x / 10)),
    function(t) rowwise(t) |> filter(qnorm(x / 10) > -2),
    function(t) rowwise(t) |> summarise(m = qnorm(x / 10))
  )
  for (f in cases) {
    got <- f88r_run(f)
    expect_length(got, 1L)
    expect_match(got[[1L]]$reason, "`qnorm()` has no GPU implementation.", fixed = TRUE)
    expect_no_match(got[[1L]]$reason, "CPU fallback was requested")
  }
})

test_that("F88: a rowwise two-table fallback counts a lazy y with pending steps", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  a <- tibble::tibble(x = c(1, 2, 3, 4), y = c(2, 3, 4, 5))
  b <- tibble::tibble(x = c(1, 3, 5), y = c(1, 4, 6))
  y <- tbl_gpu(b, lazy = TRUE) |> filter(x > 1)
  got <- f88r_cpu(union(rowwise(tbl_gpu(a)), y) |> collect())
  expect_length(got, 1L)
  expect_equal(got[[1L]]$rows, 6)
  expect_equal(got[[1L]]$cols, 4)
})

test_that("F88: an invalid-option warning survives an integer sum() overflow rerun", {
  held <- structure(class = c("cuplyr_warning_invalid_option", "warning", "condition"),
                    list(message = "invalid option", call = NULL))
  expect_warning(
    res <- retry_on_sum_overflow(
      function() {
        warning(held)
        rlang::abort("overflow", class = "cuplyr_error_sum_overflow")
      },
      function() "rerun"
    ),
    class = "cuplyr_warning_invalid_option"
  )
  expect_identical(res, "rerun")
  # any other warning of the overflowing attempt is dropped, as before
  expect_no_warning(
    retry_on_sum_overflow(
      function() {
        warning("noise")
        rlang::abort("overflow", class = "cuplyr_error_sum_overflow")
      },
      function() "rerun"
    )
  )
})

test_that("F88: the decision that sends a dot to the CPU names it, for every verb and shape", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  cases <- list(
    list(f = function(t) filter(t, x > 0, if_any(c(x, y), ~ qnorm(.x / 10) > -2)),
         expr = "if_any(c(x, y), ~ qnorm(.x / 10) > -2)", reason = "`qnorm()` has no GPU implementation."),
    list(f = function(t) transmute(t, across(c(x, y), qnorm)),
         expr = "across(c(x, y), qnorm)", reason = "`qnorm()` has no GPU implementation."),
    list(f = function(t) mutate(t, a = x + 1:6), expr = "x + 1:6", reason = "vector constant"),
    # (dplyr:: as dplyr is not attached under tools/test-file, and dplyr's rerun
    # evaluates these by name)
    list(f = function(t) mutate(t, z = dplyr::across(c(x, y), abs)), expr = "dplyr::across(c(x, y), abs)",
         reason = "named `across()`"),
    list(f = function(t) mutate(t, n = nrow(dplyr::pick(x, y))), expr = "nrow(dplyr::pick(x, y))",
         reason = "`pick()`"),
    list(f = function(t) reframe(t, c = cor(x, y)), expr = "cor(x, y)", reason = "`cor()`"),
    list(f = function(t) reframe(t, t = range(x), range(y)), expr = "range(y)", reason = "unnamed"),
    list(f = function(t) summarise(t, n = n_distinct(x, y)), expr = "n_distinct(x, y)",
         reason = "`n_distinct()` is not on the GPU with these arguments."),
    list(f = function(t) mutate(rowwise(t), a = x + runif(1)), expr = "x + runif(1)",
         reason = "rowwise table: a call that reads no column"),
    list(f = function(t) count(t, q = qnorm(x / 10)), expr = "qnorm(x / 10)",
         reason = "`qnorm()` has no GPU implementation."),
    list(f = function(t) slice_max(t, qnorm(x / 10), n = 2), expr = "qnorm(x / 10)",
         reason = "`qnorm()` has no GPU implementation.")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- withr::with_seed(1, f88r_run(cs$f, lazy))
      expect_length(got, 1L)
      expect_identical(got[[1L]]$expr, cs$expr)
      expect_match(got[[1L]]$reason, cs$reason, fixed = TRUE)
    }
  }
  # a whole-call fallback names no dot
  got <- f88r_run(function(t) summarise(t, x = max(x), m = mean(x)))
  expect_identical(got[[1L]]$expr, NA_character_)
  expect_match(got[[1L]]$reason, "in sequence", fixed = TRUE)
})

test_that("F88: a vctrs notice puts its description in the reason, with no expression", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_cpu(vctrs::vec_proxy_equal(tbl_gpu(f88r_df())))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, NA_character_)
  expect_match(got[[1L]]$reason, "vec_proxy_equal() of a tbl_gpu: ", fixed = TRUE)
})

# ---------------------------------------------------------------------------
# Round 4: the two invariants by construction, and N19-N21.

test_that("F88: `f()` has no GPU implementation is said only of a head with no GPU handler", {
  heads <- c(setdiff(names(ir_call_registry), ir_internal_ops), ir_pre_registry_fns,
             agg_functions, "range", "desc", "$", "[[", "(", "~", "-", "c",
             "cur_group_id", "if_any", "if_all", "across", "pick", "ceiling")
  for (h in heads) {
    expect_no_match(fallback_no_gpu_text(h), "has no GPU implementation", fixed = TRUE, info = h)
  }
  expect_identical(fallback_no_gpu_text("sqrt", "my shape."), "my shape.")
  for (h in c("qnorm", "cor", "weighted.mean", "pmin", "rw_agg_f64")) {
    expect_identical(fallback_no_gpu_text(h, "my shape."), sprintf("`%s()` has no GPU implementation.", h))
  }
})

test_that("F88: a probe leaves the pending cause alone, and a new dot drops it", {
  prev <- fallback_reasons_open()
  on.exit(fallback_reasons_close(prev), add = TRUE)
  fallback_cause("cause of dot 1")
  fallback_probe(fallback_cause_take())
  fallback_probe(ir_parse_quo(rlang::quo(qnorm(x)), list(names = "x", types = "FLOAT64")))
  expect_identical(fallback_cause_take(), "cause of dot 1")
  fallback_probe(ir_parse_quo(rlang::quo(qnorm(x)), list(names = "x", types = "FLOAT64")))
  expect_null(fallback_cause_take())
  fallback_cause("stale")
  fallback_dot_begin()
  expect_null(fallback_cause_take())
})

test_that("F88: summarise() shapes around reducers get summarise()'s rule, unknown reducers their name", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  around <- list(
    list(f = function(t) summarise(t, r = mean(x) * 100), expr = "mean(x) * 100"),
    list(f = function(t) summarise(t, r = sum(x > 2) / n()), expr = "sum(x > 2) / n()"),
    list(f = function(t) summarise(t, r = dplyr::first(x) + dplyr::last(x)),
         expr = "dplyr::first(x) + dplyr::last(x)"),
    list(f = function(t) group_by(t, g) |> summarise(id = dplyr::cur_group_id(), r = sqrt(mean(x))),
         expr = "sqrt(mean(x))")
  )
  for (cs in around) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88r_run(cs$f, lazy)
      expect_length(got, 1L)
      expect_identical(got[[1L]]$expr, cs$expr)
      expect_match(got[[1L]]$reason, "one reducer call per expression", fixed = TRUE)
    }
  }
  got <- f88r_run(function(t) summarise(t, r = cor(x, y)))
  expect_identical(got[[1L]]$reason, "`cor()` has no GPU implementation.")
  # cur_group_id() inside a larger expression is summarise()'s rule, not "per-group data"
  got <- f88r_run(function(t) group_by(t, g) |> summarise(id = dplyr::cur_group_id() * 2, m = mean(x)))
  expect_identical(got[[1L]]$expr, "dplyr::cur_group_id() * 2")
  expect_match(got[[1L]]$reason, "only as a whole expression", fixed = TRUE)
})

test_that("F88: a grouped .by summarise() with cur_group_id() blames the dot that fell back", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  for (lazy in c(FALSE, TRUE)) {
    got <- f88r_run(function(t) {
      summarise(t, id = dplyr::cur_group_id(), m = mean(qnorm(x / 10)), .by = g)
    }, lazy)
    expect_length(got, 1L)
    expect_identical(got[[1L]]$expr, "mean(qnorm(x / 10))")
    expect_identical(got[[1L]]$reason, "`qnorm()` has no GPU implementation.")
  }
})

test_that("F88: a pre-registry head in a shape the GPU has no form for is not said to have none", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_run(function(t) mutate(t, a = sum(x, y)))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "sum(x, y)")
  expect_match(got[[1L]]$reason, "`sum()` runs on the GPU, but not in this form", fixed = TRUE)
})

test_that("F88: a dot's own cause survives an across()/if_all() probe of a later dot", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  pipelines <- list(
    # the if_all() piece is test-parsed to type it (check_across_predicate_piece())
    function(t) {
      transmute(t, p = as.difftime(2, units = "hours"),
                a = dplyr::if_all(c(x, y), ~ qnorm(.x / 10) > -2), p = NULL)
    },
    # the across() output before a later if_any() is test-parsed to type it
    # (across_infer_output()), then the if_any() piece too
    function(t) {
      mutate(t, p = as.difftime(2, units = "hours"),
             dplyr::across(c(x, y), ~ qnorm(.x / 10), .names = "q_{.col}"),
             a = dplyr::if_any(c(x, y), ~ .x > 2), p = NULL)
    }
  )
  for (f in pipelines) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88r_run(f, lazy)
      expect_length(got, 1L)
      expect_identical(got[[1L]]$expr, 'as.difftime(2, units = "hours")')
      expect_match(got[[1L]]$reason, "^unsupported type")
      expect_no_match(got[[1L]]$reason, "qnorm", fixed = TRUE)
    }
  }
})

test_that("F88: count(), tally() and add_count() report a `wt` fallback as theirs (N19)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  cases <- list(
    list(f = function(t) count(t, g, wt = qnorm(x / 10)), verb = "count"),
    list(f = function(t) tally(t, wt = qnorm(x / 10)), verb = "tally"),
    list(f = function(t) add_count(t, g, wt = qnorm(x / 10)), verb = "add_count")
  )
  for (cs in cases) {
    for (lazy in c(FALSE, TRUE)) {
      got <- f88r_run(cs$f, lazy)
      expect_length(got, 1L)
      expect_identical(got[[1L]]$verb, cs$verb)
      expect_identical(got[[1L]]$expr, "qnorm(x / 10)")
      expect_identical(got[[1L]]$reason, "`qnorm()` has no GPU implementation.")
    }
  }
  # the relabel is scoped to the call
  got <- f88r_run(function(t) summarise(count(t, g, wt = x), m = mean(qnorm(n / 10))))
  expect_identical(got[[1L]]$verb, "summarise")
  expect_null(fallback_reasons$as)
})

test_that("F88: reframe() and pick() reasons are true where they are said (N20, N21)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  k <- 2
  got <- f88r_run(function(t) reframe(t, r = sum(log1p(k))))
  expect_length(got, 1L)
  expect_no_match(got[[1L]]$reason, "no GPU implementation", fixed = TRUE)
  expect_match(got[[1L]]$reason, "reads no column", fixed = TRUE)
  got <- f88r_run(function(t) summarise(t, k = nrow(dplyr::pick(x))))
  expect_length(got, 1L)
  expect_match(got[[1L]]$reason, "the whole table when ungrouped", fixed = TRUE)
})

test_that("F88: an impure-call reason names the call once, as a sentence", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- f88r_run(function(t) group_by(t, g) |> mutate(m = mean(x), z = besselI(m / 10, 1)))
  expect_length(got, 1L)
  expect_identical(got[[1L]]$expr, "besselI(m / 10, 1)")
  expect_identical(got[[1L]]$reason,
                   "`besselI(m/10, 1)` is not known to be pure, and dplyr evaluates it once per group.")
})
