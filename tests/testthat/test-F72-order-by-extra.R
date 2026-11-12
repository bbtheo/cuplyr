# F72 rounds 1-2: shapes beyond the acceptance file.

f72x_df <- function() {
  tibble::tibble(
    g = c("a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"),
    o = c(3L, 1L, NA, 2L, 1L, 5L, 2L, NA, 4L, 1L, 3L, 2L),
    x = as.double(1:12),
    y = c(2.5, NA, 1, 4, -3, 6, 0.5, 8, 2, -1, 7, 3)
  )
}

# dplyr raises an error for this pipeline; the GPU table must raise one of the
# same class, eager and lazy, and not return a value.
f72x_same_error <- function(df, pipeline) {
  cpu <- tryCatch(pipeline(tibble::as_tibble(df)), error = identity)
  expect_s3_class(cpu, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tryCatch({
      t <- tbl_gpu(df, lazy = lazy)
      collect(pipeline(t))
    }, error = identity)
    expect_s3_class(gpu, class(cpu)[[1]])
  }
}

f72x_same <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("order_by() with scalar extra arguments given as expressions stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, NA, 2L, 1L), x = c(1, 2, 3, 4, 5))
  k <- 2L
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lead(x, default = -1))))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, k))))
})

test_that("an extra argument is evaluated once on the native path", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, NA, 2L, 1L), x = c(1, 2, 3, 4, 5))
  calls <- 0L
  bump <- function() {
    calls <<- calls + 1L
    1L
  }
  dplyr::mutate(tbl_gpu(df), r = dplyr::order_by(o, dplyr::lag(x, bump())))
  expect_identical(calls, 1L)
})

test_that("order_by() with a row-sized extra argument runs on the CPU, not unordered", {
  skip_if_no_gpu()
  df <- tibble::tibble(o = c(3L, 1L, NA, 2L, 1L), x = c(1, 2, 3, 4, 5))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) / sum(x))))
})

# ---------------------------------------------------------------------------
# A NULL (or empty) extra argument is not a scalar: dplyr errors, so must we
# ---------------------------------------------------------------------------

test_that("a NULL extra argument raises dplyr's error", {
  skip_if_no_gpu()
  df <- f72x_df()
  f72x_same_error(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lag, x, NULL)))
  f72x_same_error(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lag, x, n = NULL)))
  f72x_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, n = NULL))))
  f72x_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) + NULL)))
})

test_that("a NULL lag() default under order_by() gives dplyr's value", {
  skip_if_no_gpu()
  df <- f72x_df()
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, default = NULL))))
})

# ---------------------------------------------------------------------------
# Rank functions over desc(): order-invariant, so native with or without order_by()
# ---------------------------------------------------------------------------

test_that("order_by(o, rank(desc(x))) stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f72x_df()
  f72x_same(df, function(d) {
    dplyr::mutate(d,
      r1 = dplyr::order_by(o, dplyr::min_rank(dplyr::desc(x))),
      r2 = dplyr::order_by(o, dplyr::dense_rank(dplyr::desc(y))),
      r3 = dplyr::order_by(o, dplyr::percent_rank(dplyr::desc(y))),
      r4 = dplyr::order_by(o, dplyr::cume_dist(dplyr::desc(x)))
    )
  })
  f72x_same(df, function(d) {
    dplyr::mutate(d, r = dplyr::with_order(o, dplyr::min_rank, dplyr::desc(x)), .by = g)
  })
  # and the value is the one without order_by()
  plain <- dplyr::mutate(df, r = dplyr::min_rank(dplyr::desc(y)))
  gpu <- collect(dplyr::mutate(tbl_gpu(df), r = dplyr::order_by(o, dplyr::min_rank(dplyr::desc(y)))))
  expect_equal(gpu$r, plain$r)
})

# ---------------------------------------------------------------------------
# Pipelines around the window mutate
# ---------------------------------------------------------------------------

test_that("select()/filter() after an order_by() window mutate keep the order column alive", {
  skip_if_no_gpu()
  df <- f72x_df()
  f72x_same(df, function(d) {
    d |>
      dplyr::mutate(r = dplyr::order_by(o, cumsum(x))) |>
      dplyr::select(g, r) |>
      dplyr::filter(r > 10)
  })
  f72x_same(df, function(d) {
    d |>
      dplyr::select(o, x) |>
      dplyr::mutate(r = dplyr::order_by(o, dplyr::lag(x))) |>
      dplyr::select(r)
  })
})

test_that("an order key created earlier in the same mutate() call is used", {
  skip_if_no_gpu()
  df <- f72x_df()
  f72x_same(df, function(d) dplyr::mutate(d, k = o * 2L, r = dplyr::order_by(k, cumsum(x))))
  f72x_same(df, function(d) dplyr::mutate(d, o = -o, r = dplyr::order_by(o, cumsum(x)), .by = g))
})

test_that("roll_*() wider than the F59 crossover run in the order_by() order", {
  skip_if_no_gpu()
  set.seed(7201)
  n <- 1500L
  df <- tibble::tibble(
    g = sample(c("p", "q"), n, replace = TRUE),
    o = sample(c(1:50, NA), n, replace = TRUE),
    x = round(stats::rnorm(n) * 10, 1)
  )
  for (path in c("auto", "scans", "cudf")) {
    withr::local_options(cuplyr.roll_path = path)
    f72x_same(df, function(d) {
      dplyr::mutate(d,
        r1 = dplyr::order_by(o, roll_sum(x, 100)),
        r2 = dplyr::order_by(o, roll_mean(x, 300)),
        r3 = dplyr::order_by(dplyr::desc(o), roll_max(x, 150)),
        r4 = dplyr::order_by(o, roll_sd(x, 64)),
        .by = g
      )
    })
  }
})

# ---------------------------------------------------------------------------
# Round 3: each user sub-expression is evaluated once, and the IR that runs
# is built from that one evaluation
# ---------------------------------------------------------------------------

# Run `pipeline` on a tibble and on a GPU table (eager or lazy) with a fresh
# counter each time; return both call counts and both results.
f72x_count <- function(df, pipeline, lazy = FALSE) {
  calls <- 0L
  bump <- function() {
    calls <<- calls + 1L
    calls
  }
  cpu <- pipeline(df, bump)
  cpu_calls <- calls
  calls <- 0L
  t <- tbl_gpu(df, lazy = lazy)
  gpu <- tibble::as_tibble(collect(pipeline(t, bump)))
  list(cpu = tibble::as_tibble(cpu), cpu_calls = cpu_calls, gpu = gpu, gpu_calls = calls)
}

test_that("a column-free call inside the first argument runs once, natively (mutate, eager and lazy)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, 2L, 1L), x = c(1, 2, 3, 4))
  pipeline <- function(d, bump) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x + bump())))
  for (lazy in c(FALSE, TRUE)) {
    res <- f72x_count(df, pipeline, lazy = lazy)
    expect_identical(res$cpu_calls, 1L)
    expect_identical(res$gpu_calls, 1L)
    expect_equal(res$cpu$r, c(14, 3, 12, 8))
    expect_equal(res$gpu, res$cpu)
  }
})

test_that("a column-free call inside the first argument runs once (filter, with_order)", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, 2L, 1L), x = c(1, 2, 3, 4))
  res <- f72x_count(df, function(d, bump) dplyr::filter(d, dplyr::order_by(o, cumsum(x + bump())) > 8))
  expect_identical(res$gpu_calls, 1L)
  expect_identical(nrow(res$cpu), 2L)
  expect_equal(res$gpu, res$cpu)
  res <- f72x_count(df, function(d, bump) dplyr::mutate(d, r = dplyr::with_order(o, cumsum, x + bump())))
  expect_identical(res$gpu_calls, 1L)
  expect_equal(res$gpu, res$cpu)
  # a window inside the first argument, and a non-window head
  res <- f72x_count(df, function(d, bump) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(cumsum(x) * bump()))))
  expect_identical(res$gpu_calls, 1L)
  expect_equal(res$gpu, res$cpu)
  res <- f72x_count(df, function(d, bump) dplyr::mutate(d, r = dplyr::order_by(o, abs(x - bump()))))
  expect_identical(res$gpu_calls, 1L)
  expect_equal(res$gpu, res$cpu)
})

test_that("a random first argument draws once: the value is dplyr's under the same seed", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, 2L, 1L), x = c(1, 2, 3, 4))
  withr::local_seed(1)
  cpu <- dplyr::mutate(df, r = dplyr::order_by(o, cumsum(x + stats::runif(1))))
  withr::local_seed(1)
  gpu <- collect(dplyr::mutate(tbl_gpu(df), r = dplyr::order_by(o, cumsum(x + stats::runif(1)))))
  expect_equal(gpu$r, cpu$r)
  expect_equal(cpu$r, c(11.062035, 2.265509, 9.796526, 6.531017), tolerance = 1e-6)
})

test_that("first argument and extras are each evaluated once, the argument first", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, 2L, 1L), x = c(1, 2, 3, 4))
  seen <- character()
  note <- function(tag, value) {
    seen <<- c(seen, tag)
    value
  }
  pipeline <- function(d) {
    dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x * note("a1", 2), note("n", 1L), default = note("def", 0))))
  }
  cpu <- pipeline(df)
  cpu_seen <- seen
  seen <- character()
  gpu <- collect(pipeline(tbl_gpu(df)))
  expect_identical(seen, cpu_seen)
  expect_equal(tibble::as_tibble(gpu), cpu)
})

test_that("an extra that reads a column is found before any extra is evaluated", {
  skip_if_no_gpu()
  df <- tibble::tibble(o = c(3L, 1L, 2L, 1L), x = c(1, 2, 3, 4), y = c(5, 6, 7, 8))
  calls <- 0L
  bump <- function() {
    calls <<- calls + 1L
    1
  }
  pipeline <- function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::if_else(cumsum(x) > 3, bump(), y)))
  cpu <- pipeline(df)
  expect_identical(calls, 1L)
  calls <- 0L
  gpu <- collect(pipeline(tbl_gpu(df)))
  expect_identical(calls, 1L)  # the CPU run only
  expect_equal(tibble::as_tibble(gpu), cpu)
})

test_that("a first argument of a classed type that is not a column stays native with dplyr's value", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    o = c(3L, 1L, NA, 2L, 1L),
    t = as.POSIXct(c("2020-01-01 10:00", "2020-01-02 10:00", NA, "2020-01-04 10:00", "2020-01-05 10:00"),
                   tz = "America/New_York"),
    u = as.POSIXct(c("2021-01-01 10:00", "2021-01-02 10:00", "2021-01-03 10:00", NA, "2021-01-05 10:00"),
                   tz = "Asia/Tokyo"),
    d = as.Date(c("2020-01-01", NA, "2020-01-03", "2020-01-04", NA)),
    d2 = as.Date("2021-05-05") + 0:4
  )
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(dplyr::coalesce(t, u)))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(dplyr::coalesce(u, t)))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lead(dplyr::coalesce(d, d2)))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(dplyr::desc(o), dplyr::lag(.data$t))))
  gpu <- collect(dplyr::mutate(tbl_gpu(df), r = dplyr::order_by(o, dplyr::lag(dplyr::coalesce(u, t)))))
  expect_identical(attr(gpu$r, "tzone"), "Asia/Tokyo")
})

test_that("a factor first argument gives dplyr's value", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    o = c(3L, 1L, NA, 2L, 1L),
    f = factor(c("a", "b", NA, "a", "c")),
    f2 = factor(c("b", "b", "a", "a", "c"))
  )
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(.data$f))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(dplyr::coalesce(f, f2)))))
})

# ---------------------------------------------------------------------------
# Classed extras and names shared by a column and an environment variable
# ---------------------------------------------------------------------------

test_that("Date and POSIXct extras keep their class and zone, natively", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(
    o = c(3L, 1L, NA, 2L, 1L),
    d = as.Date(c("2020-01-01", NA, "2020-01-03", "2020-01-04", "2020-01-05")),
    t = as.POSIXct(c("2020-01-01 10:00", "2020-01-02 10:00", NA, "2020-01-04 10:00", "2020-01-05 10:00"),
                   tz = "America/New_York")
  )
  dflt_d <- as.Date("1999-12-31")
  dflt_t <- as.POSIXct("1999-12-31 23:00", tz = "Asia/Tokyo")
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(d, default = dflt_d))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lead(t, default = dflt_t))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lead, t, 1L, dflt_t)))
  gpu <- collect(dplyr::mutate(tbl_gpu(df), r = dplyr::order_by(o, dplyr::lead(t, default = dflt_t))))
  expect_identical(attr(gpu$r, "tzone"), "America/New_York")
})

test_that("an extra named like both a column and an environment variable reads the column", {
  skip_if_no_gpu()
  df <- tibble::tibble(o = c(3L, 1L, NA, 2L, 1L), x = c(1, 2, 3, 4, 5), k = c(10L, 20L, 30L, 40L, 50L))
  k <- 1L
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) + k)))
  f72x_same_error(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, k))))
})

test_that("`.env$k` extras stay native and read the environment", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(o = c(3L, 1L, NA, 2L, 1L), x = c(1, 2, 3, 4, 5), k = c(10L, 20L, 30L, 40L, 50L))
  k <- 2L
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, cumsum(x) * .env$k)))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::order_by(o, dplyr::lag(x, .env$k))))
  f72x_same(df, function(d) dplyr::mutate(d, r = dplyr::with_order(o, dplyr::lag, x, .env[["k"]])))
})
