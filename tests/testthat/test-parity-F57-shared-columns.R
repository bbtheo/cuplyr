# F57: eager results share unchanged input columns instead of deep-copying
# them, and an eager window mutate() executes as one result table.
#
# Contract (scratchpad/parity/features/F57.md, Design A/B):
# - A verb that passes a column through unchanged (select, relocate, rename,
#   the untouched columns of mutate/transmute, a bare copy `a = x`, the input
#   columns of a window mutate, bind_cols) shares it: no device allocation.
#   `mutate(r = f(x))` allocates only `r` (plus scratch).
# - An eager window mutate() holds only what its result needs once it returns.
#   (Freeing intermediates deterministically is F58's; the bounds below allow
#   for a window temp column that is not the output.)
# - Sharing never lets one table change another: columns are immutable, and a
#   result stays valid after its source table is freed.
#
# Allocations are measured at the RMM allocator with gpu_alloc_tracking()
# (F56), which is exact and independent of the pool.

f57_col <- 8e6  # one FLOAT64 column at 1e6 rows
f57_small <- 64 * 1024

f57_big <- function(n = 1e6) {
  set.seed(57)
  data.frame(
    g = sample.int(10L, n, replace = TRUE),
    x = runif(n), y = runif(n), z = runif(n), w = runif(n)
  )
}

f57_track <- function(expr) {
  gc_gpu()
  gpu_alloc_tracking(expr)
}

f57_same <- function(gpu, expected) {
  expect_equal(tibble::as_tibble(collect(gpu)), tibble::as_tibble(expected))
}

# Small table with every type the sharing has to carry, NAs included.
f57_mixed <- function(n = 40L) {
  set.seed(5757)
  x <- runif(n)
  x[c(3, 9)] <- NA
  y <- sample.int(4L, n, replace = TRUE)
  y[c(5, 17)] <- NA
  s <- sprintf("s%02d", sample.int(30L, n, replace = TRUE))
  s[c(2, 11)] <- NA
  data.frame(
    x = x,
    y = y,
    s = s,
    f = factor(sample(c("lo", "mid", "hi"), n, replace = TRUE), levels = c("lo", "mid", "hi")),
    d = as.Date("2024-01-01") + sample.int(100L, n, replace = TRUE),
    b = sample(c(TRUE, FALSE, NA), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

# -----------------------------------------------------------------------------
# Allocator-level bounds
# -----------------------------------------------------------------------------

test_that("eager select(), relocate() and rename() allocate nothing", {
  skip_if_no_gpu()
  df <- f57_big()
  g <- tbl_gpu(df, lazy = FALSE)

  s <- f57_track(select(g, x, y))
  expect_lt(s$allocated_bytes, f57_small)
  f57_same(s$value, dplyr::select(df, x, y))

  r <- f57_track(relocate(g, w, .before = 1))
  expect_lt(r$allocated_bytes, f57_small)
  f57_same(r$value, dplyr::relocate(df, w, .before = 1))

  n <- f57_track(rename(g, xx = x))
  expect_lt(n$allocated_bytes, f57_small)
  f57_same(n$value, dplyr::rename(df, xx = x))
})

test_that("eager mutate() allocates only the column it computes", {
  skip_if_no_gpu()
  df <- f57_big()
  g <- tbl_gpu(df, lazy = FALSE)

  add <- f57_track(mutate(g, r = x + 1))
  expect_lte(add$allocated_bytes, 1.5 * f57_col)
  expect_gte(add$held_bytes, f57_col)
  expect_lte(add$held_bytes, 1.5 * f57_col)
  f57_same(add$value, dplyr::mutate(df, r = x + 1))

  repl <- f57_track(mutate(g, x = x + 1))
  expect_lte(repl$allocated_bytes, 1.5 * f57_col)
  f57_same(repl$value, dplyr::mutate(df, x = x + 1))

  copy <- f57_track(mutate(g, a = x, b = y))
  expect_lt(copy$allocated_bytes, f57_small)
  f57_same(copy$value, dplyr::mutate(df, a = x, b = y))

  tm <- f57_track(transmute(g, r = x * 2, y))
  expect_lte(tm$held_bytes, 1.5 * f57_col)
  f57_same(tm$value, dplyr::transmute(df, r = x * 2, y))

  unused <- f57_track(mutate(g, r = x + y, .keep = "unused"))
  expect_lte(unused$held_bytes, 1.5 * f57_col)
  f57_same(unused$value, dplyr::mutate(df, r = x + y, .keep = "unused"))
})

test_that("an eager window mutate() holds only its result", {
  skip_if_no_gpu()
  df <- f57_big()
  g <- tbl_gpu(df, lazy = FALSE)

  cs <- f57_track(mutate(g, r = cumsum(x)))
  expect_lte(cs$held_bytes, 1.5 * f57_col)
  expect_lte(cs$allocated_bytes, 4 * f57_col)
  f57_same(cs$value, dplyr::mutate(df, r = cumsum(x)))

  lg <- f57_track(mutate(g, r = dplyr::lag(x)))
  expect_lte(lg$held_bytes, 1.5 * f57_col)
  f57_same(lg$value, dplyr::mutate(df, r = dplyr::lag(x)))

  two <- f57_track(mutate(g, a = cumsum(x), b = dplyr::lead(y)))
  expect_lte(two$held_bytes, 2.5 * f57_col)
  f57_same(two$value, dplyr::mutate(df, a = cumsum(x), b = dplyr::lead(y)))

  # A window call inside arithmetic: the window temp column is not the
  # output, so allow for it until F58 frees intermediates deterministically.
  diff <- f57_track(mutate(g, r = x - dplyr::lag(x)))
  expect_lte(diff$held_bytes, 2.5 * f57_col)
  f57_same(diff$value, dplyr::mutate(df, r = x - dplyr::lag(x)))
})

test_that("a grouped eager window mutate() holds only its result", {
  skip_if_no_gpu()
  df <- f57_big()
  gg <- group_by(tbl_gpu(df, lazy = FALSE), g)

  cs <- f57_track(mutate(gg, r = cumsum(x)))
  expect_lte(cs$held_bytes, 1.5 * f57_col)
  f57_same(cs$value, dplyr::mutate(dplyr::group_by(df, g), r = cumsum(x)))

  ug <- tbl_gpu(df, lazy = FALSE)
  by <- f57_track(mutate(ug, r = mean(x), .by = g))
  expect_lte(by$held_bytes, 1.5 * f57_col)
  f57_same(by$value, dplyr::mutate(df, r = mean(x), .by = g))
})

test_that("lazy compute() shares the columns it passes through", {
  skip_if_no_gpu()
  df <- f57_big()
  lz <- tbl_gpu(df, lazy = TRUE) |>
    mutate(a = x + 1) |>
    mutate(b = a * 2) |>
    select(g, x, a, b)

  cm <- f57_track(compute(lz))
  expect_lte(cm$held_bytes, 2.5 * f57_col)
  f57_same(cm$value, df |> dplyr::mutate(a = x + 1, b = a * 2) |> dplyr::select(g, x, a, b))

  lw <- tbl_gpu(df, lazy = TRUE) |> mutate(r = cumsum(x))
  cw <- f57_track(compute(lw))
  expect_lte(cw$held_bytes, 1.5 * f57_col)
  f57_same(cw$value, dplyr::mutate(df, r = cumsum(x)))
})

test_that("bind_cols() of GPU tables shares their columns", {
  skip_if_no_gpu()
  df <- f57_big()
  g <- tbl_gpu(df, lazy = FALSE)

  b <- f57_track(bind_cols(select(g, x), select(g, y, z)))
  expect_lt(b$allocated_bytes, f57_small)
  f57_same(b$value, dplyr::bind_cols(dplyr::select(df, x), dplyr::select(df, y, z)))
})

test_that("string, factor and Date columns are shared too", {
  skip_if_no_gpu()
  n <- 2e5
  set.seed(1)
  df <- data.frame(
    s = sprintf("string-%07d", sample.int(n)),
    f = factor(sample(letters, n, replace = TRUE), levels = letters),
    d = as.Date("2020-01-01") + sample.int(5000L, n, replace = TRUE),
    x = runif(n),
    stringsAsFactors = FALSE
  )
  g <- tbl_gpu(df, lazy = FALSE)

  s <- f57_track(select(g, s, f, d))
  expect_lt(s$allocated_bytes, f57_small)
  f57_same(s$value, dplyr::select(df, s, f, d))

  m <- f57_track(mutate(g, y = x * 2))
  expect_lte(m$allocated_bytes, 1.5 * n * 8)
  f57_same(m$value, dplyr::mutate(df, y = x * 2))

  cp <- f57_track(mutate(g, s2 = s, f2 = f))
  expect_lt(cp$allocated_bytes, f57_small)
  f57_same(cp$value, dplyr::mutate(df, s2 = s, f2 = f))
})

# -----------------------------------------------------------------------------
# Lifetime and immutability
# -----------------------------------------------------------------------------

test_that("results stay valid after the table they share columns with is freed", {
  skip_if_no_gpu()
  df <- f57_mixed(5000L)
  g <- tbl_gpu(df, lazy = FALSE)

  a <- mutate(g, z = x * 2)
  b <- select(g, s, x, f)
  c <- mutate(g, r = cumsum(dplyr::coalesce(x, 0)), .keep = "unused")
  d <- bind_cols(select(g, d), select(g, y))
  rm(g)
  gc_gpu()
  gc_gpu()

  f57_same(a, dplyr::mutate(df, z = x * 2))
  f57_same(b, dplyr::select(df, s, x, f))
  f57_same(c, dplyr::mutate(df, r = cumsum(dplyr::coalesce(x, 0)), .keep = "unused"))
  f57_same(d, dplyr::bind_cols(dplyr::select(df, d), dplyr::select(df, y)))

  rm(a)
  gc_gpu()
  f57_same(b, dplyr::select(df, s, x, f))
  f57_same(c, dplyr::mutate(df, r = cumsum(dplyr::coalesce(x, 0)), .keep = "unused"))
})

test_that("verbs on a derived table never change the columns it shares", {
  skip_if_no_gpu()
  df <- f57_mixed(200L)
  df$x[c(20, 21)] <- NaN
  g <- tbl_gpu(df, lazy = FALSE)
  original <- tibble::as_tibble(collect(g))
  shared <- mutate(g, k = 1L)

  battery <- list(
    function(t) filter(t, is.na(x)),
    function(t) arrange(t, desc(x), s),
    function(t) mutate(t, x = dplyr::if_else(is.na(x), 0, x)),
    function(t) mutate(t, x = dplyr::coalesce(x, -1), s = dplyr::coalesce(s, "none")),
    function(t) mutate(t, x = dplyr::na_if(x, 0), y = y + 1L),
    function(t) summarise(t, m = mean(x, na.rm = TRUE), n = n()),
    function(t) summarise(group_by(t, y), s = sum(x, na.rm = TRUE), mx = max(x, na.rm = TRUE)),
    function(t) mutate(group_by(t, f), r = cumsum(dplyr::coalesce(x, 0)), l = dplyr::lag(s)),
    function(t) slice(t, 2:5),
    function(t) distinct(t, y, f),
    function(t) left_join(t, distinct(select(t, y, k)), by = c("y", "k")),
    function(t) bind_rows(t, t),
    function(t) select(t, -x),
    function(t) rename(t, xx = x),
    function(t) relocate(t, s),
    function(t) transmute(t, x = x * 2)
  )
  for (fn in battery) {
    res <- fn(shared)
    invisible(collect(res))
  }

  expect_equal(tibble::as_tibble(collect(g)), original)
  expect_equal(tibble::as_tibble(collect(shared)), dplyr::mutate(original, k = 1L))
})

test_that("a column copied to two names changes through one name only", {
  skip_if_no_gpu()
  df <- f57_mixed()
  pipeline <- function(t) {
    t |>
      mutate(a = x, b = x, s2 = s) |>
      mutate(a = a + 1, s2 = paste0(s2, "!")) |>
      select(x, a, b, s, s2)
  }
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)

  g <- tbl_gpu(df, lazy = FALSE)
  m <- mutate(g, a = x, b = x)
  m2 <- mutate(m, a = a + 1)
  invisible(collect(m2))
  f57_same(m, dplyr::mutate(df, a = x, b = x))
  f57_same(g, df)
})

# -----------------------------------------------------------------------------
# Parity with dplyr, eager and lazy, incl. 0 rows
# -----------------------------------------------------------------------------

test_that("pass-through verbs match dplyr eager and lazy, with and without rows", {
  skip_if_no_gpu()
  df <- f57_mixed()
  pipelines <- list(
    function(t) select(t, s, x, f),
    function(t) mutate(t, r = x + 1, x = y),
    function(t) mutate(t, r = x + 1, .before = 1),
    function(t) mutate(t, r = cumsum(dplyr::coalesce(x, 0)), .keep = "unused"),
    function(t) mutate(t, r = x - dplyr::lag(x), n = dplyr::row_number()),
    function(t) t |> group_by(y) |> mutate(r = dplyr::lag(x), n = dplyr::row_number()) |> ungroup(),
    function(t) t |> group_by(f) |> mutate(m = mean(x, na.rm = TRUE)),
    function(t) transmute(t, s, r = x * 2),
    function(t) relocate(t, s, .before = 1) |> rename(xx = x),
    function(t) t |> mutate(f2 = f, d2 = d + 1) |> select(f2, d2, f),
    function(t) mutate(t, b2 = !b, .keep = "none")
  )
  for (i in seq_along(pipelines)) {
    p <- pipelines[[i]]
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_same_as_dplyr(df[0, ], p)
    expect_same_as_dplyr_lazy(df[0, ], p)
  }
})

test_that("an eager window mutate() still warns once and still retries an integer sum overflow", {
  skip_if_no_gpu()
  big <- data.frame(v = c(.Machine$integer.max, 1L, 2L), x = c(1, 2, 3))
  g <- tbl_gpu(big, lazy = FALSE)

  n_warn <- 0L
  res <- withCallingHandlers(
    mutate(g, r = cumsum(v)),
    warning = function(w) {
      n_warn <<- n_warn + 1L
      invokeRestart("muffleWarning")
    }
  )
  expect_identical(n_warn, 1L)
  expect_identical(collect(res)$r, suppressWarnings(cumsum(big$v)))

  # sum() leaves the integer range: dplyr returns a double; the eager GPU path
  # reruns on the CPU (retry_on_sum_overflow()).
  expect_same_as_dplyr(big, function(t) mutate(t, s = sum(v), r = x * 2))
})
