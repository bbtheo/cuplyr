# F29 acceptance tests, part B: native sample_n()/sample_frac().
#
# dplyr runs `slice(tbl, local({ size <- check_size(size, n(), replace);
# sample.int(n(), size, replace = replace, prob = weight) }))` per group
# (sample_frac: `size <- round(n() * check_frac(size, replace))`). cuplyr
# draws the same sample.int() calls on the CPU, in the same group order, and
# gathers the rows on the GPU. With the same seed, the rows AND the RNG
# state left behind must equal dplyr's. `weight =` and a size that needs the
# data mask (n(), a column) stay on the CPU; dplyr's errors come from dplyr.

f29s_df <- function() {
  tibble::tibble(
    g = c(2, 1, 2, 1, 3, 2, 1, 2, 3, 3, 1, 2, NA, NA),
    s = c("b", "a", "b", "a", "c", "b", "a", "b", "c", "c", "a", "b", "d", "d"),
    x = 1:14,
    y = c(0.5, 1, 2, 0, 3, 1, 1, 4, 0.25, 2, 1, 1, 3, 1)
  )
}

f29s_fdf <- function() {
  df <- f29s_df()
  df$f <- factor(df$s, levels = c("a", "b", "c", "d", "e"))
  df
}

# Seeded run of `p` on dplyr and on the GPU (eager and lazy): same rows, same
# groups and .drop, same next random number. With `native = TRUE` the GPU
# runs must not fall back to the CPU.
f29s_same <- function(df, p, native = TRUE, seed = 42) {
  set.seed(seed)
  want <- p(tibble::as_tibble(df))
  want_next <- stats::runif(1)
  if (native) withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    set.seed(seed)
    got <- p(tbl_gpu(df, lazy = lazy))
    got_next <- stats::runif(1)
    expect_equal(dplyr::group_vars(got), dplyr::group_vars(want))
    got_c <- collect(got)
    expect_equal(dplyr::ungroup(tibble::as_tibble(got_c)), dplyr::ungroup(tibble::as_tibble(want)))
    expect_identical(dplyr::group_by_drop_default(got_c), dplyr::group_by_drop_default(want))
    expect_identical(got_next, want_next)
  }
}

# Both sides raise an error with the same class and message.
f29s_same_error <- function(df, p) {
  set.seed(1)
  want <- tryCatch(p(tibble::as_tibble(df)), error = identity)
  expect_s3_class(want, "error")
  for (lazy in c(FALSE, TRUE)) {
    set.seed(1)
    got <- tryCatch(collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(got, "error")
    expect_identical(class(got), class(want))
    expect_identical(conditionMessage(got), conditionMessage(want))
  }
}

test_that("F29 B: ungrouped sample_n()/sample_frac() are native and match dplyr's RNG stream", {
  skip_if_no_gpu()
  k <- 4
  pipes <- list(
    n3 = function(d) sample_n(d, 3),
    n_all = function(d) sample_n(d, 14),
    n0 = function(d) sample_n(d, 0),
    n_replace = function(d) sample_n(d, 20, replace = TRUE),
    n_frac_size = function(d) sample_n(d, 2.7),
    n_int = function(d) sample_n(d, 5L),
    n_var = function(d) sample_n(d, k),
    frac_default = function(d) sample_frac(d),
    frac_half = function(d) sample_frac(d, 0.5),
    frac_quarter = function(d) sample_frac(d, 0.25),
    frac_zero = function(d) sample_frac(d, 0),
    frac_replace = function(d) sample_frac(d, 1.5, replace = TRUE),
    null_weight = function(d) sample_n(d, 3, weight = NULL)
  )
  for (nm in names(pipes)) {
    f29s_same(f29s_df(), pipes[[nm]])
  }
})

test_that("F29 B: grouped sample_n()/sample_frac() are native and match dplyr's RNG stream", {
  skip_if_no_gpu()
  pipes <- list(
    n2 = function(d) d |> group_by(g) |> sample_n(2),
    n_min = function(d) d |> group_by(g) |> sample_n(2L),
    n0 = function(d) d |> group_by(g) |> sample_n(0),
    n_replace = function(d) d |> group_by(g) |> sample_n(6, replace = TRUE),
    frac = function(d) d |> group_by(g) |> sample_frac(0.5),
    frac_all = function(d) d |> group_by(g) |> sample_frac(),
    frac_replace = function(d) d |> group_by(g) |> sample_frac(2, replace = TRUE),
    string_key = function(d) d |> group_by(s) |> sample_n(2),
    two_keys = function(d) d |> group_by(s, g) |> sample_frac(0.5),
    prior_ops = function(d) d |> filter(x > 1L) |> group_by(g) |> sample_n(1),
    mutate_first = function(d) d |> mutate(z = x * 2) |> group_by(s) |> sample_frac(0.6),
    zero_rows = function(d) d |> filter(x > 100L) |> sample_n(0),
    zero_rows_frac = function(d) d |> filter(x > 100L) |> group_by(g) |> sample_frac(0.5)
  )
  for (nm in names(pipes)) {
    f29s_same(f29s_df(), pipes[[nm]])
  }
})

test_that("F29 B: .drop = FALSE groups (empty levels) keep their structure", {
  skip_if_no_gpu()
  f29s_same(f29s_fdf(), function(d) d |> group_by(f, .drop = FALSE) |> sample_n(0))
  f29s_same(f29s_fdf(), function(d) d |> group_by(f, .drop = FALSE) |> sample_frac(0.5))
  f29s_same(f29s_fdf(), function(d) d |> group_by(f, .drop = FALSE) |> sample_frac(1.5, replace = TRUE))
})

test_that("F29 B: the deprecated .env argument still informs, natively", {
  skip_if_no_gpu()
  capture <- function(expr) {
    msgs <- character()
    withCallingHandlers(expr, message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    msgs
  }
  df <- f29s_df()
  e <- environment()
  want_n <- capture(sample_n(tibble::as_tibble(df), 2, .env = e))
  want_f <- capture(sample_frac(tibble::as_tibble(df), 0.5, .env = e))
  expect_length(want_n, 1L)
  expect_length(want_f, 1L)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    expect_identical(capture(sample_n(tbl_gpu(df, lazy = lazy), 2, .env = e)), want_n)
    expect_identical(capture(sample_frac(tbl_gpu(df, lazy = lazy), 0.5, .env = e)), want_f)
  }
})

test_that("F29 B: weight and data-mask sizes stay on the CPU and match dplyr", {
  skip_if_no_gpu()
  w <- NULL
  pipes <- list(
    weight = function(d) sample_n(d, 3, weight = y),
    weight_frac = function(d) sample_frac(d, 0.5, weight = y),
    weight_grouped = function(d) d |> group_by(s) |> sample_n(1, weight = y),
    weight_null_var = function(d) sample_n(d, 3, weight = w),
    size_n = function(d) d |> group_by(g) |> sample_n(n() - 1L),
    size_n_ungrouped = function(d) sample_n(d, n()),
    size_col = function(d) d |> group_by(g) |> sample_n(min(x) %% 2 + 1),
    # the size expression draws random numbers itself: dplyr evaluates it
    # once per group, between that group's draws
    size_rng = function(d) d |> group_by(g) |> sample_n(sample(1:2, 1))
  )
  for (nm in names(pipes)) {
    f29s_same(f29s_df(), pipes[[nm]], native = FALSE)
  }
})

test_that("F29 B: sample_n()/sample_frac() errors match dplyr", {
  skip_if_no_gpu()
  df <- f29s_df()
  f29s_same_error(df, function(d) sample_n(d, 15))
  f29s_same_error(df, function(d) d |> group_by(g) |> sample_n(3))
  f29s_same_error(df, function(d) sample_n(d, -1))
  f29s_same_error(df, function(d) sample_n(d, NA))
  f29s_same_error(df, function(d) sample_n(d, NA, replace = TRUE))
  f29s_same_error(df, function(d) sample_frac(d, 2))
  f29s_same_error(df, function(d) d |> group_by(g) |> sample_frac(1.1))
  f29s_same_error(df, function(d) sample_n(d, 2, weight = x - 10L))
  f29s_same_error(df |> dplyr::filter(x > 100L), function(d) sample_n(d, 1))
  f29s_same_error(f29s_fdf(), function(d) d |> group_by(f, .drop = FALSE) |> sample_n(1))
  f29s_same_error(f29s_fdf(), function(d) d |> group_by(f, .drop = FALSE) |> sample_n(1, replace = TRUE))
})

# --- Added after round 1 review (B2) -----------------------------------------

test_that("F29 B: empty groups kept by filter() (.drop = FALSE, .preserve) are sampled like dplyr (R1 B2)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1, 1, 2, 2, 2), x = c(1, 2, 3, 4, 5))
  f29s_same_error(df, function(d) d |> group_by(g, .drop = FALSE) |> filter(x > 2) |> sample_n(1))
  f29s_same_error(df, function(d) d |> group_by(g) |> filter(x > 2, .preserve = TRUE) |> sample_n(1))

  # group structure, empty groups included, matches dplyr
  same_groups <- function(p) {
    set.seed(3)
    want <- p(tibble::as_tibble(df))
    want_next <- stats::runif(1)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(3)
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      got_next <- stats::runif(1)
      expect_equal(tibble::as_tibble(dplyr::group_data(got)), tibble::as_tibble(dplyr::group_data(want)))
      expect_equal(dplyr::ungroup(tibble::as_tibble(got)), dplyr::ungroup(tibble::as_tibble(want)))
      expect_identical(got_next, want_next)
    }
  }
  same_groups(function(d) d |> group_by(g, .drop = FALSE) |> filter(x > 2) |> sample_frac(0.5))
  same_groups(function(d) d |> group_by(g, .drop = FALSE) |> filter(x > 2) |> sample_n(0))
  same_groups(function(d) d |> group_by(g) |> filter(x > 2, .preserve = TRUE) |> sample_frac(1))
  same_groups(function(d) d |> group_by(g, .drop = FALSE) |> filter(x > 2) |> sample_frac(0.5) |> summarise(k = dplyr::n()))
})
