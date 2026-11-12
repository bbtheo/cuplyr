# F29 acceptance tests, part A: native top_n()/top_frac().
#
# dplyr's top_n() is a plain function that calls
# `filter(x, top_n_rank(n, wt))`, and
# `top_n_rank(n, wt)` is `min_rank(desc(wt)) <= n` for n > 0, else
# `min_rank(wt) <= abs(n)`. cuplyr lowers that dot to its native window
# rank path when n is a scalar numeric constant or `<const> * n()`. These
# tests check the result against dplyr AND that the listed shapes run with
# no CPU fallback (`options(cuplyr.fallback = "error")`), eager and lazy.
#
# Part A0: rank-family calls with a `desc()` argument
# (`min_rank(desc(x))` etc.) run natively in mutate() and filter().

# dplyr is not attached in the test env; bind the plain functions the
# pipelines call by their bare names (the dplyr oracle evaluates them).
top_n <- dplyr::top_n
top_frac <- dplyr::top_frac
min_rank <- dplyr::min_rank
dense_rank <- dplyr::dense_rank
percent_rank <- dplyr::percent_rank
cume_dist <- dplyr::cume_dist
row_number <- dplyr::row_number
desc <- dplyr::desc

f29_df <- function() {
  tibble::tibble(
    g = c("a", "b", "a", "b", "a", "c", "b", "a", "c"),
    x = c(3, 1, 5, NA, 5, 2, 8, 0.5, 2),
    i = c(4L, 2L, 9L, 9L, 1L, NA, 3L, 7L, 2L),
    s = c("m", "z", "b", "z", NA, "q", "a", "m", "c"),
    d = as.Date("2024-01-01") + c(5, 1, NA, 3, 3, 9, 0, 2, 7)
  )
}

f29_both <- function(df, p) {
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
}

# Same result as dplyr AND no CPU fallback, eager and lazy.
f29_native <- function(df, p, label) {
  f29_both(df, p)
  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    lab <- paste(label, if (lazy) "lazy" else "eager")
    expect_no_error(suppressMessages(collect(p(tbl_gpu(df, lazy = lazy)))), message = lab)
  }
}

# Both sides raise an error with the same class and message.
f29_same_error <- function(df, p) {
  want <- tryCatch(p(tibble::as_tibble(df)), error = identity)
  expect_s3_class(want, "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(p(tbl_gpu(df, lazy = lazy))), error = identity)
    expect_s3_class(got, "error")
    expect_identical(class(got), class(want))
    expect_identical(conditionMessage(got), conditionMessage(want))
  }
}

test_that("F29 A0: rank functions with a desc() argument are native in mutate()", {
  skip_if_no_gpu()
  pipes <- list(
    min_rank_x = function(d) d |> mutate(r = min_rank(desc(x))),
    dense_rank_x = function(d) d |> mutate(r = dense_rank(desc(x))),
    percent_rank_x = function(d) d |> mutate(r = percent_rank(desc(x))),
    cume_dist_x = function(d) d |> mutate(r = cume_dist(desc(x))),
    ns_desc = function(d) d |> mutate(r = min_rank(dplyr::desc(x))),
    int_col = function(d) d |> mutate(r = min_rank(desc(i))),
    string_col = function(d) d |> mutate(r = min_rank(desc(s)), q = dense_rank(desc(s))),
    date_col = function(d) d |> mutate(r = min_rank(desc(d))),
    expr_arg = function(d) d |> mutate(r = min_rank(desc(x * 2 + i))),
    mixed = function(d) d |> mutate(a = min_rank(x), b = min_rank(desc(x)), c = dense_rank(desc(i))),
    ns_both = function(d) d |> mutate(r = dplyr::min_rank(dplyr::desc(x))),
    grouped = function(d) d |> group_by(g) |> mutate(r = min_rank(desc(x)), q = dense_rank(desc(s))),
    grouped_pct = function(d) d |> group_by(g) |> mutate(r = percent_rank(desc(i)), q = cume_dist(desc(x)))
  )
  for (nm in names(pipes)) {
    f29_native(f29_df(), pipes[[nm]], nm)
  }
})

test_that("F29 A0: rank functions with a desc() argument are native in filter()", {
  skip_if_no_gpu()
  pipes <- list(
    ungrouped = function(d) d |> filter(min_rank(desc(x)) <= 2),
    grouped = function(d) d |> group_by(g) |> filter(min_rank(desc(x)) <= 1),
    grouped_n = function(d) d |> group_by(g) |> filter(dense_rank(desc(i)) <= 0.5 * n()),
    string = function(d) d |> filter(min_rank(desc(s)) <= 3),
    by = function(d) d |> filter(min_rank(desc(x)) <= 1, .by = g)
  )
  for (nm in names(pipes)) {
    f29_native(f29_df(), pipes[[nm]], nm)
  }
})

test_that("F29 A0: row_number(desc(x)) and nested desc() still match dplyr", {
  skip_if_no_gpu()
  f29_both(f29_df(), function(d) d |> mutate(r = row_number(desc(x))))
  f29_both(f29_df(), function(d) d |> group_by(g) |> mutate(r = row_number(desc(i))))
  f29_both(f29_df(), function(d) d |> mutate(r = min_rank(desc(desc(x)))))
})

test_that("F29 A: top_n() with a constant n is native, ungrouped", {
  skip_if_no_gpu()
  k <- 2
  pipes <- list(
    top2 = function(d) top_n(d, 2, x),
    top1 = function(d) top_n(d, 1, x),
    ties = function(d) top_n(d, 3, x),
    neg = function(d) top_n(d, -2, x),
    zero = function(d) top_n(d, 0, x),
    frac_n = function(d) top_n(d, 2.5, x),
    int_n = function(d) top_n(d, 2L, x),
    int_wt = function(d) top_n(d, 2L, i),
    string_wt = function(d) top_n(d, 2, s),
    string_wt_neg = function(d) top_n(d, -2, s),
    date_wt = function(d) top_n(d, 2, d),
    expr_wt = function(d) top_n(d, 2, x * -1),
    neg_wt = function(d) top_n(d, 2, -x),
    sum_wt = function(d) top_n(d, -3, x + i),
    var_n = function(d) top_n(d, k, x),
    big_n = function(d) top_n(d, 100, x)
  )
  for (nm in names(pipes)) {
    f29_native(f29_df(), pipes[[nm]], nm)
  }
})

test_that("F29 A: top_frac() and top_n(n() * c) are native", {
  skip_if_no_gpu()
  pipes <- list(
    frac = function(d) top_frac(d, 0.5, x),
    frac_neg = function(d) top_frac(d, -0.3, x),
    frac_zero = function(d) top_frac(d, 0, i),
    n_times = function(d) top_n(d, n() * .5, x),
    times_n = function(d) top_n(d, .25 * n(), i),
    grouped_frac = function(d) d |> group_by(g) |> top_frac(0.5, x),
    grouped_frac_neg = function(d) d |> group_by(g) |> top_frac(-0.5, i),
    grouped_n_times = function(d) d |> group_by(g) |> top_n(n() * .5, s)
  )
  for (nm in names(pipes)) {
    f29_native(f29_df(), pipes[[nm]], nm)
  }
})

test_that("F29 A: grouped top_n() is native and keeps the groups", {
  skip_if_no_gpu()
  pipes <- list(
    top1 = function(d) d |> group_by(g) |> top_n(1, x),
    neg1 = function(d) d |> group_by(g) |> top_n(-1, x),
    string = function(d) d |> group_by(g) |> top_n(2, s),
    two_keys = function(d) d |> mutate(h = i > 3L) |> group_by(g, h) |> top_n(1, x),
    prior_ops = function(d) d |> filter(i > 1L) |> group_by(g) |> top_n(1, x),
    mutate_first = function(d) d |> mutate(y = x * 2) |> group_by(g) |> top_n(2, y),
    then_summarise = function(d) d |> group_by(g) |> top_n(2, x) |> summarise(m = sum(x))
  )
  for (nm in names(pipes)) {
    f29_native(f29_df(), pipes[[nm]], nm)
  }
})

test_that("F29 A: top_n() on .drop = FALSE groups and on 0 rows", {
  skip_if_no_gpu()
  df <- f29_df()
  df$f <- factor(df$g, levels = c("a", "b", "c", "z"))
  pipes <- list(
    drop_false = function(d) d |> group_by(f, .drop = FALSE) |> top_n(1, x),
    zero_rows = function(d) d |> filter(x > 100) |> top_n(2, x),
    zero_rows_grouped = function(d) d |> filter(x > 100) |> group_by(g) |> top_frac(0.5, x)
  )
  for (nm in names(pipes)) {
    f29_native(df, pipes[[nm]], nm)
  }
  # The group keys (empty level kept) match dplyr. Not pinned native: a
  # summarise() after any filter() on a .drop = FALSE table falls back today
  # (summarise_impl() sends fixed-key tables to the CPU), unrelated to F29.
  f29_both(df, function(d) d |> group_by(f, .drop = FALSE) |> top_n(1, x) |> summarise(k = dplyr::n()))
})

test_that("F29 A: NA and NaN in wt get no rank, so those rows are dropped", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, NaN, 3, NA, 2, 3), y = 1:6)
  f29_native(df, function(d) top_n(d, 2, x), "na_pos")
  f29_native(df, function(d) top_n(d, -2, x), "na_neg")
  f29_native(df, function(d) top_n(d, 10, x), "na_all")
})

test_that("F29 A: top_n() with no wt selects by the last column, messages once, is native", {
  skip_if_no_gpu()
  df <- f29_df() |> dplyr::select(g, i, x)
  capture <- function(expr) {
    msgs <- character()
    res <- withCallingHandlers(expr, message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    list(res = res, msgs = msgs)
  }
  want <- capture(top_n(tibble::as_tibble(df), 2))
  want_g <- capture(top_n(dplyr::group_by(tibble::as_tibble(df), g), 1))
  expect_identical(want$msgs, "Selecting by x")

  withr::local_options(cuplyr.fallback = "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- capture(collect(top_n(tbl_gpu(df, lazy = lazy), 2)))
    expect_identical(got$msgs, want$msgs)
    expect_equal(tibble::as_tibble(got$res), want$res)
    got_g <- capture(collect(top_n(dplyr::group_by(tbl_gpu(df, lazy = lazy), g), 1)))
    expect_identical(got_g$msgs, want_g$msgs)
    expect_equal(dplyr::ungroup(tibble::as_tibble(got_g$res)), dplyr::ungroup(want_g$res))
  }
})

test_that("F29 A: shapes outside the native allowlist still match dplyr", {
  skip_if_no_gpu()
  df <- f29_df()
  df$f <- factor(df$s, levels = c("z", "q", "m", "c", "b", "a"))
  # n that reads a column, a factor wt, an opaque wt function
  f29_both(df, function(d) top_n(d, i[1] - 2L, x))
  f29_both(df, function(d) top_n(d, 2, f))
  f29_both(df, function(d) d |> group_by(g) |> top_n(-1, f))
  f29_both(df, function(d) top_n(d, 2, nchar(s)))
})

test_that("F29 A: a user-defined top_n_rank() is not rewritten", {
  skip_if_no_gpu()
  top_n_rank <- function(n, wt) wt > n
  f29_both(f29_df(), function(d) filter(d, top_n_rank(2, x)))
})

test_that("F29 A: top_n() errors match dplyr", {
  skip_if_no_gpu()
  f29_same_error(f29_df(), function(d) top_n(d, NA, x))
  f29_same_error(f29_df(), function(d) top_n(d, 2, nope))
})

# --- Added after round 1 review (B1, B3, B4) ---------------------------------

test_that("F29 A0: string desc() ranks follow the session collation (R1 B1)", {
  skip_if_no_gpu()
  # dplyr's desc() is -xtfrm(x); for strings xtfrm() ranks in LC_COLLATE
  # order, while a bare min_rank(s) (vec_rank) is C-ordered.
  withr::local_collate("en_US.UTF-8")
  df <- tibble::tibble(s = c("a", "B", "c", NA, "b", "A"), x = c(1, 2, 3, 4, 5, 6))
  f29_both(df, function(d) top_n(d, 2, s))
  f29_both(df, function(d) top_n(d, -2, s))
  f29_both(df, function(d) top_frac(d, 0.5, s))
  f29_both(df, function(d) mutate(d, r = min_rank(desc(s)), q = dense_rank(desc(s))))
  f29_both(df, function(d) filter(d, min_rank(desc(s)) <= 2))
})

test_that("F29 A: an n that draws random numbers is evaluated per group, as in dplyr (R1 B3)", {
  skip_if_no_gpu()
  check <- function(df, p, seed) {
    set.seed(seed)
    want <- p(tibble::as_tibble(df))
    want_next <- stats::runif(1)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(seed)
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      got_next <- stats::runif(1)
      expect_equal(dplyr::ungroup(tibble::as_tibble(got)), dplyr::ungroup(want))
      expect_identical(got_next, want_next)
    }
  }
  df <- tibble::tibble(g = rep(c("a", "b", "c", "d"), each = 5), x = as.numeric(1:20))
  df2 <- tibble::tibble(x = c(5, 3, 8, 1, 9), f = factor(c("a", "b", "c", "d", "e")))
  check(df, function(d) d |> group_by(g) |> top_n(sample(1:4, 1), x), 11)
  check(df2, function(d) top_n(d, sample(1:4, 1), f), 5)
})

test_that("F29 A0: a desc() or wt forwarded with {{ }} keeps its own environment (R1 B4)", {
  skip_if_no_gpu()
  rank_by <- function(d, ord) mutate(d, r = min_rank({{ ord }}))
  outer_rank <- function(d) {
    k <- -1
    rank_by(d, desc(x * k))
  }
  f29_both(tibble::tibble(x = c(3, 1, 2)), outer_rank)

  top_by <- function(d, w) top_n(d, 2, {{ w }})
  outer_top <- function(d) {
    k <- -1
    top_by(d, x * k)
  }
  f29_both(tibble::tibble(x = c(3, 1, 2, 5)), outer_top)
})

# --- Added after round 2 review (B1, B2) -------------------------------------

test_that("F29 A: a NaN computed on the GPU gets no rank, natively (R2 B1)", {
  skip_if_no_gpu()
  # An uploaded NaN becomes null; 0 / 0 computed on the GPU is a real NaN.
  # vec_rank() treats NaN as incomplete, so its rank is NA.
  df <- tibble::tibble(
    g = c("a", "a", "a", "b", "b", "b"),
    a = c(0, 1, 2, 0, 5, 3),
    b = c(0, 1, 1, 0, 1, 1)
  )
  pipes <- list(
    top_pos = function(d) d |> mutate(r = a / b) |> top_n(2, r),
    top_neg = function(d) d |> mutate(r = a / b) |> top_n(-5, r),
    top_grouped = function(d) d |> mutate(r = a / b) |> group_by(g) |> top_n(1, r),
    top_frac = function(d) d |> mutate(r = a / b) |> top_frac(0.5, r),
    top_expr_wt = function(d) top_n(d, 2, a / b),
    ranks = function(d) d |> mutate(
      k = min_rank(desc(a / b)), j = min_rank(a / b), q = dense_rank(desc(a / b)),
      p = percent_rank(desc(a / b)), c = cume_dist(a / b)
    ),
    ranks_grouped = function(d) d |> group_by(g) |> mutate(k = min_rank(desc(a / b)), j = dense_rank(a / b))
  )
  for (nm in names(pipes)) {
    f29_native(df, pipes[[nm]], nm)
  }
})

test_that("F29 A: a wt that draws random numbers is evaluated as in dplyr (R2 B2)", {
  skip_if_no_gpu()
  check <- function(df, p, seed) {
    set.seed(seed)
    want <- p(tibble::as_tibble(df))
    want_next <- stats::runif(1)
    for (lazy in c(FALSE, TRUE)) {
      set.seed(seed)
      got <- collect(p(tbl_gpu(df, lazy = lazy)))
      got_next <- stats::runif(1)
      expect_equal(dplyr::ungroup(tibble::as_tibble(got)), dplyr::ungroup(want))
      expect_identical(got_next, want_next)
    }
  }
  df <- tibble::tibble(x = c(5, 3, 8, 1, 9, 2), g = c("a", "b", "a", "b", "a", "b"))
  check(df, function(d) top_n(d, 2, x * stats::rnorm(1)), 2)
  check(df, function(d) d |> group_by(g) |> top_n(1, x * stats::rnorm(1)), 2)
  check(df, function(d) top_n(d, -2, x + stats::runif(1)), 7)
})

# --- Added after round 3 review (B1) -----------------------------------------

test_that("F29: grouped slice_min()/slice_max() keep a GPU-computed NaN tied with NA, natively (R3 B1)", {
  skip_if_no_gpu()
  # The rank machinery now gives a GPU NaN an NA rank; the grouped slice
  # desugar must count it as missing too (dplyr: vec_rank(incomplete =
  # "rank"), NaN tied with NA after every value).
  df <- tibble::tibble(
    g = c("a", "a", "a", "b", "b", "b"),
    a = c(1, 3, 0, 2, 5, 4),
    b = c(1, 1, 0, 1, 1, 1)
  )
  pipes <- list(
    min_n3 = function(d) d |> mutate(r = a / b) |> group_by(g) |> slice_min(r, n = 3),
    min_prop = function(d) d |> mutate(r = a / b) |> group_by(g) |> slice_min(r, prop = 1),
    min_by = function(d) d |> mutate(r = a / b) |> slice_min(r, n = 3, by = g),
    max_n3 = function(d) d |> mutate(r = a / b) |> group_by(g) |> slice_max(r, n = 3),
    max_n1 = function(d) d |> mutate(r = a / b) |> group_by(g) |> slice_max(r, n = 1)
  )
  for (nm in names(pipes)) {
    f29_native(df, pipes[[nm]], nm)
  }
})
