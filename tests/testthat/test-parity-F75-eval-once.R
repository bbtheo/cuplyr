# F75: an expression's column-free calls run as often as dplyr runs them:
# once for an ungrouped verb, once per group for a grouped one (dplyr's
# eval_all_summarise()/eval_all_mutate()/eval_all_filter() loop over groups),
# and the result matches dplyr's values exactly.

f75_df <- function() {
  tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), x = c(1, 2, 3, NA, 5), y = c(10L, 20L, 30L, 40L, 50L))
}

# Runs `verb(d, bump)` on dplyr and on the GPU (eager or lazy) with a fresh
# counter each time and checks values and the number of bump() calls. The
# pipeline collects, so a call evaluated only at collect() is counted too.
f75_expect_counts <- function(df, verb, lazy = FALSE, arrange_by = NULL) {
  counts <- integer()
  pipeline <- function(d) {
    k <- 0L
    bump <- function() {
      k <<- k + 1L
      k
    }
    out <- collect(verb(d, bump))
    counts <<- c(counts, k)
    out
  }
  if (lazy) {
    expect_same_as_dplyr_lazy(df, pipeline, arrange_by = arrange_by)
  } else {
    expect_same_as_dplyr(df, pipeline, arrange_by = arrange_by)
  }
  expect_length(counts, 2L)
  expect_identical(counts[[2]], counts[[1]],
                   info = sprintf("bump() calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
}

# Same with the RNG: the seed is set right before the call, and the next draw
# after the call shows how many draws the call consumed.
f75_expect_rng <- function(df, verb, lazy = FALSE) {
  after <- numeric()
  pipeline <- function(d) {
    set.seed(75)
    out <- collect(verb(d))
    after <<- c(after, stats::runif(1))
    out
  }
  if (lazy) {
    expect_same_as_dplyr_lazy(df, pipeline)
  } else {
    expect_same_as_dplyr(df, pipeline)
  }
  expect_identical(after[[2]], after[[1]])
}

f75_both <- function(df, verb, ...) {
  f75_expect_counts(df, verb, lazy = FALSE, ...)
  f75_expect_counts(df, verb, lazy = TRUE, ...)
}

# --- summarise(): ungrouped --------------------------------------------------

test_that("F75: ungrouped summarise() evaluates a column-free call once", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(x + bump(), na.rm = TRUE)))
  f75_both(df, function(d, bump) summarise(d, s = sum(x + bump())))
  f75_both(df, function(d, bump) summarise(d, m = mean(y * bump())))
  f75_both(df, function(d, bump) summarise(d, m = max(y - bump()), n = n()))
})

test_that("F75: a column-free call in two dots runs once per dot, in order", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump()), t = max(y - bump())))
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump() + bump())))
})

test_that("F75: summarise() with a sibling dot that needs dplyr still calls once", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump()), r = cor(y, y)))
  f75_both(df, function(d, bump) summarise(d, r = cor(y, y), s = sum(y + bump())))
})

test_that("F75: column-free calls in na.rm=, probs= and nth()'s n run once", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(x, na.rm = bump() > 0)))
  f75_both(df, function(d, bump) summarise(d, q = quantile(y, probs = bump() / 4, names = FALSE)))
  f75_both(df, function(d, bump) summarise(d, v = dplyr::nth(y, bump() + 1L)))
})

test_that("F75: window arguments and across() bodies run as dplyr runs them", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(cumsum(y + bump()))))
  f75_both(df, function(d, bump) summarise(d, s = dplyr::last(cumsum(y * bump()))))
  f75_both(df, function(d, bump) summarise(d, across(c(x, y), ~ sum(.x + bump(), na.rm = TRUE))))
  f75_both(df, function(d, bump) summarise(d, across(c(x, y), list(a = ~ max(.x - bump(), na.rm = TRUE), b = ~ min(.x)))))
})

test_that("F75: a bare column-free dot and a column-free reducer run as in dplyr", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = bump(), m = mean(y)))
  f75_both(df, function(d, bump) summarise(d, s = sum(bump()), m = mean(y)))
  f75_both(df, function(d, bump) summarise(d, s = sum(y) + bump()))
})

test_that("F75: ungrouped summarise() on zero rows", {
  skip_if_no_gpu()
  df <- f75_df()[0, ]
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump()), n = n()))
})

test_that("F75: runif() in an ungrouped aggregate draws like dplyr", {
  skip_if_no_gpu()
  df <- f75_df()
  for (lazy in c(FALSE, TRUE)) {
    f75_expect_rng(df, function(d) summarise(d, s = sum(y + runif(1))), lazy = lazy)
    f75_expect_rng(df, function(d) summarise(d, s = sum(y * runif(1)), t = mean(y + rnorm(1))), lazy = lazy)
  }
})

# --- summarise(): grouped, .by, empty groups ---------------------------------

test_that("F75: grouped summarise() evaluates a column-free call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(group_by(d, g), s = sum(y + bump())))
  f75_both(df, function(d, bump) summarise(group_by(d, g), s = sum(x + bump(), na.rm = TRUE), m = mean(y * bump())))
  f75_both(df, function(d, bump) summarise(group_by(d, g), s = min(y + bump())))
  f75_both(df, function(d, bump) summarise(group_by(d, g), v = dplyr::nth(y, bump())))
  f75_both(df, function(d, bump) summarise(group_by(d, g), s = dplyr::last(cumsum(y + bump()))))
  f75_both(df, function(d, bump) summarise(group_by(d, g), across(c(x, y), ~ sum(.x * bump(), na.rm = TRUE))))
})

test_that("F75: summarise(.by =) evaluates a column-free call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump()), .by = g))
  f75_both(df, function(d, bump) summarise(d, s = sum(y + bump()), n = n(), .by = g))
})

test_that("F75: grouped summarise() on zero rows and with empty groups", {
  skip_if_no_gpu()
  df0 <- f75_df()[0, ]
  f75_both(df0, function(d, bump) summarise(group_by(d, g), s = sum(y + bump())))
  ff <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), y = c(1, 2, 3))
  f75_both(ff, function(d, bump) summarise(group_by(d, f, .drop = FALSE), s = sum(y + bump())))
})

test_that("F75: runif() in a grouped aggregate draws once per group like dplyr", {
  skip_if_no_gpu()
  df <- f75_df()
  for (lazy in c(FALSE, TRUE)) {
    f75_expect_rng(df, function(d) summarise(group_by(d, g), s = sum(y + runif(1))), lazy = lazy)
    f75_expect_rng(df, function(d) summarise(d, s = mean(y * runif(1)), .by = g), lazy = lazy)
  }
})

test_that("F75: a user function that masks a base function counts as a call", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) {
    c <- function(...) {
      bump()
      base::c(...)
    }
    summarise(group_by(d, g), s = sum(y %in% c(10L, 30L)))
  })
})

# --- grouped mutate()/transmute()/filter() -----------------------------------

test_that("F75: grouped mutate() evaluates a column-free call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) mutate(group_by(d, g), s = y + bump()))
  f75_both(df, function(d, bump) mutate(group_by(d, g), s = bump()))
  f75_both(df, function(d, bump) mutate(group_by(d, g), s = cumsum(y + bump())))
  f75_both(df, function(d, bump) mutate(group_by(d, g), s = dplyr::lag(y, default = bump())))
  f75_both(df, function(d, bump) mutate(group_by(d, g), across(c(x, y), ~ .x + bump())))
  f75_both(df, function(d, bump) mutate(d, s = y + bump(), .by = g))
  f75_both(df, function(d, bump) {
    c <- function(...) {
      bump()
      base::c(...)
    }
    mutate(group_by(d, g), s = y %in% c(10L, 30L))
  })
})

test_that("F75: grouped transmute() evaluates a column-free call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) transmute(group_by(d, g), s = y + bump()))
})

test_that("F75: grouped filter() evaluates a column-free call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) filter(group_by(d, g), y > 10 * bump()))
  f75_both(df, function(d, bump) filter(group_by(d, g), if_any(c(x, y), ~ .x > bump())))
  f75_both(df, function(d, bump) filter(d, y > 10 * bump(), .by = g))
})

test_that("F75: grouped verbs on zero rows", {
  skip_if_no_gpu()
  df0 <- f75_df()[0, ]
  f75_both(df0, function(d, bump) mutate(group_by(d, g), s = y + bump()))
  f75_both(df0, function(d, bump) filter(group_by(d, g), y > bump()))
})

test_that("F75: runif() in grouped mutate()/filter() draws once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  for (lazy in c(FALSE, TRUE)) {
    f75_expect_rng(df, function(d) mutate(group_by(d, g), r = y + runif(1)), lazy = lazy)
    f75_expect_rng(df, function(d) mutate(group_by(d, g), r = runif(1)), lazy = lazy)
    f75_expect_rng(df, function(d) filter(group_by(d, g), y > 30 * runif(1)), lazy = lazy)
  }
})

test_that("F75: ungrouped mutate()/filter() still evaluate once (unchanged)", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) mutate(d, s = y + bump(), t = bump()))
  f75_both(df, function(d, bump) filter(d, y > 10 * bump()))
  f75_both(df, function(d, bump) mutate(d, s = y + bump(), across(x, ~ .x * 2)))
})

# --- count()/tally()/add_count() wt= -----------------------------------------

test_that("F75: count(wt =) keeps the caller's environment", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) {
    kk <- 3L
    count(d, g, wt = y * kk)
  })
  f75_both(df, function(d, bump) {
    kk <- 3L
    tally(group_by(d, g), wt = y * kk)
  })
  f75_both(df, function(d, bump) {
    kk <- 3L
    add_count(d, g, wt = y * kk)
  })
})

test_that("F75: count()/tally()/add_count() run a column-free wt call once per group", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) count(d, g, wt = y + bump()))
  f75_both(df, function(d, bump) count(d, wt = y + bump()))
  f75_both(df, function(d, bump) tally(group_by(d, g), wt = y + bump()))
  f75_both(df, function(d, bump) add_count(d, g, wt = y + bump()))
})

# --- pure folds stay native, and are evaluated once ---------------------------

test_that("F75: pure column-free calls stay on the GPU (eager)", {
  skip_if_no_gpu()
  df <- f75_df()
  lim <- list(v = 2)
  k <- 2
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) summarise(group_by(d, g), s = sum(y %in% c(10L, 30L))),
    function(d) summarise(d, s = sum(y > as.integer("15")), m = mean(x * k, na.rm = TRUE)),
    function(d) summarise(group_by(d, g), s = sum(y * seq_len(1)), t = max(x - lim$v, na.rm = TRUE)),
    function(d) summarise(d, s = sum(y + .env$k), .by = g),
    function(d) mutate(group_by(d, g), s = y %in% c(10L, 50L), t = y * abs(-2L), u = 5L),
    function(d) filter(group_by(d, g), y > max(c(15L, 5L))),
    function(d) count(d, g, wt = y),
    function(d) count(d, g, wt = y * k)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
  }
})

test_that("F75: a pure call's warning is raised once by an ungrouped summarise()", {
  skip_if_no_gpu()
  df <- f75_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df)
    if (lazy) gt <- as_lazy(gt)
    w <- testthat::capture_warnings(
      res <- collect(summarise(gt, s = sum(y + as.integer("a"))))
    )
    expect_length(w, 1L)
    expect_match(w[[1]], "NAs introduced by coercion")
    expect_identical(res$s, NA_integer_)
  }
})

# --- added after review round 1 ----------------------------------------------

test_that("F75: across()/if_any() lambdas over native functions stay on the GPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 4, 5, 6),
                       y = c(6, 5, 4, 3, 2, 1), s = c("ab", "ba", "ab", "bb", "aa", "ab"))
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) mutate(group_by(d, g), across(c(x, y), ~ roll_mean(.x, 2))),
    function(d) mutate(group_by(d, g), across(c(x, y), function(v) cumsum(v) * 2)),
    function(d) mutate(d, across(c(x, y), ~ .x - mean(.x)), .by = g),
    function(d) filter(group_by(d, g), if_any(c(s), ~ startsWith(.x, "a"))),
    function(d) filter(group_by(d, g), if_all(c(x, y), ~ .x > 1)),
    function(d) summarise(d, across(c(x), ~ max(roll_mean(.x, 2), na.rm = TRUE))),
    function(d) summarise(group_by(d, g), across(c(x, y), list(m = ~ mean(.x), s = ~ sum(.x * 2))))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
  }
})

test_that("F75: lambdas over native functions stay on the GPU (lazy, grouped mutate)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 4, 5, 6), y = c(6, 5, 4, 3, 2, 1))
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr_lazy(df, function(d) mutate(group_by(d, g), across(c(x, y), ~ roll_mean(.x, 2))))
  expect_same_as_dplyr_lazy(df, function(d) filter(group_by(d, g), if_all(c(x, y), ~ .x > 1)))
})

f75_expect_one_warning <- function(make) {
  for (lazy in c(FALSE, TRUE)) {
    w <- testthat::capture_warnings(collect(make(lazy)))
    expect_length(w, 1L)
  }
}

test_that("F75: a pure call in na.rm=, probs= or nth()'s n warns once", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(10, 20, 30))
  gt <- function(lazy, d = df) {
    t <- tbl_gpu(d)
    if (lazy) as_lazy(t) else t
  }
  f75_expect_one_warning(function(lazy) summarise(gt(lazy), s = sum(y, na.rm = is.na(as.integer("a")))))
  f75_expect_one_warning(function(lazy) {
    summarise(gt(lazy), q = quantile(y, probs = mean(as.numeric(c("0.5", "x")), na.rm = TRUE), names = FALSE))
  })
  f75_expect_one_warning(function(lazy) summarise(gt(lazy), v = dplyr::nth(y, length(as.integer(c("1", "a"))))))
})

test_that("F75: a pure call warns once on the .by and .drop = FALSE paths", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(10, 20, 30))
  ff <- tibble::tibble(f = factor(c("a", "a", "c"), levels = c("a", "b", "c")), y = c(1, 2, 3))
  gt <- function(lazy, d) {
    t <- tbl_gpu(d)
    if (lazy) as_lazy(t) else t
  }
  f75_expect_one_warning(function(lazy) summarise(gt(lazy, df), s = sum(y + as.integer("a")), .by = g))
  f75_expect_one_warning(function(lazy) {
    summarise(group_by(gt(lazy, ff), f, .drop = FALSE), s = sum(y + as.integer("a")))
  })
  f75_expect_one_warning(function(lazy) {
    summarise(group_by(gt(lazy, ff), f, .drop = FALSE), s = sum(y, na.rm = is.na(as.integer("a"))))
  })
})

test_that("F75: count(wt =) with a bare non-column symbol uses the caller's value", {
  skip_if_no_gpu()
  df <- f75_df()
  f75_both(df, function(d, bump) {
    w <- c(1, 2, 3, 4, 5)
    count(d, g, wt = w)
  })
  f75_both(df, function(d, bump) {
    w <- c(1, 2, 3, 4, 5)
    add_count(d, g, wt = w)
  })
})

# --- added after review round 2 ----------------------------------------------

# A free `.x` (a function argument of the caller) inside a two-sided formula
# such as case_when()'s `cond ~ value` is not a lambda formal: `bump(.x)` is a
# column-free call that dplyr runs once per group.
f75_expect_counts_dotx <- function(df, verb, lazy = FALSE) {
  counts <- integer()
  pipeline <- function(d) {
    k <- 0L
    bump <- function(v) {
      k <<- k + 1L
      k * v
    }
    out <- collect(verb(d, bump, 10))
    counts <<- c(counts, k)
    out
  }
  if (lazy) {
    expect_same_as_dplyr_lazy(df, pipeline)
  } else {
    expect_same_as_dplyr(df, pipeline)
  }
  expect_identical(counts[[2]], counts[[1]],
                   info = sprintf("bump() calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
}

test_that("F75: a free .x in a case_when() formula is not a lambda formal", {
  skip_if_no_gpu()
  df <- f75_df()
  verbs <- list(
    function(d, bump, .x) mutate(group_by(d, g), s = dplyr::case_when(y > 0L ~ bump(.x))),
    function(d, bump, .x) summarise(group_by(d, g), s = sum(dplyr::case_when(y > 0L ~ y * bump(.x)))),
    function(d, bump, .x) summarise(d, s = sum(dplyr::case_when(y > 0L ~ y * bump(.x))), r = cor(y, y)),
    function(d, bump, .x) filter(group_by(d, g), dplyr::case_when(y > 0L ~ y > bump(.x)))
  )
  for (v in verbs) {
    f75_expect_counts_dotx(df, v, lazy = FALSE)
    f75_expect_counts_dotx(df, v, lazy = TRUE)
  }
})

test_that("F75: runif() over a free .x in a case_when() formula draws per group", {
  skip_if_no_gpu()
  df <- f75_df()
  for (lazy in c(FALSE, TRUE)) {
    f75_expect_rng(df, function(d) {
      (function(.x) mutate(group_by(d, g), s = dplyr::case_when(y > 0L ~ runif(1, max = .x))))(5)
    }, lazy = lazy)
  }
})

# --- added after review round 3 ----------------------------------------------

test_that("F75: type-test where() lambdas and name builders in .cols stay on the GPU", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1.5, 2.5, 3.5, 4.5), n = 1:4, s = c("a", "b", "a", "b"))
  df2 <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1.5, 2.5, 3.5, 4.5), f = factor(c("a", "b", "a", "b")))
  df3 <- tibble::tibble(g = c(1L, 1L, 2L, 2L), d = as.Date("2024-01-01") + 0:3, x = c(1.5, 2.5, 3.5, 4.5))
  lk <- data.frame(x = 1, n = 1L)
  withr::local_options(cuplyr.fallback = "error")
  cases <- list(
    list(df, function(d) summarise(d, across(where(~ is.double(.x)), sum))),
    list(df, function(d) summarise(group_by(d, g), across(where(~ is.integer(.x)), sum))),
    list(df3, function(d) mutate(group_by(d, g), across(where(\(v) inherits(v, "Date")), ~ .x + 1))),
    list(df, function(d) filter(group_by(d, g), if_any(where(~ is.double(.x)), ~ .x > 2))),
    list(df, function(d) summarise(group_by(d, g), across(any_of(colnames(lk)), sum))),
    list(df2, function(d) mutate(group_by(d, g), across(where(~ !is.factor(.x)), ~ .x * 2))),
    list(df, function(d) summarise(group_by(d, g), across(all_of(grep("^x", names(lk), value = TRUE)), sum)))
  )
  for (cs in cases) {
    expect_same_as_dplyr(cs[[1]], cs[[2]])
    expect_same_as_dplyr_lazy(cs[[1]], cs[[2]])
  }
})

test_that("F75: with_order() given a function by name runs it as dplyr does", {
  skip_if_no_gpu()
  df <- f75_df()
  verbs <- list(
    function(d, bump, .x) mutate(group_by(d, g), s = y + dplyr::with_order(c(2, 1), bump, c(1, 2))[1]),
    function(d, bump, .x) filter(group_by(d, g), y < 10 * dplyr::with_order(c(2, 1), bump, c(1, 2))[1] + 15),
    function(d, bump, .x) summarise(d, s = sum(y + dplyr::with_order(c(2, 1), bump, c(1, 2))[1]), r = cor(y, y))
  )
  for (v in verbs) {
    f75_expect_counts_dotx(df, v, lazy = FALSE)
    f75_expect_counts_dotx(df, v, lazy = TRUE)
  }
})
