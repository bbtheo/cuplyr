# F82: a filter()/filter_out() predicate that reads no column is evaluated
# exactly as often as dplyr evaluates it: once per ungrouped call (zero rows
# too), in dot order, and the rows kept are the ones that single evaluation
# selects. The F37/F75 tests cover grouped, `.by` and mutate/summarise.

f82o_df <- function() {
  tibble::tibble(x = c(1, 5, 10, NA, 3), s = c("a", "zz", "b", NA, "zz"), g = c(1L, 1L, 2L, 2L, 2L))
}

# Runs `verb(d, bump)` on dplyr and on the GPU (eager or lazy) with a fresh
# counter each time; checks the rows and the number of bump() calls. The
# pipeline collects, so a call made only at collect() is counted too.
f82o_expect_counts <- function(df, verb, lazy = FALSE, arrange_by = NULL) {
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

f82o_both <- function(df, verb, ...) {
  f82o_expect_counts(df, verb, lazy = FALSE, ...)
  f82o_expect_counts(df, verb, lazy = TRUE, ...)
}

# The seed is set right before the call; the next draw after it shows how many
# draws the call consumed. Several seeds, so a second draw can't agree by luck.
f82o_expect_rng <- function(df, verb, seeds = 1:6) {
  for (seed in seeds) {
    for (lazy in c(FALSE, TRUE)) {
      after <- numeric()
      pipeline <- function(d) {
        set.seed(seed)
        out <- collect(verb(d))
        after <<- c(after, stats::runif(1))
        out
      }
      if (lazy) {
        expect_same_as_dplyr_lazy(df, pipeline)
      } else {
        expect_same_as_dplyr(df, pipeline)
      }
      expect_identical(after[[2]], after[[1]], info = sprintf("seed %d, lazy %s", seed, lazy))
    }
  }
}

test_that("F82: an ungrouped column-free predicate is called once", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, bump() > 0))
  f82o_both(df, function(d, bump) filter(d, bump() == 1))
  # a second call returns 2: the rows of the first call are kept
  f82o_both(df, function(d, bump) filter(d, bump() == 2))
  f82o_both(df, function(d, bump) filter(d, bump() > 0 & TRUE))
  f82o_both(df, function(d, bump) filter(d, isTRUE(bump() == 1)))
})

test_that("F82: filter_out() calls a column-free predicate once", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter_out(d, bump() == 1))
  f82o_both(df, function(d, bump) filter_out(d, bump() == 2))
  f82o_both(df, function(d, bump) filter_out(d, x > 2, bump() == 1))
})

test_that("F82: a column-free predicate next to column predicates is called once", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, x > 2, bump() == 1))
  f82o_both(df, function(d, bump) filter(d, bump() == 1, x > 2))
  f82o_both(df, function(d, bump) filter(d, bump() == 1, bump() == 2, !is.na(x)))
})

test_that("F82: column-free and column-reading dots call in dot order", {
  skip_if_no_gpu()
  df <- f82o_df()
  # dplyr: dot 1 gets the first call, dot 2 the second
  f82o_both(df, function(d, bump) filter(d, bump() > 0, x > bump()))
  f82o_both(df, function(d, bump) filter(d, x > bump(), bump() == 2))
  f82o_both(df, function(d, bump) filter(d, bump() == 1, x > bump() * 2))
})

test_that("F82: a column-free logical vector predicate is called once", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, rep(bump() == 1, 5)))
  f82o_both(df, function(d, bump) filter(d, rep(bump() == 2, 5)))
  f82o_both(df, function(d, bump) filter(d, c(TRUE, FALSE, TRUE, TRUE, FALSE) & bump() == 1))
  f82o_both(df, function(d, bump) filter_out(d, rep(bump() == 1, 5)))
  f82o_both(df, function(d, bump) filter_out(d, c(TRUE, FALSE, TRUE, TRUE, FALSE) & bump() == 1))
})

test_that("F82: a column-free predicate is called once when dplyr runs the call", {
  skip_if_no_gpu()
  df <- f82o_df()
  # a local function the GPU can't run: the whole call goes to dplyr
  keep <- function(v) !is.na(v) & v > 2
  f82o_both(df, function(d, bump) filter(d, bump() == 1, keep(x)))
  f82o_both(df, function(d, bump) filter(d, keep(x), bump() == 1))
  f82o_both(df, function(d, bump) filter(d, bump() == 2, keep(x)))
  f82o_both(df, function(d, bump) filter_out(d, bump() == 1, keep(x)))
  f82o_both(df, function(d, bump) filter(d, rep(bump() == 1, 5), keep(x)))
})

test_that("F82: a column-free predicate is called once on zero rows", {
  skip_if_no_gpu()
  df <- f82o_df()[0, ]
  f82o_both(df, function(d, bump) filter(d, bump() > 0))
  f82o_both(df, function(d, bump) filter_out(d, bump() > 0))
  f82o_both(df, function(d, bump) filter(d, x > 1, bump() > 0))
})

test_that("F82: if_any()/if_all() with a column-free body call as dplyr does", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, if_any(c(x, g), ~ bump() > 1)))
  f82o_both(df, function(d, bump) filter(d, if_all(c(x, g), ~ bump() == 1)))
})

test_that("F82: grouped and .by column-free predicates still call once per group", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(group_by(d, g), bump() == 1))
  f82o_both(df, function(d, bump) filter(d, bump() == 2, .by = g))
  f82o_both(df, function(d, bump) filter_out(group_by(d, g), bump() == 1))
})

test_that("F82: a seeded runif() predicate keeps dplyr's rows", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_expect_rng(df, function(d) filter(d, runif(1) > 0.5))
  f82o_expect_rng(df, function(d) filter_out(d, runif(1) > 0.5))
  f82o_expect_rng(df, function(d) filter(d, runif(5) > 0.5))
  f82o_expect_rng(df, function(d) filter_out(d, runif(5) > 0.5))
  f82o_expect_rng(df, function(d) filter(d, x > 0, runif(1) > 0.5))
  f82o_expect_rng(df, function(d) filter(d, runif(1) > 0.5, x > runif(1) * 5))
})

# --- a column-free predicate that errors (round-1 review B1/B3) ---------------

test_that("F82: a misspelled name in a column-free predicate errors without a fallback", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f82o_df()
  for (lazy in c(FALSE, TRUE)) {
    err <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), made_up_f82_col > 5)), error = identity)
    expect_s3_class(err, "error")
    expect_false(inherits(err, "cuplyr_error_fallback"), info = conditionMessage(err))
    expect_match(conditionMessage(err), "not found")
  }
})

test_that("F82: a column-free predicate that errors gives dplyr's error, naming that dot", {
  skip_if_no_gpu()
  df <- f82o_df()
  fail <- function() stop("f82 boom")
  ref <- tryCatch(filter(df, fail(), x > 1), error = identity)
  expect_s3_class(ref, "error")
  for (lazy in c(FALSE, TRUE)) {
    got <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), fail(), x > 1)), error = identity)
    expect_identical(class(got), class(ref))
    expect_identical(conditionMessage(got), conditionMessage(ref))
    # the fallback notice names the dot that failed
    withr::with_options(list(cuplyr.fallback = "error"), {
      note <- tryCatch(collect(filter(tbl_gpu(df, lazy = lazy), fail(), x > 1)), error = identity)
    })
    expect_s3_class(note, "cuplyr_error_fallback")
    expect_match(conditionMessage(note), "for: fail()", fixed = TRUE)
    expect_match(conditionMessage(note), "reads no column", fixed = TRUE)
  }
})

test_that("F82: if_any() with a column-reading impure piece calls as dplyr does", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, if_any(c(x, g), ~ .x > bump())))
  f82o_both(df, function(d, bump) filter(d, if_all(c(x, g), ~ .x >= bump())))
})

# --- round-2 review B4: names bound only while the dot runs -------------------

test_that("F82: a column-free predicate that binds its own names is called once", {
  skip_if_no_gpu()
  df <- f82o_df()
  f82o_both(df, function(d, bump) filter(d, { v <- bump(); v > 0 }))
  f82o_both(df, function(d, bump) filter(d, { v <- bump(); v == 2 }))
  f82o_both(df, function(d, bump) filter(d, if (bump() > 0) TRUE else not_defined_f82))
  f82o_both(df, function(d, bump) filter(d, local({ z <- bump(); z == 1 })))
  f82o_both(df, function(d, bump) filter_out(d, { v <- bump(); v == 1 }))
  f82o_expect_rng(df, function(d) filter(d, { p <- runif(1); p > 0.5 }))
})

test_that("F82: a short-circuited unbound name in a column-free predicate is never read", {
  skip_if_no_gpu()
  df <- f82o_df()
  expect_same_as_dplyr(df, function(d) filter(d, TRUE || not_defined_f82))
  expect_same_as_dplyr_lazy(df, function(d) filter(d, TRUE || not_defined_f82))
  expect_same_as_dplyr(df, function(d) filter(d, FALSE && not_defined_f82))
  expect_same_as_dplyr_lazy(df, function(d) filter_out(d, FALSE && not_defined_f82))
})

# --- round-3 review B3: an impure if_any() lambda next to a CPU route ---------

test_that("F82: an impure if_any() lambda is called as dplyr calls it when the call goes to dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), s = c("3", "1", "0", NA, "5"),
                       g = c(5L, 4L, 3L, 2L, 1L))
  twice <- function(v) v * 2
  f82o_both(df, function(d, bump) filter(d, if_any(c(x, s), ~ .x > bump())))
  f82o_both(df, function(d, bump) filter(d, if_all(c(x, g), ~ .x > bump()), twice(g) > 0))
  f82o_both(df, function(d, bump) mutate(d, a = dplyr::if_any(c(x, g), ~ .x > bump()), w = twice(x)))
  f82o_both(df, function(d, bump) mutate(d, a = dplyr::if_all(c(x, g), ~ .x > bump())))
})

# --- round-4 review N1: side-effect-free reads in an if_any() lambda ----------

test_that("F82: an if_any()/if_all() lambda reading a variable by $, [[ or [ stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), y = c(4, 0, 2, 8, NA),
                       s1 = c("a", "b", "c", NA, "a"), s2 = c("c", "a", NA, "b", "b"))
  thr <- 4
  cfg <- list(thr = 4, lv = c("a", "b"))
  vals <- c(2, 4, 6)
  withr::local_options(cuplyr.fallback = "error")
  f82_native <- function(pipeline, ...) {
    expect_same_as_dplyr(df, pipeline, ...)
    expect_same_as_dplyr_lazy(df, pipeline, ...)
  }
  f82_native(function(d) filter(d, if_any(c(x, y), ~ .x > cfg$thr)))
  f82_native(function(d) filter(d, if_any(c(x, y), ~ .x > .env$thr)))
  f82_native(function(d) filter(d, if_any(c(x, y), ~ .x > .env[["thr"]])))
  f82_native(function(d) filter(d, if_all(c(x, y), ~ .x > vals[2])))
  f82_native(function(d) filter(d, if_any(c(s1, s2), ~ .x %in% cfg[["lv"]])))
  f82_native(function(d) mutate(d, a = dplyr::if_any(c(x, y), ~ .x > cfg$thr)))
  f82_native(function(d) mutate(d, a = dplyr::if_all(c(x, y), ~ .x > .env$thr)))
})

test_that("F82: an if_any() lambda whose subscript calls a function still calls it as dplyr does", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), y = c(4, 0, 2, 8, NA))
  f82o_both(df, function(d, bump) {
    cfg <- list(4, 6)
    filter(d, if_any(c(x, y), ~ .x > cfg[[bump()]]))
  })
})

test_that("F82: a missing argument read in an if_any() lambda gives dplyr's error", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), y = c(4, 0, 2, 8, NA))
  f <- function(d, cfg) filter(d, if_any(c(x, y), ~ .x > cfg$thr))
  g <- function(d, cfg) mutate(d, a = dplyr::if_all(c(x, y), ~ .x > cfg[["thr"]]))
  h <- function(d, vals) filter(d, if_all(c(x, y), ~ .x > vals[2]))
  for (run in list(function(d) f(d), function(d) g(d), function(d) h(d, vals = c(nope_f82_var, 4)))) {
    ref <- tryCatch(run(df), error = identity)
    expect_s3_class(ref, "error")
    for (lazy in c(FALSE, TRUE)) {
      got <- tryCatch(collect(run(tbl_gpu(df, lazy = lazy))), error = identity)
      expect_identical(class(got), class(ref), info = conditionMessage(got))
      expect_identical(conditionMessage(got), conditionMessage(ref))
    }
  }
  # with the argument given, the lambda stays native
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, function(d) f(d, list(thr = 4)))
  expect_same_as_dplyr_lazy(df, function(d) g(d, list(thr = 4)))
})

test_that("F82: an active binding read through .env[[k]] in an if_any() lambda runs as often as in dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 5, 10, NA, 3), y = c(4, 0, 2, 8, NA))
  counts <- integer()
  for (lazy in c(NA, FALSE, TRUE)) {
    e <- new.env()
    n <- 0L
    makeActiveBinding("thr_f82", function() { n <<- n + 1L; 4 }, e)
    run <- local({
      k <- "thr_f82"
      function(d) filter(d, if_any(c(x, y), ~ .x > .env[[k]]))
    })
    environment(run) <- rlang::env(e, k = "thr_f82")
    out <- if (is.na(lazy)) run(df) else collect(run(tbl_gpu(df, lazy = lazy)))
    counts <- c(counts, n)
    if (is.na(lazy)) ref <- out else expect_equal(tibble::as_tibble(out), ref)
  }
  expect_identical(counts[2:3], rep(counts[[1]], 2))
})
