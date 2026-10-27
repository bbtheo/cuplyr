# F75 (attempt 2): top_n()/top_frac() evaluate `wt` as often as dplyr does.
# dplyr's top_n() is filter(x, top_n_rank(n, wt)): `wt` is evaluated once for
# an ungrouped table and once per group for a grouped one. cuplyr rewrites the
# dot to a native min_rank() predicate (rewrite_top_n_rank_dot(), R/filter.R);
# a `wt` it can't fold without running a side effect must reach dplyr
# unevaluated.

f75t_df <- function() {
  tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), y = c(4, 1, 2, 3))
}

# Runs `verb(d, f, bump)` on dplyr and on the GPU (eager or lazy) with a fresh
# counter each time; checks the rows and that `f`/`bump` together ran as many
# times as under dplyr. `f(v)` sums a vector or the rows of a data frame.
f75t_expect_counts <- function(df, verb, lazy = FALSE) {
  counts <- integer()
  pipeline <- function(d) {
    k <- 0L
    bump <- function() {
      k <<- k + 1L
      k
    }
    f <- function(v) {
      k <<- k + 1L
      if (is.data.frame(v)) rowSums(v) else sum(v)
    }
    out <- collect(verb(d, f, bump))
    counts <<- c(counts, k)
    out
  }
  if (lazy) {
    expect_same_as_dplyr_lazy(df, pipeline)
  } else {
    expect_same_as_dplyr(df, pipeline)
  }
  expect_length(counts, 2L)
  expect_identical(counts[[2]], counts[[1]],
                   info = sprintf("calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
}

f75t_both <- function(df, verb) {
  f75t_expect_counts(df, verb, lazy = FALSE)
  f75t_expect_counts(df, verb, lazy = TRUE)
}

# The seed is set right before the call; the next draw after it shows how many
# draws the call consumed.
f75t_expect_rng <- function(df, verb, lazy = FALSE) {
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

# --- B6: a user function over pick()/c_across() in a grouped wt --------------

test_that("F75: grouped top_n() with wt = f(pick(everything())) calls f once per group", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_n(group_by(d, g), 1, wt = f(pick(everything())))
  })
})

test_that("F75: grouped top_n() with wt = f(pick(\"x\", \"y\")) calls f once per group", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_n(group_by(d, g), 1, wt = f(pick("x", "y")))
  })
})

test_that("F75: grouped top_frac() with wt = f(pick(where(is.numeric))) calls f once per group", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_frac(group_by(d, g), 0.5, wt = f(pick(where(is.numeric))))
  })
})

test_that("F75: grouped top_n() with wt = f(c_across(everything())) calls f once per group", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_n(group_by(d, g), 1, wt = f(dplyr::c_across(everything())))
  })
})

test_that("F75: grouped top_n() with wt = f(pick(x, y)) (bare names) calls f once per group", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_n(group_by(d, g), 1, wt = f(pick(x, y)))
  })
})

test_that("F75: ungrouped top_n() with wt = f(pick(...)) calls f once", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_n(d, 2, wt = f(pick(everything())))
  })
  f75t_both(f75t_df(), function(d, f, bump) {
    dplyr::top_frac(d, 0.5, wt = f(pick("x", "y")))
  })
})

# --- B7: a column-free call in an ungrouped wt ------------------------------

test_that("F75: ungrouped top_n() runs a column-free call in wt once and keeps dplyr's rows", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, 2, wt = x * (bump() - 2)))
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, -2, wt = x * (bump() - 2)))
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, 1, wt = x - 10 * bump()))
})

test_that("F75: ungrouped top_frac() runs a column-free call in wt once", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_frac(d, 0.5, wt = x * (bump() - 2)))
  f75t_both(df, function(d, f, bump) dplyr::top_frac(d, -0.25, wt = y * (bump() - 2)))
})

test_that("F75: ungrouped top_n() with an impure wt on NA and zero-row inputs", {
  skip_if_no_gpu()
  na_df <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), x = c(1, NA, 3, 4, NA))
  f75t_both(na_df, function(d, f, bump) dplyr::top_n(d, 2, wt = x * (bump() - 2)))
  empty <- f75t_df()[0, ]
  f75t_both(empty, function(d, f, bump) dplyr::top_n(d, 2, wt = x * (bump() - 2)))
})

test_that("F75: a user function of a column in wt runs as often as in dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, 2, wt = f(x) + x))
  f75t_both(df, function(d, f, bump) dplyr::top_n(group_by(d, g), 1, wt = f(x) + x))
})

test_that("F75: runif() in an ungrouped top_n() wt draws like dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  for (lazy in c(FALSE, TRUE)) {
    f75t_expect_rng(df, function(d) dplyr::top_n(d, 2, wt = x * (stats::runif(1) - 0.5)), lazy = lazy)
  }
})

# --- controls (already matched before the fix) ---------------------------------

test_that("F75: grouped top_n() with a column-free call and plain filter() still match", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_n(group_by(d, g), 1, wt = x * (bump() - 2)))
  f75t_both(df, function(d, f, bump) filter(d, x * (bump() - 2) < -1))
})

test_that("F75: a pure wt keeps top_n()/top_frac() on the GPU", {
  skip_if_no_gpu()
  df <- f75t_df()
  k <- 2
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) dplyr::top_n(d, 2, x),
    function(d) dplyr::top_n(d, 2, wt = x * abs(-2)),
    function(d) dplyr::top_n(d, -1, wt = y - max(c(1, 2))),
    function(d) dplyr::top_n(d, 2, wt = x * k),
    function(d) dplyr::top_frac(d, 0.5, wt = y + as.numeric("1")),
    function(d) dplyr::top_n(group_by(d, g), 1, wt = x * abs(-2)),
    function(d) dplyr::top_frac(group_by(d, g), 0.5, wt = y - k)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

# --- added after attempt-2 review round 1 --------------------------------------
# A wt that names no column by symbol (column-free, `.data[["x"]]`, through
# `{{ }}`): filter()'s own route would evaluate it before dplyr does.

f75t_wrap_c <- function(d, w) dplyr::top_n(d, 2, wt = {{ w }} * c(1, 2, 3, 4))
f75t_wrap_g <- function(d, w) dplyr::top_n(group_by(d, g), 1, wt = {{ w }})
f75t_wrap_g0 <- function(d, w) dplyr::top_n(group_by(d, g), 1, wt = 0 + {{ w }})
f75t_wrap_u <- function(d, w) dplyr::top_n(d, 2, wt = {{ w }})
f75t_wrap_n <- function(d, w) dplyr::top_n(d, 2, wt = x * {{ w }})

test_that("F75: ungrouped top_n() with a column-free vector wt runs it once", {
  skip_if_no_gpu()
  df <- f75t_df()
  # bump() is the first call evaluated, so the wt sign tells which call was used
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, 2, wt = c(1, 2, 3, 4) * (bump() - 1.5)))
  f75t_both(df, function(d, f, bump) dplyr::top_frac(d, 0.5, wt = c(1, 2, 3, 4) * (bump() - 1.5)))
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, -1, wt = c(4, 1, 3, 2) * (bump() - 1.5)))
})

test_that("F75: ungrouped top_n() with a scalar column-free wt runs it once", {
  skip_if_no_gpu()
  f75t_both(f75t_df(), function(d, f, bump) dplyr::top_n(d, 2, wt = bump()))
})

test_that("F75: ungrouped top_n() with wt = runif(<n rows>) draws like dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  for (lazy in c(FALSE, TRUE)) {
    f75t_expect_rng(df, function(d) dplyr::top_n(d, 2, wt = stats::runif(4)), lazy = lazy)
  }
})

test_that("F75: ungrouped top_n() with a column read through .data runs wt once", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_n(d, 2, wt = (bump() - 1.5) * .data[["x"]]))
  f75t_both(df, function(d, f, bump) dplyr::top_frac(d, 0.5, wt = (bump() - 2) * .data$x))
})

test_that("F75: top_n() with an impure wt forwarded through {{ }} runs it as in dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) f75t_wrap_c(d, bump() - 1.5))
  f75t_both(df, function(d, f, bump) f75t_wrap_u(d, x * (bump() - 2)))
  f75t_both(df, function(d, f, bump) f75t_wrap_n(d, (bump() - 2)))
  f75t_both(df, function(d, f, bump) f75t_wrap_g(d, f(pick(everything()))))
  f75t_both(df, function(d, f, bump) f75t_wrap_g0(d, f(pick(x, y))))
})

test_that("F75: dplyr::pick() in a grouped wt and other passing shapes", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) dplyr::top_n(group_by(d, g), 1, wt = f(dplyr::pick(everything()))))
  f75t_both(df, function(d, f, bump) dplyr::top_frac(group_by(d, g), 0.5, wt = f(dplyr::pick(x, y))))
  f75t_both(df, function(d, f, bump) dplyr::top_frac(group_by(d, g), 0.5, wt = x * (bump() - 2)))
})

test_that("F75: pure wt shapes without a bare column stay on the GPU", {
  skip_if_no_gpu()
  df <- f75t_df()
  kk <- 2
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) dplyr::top_n(d, 2, wt = x * nrow(df)),
    function(d) dplyr::top_n(d, 2, wt = .data[["x"]] * 2),
    function(d) dplyr::top_n(d, 2, wt = x * .env$kk),
    function(d) dplyr::top_n(group_by(d, g), 1, wt = y * kk + abs(-1))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

# --- added after attempt-2 review round 2 --------------------------------------

test_that("F75: a call in a .env[[ ]] subscript of a top_n() wt runs once", {
  skip_if_no_gpu()
  df <- f75t_df()
  # name_fn() gives "w1" on its first call, "w2" after
  verb <- function(d, f, bump) {
    w1 <- c(1, 2, 3, 4)
    w2 <- c(4, 3, 2, 1)
    name_fn <- function() if (bump() == 1L) "w1" else "w2"
    dplyr::top_n(d, 2, wt = .env[[name_fn()]])
  }
  f75t_both(df, verb)
  verb2 <- function(d, f, bump) {
    w1 <- c(1, 2, 3, 4)
    w2 <- c(4, 3, 2, 1)
    names_k <- c("w1", "w2", "w2")
    dplyr::top_frac(d, 0.5, wt = .env[[names_k[bump()]]])
  }
  f75t_both(df, verb2)
})

test_that("F75: a top_n() wt read through .env[[<variable>]] matches dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  verb <- function(d, f, bump) {
    w1 <- c(1, 2, 3, 4)
    nm <- "w1"
    dplyr::top_n(d, 2, wt = .env[[nm]])
  }
  f75t_both(df, verb)
})

test_that("F75: a hand-written top_n_rank() dot and other filter() shapes run wt as in dplyr", {
  skip_if_no_gpu()
  df <- f75t_df()
  f75t_both(df, function(d, f, bump) filter(d, dplyr:::top_n_rank(2, c(1, 2, 3, 4) * (bump() - 1.5))))
  f75t_both(df, function(d, f, bump) dplyr::filter_out(d, dplyr:::top_n_rank(2, c(1, 2, 3, 4) * (bump() - 1.5))))
  f75t_both(df, function(d, f, bump) filter(d, dplyr:::top_n_rank(1, x * (bump() - 2)), .by = g))
  f75t_both(df, function(d, f, bump) filter(d, x > 1, dplyr:::top_n_rank(2, c(1, 2, 3, 4) * (bump() - 1.5))))
  f75t_both(df, function(d, f, bump) dplyr::top_n(mutate(d, z = x * 2), 2, wt = z * (bump() - 2)))
  f75t_both(df, function(d, f, bump) dplyr::top_n(dplyr::rowwise(d), 2, wt = x * (bump() - 2)))
})

# --- added after attempt-2 review round 3 --------------------------------------

test_that("F75: a top_n() wt with an empty argument (`[, 1]`, switch() fall-through) works", {
  skip_if_no_gpu()
  df <- f75t_df()
  m <- cbind(c(4, 3, 2, 1))
  pc <- list(scores = cbind(PC1 = c(10, 40, 20, 30)))
  kind <- "a"
  pipes <- list(
    function(d) dplyr::top_n(d, 2, wt = scale(x)[, 1]),
    function(d) dplyr::top_n(group_by(d, g), 1, wt = scale(y)[, 1]),
    function(d) dplyr::top_frac(d, 0.5, wt = m[, 1]),
    function(d) dplyr::top_n(d, 2, wt = pc$scores[, "PC1"]),
    function(d) dplyr::top_n(d, 2, wt = switch(kind, a = , b = y, x))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F75: a grouped top_n() with a call in a .env[[ ]] subscript runs it once per group", {
  skip_if_no_gpu()
  df <- f75t_df()
  verb <- function(d, f, bump) {
    w1 <- c(1, 2, 3, 4)
    w2 <- c(4, 3, 2, 1)
    name_fn <- function() if (bump() == 1L) "w1" else "w2"
    dplyr::top_n(group_by(d, g), 1, wt = .env[[name_fn()]][dplyr::row_number()])
  }
  f75t_both(df, verb)
})
