# F75 (attempt 2, round 3): a call in a `.env` subscript of a top_n() wt.
# The protected file pins the plain `.env[[name_fn()]]` shape; these cover the
# same subscript behind `{{ }}`, the notice text, and the pronoun shapes that
# must stay on the GPU.

f75s_df <- function() {
  tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), y = c(4, 1, 2, 3))
}

# A wt that reads `.env[[name_fn()]]`, passed through `{{ }}` (embed) or
# nested in a call under top_frac(); name_fn() gives "w1" on its first call,
# "w2" after.
f75s_run <- function(d, embed) {
  k <- 0L
  w1 <- c(1, 2, 3, 4)
  w2 <- c(4, 3, 2, 1)
  name_fn <- function() {
    k <<- k + 1L
    if (k == 1L) "w1" else "w2"
  }
  out <- if (embed) {
    wrap <- function(d, wt) dplyr::top_n(d, 2, wt = {{ wt }})
    wrap(d, .env[[name_fn()]])
  } else {
    dplyr::top_frac(d, 0.5, wt = -.env[[name_fn()]])
  }
  list(calls = k, x = collect(out)$x)
}

test_that("F75: a call in a .env[[ ]] subscript behind {{ }} or under a call in wt runs once", {
  skip_if_no_gpu()
  df <- f75s_df()
  for (embed in c(TRUE, FALSE)) {
    want <- f75s_run(df, embed)
    expect_identical(f75s_run(tbl_gpu(df), embed), want)
    expect_identical(f75s_run(as_lazy(tbl_gpu(df)), embed), want)
  }
})

test_that("F75: the fallback notice names the .env subscript and keeps the deferred wording", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f75s_df()
  w1 <- c(1, 2, 3, 4)
  name_fn <- function() "w1"
  expect_error(
    collect(dplyr::top_n(tbl_gpu(df), 2, wt = .env[[name_fn()]])),
    "`.env[[name_fn()]]`, whose subscript dplyr evaluates inside the data mask",
    fixed = TRUE
  )
  expect_error(
    collect(dplyr::top_n(tbl_gpu(df), 2, wt = pick(x)[[1]])),
    "which dplyr evaluates inside the data mask",
    fixed = TRUE
  )
})

test_that("F75: a pronoun with a string or name subscript keeps top_n() on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- f75s_df()
  col <- "x"
  kk <- 2
  pipes <- list(
    function(d) dplyr::top_n(d, 2, wt = .data[[col]]),
    function(d) dplyr::top_n(d, 2, wt = .data[["x"]] * 2),
    function(d) dplyr::top_n(d, 2, x),
    function(d) dplyr::top_n(d, 2, wt = x * .env[["kk"]]),
    function(d) dplyr::top_n(d, 2, wt = x * .env$kk)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

# F75 (attempt 2, round 4): an empty argument (`m[, 1]`, `switch(k, a = , x)`)
# in a top_n() wt, plain and behind `{{ }}`. The walkers over `wt` must skip
# R's missing-argument marker rather than force it.

test_that("F75: the top_n() wt walkers skip an empty argument", {
  e <- quote(scale(x)[, 1])
  expect_null(top_n_env_call_subscript(e))
  expect_null(top_n_env_call_subscript(quote(switch(kind, a = , b = y, x))))
  expect_null(top_n_env_call_subscript(rlang::quo(m[, 1])))
  expect_identical(
    top_n_env_call_subscript(quote(.env[[f()]][, 1])),
    quote(.env[[f()]])
  )
  expect_null(top_n_env_call_subscript(quote(.env[[]])))
  expect_null(top_n_env_call_subscript(quote(x * .env[[]][, 1])))
  expect_false(top_n_has_quosure(e))
  schema <- list(names = c("g", "x", "y"), types = c("INT32", "FLOAT64", "FLOAT64"))
  expect_null(top_n_wt_decline(list(expr = quote(m[, 1]), env = globalenv()), schema))
})

test_that("F75: a top_n() wt with an empty argument behind {{ }} works", {
  skip_if_no_gpu()
  df <- f75s_df()
  m <- cbind(c(4, 3, 2, 1))
  kind <- "a"
  w <- function(d, v) dplyr::top_n(d, 2, wt = {{ v }})
  wg <- function(d, v) dplyr::top_n(group_by(d, g), 1, wt = {{ v }})
  wsub <- function(d, v) dplyr::top_frac(d, 0.5, wt = {{ v }}[, 1])
  pipes <- list(
    function(d) w(d, scale(x)[, 1]),
    function(d) wg(d, scale(y)[, 1]),
    function(d) w(d, m[, 1]),
    function(d) w(d, switch(kind, a = , b = y, x)),
    function(d) w(d, .env$m[, 1]),
    function(d) w(d, .env[["m"]][, 1]),
    function(d) wsub(d, m),
    function(d) wsub(d, scale(x))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F75: a .env[[ ]] call subscript next to an empty argument still runs once", {
  skip_if_no_gpu()
  df <- f75s_df()
  run <- function(d, embed) {
    k <- 0L
    w1 <- cbind(c(1, 2, 3, 4))
    w2 <- cbind(c(4, 3, 2, 1))
    name_fn <- function() {
      k <<- k + 1L
      if (k == 1L) "w1" else "w2"
    }
    out <- if (embed) {
      wrap <- function(d, v) dplyr::top_n(d, 2, wt = {{ v }})
      wrap(d, .env[[name_fn()]][, 1])
    } else {
      dplyr::top_n(d, 2, wt = .env[[name_fn()]][, 1])
    }
    list(calls = k, x = collect(out)$x)
  }
  for (embed in c(TRUE, FALSE)) {
    want <- run(df, embed)
    expect_identical(run(tbl_gpu(df), embed), want)
    expect_identical(run(as_lazy(tbl_gpu(df)), embed), want)
  }
})

test_that("F75: an empty .env[[ ]] subscript in wt raises dplyr's own error", {
  skip_if_no_gpu()
  df <- f75s_df()
  expect_error(dplyr::top_n(df, 2, wt = .env[[]]), "argument \"i\" is missing")
  suppressWarnings({
    expect_error(collect(dplyr::top_n(tbl_gpu(df), 2, wt = .env[[]])), "argument \"i\" is missing")
    expect_error(collect(dplyr::top_n(as_lazy(tbl_gpu(df)), 2, wt = .env[[]])), "argument \"i\" is missing")
  })
})
