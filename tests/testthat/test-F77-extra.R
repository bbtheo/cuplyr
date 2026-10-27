# F77: the screen itself (shadow_screen_text()), beyond the acceptance tests

f77x_text <- function(expr_text, env = parent.frame()) {
  shadow_screen_text(list(rlang::new_quosure(rlang::parse_expr(expr_text), env)))
}

test_that("F77: the screen names the dot and the shadowed head", {
  skip_if_no_gpu()
  abs <- function(x) -x
  expect_match(f77x_text("y + abs(x)"), "`abs`")
  expect_match(f77x_text("y + abs(x)"), "y \\+ abs\\(x\\)", fixed = FALSE)
  expect_null(f77x_text("y + base::abs(x)"))
  expect_null(f77x_text("y + sqrt(x)"))
  expect_null(f77x_text("x"))
})

test_that("F77: a call head found through a formula, lambda or embedded quosure is seen", {
  skip_if_no_gpu()
  abs <- function(x) -x
  expect_match(f77x_text("case_when(x > 0 ~ abs(x), .default = 0)"), "abs")
  expect_match(f77x_text("across(c(x, y), function(v) abs(v))"), "abs")
  expect_match(f77x_text("across(c(x, y), abs)"), "abs")
  expect_match(f77x_text("across(c(x, y), list(a = abs, b = sqrt))"), "abs")
  # an embedded quosure has its own env: the shadow lives only in the other one
  inner_env <- local({
    sqrt <- function(x) x
    environment()
  })
  q <- rlang::new_quosure(quote(sqrt(x)), inner_env)
  outer <- rlang::new_quosure(rlang::expr(1 + !!q), globalenv())
  expect_match(shadow_screen_text(list(outer)), "sqrt")
  expect_null(shadow_screen_text(list(rlang::new_quosure(quote(1 + sqrt(x)), globalenv()))))
})

test_that("F77: names that are not checked, and non-function bindings, are not shadows", {
  skip_if_no_gpu()
  myfun <- function(x) x
  abs <- 3
  expect_null(f77x_text("myfun(x) + abs"))
  expect_null(f77x_text("abs(x)"))
  # a pronoun call is not a head
  mean <- function(x) 1
  expect_null(f77x_text(".data$mean + .env$mean"))
  expect_match(f77x_text(".data[[paste0('a', 'b')]] + mean(x)"), "mean")
})

test_that("F77: the data.table and stats homonyms are not the native op", {
  skip_if_no_gpu()
  skip_if_not_installed("data.table")
  expect_match(f77x_text("data.table::first(x)"), "data.table::first")
  expect_match(f77x_text("stats::lag(x)"), "stats::lag")
  expect_null(f77x_text("dplyr::first(x)"))
  expect_null(f77x_text("dplyr::lag(x)"))
})

test_that("F77: the shadow falls back with a message naming it", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  abs <- function(x) -x
  df <- tbl_gpu(tibble::tibble(x = c(1, -2)))
  expect_error(mutate(df, y = abs(x)), "abs")
  expect_error(filter(df, abs(x) > 1), "abs")
  expect_error(arrange(df, abs(x)), "abs")
  expect_error(summarise(df, m = sum(abs(x))), "abs")
  # a symbol head of the same name that is the native function is fine
  rm(abs)
  expect_no_error(mutate(df, y = abs(x)))
})

# --- round 2 -------------------------------------------------------------------

test_that("F77: the screen forces no promise bound to a column or variable name", {
  skip_if_no_gpu()
  f <- function(x = stop("forced x"), y = stop("forced y")) f77x_text("x + y + abs(1)")
  expect_null(f())
  g <- function(x = stop("forced x")) f77x_text("with_order(o, cumsum, x)")
  expect_null(g())
  # a head's own name is looked up, as R's call lookup does
  abs <- function(x) -x
  expect_match(f77x_text("abs(x) + y"), "abs")
})

test_that("F77: qualified across()/if_any() and an embedded quosure .fns keep their env", {
  skip_if_no_gpu()
  abs <- function(x) -x
  expect_match(f77x_text("dplyr::across(c(x, y), abs)"), "abs")
  expect_match(f77x_text("dplyr::if_any(c(x, y), abs)"), "abs")
  expect_match(f77x_text("dplyr::if_all(c(x, y), list(a = abs))"), "abs")
  inner_env <- local({
    sqrt <- function(x) x
    environment()
  })
  q <- rlang::new_quosure(quote(sqrt), inner_env)
  dot <- rlang::new_quosure(rlang::expr(across(c(x, y), list(a = !!q, b = base::abs))), globalenv())
  # sqrt is shadowed in the quosure's env only
  expect_match(shadow_screen_text(list(dot)), "sqrt")
  dot2 <- rlang::new_quosure(rlang::expr(across(c(x, y), list(a = sqrt, b = base::abs))), globalenv())
  expect_null(shadow_screen_text(list(dot2)))
})

test_that("F77: with_order()'s fun is checked, qualified or not", {
  skip_if_no_gpu()
  cumsum <- function(x) x
  expect_match(f77x_text("with_order(o, cumsum, x)"), "cumsum")
  expect_match(f77x_text("dplyr::with_order(o, cumsum, x)"), "cumsum")
  expect_null(f77x_text("dplyr::with_order(o, base::cumsum, x)"))
  expect_null(f77x_text("dplyr::with_order(o, dplyr::lag, x)"))
  expect_match(f77x_text("dplyr::with_order(o, stats::lag, x)"), "stats::lag")
})

test_that("F77: a pkg::fn that is not exported is a shadow; pkg:::fn is the namespace's", {
  skip_if_no_gpu()
  expect_match(f77x_text("dplyr::abs(x)"), "dplyr::abs")
  expect_match(f77x_text("nonexistentpkg::abs(x)"), "nonexistentpkg::abs")
  expect_match(f77x_text("dplyr:::abs(x)"), "dplyr:::abs")
  expect_null(f77x_text("base:::abs(x)"))
  expect_null(f77x_text("dplyr:::n()"))
  expect_null(f77x_text("dplyr::n()"))
  expect_null(f77x_text("base::sum(x, na.rm = TRUE)"))
})

test_that("F77: a formula value with a shadow in its own env goes to dplyr, eager and lazy", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(-1.5, 2, 4), o = c(2L, 1L, 3L))
  make <- function() {
    cumsum <- function(x) x * 100
    ord <- c(2L, 1L, 3L)
    ~ dplyr::with_order(ord, cumsum, .x)
  }
  fml <- make()
  expect_same_as_dplyr(df, function(d) mutate(d, across(x, fml)))
  expect_same_as_dplyr_lazy(df, function(d) mutate(d, across(x, fml)))
})

test_that("F77: count() and tally() take no deparse for their n()", {
  skip_if_no_gpu()
  expect_true(summarise_is_bare_n(rlang::new_quosure(quote(dplyr::n()), globalenv())))
  expect_true(summarise_is_bare_n(rlang::new_quosure(quote(n()), globalenv())))
  expect_false(summarise_is_bare_n(rlang::new_quosure(quote(dplyr::sum()), globalenv())))
  expect_false(summarise_is_bare_n(rlang::new_quosure(quote(dplyr::n(x)), globalenv())))
})

# --- round 3: function values seen through a namespace, scoped verbs ---------

test_that("F77: a registry name's function value is canonical only by F77's table", {
  skip_if_no_gpu()
  # A namespace's lookup chain ends in the global env: a global `median` must
  # not pass for stats::median because dplyr's namespace does not define one.
  ge <- globalenv()
  assign("median", function(x, ...) -1, envir = ge)
  withr::defer(rm("median", envir = ge))
  user_median <- get("median", envir = ge)
  expect_null(across_known_fn_name(user_median, ge))
  expect_null(across_known_fn_name(user_median, ge, quote(median)))
  expect_identical(across_known_fn_name(stats::median, ge, quote(stats::median)), "median")
  # a canonical value with no name: found only where its name resolves to it
  expect_null(across_known_fn_name(stats::median, ge))
  expect_identical(across_known_fn_name(stats::sd, ge), "sd")
  expect_identical(across_known_fn_name(base::mean, ge), "mean")
  # the homonym is not the native op, qualified or not
  expect_null(across_known_fn_name(stats::lag, ge, quote(stats::lag)))
  expect_identical(across_known_fn_name(dplyr::lag, ge, quote(dplyr::lag)), "lag")
})

test_that("F77: the scoped verbs' rewrite gives up on a shadowed lambda body", {
  skip_if_no_gpu()
  abs <- function(x) -x * 10
  schema <- list(names = c("x", "y"), types = c("FLOAT64", "FLOAT64"))
  fml_fn <- structure(function(...) NULL, class = c("inline_colwise_function", "function"),
                      formula = ~ abs(.))
  dot <- rlang::new_quosure(rlang::call2(fml_fn, quote(x)), environment())
  out <- colwise_rewrite_dots(list(x = dot), schema)
  expect_true(is.function(rlang::quo_get_expr(out$x)[[1L]]))
  lam <- rlang::as_function(~ abs(.x) > 10)
  dot2 <- rlang::new_quosure(rlang::call2(lam, quote(x)), environment())
  out2 <- colwise_rewrite_dots(list(dot2), schema)
  expect_true(is.function(rlang::quo_get_expr(out2[[1L]])[[1L]]))
  # without the shadow both are rewritten to plain calls
  rm(abs)
  out3 <- colwise_rewrite_dots(list(x = dot), schema)
  expect_true(is.symbol(rlang::quo_get_expr(out3$x)[[1L]]))
})

test_that("F77: scoped verbs with a shadow match dplyr in every verb, eager and lazy", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(-1.5, 2, 4), y = c(3, -2, 5))
  abs <- function(x) -x * 10
  mean <- function(x, ...) -1
  pipes <- list(
    function(d) dplyr::transmute_at(d, "x", ~ abs(.)),
    function(d) dplyr::mutate_at(dplyr::group_by(d, g), "x", ~ abs(.)),
    function(d) dplyr::filter_at(dplyr::group_by(d, g), "x", ~ abs(.x) > 10),
    function(d) dplyr::filter_if(d, is.double, dplyr::all_vars(abs(.) < 100)),
    function(d) dplyr::arrange_at(d, "x", ~ abs(.)),
    function(d) dplyr::arrange_at(dplyr::group_by(d, g), "x", ~ abs(.), .by_group = TRUE),
    function(d) dplyr::group_by_at(d, "x", ~ abs(.)),
    function(d) dplyr::summarise_at(d, "x", list(a = ~ mean(.), b = ~ sum(.))),
    function(d) dplyr::mutate_at(d, dplyr::vars(x, y), function(v) abs(v))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F77: a global function value named like a registry op is the user's (summarise_at/mutate_at/across)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, NA, 10, 2))
  ge <- globalenv()
  assign("var", function(x, ...) 42, envir = ge)
  assign("f77x_fns", list(v = get("var", envir = ge), m = base::mean), envir = ge)
  withr::defer(rm(list = c("var", "f77x_fns"), envir = ge))
  pipes <- list(
    function(d) dplyr::mutate(dplyr::group_by(d, g), dplyr::across(x, f77x_fns)),
    function(d) dplyr::mutate_at(dplyr::group_by(d, g), "x", var),
    function(d) dplyr::summarise(d, dplyr::across(x, !!var))
  )
  for (p in pipes) {
    environment(p) <- ge
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F77: the screen looks up no name inside a call-valued head", {
  skip_if_no_gpu()
  obj <- list(fn = function(v) v)
  # dplyr evaluates `obj$fn` (not a variable `fn`) and calls the literal
  # (never looking `v` up from the env): neither promise is forced
  screen <- function(fn = stop("forced fn"), v = stop("forced v")) {
    c(
      shadow_screen_text(list(rlang::new_quosure(quote(obj$fn(x) + 1), environment()))),
      shadow_screen_text(list(rlang::new_quosure(quote((function(v) v * 2)(x)), environment())))
    )
  }
  expect_null(screen())
  # a real head with a checked name is still looked up
  abs <- function(x) -x
  expect_match(f77x_text("obj$fn(abs(x))"), "abs")
})
