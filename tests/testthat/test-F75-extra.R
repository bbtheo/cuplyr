# F75: the syntactic classifier of impure column-free calls, and the shapes
# the acceptance file cannot run (it uses unqualified nth()/last(), which a
# test environment without dplyr attached does not find, in dplyr too).

f75x_schema <- list(names = c("g", "x", "y"), types = c("INT32", "FLOAT64", "INT32"))

# (dplyr's namespace as the parent: the functions a user with dplyr attached
# finds; the test environment itself has no dplyr attached)
f75x_impure <- function(expr, env = rlang::env(asNamespace("dplyr"))) {
  !is.null(impure_const_call_text(list(rlang::new_quosure(expr, env)), f75x_schema))
}

test_that("F75: the classifier allows pure shapes and flags the rest", {
  k <- 2
  lim <- list(v = 2)
  bump <- function() 1
  expect_false(f75x_impure(quote(sum(x))))
  expect_false(f75x_impure(quote(x %in% c(1, 3))))
  expect_false(f75x_impure(quote(sum(y * seq_len(1)))))
  expect_false(f75x_impure(quote(max(x - lim$v, na.rm = TRUE))))
  expect_false(f75x_impure(quote(y + .env$k)))
  expect_false(f75x_impure(quote(x > as.Date("2020-01-01"))))
  expect_false(f75x_impure(quote(base::abs(x) + dplyr::n())))
  expect_false(f75x_impure(quote(across(c(x, y), ~ .x + 1))))
  expect_false(f75x_impure(quote(across(where(is.numeric), mean))))
  env <- rlang::env(asNamespace("dplyr"), bump = bump, lim = lim, k = k)
  expect_true(f75x_impure(quote(x + bump()), env))
  expect_true(f75x_impure(quote(sum(x + runif(1))), env))
  expect_true(f75x_impure(quote(x + Sys.time()), env))
  expect_true(f75x_impure(quote({ z <- 1; x + z }), env))
  expect_true(f75x_impure(quote(sapply(1:3, identity)), env))
  expect_true(f75x_impure(quote(x + lim$fn()), env))
  expect_true(f75x_impure(quote(across(c(x, y), ~ .x + bump())), env))
  expect_true(f75x_impure(quote(across(c(x, y), list(a = function(v) v + bump()))), env))
  expect_true(f75x_impure(quote(f(n())), env))
  expect_true(f75x_impure(quote(mypkg::c(1, 2)), env))
})

test_that("F75: a masked base name is impure, pick() is not walked", {
  expect_false(f75x_impure(quote(y %in% c(1, 2))))
  env <- rlang::env(asNamespace("dplyr"), c = function(...) base::c(...))
  expect_true(f75x_impure(quote(y %in% c(1, 2)), env))
  env <- rlang::env(asNamespace("dplyr"), bump = function() 1)
  # pick() goes to dplyr before cuplyr evaluates its selection
  expect_false(f75x_impure(quote(sum(pick(all_of(bump())))), env))
  expect_false(f75x_impure(quote(if_any(c(x, y), ~ .x > 1))))
  # an embedded quosure keeps its own environment
  e <- rlang::env(asNamespace("dplyr"), bump = function() 1)
  q <- rlang::new_quosure(quote(bump()), e)
  expect_true(f75x_impure(rlang::expr(x + !!q)))
})

test_that("F75: nth()/last() with a column-free call run as dplyr runs them", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), x = c(1, 2, 3, NA, 5), y = c(10L, 20L, 30L, 40L, 50L))
  counts <- integer()
  mk <- function(verb) {
    function(d) {
      k <- 0L
      bump <- function() {
        k <<- k + 1L
        k
      }
      out <- collect(verb(d, bump))
      counts <<- c(counts, k)
      out
    }
  }
  verbs <- list(
    function(d, bump) summarise(d, v = dplyr::nth(y, bump() + 1L)),
    function(d, bump) summarise(d, s = dplyr::last(cumsum(y * bump()))),
    function(d, bump) summarise(group_by(d, g), v = dplyr::nth(y, bump()))
  )
  for (verb in verbs) {
    counts <- integer()
    expect_same_as_dplyr(df, mk(verb))
    expect_identical(counts[[2]], counts[[1]])
    counts <- integer()
    expect_same_as_dplyr_lazy(df, mk(verb))
    expect_identical(counts[[2]], counts[[1]])
  }
})

test_that("F75: quantile() with a column-free probs runs once in both", {
  skip_if_no_gpu()
  df <- tibble::tibble(y = c(10L, 20L, 30L, 40L, 50L))
  counts <- integer()
  pipeline <- function(d) {
    k <- 0L
    bump <- function() {
      k <<- k + 1L
      k
    }
    out <- collect(summarise(d, q = quantile(y, probs = bump() / 4, names = FALSE)))
    counts <<- c(counts, k)
    out
  }
  expect_same_as_dplyr(df, pipeline)
  expect_identical(counts[[2]], counts[[1]])
})

test_that("F75: count(wt =) stays native and keeps the caller's environment", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(1, 2, 3))
  kk <- 3
  withr::local_options(cuplyr.fallback = "error")
  expect_same_as_dplyr(df, function(d) count(d, g, wt = y * kk))
  expect_same_as_dplyr_lazy(df, function(d) add_count(d, g, wt = y * kk))
})

# --- round 2 -------------------------------------------------------------------

test_that("F75: the formals of a lambda stand for columns, a side effect in its body is still caught", {
  env <- rlang::env(asNamespace("dplyr"), roll_mean = roll_mean, bump = function() 1)
  expect_false(f75x_impure(quote(across(c(x, y), ~ roll_mean(.x, 2))), env))
  expect_false(f75x_impure(quote(across(c(x, y), ~ startsWith(as.character(.x), "a"))), env))
  expect_false(f75x_impure(quote(across(c(x, y), function(v) max(v) - mean(v))), env))
  expect_false(f75x_impure(quote(across(c(x, y), \(v) roll_mean(v, 2))), env))
  expect_false(f75x_impure(quote(if_any(c(x), list(a = ~ .x > 1, b = ~ .x < 0))), env))
  expect_true(f75x_impure(quote(across(c(x, y), ~ .x + bump())), env))
  expect_true(f75x_impure(quote(across(c(x, y), function(v) v + bump())), env))
  # a lambda does not make the call around it read a column
  expect_true(f75x_impure(quote(x + sapply(1:3, function(i) i + 1)), env))
  expect_true(f75x_impure(quote(sapply(1:3, function(i) { n_calls <<- n_calls + i })), env))
  expect_true(f75x_impure(quote(sapply(1:3, ~ .x + 1)), env))
  # a lambda over a real column
  expect_false(f75x_impure(quote(sapply(x, function(i) i + 1)), env))
})

test_that("F75: the allowlist holds the pre-registry natives and common deterministic base functions", {
  env <- rlang::env(asNamespace("dplyr"), roll_mean = roll_mean)
  for (e in list(
    quote(startsWith("abc", "a")), quote(endsWith("abc", "c")), quote(roll_mean(1:5, 2)),
    quote(head(c(1, 2, 3), 2)), quote(tail(c(1, 2, 3), 1)), quote(format(2.5)),
    quote(match("b", c("a", "b"))), quote(unlist(list(1, 2))), quote(as.vector(1:3)),
    quote(getOption("digits")), quote(switch("a", a = 1, 2)), quote(suppressWarnings(as.integer("a")))
  )) {
    expect_false(f75x_impure(e, env), info = deparse(e))
  }
})

test_that("F75: the canonical lookup does not load a namespace or cache its absence", {
  skip_if(isNamespaceLoaded("lubridate"), "lubridate is already loaded in this session")
  fns <- ir_canonical_fns("ymd")
  expect_false(isNamespaceLoaded("lubridate"))
  expect_true("lubridate" %in% ir_canonical_memo[["ymd"]]$unloaded)
  expect_length(fns, 0L)
})

test_that("F75: the canonical memo is redone once a namespace it skipped is loaded", {
  # a memo taken while `stats` was "not loaded" holds no stats function; stats
  # is loaded now, so the lookup is redone and finds stats::sd
  withr::defer(rm(list = "sd", envir = ir_canonical_memo))
  assign("sd", list(fns = list(), unloaded = "stats"), envir = ir_canonical_memo)
  fns <- ir_canonical_fns("sd")
  expect_true(any(vapply(fns, identical, logical(1), stats::sd)))
  expect_false("stats" %in% ir_canonical_memo[["sd"]]$unloaded)
  # and a memo whose skipped namespaces are all still unloaded is kept
  assign("sd", list(fns = list("kept"), unloaded = "f75.no.such.namespace"), envir = ir_canonical_memo)
  expect_identical(ir_canonical_fns("sd"), list("kept"))
})

test_that("F75: lambdas over native functions stay native (every shape, eager and lazy)", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L, 2L), x = c(1, 2, 3, 4, 5, 6), y = c(6, 5, 4, 3, 2, 1),
                       s = c("ab", "ba", "ab", "bb", "aa", "ab"))
  withr::local_options(cuplyr.fallback = "error")
  pipes <- list(
    function(d) mutate(group_by(d, g), across(c(x, y), \(v) roll_sum(v, 2))),
    function(d) mutate(group_by(d, g), across(c(x, y), list(a = ~ .x * 2, b = ~ cumsum(.x)))),
    function(d) filter(group_by(d, g), if_any(c(s), ~ endsWith(.x, "b"))),
    function(d) transmute(group_by(d, g), across(c(x), ~ abs(.x - 3) * 2)),
    function(d) summarise(group_by(d, g), across(c(x, y), ~ max(roll_mean(.x, 2), na.rm = TRUE)))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F75: the fallback notice names the offending call", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), x = c(1, 2, 3))
  bump <- function() 1
  withr::local_options(cuplyr.fallback = "error")
  gd <- group_by(tbl_gpu(df), g)
  expect_error(summarise(gd, s = sum(x + bump())), "bump\\(\\)", class = "cuplyr_error_fallback")
  expect_error(mutate(gd, y = x + bump()), "bump\\(\\)", class = "cuplyr_error_fallback")
  expect_error(filter(gd, x > bump()), "bump\\(\\)", class = "cuplyr_error_fallback")
  expect_error(transmute(gd, y = x + bump()), "bump\\(\\)", class = "cuplyr_error_fallback")
})

test_that("F75: nth()'s n and na.rm warn once on a grouped table", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(10, 20, 30))
  for (lazy in c(FALSE, TRUE)) {
    t <- group_by(if (lazy) as_lazy(tbl_gpu(df)) else tbl_gpu(df), g)
    expect_length(testthat::capture_warnings(collect(
      summarise(t, v = dplyr::nth(y, length(as.integer(c("1", "a")))))
    )), 1L)
    expect_length(testthat::capture_warnings(collect(
      summarise(t, s = sum(y, na.rm = is.na(as.integer("a"))))
    )), 1L)
  }
})

test_that("F75: count()/tally() wt = <name> is the caller's variable unless it is a column", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L), y = c(1, 2, 3))
  f75_w_count <- function(d) {
    w <- c(10, 20, 30)
    count(d, g, wt = w)
  }
  f75_w_tally <- function(d) {
    w <- c(10, 20, 30)
    tally(group_by(d, g), wt = w)
  }
  expect_same_as_dplyr(df, f75_w_count)
  expect_same_as_dplyr(df, f75_w_tally)
  # a column of that name wins over the caller's variable, as in dplyr
  withr::local_options(cuplyr.fallback = "error")
  y <- c(100, 200, 300)
  expect_same_as_dplyr(df, function(d) count(d, g, wt = y))
  # a column made by count()'s own `...`
  expect_same_as_dplyr(df, function(d) count(d, z = y * 2, wt = z))
})

# --- round 3 -------------------------------------------------------------------

# A lambda's formals stand for a column only where cuplyr's across() expansion
# replaces them: the `.fns` of a dot-level across()/if_any()/if_all() that the
# expansion inlines. Anywhere else (a two-sided formula, a function literal
# passed to with_order(), a nested across()) they are ordinary variables, and
# a call over them is column-free.
test_that("F75: formals are bound only in an inlined .fns position", {
  env <- rlang::env(asNamespace("dplyr"), bump = function(v) v, .x = 10, roll_mean = roll_mean)
  # case_when()'s `cond ~ value`: `.x` is the caller's variable
  expect_true(f75x_impure(quote(case_when(x > 0 ~ bump(.x))), env))
  expect_true(f75x_impure(quote(sum(case_when(x > 0 ~ x * bump(.x)))), env))
  expect_true(f75x_impure(quote(case_when(x > 0 ~ runif(1, max = nrow(.)))), env))
  # inside an across() lambda a two-sided formula inherits the lambda's formals
  expect_false(f75x_impure(quote(across(c(x, y), ~ case_when(.x > 1 ~ bump(.x), .default = .x))), env))
  # a function literal that is not a .fns
  expect_true(f75x_impure(quote(x + with_order(c(2, 1), function(v) bump(v), c(1, 2))), env))
  expect_true(f75x_impure(quote(x + with_order(c(2, 1), \(v) bump(v), c(1, 2))), env))
  # a nested across() is not expanded (dplyr runs the call)
  expect_true(f75x_impure(quote(sum(across(x, ~ bump(.x)))), env))
  # a lambda the expansion does not inline: `.y`/`..1`, two formals, extra args
  expect_true(f75x_impure(quote(across(x, ~ bump(.y))), env))
  expect_true(f75x_impure(quote(across(x, ~ bump(..1))), env))
  expect_true(f75x_impure(quote(across(x, function(v, w) bump(v))), env))
  expect_true(f75x_impure(quote(across(x, function(v) bump(v), 2)), env))
  # every inlined shape: named .fns, a list, `dplyr::`, `{{ fn }}`
  expect_false(f75x_impure(quote(across(.fns = ~ bump(.x), .cols = c(x, y))), env))
  expect_false(f75x_impure(quote(across(x, list(a = ~ bump(.x), b = \(v) bump(v)))), env))
  expect_false(f75x_impure(quote(dplyr::if_all(c(x, y), ~ bump(.) > 1)), env))
  q <- rlang::new_quosure(quote(~ bump(.x)), env)
  expect_false(f75x_impure(rlang::expr(across(x, !!q)), env))
  expect_true(f75x_impure(rlang::expr(across(x, list(!!q, ~ .x + bump(1)))), env))
})

test_that("F75: a function literal passed to with_order() runs once per group", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4))
  counts <- integer()
  mk <- function(verb) {
    function(d) {
      k <- 0
      bump <- function(v) {
        k <<- k + 1
        k * v
      }
      out <- collect(verb(d, bump))
      counts <<- c(counts, k)
      out
    }
  }
  verbs <- list(
    function(d, bump) mutate(group_by(d, g), y = x + dplyr::with_order(c(2, 1), function(v) bump(v), c(1, 2))[1]),
    function(d, bump) filter(group_by(d, g), x > dplyr::with_order(c(2, 1), \(v) bump(v), c(1, 2))[1]),
    function(d, bump) summarise(d, s = sum(x) + dplyr::with_order(c(2, 1), function(v) bump(v), c(1, 2))[1],
                                r = cor(x, x))
  )
  for (v in verbs) {
    for (lazy in c(FALSE, TRUE)) {
      counts <- integer()
      if (lazy) expect_same_as_dplyr_lazy(df, mk(v)) else expect_same_as_dplyr(df, mk(v))
      expect_identical(counts[[2]], counts[[1]], info = sprintf("bump() calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
    }
  }
})

test_that("F75: a two-sided formula inside an across() lambda stays native", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), y = c(4, 3, 2, 1))
  withr::local_options(cuplyr.fallback = "error")
  p <- function(d) {
    mutate(group_by(d, g), across(c(x, y), ~ dplyr::case_when(.x > 2 ~ roll_sum(.x, 1), .default = -.x)))
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

# The across() expansion evaluates `.cols` (tidyselect) before the parse, and a
# CPU exit after it hands dplyr the original call, which evaluates it again; so
# a column-free call in `.cols` is checked like any other. The tidyselect
# helpers are pure.
test_that("F75: .cols is walked, tidyselect helpers are pure", {
  env <- rlang::env(asNamespace("dplyr"), cols_fn = function() "x", vars = c("x", "y"))
  expect_true(f75x_impure(quote(across(all_of(cols_fn()), mean)), env))
  expect_true(f75x_impure(quote(if_any(any_of(cols_fn()), ~ .x > 1)), env))
  for (e in list(
    quote(across(all_of(vars), mean)), quote(across(any_of(c("x", "z")), mean)),
    quote(across(where(is.numeric), mean)), quote(across(where(~ is.numeric(.x)), mean)),
    quote(across(starts_with("x") | ends_with("y"), mean)), quote(across(contains("x"), mean)),
    quote(across(matches("^x"), mean)), quote(across(everything(), mean)),
    quote(across(last_col(), mean)), quote(across(num_range("x", 1:2), mean)),
    quote(across(!c(g), mean)), quote(across(-g, mean)), quote(across(x:y, mean)),
    quote(across(c("x", "y"), mean)), quote(across(paste0("x", ""), mean)),
    quote(if_all(tidyselect::all_of(vars), ~ .x > 0))
  )) {
    expect_false(f75x_impure(e, env), info = deparse(e))
  }
  # without dplyr or tidyselect attached, tidyselect still finds its helpers
  expect_false(f75x_impure(quote(across(starts_with("x"), mean)), rlang::env(baseenv())))
})

test_that("F75: a call in .cols runs once when the across() falls back", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4), y = c(4, 3, 2, 1))
  counts <- integer()
  mk <- function(verb) {
    function(d) {
      k <- 0
      cols_fn <- function() {
        k <<- k + 1
        c("x", "y")
      }
      myf <- function(v) v * 10
      out <- collect(verb(d, cols_fn, myf))
      counts <<- c(counts, k)
      out
    }
  }
  verbs <- list(
    function(d, cols_fn, myf) mutate(group_by(d, g), across(all_of(cols_fn()), ~ myf(.x))),
    function(d, cols_fn, myf) summarise(group_by(d, g), across(all_of(cols_fn()), ~ sum(myf(.x)))),
    function(d, cols_fn, myf) summarise(d, across(all_of(cols_fn()), ~ sum(myf(.x)))),
    function(d, cols_fn, myf) filter(group_by(d, g), if_any(all_of(cols_fn()), ~ myf(.x) > 30))
  )
  for (v in verbs) {
    for (lazy in c(FALSE, TRUE)) {
      counts <- integer()
      if (lazy) expect_same_as_dplyr_lazy(df, mk(v)) else expect_same_as_dplyr(df, mk(v))
      expect_identical(counts[[2]], counts[[1]], info = sprintf("cols_fn() calls: dplyr %d, GPU %d", counts[[1]], counts[[2]]))
    }
  }
})

test_that("F75: tidyselect helpers in .cols stay native", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), x1 = c(1, 2, 3, 4), x2 = c(4, 3, 2, 1), s = c("a", "b", "a", "b"))
  withr::local_options(cuplyr.fallback = "error")
  vars <- c("x1", "x2")
  pipes <- list(
    function(d) summarise(group_by(d, g), across(where(is.numeric), ~ sum(.x))),
    function(d) mutate(group_by(d, g), across(starts_with("x"), ~ .x * 2)),
    function(d) summarise(d, across(all_of(vars), ~ max(.x))),
    function(d) filter(group_by(d, g), if_all(num_range("x", 1:2), ~ .x > 1))
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

# --- round 4 -------------------------------------------------------------------

# Everything the native `.cols` resolver (selection_is_type_only()) accepts is
# pure, or the `.cols` walk would send a native selection to the CPU.
test_that("F75: the allowlist covers the native .cols resolver", {
  expect_true(all(c(base_type_test_names, "inherits", name_builder_fns) %in% ir_pure_names()))
  lk <- data.frame(x = 1)
  env <- rlang::env(asNamespace("dplyr"), lk = lk)
  expect_false(f75x_impure(quote(across(where(~ is.double(.x)), sum)), env))
  expect_false(f75x_impure(quote(across(where(\(v) inherits(v, "Date")), ~ .x + 1)), env))
  expect_false(f75x_impure(quote(across(where(~ !is.factor(.x) && !is.ordered(.x)), ~ .x * 2)), env))
  expect_false(f75x_impure(quote(if_any(where(~ is.integer(.x)), ~ .x > 2)), env))
  expect_false(f75x_impure(quote(across(any_of(colnames(lk)), sum)), env))
  expect_false(f75x_impure(quote(across(all_of(grep("^x", names(lk), value = TRUE)), sum)), env))
  expect_false(f75x_impure(quote(across(all_of(head(sort(union(names(lk), "y")), 2)), sum)), env))
  # a user function named like a type test is not the base one
  env2 <- rlang::env(env, is.double = function(x) TRUE)
  expect_true(f75x_impure(quote(across(where(~ is.double(.x)), sum)), env2))
})

# with_order() calls its `fun`: a closure passed by name is a user call the
# walk cannot see, so with_order() itself is not pure (it stays a pre-registry
# head for the parser). order_by() takes a call, which is walked.
test_that("F75: with_order() is not pure, order_by() over a pure call is", {
  env <- rlang::env(asNamespace("dplyr"), bumpv = function(v) v)
  expect_false("with_order" %in% ir_pure_names())
  expect_true("with_order" %in% ir_pre_registry_fns)
  expect_true(f75x_impure(quote(x + with_order(c(2, 1), bumpv, c(1, 2))[1]), env))
  expect_true(f75x_impure(quote(x + dplyr::with_order(c(2, 1), cumsum, c(1, 2))[1]), env))
  expect_false(f75x_impure(quote(with_order(x, cumsum, y)), env))
  expect_false(f75x_impure(quote(x + order_by(c(2, 1), cumsum(c(1, 2)))[1]), env))
  expect_true(f75x_impure(quote(x + order_by(c(2, 1), bumpv(c(1, 2)))[1]), env))
})

# A call over pick()/c_across() goes to dplyr whole through the deferred
# fallback; the classifier does not report it as an impure call.
test_that("F75: a call over pick() or c_across() is not reported as impure", {
  env <- rlang::env(asNamespace("dplyr"), bumpv = function(v) v)
  expect_false(f75x_impure(quote(rowSums(pick(g, x))), env))
  expect_false(f75x_impure(quote(bumpv(c_across(c(x, y)))), env))
  # a column-free impure call next to one still is
  expect_true(f75x_impure(quote(rowSums(pick(g, x)) + bumpv(1)), env))
})
