# F60: colwise_rewrite_dots() resolves each distinct function once per call
# (colwise_fn_plan()). These pin what the memo must keep apart: closures that
# differ only in their environment, extra args, and the column-dependent
# checks (desc() on a string, a column landing in a call head).

f60m_schema <- function(names, types) list(names = names, types = types)

f60m_dot <- function(fn, col, ..., env = globalenv()) {
  rlang::new_quosure(as.call(c(list(fn, as.name(col)), list(...))), env)
}

f60m_inline <- function(fml) {
  structure(rlang::as_function(fml), class = "inline_colwise_function", formula = fml)
}

test_that("closures with the same body but different environments keep their own", {
  mk <- function(k) function(x) x * k
  f1 <- mk(2)
  f2 <- mk(3)
  schema <- f60m_schema(c("v1", "v2"), c("FLOAT64", "FLOAT64"))
  dots <- colwise_rewrite_dots(
    list(a = f60m_dot(f1, "v1"), b = f60m_dot(f2, "v1"), c = f60m_dot(f1, "v2")),
    schema
  )
  for (q in dots) expect_false(colwise_has_fn_head(rlang::quo_get_expr(q)))
  data <- list(v1 = 10, v2 = 100)
  expect_equal(rlang::eval_tidy(dots$a, data), 20)
  expect_equal(rlang::eval_tidy(dots$b, data), 30)
  expect_equal(rlang::eval_tidy(dots$c, data), 200)
})

test_that("the same function with different extra args is resolved per args", {
  schema <- f60m_schema(c("v1", "v2"), c("FLOAT64", "FLOAT64"))
  dots <- colwise_rewrite_dots(
    list(a = f60m_dot(round, "v1", 1), b = f60m_dot(round, "v2", digits = 2), c = f60m_dot(round, "v1", 1)),
    schema
  )
  expect_identical(rlang::quo_get_expr(dots$a), quote(round(v1, 1)))
  expect_identical(rlang::quo_get_expr(dots$b), quote(round(v2, digits = 2)))
  expect_identical(rlang::quo_get_expr(dots$c), quote(round(v1, 1)))
})

test_that("desc() is still checked per column within one call", {
  schema <- f60m_schema(c("s", "x", "t"), c("STRING", "FLOAT64", "STRING"))
  env <- new.env(parent = globalenv())
  env$desc <- dplyr::desc
  orig <- lapply(c("x", "s", "x", "t"), function(col) f60m_dot(dplyr::desc, col, env = env))
  dots <- colwise_rewrite_dots(orig, schema)
  expect_identical(rlang::quo_get_expr(dots[[1]]), quote(desc(x)))
  expect_identical(dots[[2]], orig[[2]])
  expect_identical(rlang::quo_get_expr(dots[[3]]), quote(desc(x)))
  expect_identical(dots[[4]], orig[[4]])
})

test_that("a column substituted into a call head is checked per column", {
  # `~ .x(1)` is only native when the column's name is one the IR knows
  fn <- f60m_inline(~ .x(1))
  schema <- f60m_schema(c("zz", "mean"), c("FLOAT64", "FLOAT64"))
  for (cols in list(c("zz", "mean"), c("mean", "zz"))) {
    orig <- list(f60m_dot(fn, cols[1]), f60m_dot(fn, cols[2]))
    dots <- colwise_rewrite_dots(orig, schema)
    is_mean <- cols == "mean"
    expect_identical(rlang::quo_get_expr(dots[[which(is_mean)]]), quote(mean(1)))
    expect_identical(dots[[which(!is_mean)]], orig[[which(!is_mean)]])
  }
})

test_that("a function that can't be rewritten leaves every dot untouched", {
  f <- function(x) x + nchar(cur_column())
  schema <- f60m_schema(c("v1", "v2"), c("FLOAT64", "FLOAT64"))
  orig <- list(a = f60m_dot(f, "v1"), b = f60m_dot(f, "v2"), c = f60m_dot(abs, "v2"))
  dots <- colwise_rewrite_dots(orig, schema)
  expect_identical(dots$a, orig$a)
  expect_identical(dots$b, orig$b)
  expect_identical(rlang::quo_get_expr(dots$c), quote(abs(v2)))
})

test_that("the registry name cache follows a mocked registry", {
  expect_false("f60_bogus" %in% across_known_fn_names())
  local({
    testthat::local_mocked_bindings(
      ir_call_registry = local({
        r <- ir_call_registry
        r[["f60_bogus"]] <- list(arity = 1L, parse = NULL, type = type_arg1, lower = NULL)
        r
      }),
      .package = "cuplyr"
    )
    expect_true("f60_bogus" %in% across_known_fn_names())
    expect_true("f60_bogus" %in% across_native_heads())
  })
  expect_false("f60_bogus" %in% across_known_fn_names())
  expect_false("f60_bogus" %in% across_native_heads())
})

# The per-dot screens before across() expansion skip their walk when
# all.names() rules the name out; what the walk finds must not change.
test_that("the zero-arg-call and fallback screens still find nested, quoted and pkg:: calls", {
  q <- rlang::quo(cur_group_id())
  expect_true(expr_contains_zero_arg_call(quote(f(x, dplyr::cur_group_id())), "cur_group_id"))
  expect_true(expr_contains_zero_arg_call(rlang::expr(g(!!q)), "cur_group_id"))
  expect_false(expr_contains_zero_arg_call(quote(f(cur_group_id(1))), "cur_group_id"))
  expect_false(expr_contains_zero_arg_call(quote(f(cur_group_id)), "cur_group_id"))
  expect_false(expr_contains_zero_arg_call(quote(scale(x)[, 1]), "cur_group_id"))
  expect_true(expr_contains_zero_arg_call(quote(scale(cur_group_id())[, 1]), "cur_group_id"))

  fb <- function(e) dots_need_fallback(list(rlang::new_quosure(e, globalenv())))
  expect_true(fb(quote(mean(dplyr::pick(x)))))
  expect_true(fb(rlang::expr(sum(!!rlang::quo(c_across(a:b))))))
  expect_true(fb(quote(f(across(x, mean)))))
  expect_true(fb(quote(across(x, mean, .unpack = TRUE))))
  expect_false(fb(quote(across(x, mean))))
  expect_false(fb(quote(scale(x)[, 1])))
  expect_false(fb(quote(mean(x))))
})
