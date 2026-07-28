# Tests for window-expression decomposition (R/window.R)
#
# Pure R: no skip_if_no_gpu(), no GPU calls anywhere in this file. Nothing
# calls ir_extract_windows()/plan_window_stages() in production yet (W1 is
# dormant, per scratchpad/phase5_window_design.md's sequencing table: no
# verb is wired to window functions until W3/W5/W6/W7) -- these tests
# exercise R/window.R's functions directly, self-sufficiently (this file
# defines its own fixtures rather than relying on test-ir.R's, since
# testthat doesn't share top-level test-*.R definitions across files).

schema_xyz <- list(names = c("x", "y", "z"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))

new_window_state <- function(existing_names = schema_xyz$names) {
  state <- new.env(parent = emptyenv())
  state$seen <- list()
  state$specs <- list()
  state$pre <- list()
  state$existing_names <- existing_names
  state
}

# -----------------------------------------------------------------------------
# unique_temp_name()
# -----------------------------------------------------------------------------

test_that("unique_temp_name() returns the bare base when free", {
  expect_equal(unique_temp_name("..win", c("x", "y")), "..win")
})

test_that("unique_temp_name() appends an incrementing suffix on collision", {
  expect_equal(unique_temp_name("..win", c("..win")), "..win1")
  expect_equal(unique_temp_name("..win", c("..win", "..win1")), "..win2")
})

# -----------------------------------------------------------------------------
# ir_extract_windows(): single spec, CSE, non-col args, two-pass nesting
# -----------------------------------------------------------------------------

test_that("ir_extract_windows() extracts one innermost window call: x - mean(x)", {
  ir <- ir_parse_quo(rlang::quo(x - mean(x)), schema_xyz)
  state <- new_window_state()

  post <- ir_extract_windows(ir, schema_xyz, state)

  expect_equal(length(state$specs), 1)
  spec <- state$specs[[1]]
  expect_equal(spec$fn, "mean")
  expect_equal(spec$value_col, "x")
  expect_equal(spec$output_col, "..win1..")

  # post-expr is `x - ..win1..` (window-free)
  expect_equal(post$kind, "call")
  expect_equal(post$op, "-")
  expect_equal(post$args[[1]]$kind, "col")
  expect_equal(post$args[[1]]$name, "x")
  expect_equal(post$args[[2]]$kind, "col")
  expect_equal(post$args[[2]]$name, "..win1..")
  expect_false(ir_has_window(post))
})

test_that("ir_extract_windows() CSEs two structurally identical window calls", {
  ir <- ir_call("+", list(
    ir_call("mean", list(ir_col("x"))),
    ir_call("mean", list(ir_col("x")))
  ))
  state <- new_window_state()

  post <- ir_extract_windows(ir, schema_xyz, state)

  expect_equal(length(state$specs), 1)
  expect_equal(post$args[[1]]$kind, "col")
  expect_equal(post$args[[2]]$kind, "col")
  expect_equal(post$args[[1]]$name, post$args[[2]]$name)
})

test_that("ir_extract_windows() does NOT CSE structurally different window calls", {
  ir <- ir_call("+", list(
    ir_call("mean", list(ir_col("x"))),
    ir_call("mean", list(ir_col("y")))
  ))
  state <- new_window_state()

  post <- ir_extract_windows(ir, schema_xyz, state)

  expect_equal(length(state$specs), 2)
  expect_false(identical(post$args[[1]]$name, post$args[[2]]$name))
})

test_that("ir_extract_windows() hoists a non-col window argument into a ..winarg.. pre-mutate", {
  ir <- ir_call("mean", list(ir_call("+", list(ir_col("x"), ir_col("y")))))
  state <- new_window_state()

  post <- ir_extract_windows(ir, schema_xyz, state)

  expect_equal(length(state$pre), 1)
  winarg_name <- names(state$pre)[1]
  expect_true(grepl("^\\.\\.winarg", winarg_name))
  expect_equal(state$pre[[1]]$output_col, winarg_name)
  expect_equal(state$pre[[1]]$ir$op, "+")  # the hoisted x + y expression

  expect_equal(length(state$specs), 1)
  expect_equal(state$specs[[1]]$fn, "mean")
  expect_equal(state$specs[[1]]$value_col, winarg_name)

  expect_equal(post$kind, "col")
  expect_equal(post$name, state$specs[[1]]$output_col)
})

test_that("ir_extract_windows() leaves a not-yet-innermost window call for a later pass", {
  ir <- ir_parse_quo(rlang::quo(lag(cumsum(x))), schema_xyz)
  expect_equal(ir$op, "lag")
  expect_true(ir_has_window(ir$args[[1]]))  # cumsum(x) is itself a window call

  state <- new_window_state()
  post <- ir_extract_windows(ir, schema_xyz, state)

  # Pass 1: only cumsum(x) is innermost; lag() itself is deferred.
  expect_equal(length(state$specs), 1)
  expect_equal(state$specs[[1]]$fn, "cumsum")
  expect_equal(state$specs[[1]]$value_col, "x")

  expect_equal(post$kind, "call")
  expect_equal(post$op, "lag")
  expect_equal(post$args[[1]]$kind, "col")
  expect_equal(post$args[[1]]$name, state$specs[[1]]$output_col)
  expect_true(ir_has_window(post))  # lag itself still unresolved

  # Pass 2: lag() is now innermost (its arg is a bare temp column).
  schema_after_pass1 <- list(
    names = c(schema_xyz$names, state$specs[[1]]$output_col),
    types = c(schema_xyz$types, state$specs[[1]]$output_type)
  )
  state2 <- new_window_state(schema_after_pass1$names)
  post2 <- ir_extract_windows(post, schema_after_pass1, state2)

  expect_equal(length(state2$specs), 1)
  expect_equal(state2$specs[[1]]$fn, "lag")
  expect_equal(state2$specs[[1]]$value_col, state$specs[[1]]$output_col)
  expect_false(ir_has_window(post2))
})

test_that("ir_extract_windows() passes a window-free tree through unchanged", {
  ir <- ir_parse_quo(rlang::quo(x + y * 2), schema_xyz)
  state <- new_window_state()

  post <- ir_extract_windows(ir, schema_xyz, state)

  expect_equal(length(state$specs), 0)
  expect_equal(length(state$pre), 0)
  expect_identical(post, ir)
})

# -----------------------------------------------------------------------------
# plan_window_stages()
# -----------------------------------------------------------------------------

test_that("plan_window_stages() on a window-free expr produces zero stages, untouched post", {
  exprs <- list(a = ir_parse_quo(rlang::quo(x + y), schema_xyz))
  plan <- plan_window_stages(exprs, schema_xyz, character())

  expect_equal(length(plan$stages), 0)
  expect_identical(plan$post$a, exprs$a)
  expect_equal(plan$group_cols, character())
})

test_that("plan_window_stages() output shape matches section 1.2 exactly", {
  exprs <- list(a = ir_parse_quo(rlang::quo(x - mean(x)), schema_xyz))
  plan <- plan_window_stages(exprs, schema_xyz, "z")

  expect_named(plan, c("stages", "post", "schema", "group_cols"))
  expect_equal(plan$group_cols, "z")
  expect_equal(length(plan$stages), 1)

  stage <- plan$stages[[1]]
  expect_named(stage, c("pre", "specs"))
  expect_equal(length(stage$pre), 0)
  expect_equal(length(stage$specs), 1)

  spec <- stage$specs[[1]]
  expect_named(
    spec,
    c("output_col", "fn", "value_col", "order_cols", "order_desc",
      "output_type", "n", "default", "na_rm")
  )
  expect_equal(spec$fn, "mean")

  expect_named(plan$post, "a")
  expect_false(ir_has_window(plan$post$a))

  # schema grows to include the spec's temp output column
  expect_true(spec$output_col %in% plan$schema$names)
})

test_that("plan_window_stages() on lag(cumsum(x)) produces two passes/stages", {
  exprs <- list(a = ir_parse_quo(rlang::quo(lag(cumsum(x))), schema_xyz))
  plan <- plan_window_stages(exprs, schema_xyz, character())

  expect_equal(length(plan$stages), 2)
  expect_equal(plan$stages[[1]]$specs[[1]]$fn, "cumsum")
  expect_equal(plan$stages[[2]]$specs[[1]]$fn, "lag")

  # the lag spec's value_col is exactly the cumsum spec's output_col
  expect_equal(plan$stages[[2]]$specs[[1]]$value_col, plan$stages[[1]]$specs[[1]]$output_col)

  expect_equal(plan$post$a$kind, "col")
  expect_equal(plan$post$a$name, plan$stages[[2]]$specs[[1]]$output_col)
})

test_that("plan_window_stages() batches independent window-bearing dots into one stage", {
  exprs <- list(
    a = ir_parse_quo(rlang::quo(cumsum(x)), schema_xyz),
    b = ir_parse_quo(rlang::quo(mean(y)), schema_xyz)
  )
  plan <- plan_window_stages(exprs, schema_xyz, character())

  # Neither dot's window-call args reference the other's output name, so
  # both are decomposed in the SAME pass -> one stage with two specs.
  expect_equal(length(plan$stages), 1)
  expect_equal(length(plan$stages[[1]]$specs), 2)
})

test_that("plan_window_stages() opens a new stage group when a dot references an earlier dot's window output", {
  exprs <- list(
    a = ir_call("cumsum", list(ir_col("x"))),
    b = ir_call("lag", list(ir_col("a")), meta = list(n = 1L, default = NULL))
  )
  plan <- plan_window_stages(exprs, schema_xyz, character())

  # dot b's `lag(a)` references dot a's own output name "a" -- not a real
  # schema column at plan start -- so it must be a SEPARATE stage group,
  # after "a" has been materialized into the running schema.
  expect_equal(length(plan$stages), 2)
  expect_equal(plan$stages[[1]]$specs[[1]]$fn, "cumsum")
  expect_equal(plan$stages[[2]]$specs[[1]]$fn, "lag")
  expect_equal(plan$stages[[2]]$specs[[1]]$value_col, "a")

  expect_true("a" %in% plan$schema$names)
  expect_equal(plan$post$a$kind, "col")
  expect_equal(plan$post$b$kind, "col")
})

test_that("plan_window_stages() with an empty exprs list is a no-op", {
  plan <- plan_window_stages(list(), schema_xyz, character())
  expect_equal(length(plan$stages), 0)
  expect_equal(length(plan$post), 0)
  expect_identical(plan$schema, schema_xyz)
})

test_that("plan_window_stages() requires a fully named exprs list", {
  expect_error(
    plan_window_stages(list(ir_col("x")), schema_xyz, character()),
    "named"
  )
})

# -----------------------------------------------------------------------------
# window_arg_cols() -- the multi-dot chunking rule's own building block
# -----------------------------------------------------------------------------

test_that("window_arg_cols() collects columns referenced inside window-call args only", {
  # `a + lag(b)`: "a" is an ordinary (non-window) reference, "b" is inside
  # a window call's argument -- only "b" should be reported.
  ir <- ir_call("+", list(
    ir_col("a"),
    ir_call("lag", list(ir_col("b")), meta = list(n = 1L, default = NULL))
  ))
  expect_equal(window_arg_cols(ir), "b")
})

test_that("window_arg_cols() is empty for an expression with no window calls", {
  ir <- ir_parse_quo(rlang::quo(x + y), schema_xyz)
  expect_equal(window_arg_cols(ir), character())
})
