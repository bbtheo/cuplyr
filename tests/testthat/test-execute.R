# Tests for R/execute.R - unified eager/lazy execution helpers
#
# Covers: current_schema(), input_node(), push_op(), propagate_groups(),
# cuplyr_fallback_notify(). See scratchpad/unification_design.md (D1-D4,
# S1 spec) for the design this file pins.

# =============================================================================
# input_node()
# =============================================================================

test_that("input_node() attaches source_ptr in eager mode", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = FALSE)

  node <- input_node(tbl)

  expect_s3_class(node, "ast_source")
  expect_null(node$input)
  expect_identical(node$source_ptr, tbl$ptr)
  expect_equal(node$schema, tbl$schema)
})

test_that("input_node() builds a bare source node in lazy mode with no pending ops", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = TRUE)

  expect_false(has_pending_ops(tbl))

  node <- input_node(tbl)

  expect_s3_class(node, "ast_source")
  expect_null(node$input)
  expect_null(node$source_ptr)
  expect_equal(node$schema, tbl$schema)
})

test_that("input_node() returns the existing AST when lazy ops are pending", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 1)

  expect_true(has_pending_ops(tbl))

  node <- input_node(tbl)

  expect_identical(node, tbl$lazy_ops)
  expect_s3_class(node, "ast_filter")
})

# =============================================================================
# current_schema()
# =============================================================================

test_that("current_schema() returns base schema when eager", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = FALSE)

  expect_equal(current_schema(tbl), tbl$schema)
})

test_that("current_schema() returns base schema when lazy with no pending ops", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = TRUE)

  expect_equal(current_schema(tbl), tbl$schema)
})

test_that("current_schema() infers schema from pending lazy AST", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = TRUE) |> dplyr::select(x)

  expect_equal(current_schema(tbl)$names, "x")
  expect_equal(current_schema(tbl), infer_schema(tbl$lazy_ops))
})

# =============================================================================
# push_op() - eager materialization
# =============================================================================

test_that("push_op() on eager tbl_gpu materializes a hand-built ast_select node", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = c(5.5, 4.5, 3.5, 2.5, 1.5))
  tbl <- tbl_gpu(df, lazy = FALSE)

  node <- ast_select(input_node(tbl), c("y", "x"))
  result <- push_op(tbl, node)

  expect_valid_tbl_gpu(result)
  expect_data_on_gpu(result)
  expect_identical(result$exec_mode, "eager")
  expect_null(result$lazy_ops)

  expect_equal(result$schema$names, c("y", "x"))
  expect_equal(unname(result$schema$types), c("FLOAT64", "INT32"))

  collected <- collect(result)
  expect_equal(names(collected), c("y", "x"))
  expect_equal(collected$x, df$x)
  expect_equal(collected$y, df$y)
})

test_that("push_op() eager result is a freshly materialized pointer", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = FALSE)

  node <- ast_select(input_node(tbl), "x")
  result <- push_op(tbl, node)

  # A new GPU allocation was made; the pointer should differ from the input's.
  expect_false(identical(result$ptr, tbl$ptr))
})

# =============================================================================
# push_op() - lazy deferral
# =============================================================================

test_that("push_op() on lazy tbl_gpu stores the node without executing", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1)
  tbl <- tbl_gpu(df, lazy = TRUE)

  node <- ast_select(input_node(tbl), "x")
  result <- push_op(tbl, node)

  expect_identical(result$exec_mode, "lazy")
  expect_true(has_pending_ops(result))
  expect_identical(result$lazy_ops, node)

  # No GPU work performed: the underlying pointer is unchanged.
  expect_identical(result$ptr, tbl$ptr)
  expect_data_on_gpu(result)

  expect_equal(result$schema$names, "x")
})

test_that("push_op() lazy schema matches infer_schema(node)", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:5, y = 5:1, z = c("a", "b", "c", "d", "e"))
  tbl <- tbl_gpu(df, lazy = TRUE)

  node <- ast_select(input_node(tbl), c("z", "x"))
  result <- push_op(tbl, node)

  expect_equal(result$schema, infer_schema(node))
})

# =============================================================================
# propagate_groups() truth table
# =============================================================================

test_that("propagate_groups() keeps surviving group columns for ordinary nodes", {
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))

  for (node_type in c("select", "filter", "mutate", "arrange")) {
    node <- list(type = node_type)
    expect_equal(propagate_groups(node, "g", schema), "g")
    expect_equal(propagate_groups(node, c("g", "x"), schema), c("g", "x"))
  }
})

test_that("propagate_groups() drops group columns no longer in schema", {
  schema <- list(names = "x", types = "FLOAT64")
  node <- list(type = "select")

  expect_equal(propagate_groups(node, "g", schema), character())
  expect_equal(propagate_groups(node, c("g", "x"), schema), "x")
})

test_that("propagate_groups() always clears groups for summarise and join", {
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))

  summarise_node <- list(type = "summarise")
  join_node <- list(type = "join")

  expect_equal(propagate_groups(summarise_node, "g", schema), character())
  expect_equal(propagate_groups(summarise_node, c("g", "x"), schema), character())
  expect_equal(propagate_groups(join_node, "g", schema), character())
  expect_equal(propagate_groups(join_node, c("g", "x"), schema), character())
})

test_that("propagate_groups() handles empty input groups", {
  schema <- list(names = c("g", "x"), types = c("INT32", "FLOAT64"))
  node <- list(type = "select")

  expect_equal(propagate_groups(node, character(), schema), character())
})

# =============================================================================
# factor_levels survive push_op()
# =============================================================================

test_that("push_op() select of a factor column preserves factor_levels (D3)", {
  skip_if_no_gpu()

  df <- data.frame(
    f = factor(c("low", "high", "medium"), levels = c("low", "medium", "high")),
    x = 1:3
  )
  tbl <- tbl_gpu(df, lazy = FALSE)

  expect_equal(names(tbl$schema$factor_levels), "f")

  node <- ast_select(input_node(tbl), c("f", "x"))
  result <- push_op(tbl, node)

  expect_equal(names(result$schema$factor_levels), "f")
  expect_equal(result$schema$factor_levels$f, levels(df$f))

  collected <- collect(result)
  expect_s3_class(collected$f, "factor")
  expect_equal(levels(collected$f), levels(df$f))
  expect_equal(as.character(collected$f), as.character(df$f))
})

test_that("push_op() select that drops the factor column drops its factor_levels", {
  skip_if_no_gpu()

  df <- data.frame(
    f = factor(c("low", "high", "medium")),
    x = 1:3
  )
  tbl <- tbl_gpu(df, lazy = FALSE)

  node <- ast_select(input_node(tbl), "x")
  result <- push_op(tbl, node)

  expect_null(result$schema$factor_levels)
})

# =============================================================================
# cuplyr_fallback_notify()
# =============================================================================

test_that("cuplyr_fallback_notify() is silent by default", {
  withr::local_options(cuplyr.fallback = NULL)
  expect_no_warning(cuplyr_fallback_notify("filter", "some_fn(x)"))
})

test_that("cuplyr_fallback_notify() is silent under 'silent'", {
  withr::local_options(cuplyr.fallback = "silent")
  expect_no_warning(cuplyr_fallback_notify("filter", "some_fn(x)"))
})

test_that("cuplyr_fallback_notify() warns under 'warn'", {
  withr::local_options(cuplyr.fallback = "warn")
  expect_warning(
    cuplyr_fallback_notify("filter", "some_fn(x)"),
    "cuplyr: filter\\(\\) fell back to CPU evaluation for: some_fn\\(x\\)"
  )
})

test_that("cuplyr_fallback_notify() errors under 'error'", {
  withr::local_options(cuplyr.fallback = "error")
  expect_error(
    cuplyr_fallback_notify("summarise", "sum(x > 3)"),
    "cuplyr: summarise\\(\\) fell back to CPU evaluation for: sum\\(x > 3\\)"
  )
})
