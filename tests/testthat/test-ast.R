# Tests for AST Infrastructure

test_that("ast_source creates valid source node", {
  schema <- list(names = c("x", "y"), types = c("FLOAT64", "INT32"))
  node <- ast_source(schema)

  expect_s3_class(node, "ast_source")
  expect_s3_class(node, "ast_node")
  expect_equal(node$type, "source")
  expect_equal(node$schema, schema)
  expect_null(node$input)
})

test_that("ast_filter creates valid filter node", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  pred <- make_predicate("x", ">", 5)
  node <- ast_filter(source, list(pred))

  expect_s3_class(node, "ast_filter")
  expect_equal(node$type, "filter")
  expect_length(node$predicates, 1)
  expect_equal(node$input$type, "source")
})

test_that("ast_mutate creates valid mutate node", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  expr <- make_mutate_expr("y", "x", "*", scalar = 2, input_types = "FLOAT64")
  node <- ast_mutate(source, list(expr))

  expect_s3_class(node, "ast_mutate")
  expect_equal(node$type, "mutate")
  expect_length(node$expressions, 1)
  expect_equal(node$expressions[[1]]$output_col, "y")
})

test_that("ast_select creates valid select node", {
  source <- ast_source(list(names = c("x", "y", "z"), types = c("FLOAT64", "INT32", "FLOAT64")))
  node <- ast_select(source, c("x", "z"))

  expect_s3_class(node, "ast_select")
  expect_equal(node$type, "select")
  expect_equal(node$columns, c("x", "z"))
})

test_that("ast_arrange creates valid arrange node", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  specs <- list(list(col_name = "x", descending = TRUE))
  node <- ast_arrange(source, specs)

  expect_s3_class(node, "ast_arrange")
  expect_equal(node$type, "arrange")
  expect_length(node$sort_specs, 1)
  expect_true(node$sort_specs[[1]]$descending)
})

test_that("ast_summarise stores groups explicitly", {
  source <- ast_source(list(names = c("g", "x"), types = c("INT32", "FLOAT64")))
  aggs <- list(make_aggregation("total", "x", "sum", "FLOAT64"))
  node <- ast_summarise(source, aggs, groups = "g")

  expect_s3_class(node, "ast_summarise")
  expect_equal(node$groups, "g")
  expect_length(node$aggregations, 1)
})

# Schema Inference Tests

test_that("infer_schema.ast_source returns schema", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "INT32"))
  node <- ast_source(schema)

  result <- infer_schema(node)
  expect_equal(result$names, c("a", "b"))
  expect_equal(result$types, c("FLOAT64", "INT32"))
})

test_that("infer_schema.ast_filter preserves schema", {
  source <- ast_source(list(names = c("x", "y"), types = c("FLOAT64", "INT32")))
  node <- ast_filter(source, list(make_predicate("x", ">", 0)))

  result <- infer_schema(node)
  expect_equal(result$names, c("x", "y"))
  expect_equal(result$types, c("FLOAT64", "INT32"))
})

test_that("infer_schema.ast_select subsets schema", {
  source <- ast_source(list(names = c("a", "b", "c"), types = c("FLOAT64", "INT32", "FLOAT64")))
  node <- ast_select(source, c("c", "a"))

  result <- infer_schema(node)
  expect_equal(result$names, c("c", "a"))
  expect_equal(result$types, c("FLOAT64", "FLOAT64"))
})

test_that("infer_schema.ast_mutate adds new columns", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  expr <- make_mutate_expr("y", "x", "*", scalar = 2, input_types = "FLOAT64")
  node <- ast_mutate(source, list(expr))

  result <- infer_schema(node)
  expect_equal(result$names, c("x", "y"))
  expect_equal(result$types, c("FLOAT64", "FLOAT64"))
})

test_that("infer_schema.ast_mutate replaces existing columns", {
  source <- ast_source(list(names = c("x", "y"), types = c("INT32", "INT32")))
  expr <- make_mutate_expr("x", "y", "/", scalar = 2, input_types = "INT32")
  node <- ast_mutate(source, list(expr))

  result <- infer_schema(node)
  expect_equal(result$names, c("x", "y"))
  expect_equal(result$types, c("FLOAT64", "INT32"))  # x changed to FLOAT64
})

test_that("infer_schema.ast_summarise returns group + agg columns", {
  source <- ast_source(list(names = c("g", "x", "y"), types = c("INT32", "FLOAT64", "FLOAT64")))
  aggs <- list(
    make_aggregation("total", "x", "sum", "FLOAT64"),
    make_aggregation("avg", "y", "mean", "FLOAT64")
  )
  node <- ast_summarise(source, aggs, groups = "g")

  result <- infer_schema(node)
  expect_equal(result$names, c("g", "total", "avg"))
  expect_equal(result$types, c("INT32", "FLOAT64", "FLOAT64"))
})

# Predicate and Expression Structure Tests

test_that("make_predicate creates valid structure", {
  pred <- make_predicate("col", ">=", 10)

  expect_equal(pred$col_name, "col")
  expect_equal(pred$op, ">=")
  expect_equal(pred$value, 10)
  expect_false(pred$is_col_compare)
  expect_equal(pred$estimated_cost, 1L)
  expect_true(pred$is_deterministic)
})

test_that("make_predicate handles column comparison", {
  pred <- make_predicate("a", "==", "b", is_col_compare = TRUE)

  expect_equal(pred$value, "b")
  expect_true(pred$is_col_compare)
  expect_equal(pred$estimated_cost, 2L)
})

test_that("make_mutate_expr creates valid structure", {
  expr <- make_mutate_expr("out", c("a", "b"), "+", input_types = c("FLOAT64", "INT32"))

  expect_equal(expr$output_col, "out")
  expect_equal(expr$input_cols, c("a", "b"))
  expect_equal(expr$op, "+")
  expect_null(expr$scalar)
  expect_equal(expr$output_type, "FLOAT64")
})

test_that("make_mutate_expr handles scalar operations", {
  expr <- make_mutate_expr("out", "x", "*", scalar = 2.5, input_types = "INT32")

  expect_equal(expr$scalar, 2.5)
  expect_equal(expr$output_type, "FLOAT64")  # scalar is double
})

test_that("make_mutate_expr copy operation preserves type", {
  expr <- make_mutate_expr("out", "x", "copy", input_types = "INT32")

  expect_equal(expr$op, "copy")
  expect_equal(expr$output_type, "INT32")
})

test_that("infer_mutate_output_type handles division", {
  # Division always produces FLOAT64
  result <- infer_mutate_output_type("/", c("INT32", "INT32"), NULL)
  expect_equal(result, "FLOAT64")
})

test_that("infer_mutate_output_type preserves INT32 for non-division", {
  result <- infer_mutate_output_type("+", c("INT32", "INT32"), NULL)
  expect_equal(result, "INT32")
})

# Helper Function Tests

test_that("find_calls extracts function names", {
  expr <- quote(sqrt(x + log(y)))
  calls <- find_calls(expr)

  expect_true("sqrt" %in% calls)
  expect_true("log" %in% calls)
  expect_true("+" %in% calls)
})

test_that("is_opaque_expression detects unknown functions", {
  expect_false(is_opaque_expression("x + y"))
  expect_false(is_opaque_expression("sqrt(x)"))
  expect_false(is_opaque_expression("log(x) + abs(y)"))
  expect_true(is_opaque_expression("custom_fn(x)"))
  expect_true(is_opaque_expression("my_func(x, y)"))
})

test_that("is_barrier identifies barrier nodes", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  filter_node <- ast_filter(source, list())
  arrange_node <- ast_arrange(source, list())
  head_node <- ast_head(source, 10)
  barrier_node <- ast_barrier(source)

  expect_false(is_barrier(source))
  expect_false(is_barrier(filter_node))
  expect_true(is_barrier(arrange_node))
  expect_true(is_barrier(head_node))
  expect_true(is_barrier(barrier_node))
})

test_that("ast_depth computes correct depth", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  filter1 <- ast_filter(source, list())
  filter2 <- ast_filter(filter1, list())
  mutate_node <- ast_mutate(filter2, list())

  expect_equal(ast_depth(source), 1L)
  expect_equal(ast_depth(filter1), 2L)
  expect_equal(ast_depth(filter2), 3L)
  expect_equal(ast_depth(mutate_node), 4L)
})

test_that("ast_count counts nodes correctly", {
  source <- ast_source(list(names = "x", types = "FLOAT64"))
  filter_node <- ast_filter(source, list())
  mutate_node <- ast_mutate(filter_node, list())

  expect_equal(ast_count(source), 1L)
  expect_equal(ast_count(filter_node), 2L)
  expect_equal(ast_count(mutate_node), 3L)
})

test_that("ast_to_string produces readable output", {
  source <- ast_source(list(names = c("x", "y"), types = c("FLOAT64", "INT32")))
  filter_node <- ast_filter(source, list(make_predicate("x", ">", 0)))

  str <- ast_to_string(filter_node)
  expect_true(grepl("filter", str))
  expect_true(grepl("source", str))
})

# Print method test
test_that("print.ast_node works without error", {
  source <- ast_source(list(names = c("x", "y"), types = c("FLOAT64", "INT32")))
  filter_node <- ast_filter(source, list(make_predicate("x", ">", 0)))
  mutate_node <- ast_mutate(filter_node, list(make_mutate_expr("z", "x", "*", scalar = 2)))

  expect_output(print(mutate_node), "ast_mutate")
  expect_output(print(mutate_node), "ast_filter")
  expect_output(print(mutate_node), "ast_source")
})
