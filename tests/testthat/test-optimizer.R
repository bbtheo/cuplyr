# Tests for AST optimizer passes

test_that("push_down_projections inserts select nodes for required columns", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("d", "a", "+", scalar = 1, input_types = "FLOAT64")
  mutate_node <- ast_mutate(source, list(expr))
  select_node <- ast_select(mutate_node, "d")

  optimized <- push_down_projections(select_node)

  expect_s3_class(optimized, "ast_select")
  expect_s3_class(optimized$input, "ast_mutate")
  expect_s3_class(optimized$input$input, "ast_select")
  expect_s3_class(optimized$input$input$input, "ast_source")
  expect_equal(optimized$input$input$columns, "a")
})

test_that("fuse_mutates merges consecutive mutate nodes", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr1 <- make_mutate_expr("c", "a", "+", scalar = 1, input_types = "FLOAT64")
  expr2 <- make_mutate_expr("d", "b", "*", scalar = 2, input_types = "FLOAT64")

  lower <- ast_mutate(source, list(expr1))
  upper <- ast_mutate(lower, list(expr2))

  fused <- fuse_mutates(upper)

  expect_s3_class(fused, "ast_mutate")
  expect_equal(length(fused$expressions), 2)
  expect_s3_class(fused$input, "ast_source")
})

test_that("reorder_filters orders predicates by estimated cost", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)

  pred_expensive <- make_predicate("a", "==", "b", is_col_compare = TRUE)
  pred_cheap <- make_predicate("c", ">", 5)

  filter1 <- ast_filter(source, list(pred_expensive))
  filter2 <- ast_filter(filter1, list(pred_cheap))

  reordered <- reorder_filters(filter2)

  expect_s3_class(reordered, "ast_filter")
  expect_equal(length(reordered$predicates), 2)
  expect_equal(reordered$predicates[[1]]$col_name, "c")
})

test_that("push_down_filters moves filter below mutate when safe", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("c", "a", "+", scalar = 1, input_types = "FLOAT64")
  mutate_node <- ast_mutate(source, list(expr))
  pred <- make_predicate("b", ">", 5)
  filter_node <- ast_filter(mutate_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_mutate")
  expect_s3_class(pushed$input, "ast_filter")
  expect_s3_class(pushed$input$input, "ast_source")
})

test_that("push_down_filters does not move filter when it depends on mutate output", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("b", "a", "+", scalar = 1, input_types = "FLOAT64")
  mutate_node <- ast_mutate(source, list(expr))
  pred <- make_predicate("b", ">", 5)
  filter_node <- ast_filter(mutate_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_filter")
  expect_s3_class(pushed$input, "ast_mutate")
})

test_that("push_down_filters moves filter below select when safe", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  select_node <- ast_select(source, c("a", "b"))
  pred <- make_predicate("a", ">", 5)
  filter_node <- ast_filter(select_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_select")
  expect_s3_class(pushed$input, "ast_filter")
  expect_s3_class(pushed$input$input, "ast_source")
})

test_that("push_down_filters does not move filter below select if column dropped", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  select_node <- ast_select(source, c("b"))
  pred <- make_predicate("a", ">", 5)
  filter_node <- ast_filter(select_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_filter")
  expect_s3_class(pushed$input, "ast_select")
})

test_that("push_down_filters moves side-only filters below inner join", {
  left_schema <- list(names = c("k", "a"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("k", "b"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)
  join <- ast_join("inner", left, right, by = list(left = "k", right = "k"))

  pred_left <- make_predicate("a", ">", 1)
  pred_right <- make_predicate("b", "<", 5)
  filter_node <- ast_filter(join, list(pred_left, pred_right))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_join")
  expect_s3_class(pushed$left, "ast_filter")
  expect_s3_class(pushed$right, "ast_filter")
})

test_that("push_down_filters does not move right-only filters below left join", {
  left_schema <- list(names = c("k", "a"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("k", "b"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)
  join <- ast_join("left", left, right, by = list(left = "k", right = "k"))

  pred_right <- make_predicate("b", "<", 5)
  filter_node <- ast_filter(join, list(pred_right))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_filter")
  expect_s3_class(pushed$input, "ast_join")
})

test_that("push_down_filters respects renamed join keys", {
  left_schema <- list(names = c("a", "x"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("b", "y"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)
  join <- ast_join("inner", left, right, by = list(left = "a", right = "b"))

  pred_left <- make_predicate("x", ">", 1)
  pred_right <- make_predicate("y", "<", 5)
  filter_node <- ast_filter(join, list(pred_left, pred_right))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_join")
  expect_s3_class(pushed$left, "ast_filter")
  expect_s3_class(pushed$right, "ast_filter")
})

test_that("fuse_filters marks simple predicates as fused", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)

  pred1 <- make_predicate("a", ">", 1)
  pred2 <- make_predicate("b", "<=", 3)
  filter_node <- ast_filter(source, list(pred1, pred2))

  fused <- fuse_filters(filter_node)

  expect_true(isTRUE(fused$fused))
})

test_that("optimize_ast does not reorder across barriers", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)

  pred1 <- make_predicate("a", ">", 1)
  pred2 <- make_predicate("b", "<", 5)

  filter1 <- ast_filter(source, list(pred1))
  arrange_node <- ast_arrange(filter1, list(list(col_name = "a", descending = FALSE)))
  filter2 <- ast_filter(arrange_node, list(pred2))

  optimized <- optimize_ast(filter2)

  # Verify barrier keeps arrange between the two filters (allow selects)
  ast_str <- ast_to_string(optimized)
  expect_true(grepl("arrange\\[", ast_str))
  expect_equal(length(gregexpr("filter\\[", ast_str, fixed = FALSE)[[1]]), 2)
  expect_true(grepl("filter\\[.*arrange\\[.*filter\\[", ast_str))
})
