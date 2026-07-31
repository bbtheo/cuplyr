# Tests for AST optimizer passes

test_that("push_down_projections inserts select nodes for required columns", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("d", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
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
  expr1 <- make_mutate_expr("c", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
  expr2 <- make_mutate_expr("d", ir_call("*", list(ir_col("b"), ir_lit_from_r(2))), schema)

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

  pred_expensive <- make_predicate(ir_call("==", list(ir_col("a"), ir_col("b"))), schema)
  pred_cheap <- make_predicate(ir_call(">", list(ir_col("c"), ir_lit(5, "FLOAT64"))), schema)

  filter1 <- ast_filter(source, list(pred_expensive))
  filter2 <- ast_filter(filter1, list(pred_cheap))

  reordered <- reorder_filters(filter2)

  expect_s3_class(reordered, "ast_filter")
  expect_equal(length(reordered$predicates), 2)
  expect_equal(reordered$predicates[[1]]$cols, "c")
})

test_that("push_down_filters moves filter below mutate when safe", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("c", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
  mutate_node <- ast_mutate(source, list(expr))
  pred <- make_predicate(ir_call(">", list(ir_col("b"), ir_lit(5, "FLOAT64"))), schema)
  filter_node <- ast_filter(mutate_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_mutate")
  expect_s3_class(pushed$input, "ast_filter")
  expect_s3_class(pushed$input$input, "ast_source")
})

test_that("push_down_filters does not move filter when it depends on mutate output", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("b", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
  mutate_node <- ast_mutate(source, list(expr))
  pred <- make_predicate(ir_call(">", list(ir_col("b"), ir_lit(5, "FLOAT64"))), schema)
  filter_node <- ast_filter(mutate_node, list(pred))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_filter")
  expect_s3_class(pushed$input, "ast_mutate")
})

test_that("push_down_filters moves filter below select when safe", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  select_node <- ast_select(source, c("a", "b"))
  pred <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(5, "FLOAT64"))), schema)
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
  pred <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(5, "FLOAT64"))), schema)
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

  join_schema <- list(names = c("k", "a", "b"), types = c("INT32", "FLOAT64", "FLOAT64"))
  pred_left <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(1, "FLOAT64"))), join_schema)
  pred_right <- make_predicate(ir_call("<", list(ir_col("b"), ir_lit(5, "FLOAT64"))), join_schema)
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

  join_schema <- list(names = c("k", "a", "b"), types = c("INT32", "FLOAT64", "FLOAT64"))
  pred_right <- make_predicate(ir_call("<", list(ir_col("b"), ir_lit(5, "FLOAT64"))), join_schema)
  filter_node <- ast_filter(join, list(pred_right))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_filter")
  expect_s3_class(pushed$input, "ast_join")
})

test_that("push_down_filters pushes ALL predicates below a semi/anti join, never to the right", {
  # Phase 7 J2: semi/anti's output schema is x's (left's) schema verbatim,
  # so every predicate is necessarily left-only -- unlike a mutating join,
  # there's no "stay" case and no right-side pushdown target at all.
  left_schema <- list(names = c("k", "a"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("k", "b"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)

  for (jt in c("semi", "anti")) {
    join <- ast_join(jt, left, right, by = list(left = "k", right = "k"))

    pred <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(1, "FLOAT64"))), left_schema)
    filter_node <- ast_filter(join, list(pred))

    pushed <- push_down_filters(filter_node)

    expect_s3_class(pushed, "ast_join")
    expect_s3_class(pushed$left, "ast_filter")
    expect_s3_class(pushed$right, "ast_source")
  }
})

test_that("push_down_projections only needs by$right from a semi/anti join's right side", {
  left_schema <- list(names = c("k", "a", "unused"), types = c("INT32", "FLOAT64", "FLOAT64"))
  right_schema <- list(names = c("k", "extra1", "extra2"), types = c("INT32", "FLOAT64", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)

  for (jt in c("semi", "anti")) {
    join <- ast_join(jt, left, right, by = list(left = "k", right = "k"))
    select_node <- ast_select(join, c("k", "a"))

    pushed <- push_down_projections(select_node)
    expect_s3_class(pushed$input$left, "ast_select")
    expect_equal(sort(pushed$input$left$columns), c("a", "k"), info = jt)
    expect_s3_class(pushed$input$right, "ast_select")
    expect_equal(pushed$input$right$columns, "k", info = jt)
  }
})

test_that("prune_dead_columns drops an unused mutate output feeding a semi/anti join's left side", {
  # Same rationale as the push_down_projections test above, but exercised
  # through prune_dead_columns' own "join" branch, which needs a mutate
  # (not a bare source -- prune_dead_columns' "source" case is a no-op,
  # unlike push_down_projections') to observe anything actually get pruned.
  left_schema <- list(names = c("k", "a"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("k"), types = c("INT32"))
  left_source <- ast_source(left_schema)
  waste_expr <- make_mutate_expr("waste", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), left_schema)
  left_mutate <- ast_mutate(left_source, list(waste_expr))
  right <- ast_source(right_schema)

  for (jt in c("semi", "anti")) {
    join <- ast_join(jt, left_mutate, right, by = list(left = "k", right = "k"))
    select_node <- ast_select(join, c("k", "a"))

    pruned <- prune_dead_columns(select_node, c("k", "a"))
    # "waste" isn't required anywhere downstream of the join (the join's
    # own output is x's schema verbatim, and the outer select only asks
    # for k/a) -- the mutate node producing it should be pruned away
    # entirely, leaving the join's left input as the bare source.
    expect_s3_class(pruned$input$left, "ast_source")
  }
})

test_that("infer_schema.ast_join returns x's schema verbatim for semi/anti (incl. factor_levels)", {
  left_schema <- list(names = c("k", "a"), types = c("INT32", "FLOAT64"),
                      factor_levels = list(a = c("x", "y")))
  right_schema <- list(names = c("k", "b"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)

  for (jt in c("semi", "anti")) {
    join <- ast_join(jt, left, right, by = list(left = "k", right = "k"))
    schema <- infer_schema(join)
    expect_identical(schema$names, left_schema$names, info = jt)
    expect_identical(schema$types, left_schema$types, info = jt)
    expect_identical(schema$factor_levels, left_schema$factor_levels, info = jt)
  }
})

test_that("push_down_filters respects renamed join keys", {
  left_schema <- list(names = c("a", "x"), types = c("INT32", "FLOAT64"))
  right_schema <- list(names = c("b", "y"), types = c("INT32", "FLOAT64"))
  left <- ast_source(left_schema)
  right <- ast_source(right_schema)
  join <- ast_join("inner", left, right, by = list(left = "a", right = "b"))

  join_schema <- list(names = c("a", "x", "b", "y"), types = c("INT32", "FLOAT64", "INT32", "FLOAT64"))
  pred_left <- make_predicate(ir_call(">", list(ir_col("x"), ir_lit(1, "FLOAT64"))), join_schema)
  pred_right <- make_predicate(ir_call("<", list(ir_col("y"), ir_lit(5, "FLOAT64"))), join_schema)
  filter_node <- ast_filter(join, list(pred_left, pred_right))

  pushed <- push_down_filters(filter_node)

  expect_s3_class(pushed, "ast_join")
  expect_s3_class(pushed$left, "ast_filter")
  expect_s3_class(pushed$right, "ast_filter")
})

test_that("consecutive filters merge into one predicate list", {
  # Phase 1 expression-engine cutover (task T3): fuse_filters()/ast$fused
  # are gone -- lower_filter() always folds every predicate in a single
  # filter node into one `&`-combined expression, so a filter node with N
  # predicates is already exactly one fused GPU kernel. reorder_filters()
  # is what merges *consecutive* filter nodes (from separate filter() dots
  # or calls) into that single node.
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)

  pred1 <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(1, "FLOAT64"))), schema)
  pred2 <- make_predicate(ir_call("<=", list(ir_col("b"), ir_lit(3, "FLOAT64"))), schema)
  filter1 <- ast_filter(source, list(pred1))
  filter2 <- ast_filter(filter1, list(pred2))

  merged <- reorder_filters(filter2)

  expect_s3_class(merged, "ast_filter")
  expect_equal(length(merged$predicates), 2)
  expect_s3_class(merged$input, "ast_source")
})

# -----------------------------------------------------------------------------
# Phase 12 audit: barrier-type nodes (rename/distinct/slice) reached via a
# join's left/right side
#
# `rename`/`distinct`/`slice` are optimizer barriers (`is_barrier()`,
# R/ast.R), so `push_down_projections()`/`prune_dead_columns()` never see
# them as the `ast$type` they're invoked on via the normal barrier-splitting
# entry point (`optimize_with_barriers()`). They ARE reached this way,
# though, when one of them sits directly on a join's `left`/`right` side
# (`x |> rename(...) |> left_join(y, ...)` is a completely ordinary lazy
# pipeline) -- the "join" case in both passes recurses into `ast$left`/
# `ast$right` with a *direct* `push_down_projections()`/`prune_dead_columns()`
# call, bypassing `optimize_with_barriers()` entirely. Before these three
# cases existed, that direct recursion silently corrupted the join side's
# required columns (see each test below for the specific mechanism).
# -----------------------------------------------------------------------------

test_that("push_down_projections preserves a renamed column feeding a join", {
  left_schema <- list(names = c("x", "y"), types = c("FLOAT64", "FLOAT64"))
  left_rename <- ast_rename(ast_source(left_schema), old_names = "x", new_names = "new_x")
  right_source <- ast_source(list(names = c("z", "w"), types = c("FLOAT64", "FLOAT64")))
  join <- ast_join("inner", left_rename, right_source, by = list(left = "new_x", right = "z"))

  optimized <- push_down_projections(join)

  expect_s3_class(optimized$left, "ast_rename")
  # Without translating required_cols through old_names/new_names, this
  # used to insert ast_select(source, "y") below the rename -- dropping
  # "x" (the rename's own input column) entirely.
  expect_s3_class(optimized$left$input, "ast_source")
  expect_equal(sort(infer_schema(optimized$left$input)$names), c("x", "y"))
})

test_that("prune_dead_columns preserves a mutate output consumed by a rename feeding a join", {
  base_schema <- list(names = c("a", "y"), types = c("FLOAT64", "FLOAT64"))
  x_expr <- make_mutate_expr("x", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), base_schema)
  left_mutate <- ast_mutate(ast_source(base_schema), list(x_expr))
  left_rename <- ast_rename(left_mutate, old_names = "x", new_names = "new_x")
  right_source <- ast_source(list(names = "z", types = "FLOAT64"))
  join <- ast_join("inner", left_rename, right_source, by = list(left = "new_x", right = "z"))

  pruned <- prune_dead_columns(join, c("new_x", "z"))

  # Without translating required_cols, the mutate producing "x" looked
  # unrequired (required_cols only ever held "new_x") and was dropped
  # entirely, leaving the rename node with no "x" to rename.
  expect_s3_class(pruned$left, "ast_rename")
  expect_s3_class(pruned$left$input, "ast_mutate")
  expect_equal(length(pruned$left$input$expressions), 1)
})

test_that("push_down_projections preserves distinct's own key_cols feeding a join (.keep_all = TRUE)", {
  left_source <- ast_source(list(names = c("k", "extra"), types = c("FLOAT64", "FLOAT64")))
  left_distinct <- ast_distinct(left_source, key_cols = c("k", "extra"), keep_all = TRUE)
  right_source <- ast_source(list(names = "k2", types = "FLOAT64"))
  join <- ast_join("inner", left_distinct, right_source, by = list(left = "k", right = "k2"))
  # Downstream only ever asks for "k" -- "extra" is still a required dedup key.
  select_node <- ast_select(join, "k")

  optimized <- push_down_projections(select_node)

  distinct_node <- optimized$input$left
  expect_s3_class(distinct_node, "ast_distinct")
  expect_equal(sort(infer_schema(distinct_node$input)$names), c("extra", "k"))
})

test_that("push_down_projections preserves slice(mode='rank')'s order_col feeding a join", {
  left_source <- ast_source(list(names = c("k", "rank_by"), types = c("FLOAT64", "FLOAT64")))
  left_slice <- ast_slice(left_source, mode = "rank", amount = 1, is_prop = FALSE,
                          order_col = "rank_by", order_is_temp = FALSE, descending = FALSE,
                          with_ties = FALSE, na_rm = FALSE)
  right_source <- ast_source(list(names = "k2", types = "FLOAT64"))
  join <- ast_join("inner", left_slice, right_source, by = list(left = "k", right = "k2"))
  # Downstream only ever asks for "k" -- "rank_by" is still needed to rank by.
  select_node <- ast_select(join, "k")

  optimized <- push_down_projections(select_node)

  slice_node <- optimized$input$left
  expect_s3_class(slice_node, "ast_slice")
  expect_equal(sort(infer_schema(slice_node$input)$names), c("k", "rank_by"))
})

# -----------------------------------------------------------------------------
# Phase 12 audit: fuse_mutates()/reorder_filters()/push_down_filters() used
# to skip a join's left/right subtrees entirely (a "join" node has no
# `$input` field, only `$left`/`$right`, and none of these three passes had
# a "join" case). Not a correctness bug -- unfused mutates/unreordered
# filters still execute correctly, just as more/costlier kernels -- but a
# real missed-optimization gap, closed the same way push_down_projections()/
# prune_dead_columns() already recurse into both join sides.
# -----------------------------------------------------------------------------

test_that("fuse_mutates fuses a mutate chain feeding a join's left side", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr1 <- make_mutate_expr("c", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
  expr2 <- make_mutate_expr("d", ir_call("*", list(ir_col("b"), ir_lit_from_r(2))), schema)
  left <- ast_mutate(ast_mutate(source, list(expr1)), list(expr2))
  right <- ast_source(list(names = "k", types = "FLOAT64"))
  join <- ast_join("inner", left, right, by = list(left = "a", right = "k"))

  fused <- fuse_mutates(join)

  expect_s3_class(fused$left, "ast_mutate")
  expect_equal(length(fused$left$expressions), 2)
  expect_s3_class(fused$left$input, "ast_source")
})

test_that("reorder_filters merges a filter chain feeding a join's right side", {
  schema <- list(names = c("a", "b", "c"), types = c("FLOAT64", "FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  pred_expensive <- make_predicate(ir_call("==", list(ir_col("a"), ir_col("b"))), schema)
  pred_cheap <- make_predicate(ir_call(">", list(ir_col("c"), ir_lit(5, "FLOAT64"))), schema)
  right <- ast_filter(ast_filter(source, list(pred_expensive)), list(pred_cheap))
  left <- ast_source(list(names = "k", types = "FLOAT64"))
  join <- ast_join("inner", left, right, by = list(left = "k", right = "a"))

  reordered <- reorder_filters(join)

  expect_s3_class(reordered$right, "ast_filter")
  expect_equal(length(reordered$right$predicates), 2)
  expect_equal(reordered$right$predicates[[1]]$cols, "c")
  expect_s3_class(reordered$right$input, "ast_source")
})

test_that("push_down_filters pushes a filter below a mutate entirely within a join's left side", {
  # Unlike the pre-existing "moves side-only filters below inner join" test
  # (which wraps a filter ABOVE the whole join), this filter sits WITHIN
  # the join's own left subtree, with no filter node directly wrapping the
  # join at all -- exercising the new join-recursion path, not the
  # existing one-level filter-above-join swap.
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)
  expr <- make_mutate_expr("c", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), schema)
  mutate_node <- ast_mutate(source, list(expr))
  pred <- make_predicate(ir_call(">", list(ir_col("b"), ir_lit(5, "FLOAT64"))), schema)
  left <- ast_filter(mutate_node, list(pred))
  right <- ast_source(list(names = "k", types = "FLOAT64"))
  join <- ast_join("inner", left, right, by = list(left = "a", right = "k"))

  pushed <- push_down_filters(join)

  expect_s3_class(pushed$left, "ast_mutate")
  expect_s3_class(pushed$left$input, "ast_filter")
  expect_s3_class(pushed$left$input$input, "ast_source")
})

test_that("optimize_ast does not reorder across barriers", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  source <- ast_source(schema)

  pred1 <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(1, "FLOAT64"))), schema)
  pred2 <- make_predicate(ir_call("<", list(ir_col("b"), ir_lit(5, "FLOAT64"))), schema)

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
