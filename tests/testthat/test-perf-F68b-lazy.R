# F68b round 3: the lazy-only shortcuts (compute()/collect() of a pending plan).
# Each must give exactly what the general code gives: the optimizer's
# settled-chain path (settled_chain()/settled_prune(), R/optimizer.R) is
# compared with the passes it stands for, and with optimize_ast() run with the
# shortcut switched off (settled_chain() mocked to "not settled").

f68b_schema <- list(names = c("a", "b", "c", "d"), types = c("FLOAT64", "FLOAT64", "INT32", "STRING"))

f68b_all_passes <- function(ast, req) {
  ast |>
    push_down_projections(req) |>
    fuse_mutates() |>
    prune_dead_columns(req) |>
    push_down_filters() |>
    reorder_filters()
}

# A random chain of filter/mutate (and sometimes select) nodes over a source.
# Mutate expression lists sometimes carry the attributes parse_mutate_dots()
# sets (which prune_dead_columns() drops), sometimes names, sometimes nothing;
# a few predicates read a column their input doesn't have (a malformed plan the
# shortcut must decline), and a few mutates are empty.
f68b_random_chain <- function(allow_select = TRUE) {
  sch <- f68b_schema
  node <- ast_source(sch)
  cols <- sch$names
  types <- stats::setNames(sch$types, sch$names)
  k <- 0L
  kinds <- c("filter", "filter2", "mutate", "mutate_new", "mutate2", if (allow_select) "select")
  for (i in seq_len(sample(1:6, 1))) {
    kind <- sample(kinds, 1)
    num <- cols[types[cols] %in% c("FLOAT64", "INT32")]
    if (length(num) == 0L) break
    cur <- list(names = cols, types = unname(types[cols]))
    if (kind %in% c("filter", "filter2")) {
      col <- if (runif(1) < 0.05) "zz" else sample(num, 1)
      preds <- list(make_predicate(ir_call(">", list(ir_col(col), ir_lit(5, "FLOAT64"))), cur))
      if (kind == "filter2") {
        preds <- c(preds, list(make_predicate(
          ir_call("<", list(ir_col(sample(num, 1)), ir_lit(50, "FLOAT64"))), cur)))
      }
      node <- ast_filter(node, preds)
    } else if (kind %in% c("mutate", "mutate_new", "mutate2")) {
      n_ex <- if (kind == "mutate2") 2L else 1L
      if (runif(1) < 0.04) n_ex <- 0L
      exprs <- list()
      for (j in seq_len(n_ex)) {
        k <- k + 1L
        out <- if (kind == "mutate" && length(num) > 0L) sample(num, 1) else paste0("n", k)
        ex <- make_mutate_expr(out, ir_call("+", list(ir_col(sample(num, 1)), ir_lit_from_r(1))), cur)
        exprs[[j]] <- ex
        if (!out %in% cols) cols <- c(cols, out)
        types[[out]] <- ex$output_type
      }
      shape <- sample(c("bare", "named", "attrs"), 1)
      if (shape != "bare" && length(exprs) > 0L) {
        names(exprs) <- vapply(exprs, `[[`, character(1), "output_col")
      }
      if (shape == "attrs") {
        attr(exprs, "has_window") <- FALSE
        attr(exprs, "dot_order") <- names(exprs)
        attr(exprs, "dropped") <- character()
      }
      node <- ast_mutate(node, exprs)
    } else {
      keep <- sample(cols, sample(seq_along(cols), 1))
      node <- ast_select(node, keep)
      cols <- keep
    }
  }
  node
}

f68b_no_shortcut <- function(code) {
  local_mocked_bindings(settled_chain = function(ast, required_cols) 0L)
  code
}

test_that("a settled segment optimizes to what the five passes give", {
  set.seed(6869)
  n_settled <- 0L
  for (i in 1:300) {
    plan <- f68b_random_chain(allow_select = i %% 3L == 0L)
    req <- infer_schema(plan)$names
    if (settled_chain(plan, NULL) > 0L) n_settled <- n_settled + 1L
    expect_identical(optimize_segment(plan, req), f68b_all_passes(plan, req), info = paste("plan", i))
    expect_identical(optimize_segment(plan), f68b_all_passes(plan, req), info = paste("plan", i, "NULL"))
    # a consumer that needs more columns than the segment has, or fewer
    wide <- c(req, "zz")
    expect_identical(optimize_segment(plan, wide), f68b_all_passes(plan, wide), info = paste("plan", i, "wide"))
    narrow <- req[-length(req)]
    if (length(narrow) > 0L) {
      expect_identical(optimize_segment(plan, narrow), f68b_all_passes(plan, narrow),
                       info = paste("plan", i, "narrow"))
    }
    expect_identical(optimize_ast(plan), f68b_no_shortcut(optimize_ast(plan)), info = paste("plan", i, "ast"))
  }
  # the grid really exercises the shortcut
  expect_gt(n_settled, 100L)
})

test_that("settled_chain() declines what the passes may change", {
  src <- ast_source(f68b_schema)
  pred_a <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(5, "FLOAT64"))), f68b_schema)
  ex_e <- make_mutate_expr("e", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), f68b_schema)
  f <- ast_filter(src, list(pred_a))
  m <- ast_mutate(f, list(ex_e))
  expect_identical(settled_chain(src, NULL), 2L)
  expect_identical(settled_chain(f, NULL), 2L)
  expect_identical(settled_chain(m, NULL), 2L)
  expect_identical(settled_chain(m, c("a", "b", "c", "d", "e", "x")), 2L)
  # a pair a later pass looks at: settled, but not final
  expect_identical(settled_chain(ast_filter(m, list(pred_a)), NULL), 1L)
  expect_identical(settled_chain(ast_mutate(m, list(ex_e)), NULL), 1L)
  expect_identical(settled_chain(ast_filter(f, list(pred_a)), NULL), 1L)
  # a column the consumer doesn't need: pruning can drop it
  expect_identical(settled_chain(m, c("a", "b", "c", "d")), 0L)
  # anything but filters and mutates, a truncated chain, an empty mutate, a
  # predicate on a column its input doesn't have
  expect_identical(settled_chain(ast_select(m, c("a", "e")), NULL), 0L)
  expect_identical(settled_chain(ast_mutate(NULL, list(ex_e)), NULL), 0L)
  expect_identical(settled_chain(ast_mutate(f, list()), NULL), 0L)
  pred_e <- make_predicate(ir_call(">", list(ir_col("e"), ir_lit(5, "FLOAT64"))), f68b_schema)
  expect_identical(settled_chain(ast_filter(src, list(pred_e)), NULL), 0L)
  expect_identical(settled_chain(ast_arrange(m, list(list(col_name = "a", descending = FALSE))), NULL), 0L)
})

test_that("settled_prune() keeps a node it doesn't change, and drops the parse flags as the pass does", {
  src <- ast_source(f68b_schema)
  pred_a <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(5, "FLOAT64"))), f68b_schema)
  ex_e <- make_mutate_expr("e", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), f68b_schema)
  f <- ast_filter(src, list(pred_a))
  plain <- ast_mutate(f, list(e = ex_e))
  expect_true(rlang::is_reference(settled_prune(plain), plain))
  expect_true(rlang::is_reference(settled_prune(f), f))
  flagged <- list(e = ex_e)
  attr(flagged, "dot_order") <- "e"
  m <- ast_mutate(f, flagged)
  out <- settled_prune(m)
  expect_identical(out, prune_dead_columns(m, c("a", "b", "c", "d", "e")))
  expect_identical(attributes(.subset2(out, "expressions")), list(names = "e"))
  expect_true(rlang::is_reference(.subset2(out, "input"), f))
})

test_that("lazy plans built by the verbs optimize, explain and collect as with the shortcut off", {
  skip_if_no_gpu()
  set.seed(1)
  df <- data.frame(g = sample.int(5L, 50, TRUE), x = runif(50), y = runif(50), w = rnorm(50),
                   s = sample(letters[1:3], 50, TRUE))
  lz <- tbl_gpu(df, lazy = TRUE)
  other <- tbl_gpu(data.frame(g = 1:5, label = letters[1:5]), lazy = TRUE)
  plans <- list(
    f62 = function(d) d |> filter(x > 0.1) |> mutate(z = x * y + w) |> group_by(g) |>
      summarise(m = mean(z), n = n()) |> arrange(desc(m)),
    filter = function(d) filter(d, x > 0.5),
    mutate = function(d) mutate(d, z = x + 1),
    mutate_filter = function(d) d |> mutate(z = x * 2) |> filter(y > 0.2),
    filter_on_new = function(d) d |> mutate(z = x * 2) |> filter(z > 0.2),
    mutate_mutate = function(d) d |> mutate(z = x * 2) |> mutate(q = z + y),
    filter_filter = function(d) d |> filter(x > 0.1) |> filter(y < 0.9),
    replace = function(d) d |> filter(x > 0.1) |> mutate(x = x * 2, y = NULL),
    keep = function(d) d |> mutate(z = x + y, .keep = "used"),
    transmute = function(d) d |> filter(w > 0) |> transmute(g, z = x - y),
    select_mutate = function(d) d |> select(g, x, y) |> mutate(z = x + y) |> filter(x > 0.3),
    select_after = function(d) d |> mutate(z = x + y, k = 1) |> select(g, z),
    across = function(d) d |> mutate(across(c(x, y), ~ .x * 2)) |> filter(if_any(c(x, y), ~ .x > 1)),
    grouped_mutate = function(d) d |> group_by(g) |> mutate(z = x - mean(x)) |> filter(z > 0),
    window = function(d) d |> arrange(x) |> mutate(l = lag(y)) |> filter(x > 0.2),
    rename = function(d) d |> mutate(z = x) |> rename(zz = z) |> filter(zz > 0.5),
    distinct = function(d) d |> mutate(z = g * 2L) |> distinct(z, .keep_all = TRUE),
    slice = function(d) d |> filter(x > 0.2) |> slice_head(n = 3),
    join = function(d) d |> mutate(z = x + 1) |> left_join(other, by = "g", relationship = "many-to-one") |>
      filter(z > 1.2),
    collapse = function(d) d |> mutate(z = x + 1) |> collapse() |> filter(z > 1.5) |> mutate(q = z * 2),
    by_summarise = function(d) d |> mutate(z = x * y) |> summarise(m = sum(z), .by = s)
  )
  for (nm in names(plans)) {
    plan <- plans[[nm]](lz)
    ops <- .subset2(plan, "lazy_ops")
    expect_identical(optimize_ast(ops), f68b_no_shortcut(optimize_ast(ops)), info = nm)
    expect_identical(capture.output(explain(plan)), f68b_no_shortcut(capture.output(explain(plan))), info = nm)
    expect_identical(collect(plan), f68b_no_shortcut(collect(plan)), info = nm)
  }
})

test_that("current_schema() and has_pending_ops() keep their answers", {
  skip_if_no_gpu()
  eg <- tbl_gpu(data.frame(a = 1:3), lazy = FALSE)
  lz <- tbl_gpu(data.frame(a = 1:3), lazy = TRUE)
  lzp <- dplyr::mutate(lz, b = a + 1L)
  expect_false(has_pending_ops(eg))
  expect_false(has_pending_ops(lz))
  expect_true(has_pending_ops(lzp))
  expect_false(has_pending_ops(unclass(lzp)))
  expect_false(has_pending_ops(NULL))
  expect_false(has_pending_ops(list(lazy_ops = .subset2(lzp, "lazy_ops"))))
  expect_identical(current_schema(lzp), .subset2(lzp, "schema"))
  # the debug switch still checks a lazy table's stored schema against its plan,
  # and only a tbl_gpu with pending operations
  sch <- .subset2(lzp, "schema")
  sch$names <- c("a", "zz")
  bad <- set_tbl_gpu_field(lzp, "schema", sch)
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  expect_error(current_schema(bad), class = "cuplyr_error_internal")
  expect_identical(current_schema(unclass(bad)), sch)
  expect_identical(current_schema(lz), .subset2(lz, "schema"))
})

test_that("the shortcut-off switch above really runs the passes", {
  src <- ast_source(f68b_schema)
  ex_e <- make_mutate_expr("e", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), f68b_schema)
  m <- ast_mutate(src, list(ex_e))
  ran <- 0L
  real <- push_down_projections
  local_mocked_bindings(push_down_projections = function(...) {
    ran <<- ran + 1L
    real(...)
  })
  optimize_ast(m)
  expect_identical(ran, 0L)
  f68b_no_shortcut(optimize_ast(m))
  expect_gt(ran, 0L)
})

test_that("current_schema() checks a lazy plan exactly when check_schema_cache_on() says so", {
  skip_if_no_gpu()
  lzp <- dplyr::mutate(tbl_gpu(data.frame(a = 1:3), lazy = TRUE), b = a + 1L)
  sch <- .subset2(lzp, "schema")
  sch$names <- c("a", "zz")
  bad <- set_tbl_gpu_field(lzp, "schema", sch)
  env_prev <- .schema_memo$check_env
  withr::defer(.schema_memo$check_env <- env_prev)
  for (env in list(NULL, TRUE, FALSE)) {
    for (opt in list(NULL, TRUE, FALSE, "yes", NA)) {
      .schema_memo$check_env <- env
      withr::local_options(cuplyr.check_schema_cache = opt)
      expected_on <- check_schema_cache_on()
      if (expected_on) {
        expect_error(current_schema(bad), class = "cuplyr_error_internal")
      } else {
        expect_identical(current_schema(bad), sch)
      }
      # the env var is read once per session, by whichever runs first
      .schema_memo$check_env <- env
      current_schema(lzp)
      expect_identical(.schema_memo$check_env, {
        .schema_memo$check_env <- env
        check_schema_cache_on()
        .schema_memo$check_env
      }, info = paste(deparse(env), deparse(opt)))
    }
  }
})

# set_ast_source_ptr() as it was before F68b (round 3 reads fields with
# .subset2(); the answers must not change).
f68b_set_ast_source_ptr_ref <- function(node, ptr) {
  if (is.null(node)) return(NULL)
  if (is.null(ptr)) return(node)
  if (node$type == "source") {
    if (is.null(node$source_ptr)) node$source_ptr <- ptr
    return(node)
  }
  if (!is.null(node$input)) {
    node$input <- f68b_set_ast_source_ptr_ref(node$input, ptr)
  }
  if (!is.null(node$left)) {
    node$left <- f68b_set_ast_source_ptr_ref(node$left, ptr)
  }
  if (!is.null(node$right)) {
    node$right <- f68b_set_ast_source_ptr_ref(node$right, ptr)
  }
  node
}

test_that("set_ast_source_ptr() attaches pointers as before, on every plan shape", {
  skip_if_no_gpu()
  set.seed(3)
  df <- data.frame(g = sample.int(4L, 20, TRUE), x = runif(20), y = runif(20))
  lz <- tbl_gpu(df, lazy = TRUE)
  eg <- tbl_gpu(df, lazy = FALSE)
  rt <- tbl_gpu(data.frame(g = 1:4, v = runif(4)), lazy = TRUE)
  rt2 <- tbl_gpu(data.frame(g = 1:4, u = runif(4)), lazy = TRUE)
  deep <- lz
  for (i in 1:12) deep <- mutate(deep, x = x + 1)
  j1 <- left_join(filter(deep, y > 0.2), mutate(rt, v = v * 2), by = "g")
  j2 <- inner_join(mutate(j1, w = x + v), rt2, by = "g")
  j3 <- semi_join(arrange(lz, x), filter(rt, v > 0.1), by = "g")
  plans <- list(
    source = ast_source(.subset2(lz, "schema")),
    attached = ast_source(.subset2(lz, "schema"), source_ptr = .subset2(rt, "ptr")),
    deep = .subset2(deep, "lazy_ops"),
    join = .subset2(j1, "lazy_ops"),
    join_of_join = .subset2(j2, "lazy_ops"),
    semi = .subset2(j3, "lazy_ops"),
    eager_input = input_node(filter(eg, x > 0.5))
  )
  ptrs <- list(lz = .subset2(lz, "ptr"), rt = .subset2(rt, "ptr"), none = NULL)
  for (pn in names(plans)) {
    for (qn in names(ptrs)) {
      expect_identical(set_ast_source_ptr(plans[[pn]], ptrs[[qn]]),
                       f68b_set_ast_source_ptr_ref(plans[[pn]], ptrs[[qn]]),
                       info = paste(pn, qn))
    }
  }
  expect_null(set_ast_source_ptr(NULL, ptrs$lz))
  # the joins built on these plans still collect what eager gives
  expect_identical(collect(j2),
                   collect(inner_join(mutate(left_join(filter(as_eager(deep), y > 0.2),
                                                       mutate(as_eager(rt), v = v * 2), by = "g"),
                                             w = x + v), as_eager(rt2), by = "g")))
})
