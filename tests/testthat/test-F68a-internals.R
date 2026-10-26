# F68a internals: schema inference from a known input schema, the per-compute
# schema memo, the optimizer's pass skipping and set helpers, and the
# current_schema() cache invariant. (The acceptance tests are
# test-parity-F68a-*.R; these pin the pieces they rely on.)

f68_schema <- list(names = c("a", "b", "c", "d"), types = c("FLOAT64", "FLOAT64", "INT32", "STRING"))

f68_chain <- function() {
  src <- ast_source(f68_schema)
  pred <- make_predicate(ir_call(">", list(ir_col("a"), ir_lit(5, "FLOAT64"))), f68_schema)
  expr <- make_mutate_expr("e", ir_call("+", list(ir_col("a"), ir_lit_from_r(1))), f68_schema)
  f <- ast_filter(src, list(pred))
  m <- ast_mutate(f, list(expr))
  s <- ast_select(m, c("e", "c", "a"))
  r <- ast_rename(s, "e", "e2")
  list(src = src, filter = f, mutate = m, select = s, rename = r)
}

test_that("infer_schema_from() gives what infer_schema() gives, node by node", {
  ch <- f68_chain()
  for (nm in c("filter", "mutate", "select", "rename")) {
    node <- ch[[nm]]
    expect_identical(infer_schema_from(node, infer_schema(node$input)), infer_schema(node), info = nm)
    expect_identical(infer_schema_memo(node), infer_schema(node), info = nm)
  }
  distinct <- ast_distinct(ch$mutate, c("c"), keep_all = FALSE)
  expect_identical(infer_schema_from(distinct, infer_schema(ch$mutate)), infer_schema(distinct))
  summ <- ast_summarise(ch$mutate, list(make_aggregation("t", "a", "sum", "FLOAT64")), groups = "c",
                        result_groups = character())
  expect_identical(infer_schema_from(summ, infer_schema(ch$mutate)), infer_schema(summ))
})

test_that("infer_push_schema() walks a verb's fresh node chain, not the pending plan", {
  ch <- f68_chain()
  # push_select_rename() pushes ast_rename(ast_select(input_node(.data))): the
  # rename's input is the select, whose input is the table's own input node.
  sel <- ast_select(ch$mutate, c("e", "c"))
  ren <- ast_rename(sel, "e", "e2")
  expect_identical(infer_push_schema(ren, infer_schema(ch$mutate), base = ch$mutate),
                   infer_schema(ren))
  expect_identical(infer_push_schema(ren, infer_schema(ch$mutate), base = ch$mutate)$names,
                   c("e2", "c"))
  # no pending plan: the chain ends at a source
  sel0 <- ast_select(ch$src, c("b", "a"))
  ren0 <- ast_rename(sel0, "b", "bb")
  expect_identical(infer_push_schema(ren0, f68_schema, base = NULL)$names, c("bb", "a"))
})

test_that("infer_schema_full() is a single pass that matches the recursive definition", {
  ch <- f68_chain()
  full <- infer_schema_full(ch$rename)
  expect_identical(full$names, infer_schema(ch$rename)$names)
  expect_identical(unname(full$types), unname(infer_schema(ch$rename)$types))
})

test_that("the schema memo never serves a stale entry", {
  ch <- f68_chain()
  expect_true(schema_memo_open())
  withr::defer(.schema_memo$plain <- NULL)
  expect_false(schema_memo_open())       # nested: the outer scope is shared
  before <- infer_schema_memo(ch$select)
  expect_identical(before$names, c("e", "c", "a"))
  # a rewritten node is a different object: its schema is inferred afresh
  rewritten <- ch$select
  rewritten$columns <- c("c", "a")
  expect_identical(infer_schema_memo(rewritten)$names, c("c", "a"))
  expect_identical(infer_schema_memo(ch$select)$names, c("e", "c", "a"))
  # the same node object rewired to another input
  rewired <- ch$rename
  rewired$input <- ast_select(ch$mutate, c("a", "e"))
  expect_identical(infer_schema_memo(rewired)$names, c("a", "e2"))
  expect_identical(infer_schema_memo(ch$rename)$names, c("e2", "c", "a"))
})

test_that("a memo scope is closed again, also after an error", {
  expect_null(.schema_memo$plain)
  f <- function() {
    memo_prev <- .schema_memo$plain
    on.exit(.schema_memo$plain <- memo_prev, add = TRUE)
    schema_memo_open()
    stop("boom")
  }
  expect_error(f(), "boom")
  expect_null(.schema_memo$plain)
})

test_that("compute() closes its memo scope when lowering fails", {
  skip_if_no_gpu()
  # a grouped INT32 sum that overflows raises cuplyr_error_sum_overflow at collect
  lz <- tbl_gpu(data.frame(g = c(1L, 1L), v = c(.Machine$integer.max, 1L)), lazy = TRUE) |>
    dplyr::group_by(g) |>
    dplyr::mutate(s = sum(v))
  expect_null(.schema_memo$plain)
  expect_error(dplyr::collect(lz), class = "cuplyr_error_sum_overflow")
  expect_null(.schema_memo$plain)
  expect_error(dplyr::compute(lz), class = "cuplyr_error_sum_overflow")
  expect_null(.schema_memo$plain)
  # the next plan still runs
  ok <- tbl_gpu(data.frame(a = 1:3), lazy = TRUE) |> dplyr::mutate(b = a + 1L) |> dplyr::collect()
  expect_identical(ok$b, 2:4)
})

test_that("the optimizer's set helpers match base union/setdiff/intersect", {
  set.seed(68)
  pool <- c("a", "b", "c", "d", "e", "f")
  for (i in 1:200) {
    x <- if (runif(1) < 0.1) NULL else sample(pool, sample(0:6, 1), replace = TRUE)
    y <- if (runif(1) < 0.1) NULL else sample(pool, sample(0:6, 1), replace = TRUE)
    expect_identical(opt_union(x, y), unique(c(x, y)))
    expect_identical(unname(opt_union(x, y)), union(x, y) %||% NULL)
    expect_identical(opt_diff(x, y), setdiff(x, y))
    expect_identical(opt_intersect(x, y), intersect(x, y))
  }
})

test_that("summarise_output_index() is the identity when no aggregate replaces a key", {
  expect_identical(summarise_output_index(character(), c("m", "n")), 1:2)
  expect_identical(summarise_output_index(c("g", "h"), c("m")), 1:3)
  expect_identical(summarise_output_index("g", character()), 1L)
  # an aggregate named like a key replaces it in place
  expect_identical(summarise_output_index(c("g", "h"), c("m", "g")), c(4L, 2L, 3L))
  expect_identical(summarise_output_index("g", c("g", "m")), c(2L, 3L))
})

# A pass is skipped only when the scan says it cannot change the tree, so the
# staged optimize_segment() must return exactly what running every pass does.
test_that("optimize_segment() equals running all five passes on random plans", {
  set.seed(6868)
  all_passes <- function(ast, req) {
    ast |>
      push_down_projections(req) |>
      fuse_mutates() |>
      prune_dead_columns(req) |>
      push_down_filters() |>
      reorder_filters()
  }
  build <- function() {
    sch <- f68_schema
    node <- ast_source(sch)
    cols <- sch$names
    types <- stats::setNames(sch$types, sch$names)
    n_nodes <- sample(1:7, 1)
    k <- 0L
    for (i in seq_len(n_nodes)) {
      kind <- sample(c("filter", "filter2", "mutate", "mutate_new", "select"), 1)
      num <- cols[types[cols] %in% c("FLOAT64", "INT32")]
      if (length(num) == 0L) break
      cur <- list(names = cols, types = unname(types[cols]))
      if (kind %in% c("filter", "filter2")) {
        p1 <- make_predicate(ir_call(">", list(ir_col(sample(num, 1)), ir_lit(5, "FLOAT64"))), cur)
        preds <- list(p1)
        if (kind == "filter2") {
          preds <- c(preds, list(make_predicate(
            ir_call("<", list(ir_col(sample(num, 1)), ir_lit(50, "FLOAT64"))), cur)))
        }
        node <- ast_filter(node, preds)
      } else if (kind %in% c("mutate", "mutate_new")) {
        k <- k + 1L
        out <- if (kind == "mutate" && length(num) > 0L) sample(num, 1) else paste0("n", k)
        ex <- make_mutate_expr(out, ir_call("+", list(ir_col(sample(num, 1)), ir_lit_from_r(1))), cur)
        node <- ast_mutate(node, list(ex))
        if (!out %in% cols) cols <- c(cols, out)
        types[[out]] <- ex$output_type
      } else {
        keep <- sample(cols, sample(1:length(cols), 1))
        node <- ast_select(node, keep)
        cols <- keep
      }
    }
    node
  }
  for (i in 1:150) {
    plan <- build()
    req <- infer_schema(plan)$names
    expect_identical(optimize_segment(plan, req), all_passes(plan, req), info = paste("plan", i))
    expect_identical(optimize_segment(plan), all_passes(plan, req), info = paste("plan", i, "NULL"))
  }
})

test_that("segment_features() reads what each pass looks for", {
  ch <- f68_chain()
  pred <- ch$filter$predicates
  expect_identical(
    segment_features(ch$src),
    c(mutate_pair = FALSE, column_nodes = FALSE, filter_over = FALSE, filter_pair = FALSE))
  expect_true(segment_features(ch$mutate)[["column_nodes"]])
  expect_true(segment_features(ast_mutate(ch$mutate, ch$mutate$expressions))[["mutate_pair"]])
  expect_true(segment_features(ast_filter(ch$mutate, pred))[["filter_over"]])
  expect_true(segment_features(ast_filter(ch$filter, pred))[["filter_pair"]])
  # both sides of a join are scanned
  join <- ast_join("inner", ch$filter, ast_filter(ch$mutate, pred),
                   by = list(left = "a", right = "a", op = "==", filter = "none"))
  expect_true(segment_features(join)[["filter_over"]])
})

test_that("the cache switch reads the option, then the environment variable", {
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  expect_true(check_schema_cache_on())
  withr::local_options(cuplyr.check_schema_cache = FALSE)
  expect_false(check_schema_cache_on())
})

test_that("the cache switch falls back to CUPLYR_CHECK_SCHEMA_CACHE, read once", {
  withr::local_options(cuplyr.check_schema_cache = NULL)
  saved <- .schema_memo$check_env
  withr::defer(.schema_memo$check_env <- saved)
  for (val in c("1", "true", "TRUE", "yes")) {
    withr::local_envvar(CUPLYR_CHECK_SCHEMA_CACHE = val)
    .schema_memo$check_env <- NULL
    expect_true(check_schema_cache_on(), info = val)
  }
  withr::local_envvar(CUPLYR_CHECK_SCHEMA_CACHE = "0")
  .schema_memo$check_env <- NULL
  expect_false(check_schema_cache_on())
  # read once per session: changing the variable afterwards has no effect
  withr::local_envvar(CUPLYR_CHECK_SCHEMA_CACHE = "1")
  expect_false(check_schema_cache_on())
  # the option wins over the variable
  withr::local_options(cuplyr.check_schema_cache = FALSE)
  .schema_memo$check_env <- NULL
  expect_false(check_schema_cache_on())
})

test_that("current_schema() is the stored schema, checked against the plan on request", {
  skip_if_no_gpu()
  lz <- tbl_gpu(data.frame(a = c(1, 2, 3), g = c("x", "y", "x")), lazy = TRUE) |>
    dplyr::filter(a > 1) |>
    dplyr::mutate(b = a * 2)
  expect_identical(current_schema(lz), unclass(lz)$schema)
  expect_identical(current_schema(lz), infer_schema_full(unclass(lz)$lazy_ops))

  bad <- set_tbl_gpu_field(lz, "schema", list(names = "zzz", types = "INT32"))
  withr::local_options(cuplyr.check_schema_cache = FALSE)
  expect_identical(current_schema(bad)$names, "zzz")      # no check: trusted as stored
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  expect_error(current_schema(bad), class = "cuplyr_error_internal")
  expect_identical(current_schema(lz)$names, c("a", "g", "b"))
})

test_that("a rename over a select (select(new = old)) records the right schema, eager and lazy", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  df <- data.frame(a = c(1, 2), b = c("p", "q"), c = c(TRUE, FALSE))
  for (lazy in c(FALSE, TRUE)) {
    for (pending in c(FALSE, TRUE)) {
      base <- tbl_gpu(df, lazy = lazy)
      if (pending) base <- dplyr::filter(base, a > 0)
      out <- base |> dplyr::select(z = c, y = a)
      expect_identical(current_schema(out)$names, c("z", "y"), info = c(lazy, pending))
      expect_identical(unname(current_schema(out)$types), c("BOOL8", "FLOAT64"), info = c(lazy, pending))
      expect_equal(tibble::as_tibble(dplyr::collect(out)),
                   tibble::tibble(z = c(TRUE, FALSE), y = c(1, 2)), info = c(lazy, pending))
    }
  }
})

test_that("collapse() of a table fresh from tbl_gpu() keeps schema and plan in step", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  lz <- dplyr::collapse(tbl_gpu(data.frame(a = 1:3, f = factor(c("u", "v", "u"))), lazy = TRUE))
  expect_true(has_pending_ops(lz))
  expect_identical(unclass(lz)$schema, infer_schema_full(unclass(lz)$lazy_ops))
  expect_identical(current_schema(dplyr::mutate(lz, b = a + 1L))$names, c("a", "f", "b"))
  expect_equal(dplyr::collect(lz)$f, factor(c("u", "v", "u")))
  zc <- dplyr::collapse(tbl_gpu(tibble::tibble(.rows = 5L), lazy = TRUE))
  expect_identical(current_schema(zc)$names, character())
})

test_that("set operations on lazy tables with pending ops leave their stored schema alone", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.check_schema_cache = TRUE)
  a <- tbl_gpu(data.frame(k = c("a", "b", "c"), v = 1:3), lazy = TRUE) |> dplyr::filter(v > 0L)
  b <- tbl_gpu(data.frame(k = c("b", "c", "d"), v = 2:4), lazy = TRUE) |> dplyr::filter(v > 0L)
  expect_equal(dplyr::collect(dplyr::union_all(a, b))$v, c(1:3, 2:4))
  expect_equal(dplyr::collect(dplyr::intersect(a, b))$v, 2:3)
  expect_equal(dplyr::collect(dplyr::setdiff(a, b))$v, 1L)
  expect_equal(sort(dplyr::collect(dplyr::union(a, b))$v), 1:4)
})

test_that("collect() of a lazy table computes through the stored schema and keeps its metadata", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("u", "v", "u")), x = c(1, 2, 3),
                   t = as.POSIXct("2024-05-01", tz = "America/New_York") + 0:2)
  lz <- tbl_gpu(df, lazy = TRUE) |> dplyr::filter(x > 1) |> dplyr::mutate(y = x + 1)
  got <- dplyr::collect(lz)
  expect_equal(got$f, factor(c("v", "u"), levels = c("u", "v")))
  expect_identical(attr(got$t, "tzone"), "America/New_York")
  expect_identical(got$y, c(3, 4))
})
