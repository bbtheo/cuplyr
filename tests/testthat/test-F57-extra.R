# F57 coder tests: pieces of the perf work the acceptance tests don't pin.
# - bare copies in gpu_mutate_expr() are alias slots of the evaluation context
#   (shared columns), including chains of copies, swaps and copies that later
#   dots read or overwrite
# - an eager chain (push_chain()) keeps groups, attributes and schema metadata
#   exactly like the one-node-at-a-time pushes
# - the plain-vector set operations equal base R's
# - mutate_needs_select() only says "no select" when the select is a no-op

test_that("set_union()/set_diff()/set_intersect() equal base R on plain vectors", {
  cases <- list(
    list(character(), character()),
    list(c("a", "b", "a"), c("b", "c")),
    list(c(x = "a", y = "b"), c(z = "b", w = "c")),
    list(NULL, c("a", "b")),
    list(c("a", "b"), NULL),
    list(NULL, NULL),
    list(1:5, c(2, 4, 6)),
    list(c(1.5, 2), 2L),
    list(c(TRUE, NA), c(NA, FALSE)),
    list(c("a", NA, "b"), c(NA, "c"))
  )
  for (cs in cases) {
    x <- cs[[1]]
    y <- cs[[2]]
    expect_identical(set_union(x, y), base::union(x, y))
    expect_identical(set_diff(x, y), base::setdiff(x, y))
    expect_identical(set_intersect(x, y), base::intersect(x, y))
  }
  # Not plain: the original (dplyr/base) call decides.
  f1 <- factor(c("a", "b"))
  f2 <- factor(c("b", "c"))
  expect_identical(set_union(f1, f2), base::union(f1, f2))
  expect_identical(set_diff(f1, f2), base::setdiff(f1, f2))
  expect_identical(set_intersect(f1, f2), base::intersect(f1, f2))
  expect_identical(set_union(list("a", 1), list(1, "b")), base::union(list("a", 1), list(1, "b")))
})

test_that("mutate_needs_select() only skips the select when it is a no-op", {
  e <- function(out) list(list(output_col = out))
  with_order <- function(ex, order) {
    attr(ex, "dot_order") <- order
    ex
  }
  nq <- null_quo
  ok <- function(ex, keep = "all", drop = character(), dropped = character(), before = nq) {
    mutate_needs_select(c("a", "b"), ex, keep, before, nq, drop, dropped)
  }
  expect_false(ok(with_order(c(e("c"), e("a")), c("c", "a"))))
  expect_false(ok(with_order(e("a"), "a")))
  expect_true(ok(with_order(e("c"), "c"), keep = "used"))
  expect_true(ok(with_order(e("c"), "c"), drop = "..tmp.."))
  expect_true(ok(with_order(e("c"), "c"), dropped = "b"))
  expect_true(ok(with_order(e("c"), "c"), before = rlang::quo(a)))
  # `d = NULL` first, then `c`, then `d` again: dplyr puts d before c
  expect_true(ok(with_order(c(e("c"), e("d")), c("d", "c"))))
})

test_that("bare copies in mutate() are shared slots that later dots read and overwrite", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1, NA, 3, 4), y = c(5, 6, NA, 8), s = c("a", NA, "c", "d"),
    stringsAsFactors = FALSE
  )
  pipelines <- list(
    function(t) mutate(t, a = x, b = a, c = b),
    function(t) mutate(t, a = x, b = a + 1),
    function(t) mutate(t, a = x, x = 2),
    function(t) mutate(t, x = y, y = x),
    function(t) mutate(t, a = x, a = a * 2),
    function(t) mutate(t, a = x, b = x, a = a + b),
    function(t) mutate(t, s2 = s, s3 = s2, x = x + 1),
    function(t) mutate(t, a = x, .keep = "none"),
    function(t) mutate(t, a = x, b = y, .before = 1),
    function(t) transmute(t, a = x, b = a, y),
    function(t) mutate(t, a = x) |> mutate(a = a + 1) |> select(x, a)
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("an eager chain keeps groups, attributes and schema metadata", {
  skip_if_no_gpu()
  df <- data.frame(
    g = c(1L, 1L, 2L, 2L), x = c(1, 2, 3, 4),
    f = factor(c("u", "v", "u", "v")),
    t = as.POSIXct("2024-03-01 12:00:00", tz = "America/New_York") + 0:3
  )
  attr(df, "note") <- "kept"

  gt <- tbl_gpu(df, lazy = FALSE)
  res <- mutate(gt, r = cumsum(x), f2 = f, t2 = t)
  expect_identical(attr(res, "note"), "kept")
  expect_identical(.subset2(res, "exec_mode"), "eager")
  expect_null(.subset2(res, "lazy_ops"))
  out <- collect(res)
  expect_identical(attr(out, "note"), "kept")
  expect_s3_class(out$f2, "factor")
  expect_identical(attr(out$t2, "tzone"), "America/New_York")
  expect_equal(out$r, cumsum(df$x))

  # grouped: the groups survive, no attributes cross from a grouped input
  gg <- group_by(gt, g)
  rg <- mutate(gg, r = cumsum(x), .keep = "used")
  expect_identical(group_vars(rg), "g")
  expect_identical(.subset2(rg, "exec_mode"), "eager")
  expect_equal(tibble::as_tibble(collect(rg)),
               tibble::as_tibble(dplyr::mutate(dplyr::group_by(df, g), r = cumsum(x), .keep = "used")))
})

test_that("push_chain() on a table it can't run reports the chain it ran", {
  skip_if_no_gpu()
  gt <- tbl_gpu(data.frame(x = c(1, 2, 3)), lazy = FALSE)
  # Schema work in push_op() doesn't see the missing column; lowering fails on
  # the second select, and the error names both nodes of the chain.
  bad <- function(t) {
    t1 <- push_op(t, ast_select(input_node(t), "x"), carry = FALSE)
    push_op(t1, ast_select(input_node(t1), c("x", "nope")), carry = FALSE)
  }
  expect_error(push_chain(gt, bad), "'select\\+select' failed")
})

test_that("a lazy plan around an eager chain gives the same result", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1L, 2L, 1L, 2L, 1L), x = c(5, 3, 8, 1, 9))
  p <- function(t) t |> mutate(r = dplyr::lag(x), s = x * 2, .keep = "unused") |> filter(!is.na(r))
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  q <- function(t) t |> group_by(g) |> mutate(r = cumsum(x), m = mean(x)) |> ungroup()
  expect_same_as_dplyr(df, q)
  expect_same_as_dplyr_lazy(df, q)
})

test_that("gpu_gc() reports a freed amount and a message under every resource", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(x = runif(1e6)), lazy = FALSE)
  rm(g)
  res <- gpu_gc(verbose = FALSE)
  expect_true(is.numeric(res$freed_bytes))
  expect_equal(res$freed_gb, res$freed_bytes / 1e9)
  expect_true(all(c("resource", "pool_bytes", "pool_used_bytes") %in% names(res$after)))
})
