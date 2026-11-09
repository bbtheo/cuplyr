# F99 round 3: under a barrier's narrower requirement the optimizer removes
# columns only. A mutate expression nothing reads stays (optimize_segment(
# keep_mutates = TRUE)), and the settled-chain shortcut must give the plan the
# full passes give.

plain_ops <- function(x) .subset2(x, "lazy_ops")

mutate_outputs <- function(node) {
  out <- character()
  while (!is.null(node)) {
    if (identical(node$type, "mutate")) {
      out <- c(out, vapply(node$expressions, `[[`, character(1), "output_col"))
    }
    node <- node$input
  }
  out
}

test_that("keep_mutates: the settled shortcut gives the plan the passes give", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 2, 3), b = c(4, 5, 6), c = c(7, 8, 9))
  g <- tbl_gpu(df, lazy = TRUE)
  plans <- list(
    g |> dplyr::mutate(d = a + 1),
    g |> dplyr::filter(a > 0) |> dplyr::mutate(d = a + b + c),
    g |> dplyr::mutate(d = a + 1) |> dplyr::mutate(e = d + b + c),
    g |> dplyr::mutate(d = a + b + c) |> dplyr::filter(d > 0),
    g |> dplyr::filter(a > 0) |> dplyr::filter(b > 0) |> dplyr::mutate(d = c + 1),
    g |> dplyr::mutate(a = b * 2, d = a + c)
  )
  no_shortcut <- function(plan, req) {
    local_mocked_bindings(settled_chain = function(ast, required_cols) 0L)
    optimize_segment(plan, req, keep_mutates = TRUE)
  }
  for (i in seq_along(plans)) {
    plan <- plain_ops(plans[[i]])
    out <- infer_schema(plan)$names
    subsets <- unlist(lapply(seq_along(out), function(k) utils::combn(out, k, simplify = FALSE)),
                      recursive = FALSE)
    for (req in subsets) {
      kept <- optimize_segment(plan, req, keep_mutates = TRUE)
      expect_identical(kept, no_shortcut(plan, req), info = paste(i, toString(req)))
      # every expression stays, whatever the requirement
      expect_setequal(mutate_outputs(kept), mutate_outputs(plan))
    }
  }
})

test_that("a dead replace-in-place below a barrier matches dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(x1 = c(3, 1, 2), x2 = c(6, 4, 5), x3 = c(9, 7, 8))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> dplyr::mutate(x1 = x2 * 2) |> dplyr::arrange(x2) |> dplyr::select(x2, x3)
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> dplyr::mutate(x1 = x2 * 2) |> dplyr::summarise(s = sum(x2))
  })
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> dplyr::mutate(x3 = x2 + 1, z = x3 * 2) |> dplyr::arrange(x2) |> dplyr::select(x2, x1)
  })
  # read after the replacement: not dead
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> dplyr::mutate(x1 = x2 * 2) |> dplyr::arrange(x1) |> dplyr::select(x3)
  })
})

test_that("a dead rowwise min() below a barrier keeps dplyr's warning", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(NA_real_, 1), y = c(NA_real_, 2), z = c(2, 1))
  pipe <- function(d) {
    d |> dplyr::rowwise() |> dplyr::mutate(m = min(c(x, y), na.rm = TRUE)) |> dplyr::ungroup()
  }
  msg <- "no non-missing arguments to min"
  g <- tbl_gpu(df, lazy = TRUE)
  expect_warning(res <- pipe(g) |> dplyr::arrange(z) |> dplyr::select(z) |> dplyr::collect(), msg)
  expect_equal(res$z, c(1, 2))
  expect_warning(res <- pipe(g) |> dplyr::summarise(n = n()) |> dplyr::collect(), msg)
  expect_equal(res$n, 2L)
  expect_warning(res <- pipe(g) |> dplyr::filter(z > 0) |> dplyr::summarise(s = sum(z)) |> dplyr::collect(), msg)
  expect_equal(res$s, 3)
})

test_that("an empty mutate under a narrower requirement still lowers", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 2, 3), b = c(4, 5, 6), c = c(7, 8, 9))
  g <- tbl_gpu(df, lazy = TRUE)
  src <- input_node(g)
  agg <- .subset2(input_node(g |> dplyr::summarise(s = sum(a))), "aggregations")
  plan <- ast_summarise(ast_mutate(src, list()), agg, character())
  ptr <- lower_and_execute(optimize_ast(plan), .subset2(g, "ptr"))
  expect_equal(gpu_dim(ptr), c(1L, 1L))
})
