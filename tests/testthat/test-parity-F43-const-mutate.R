# F43 acceptance tests: a lazy filter under a mutate whose surviving
# expressions are all constants must keep its row count. Projection pruning
# must never cut a filter's input down to zero columns.

f43_df <- function() {
  tibble::tibble(
    g = c("a", "a", "b", "b", "c"),
    x = c(1, 2, NA, 4, 5),
    s = c("p", "q", "r", "s", "t")
  )
}

f43_y <- function(d) {
  y <- tibble::tibble(g = c("a", "b"), z = c(10, 20))
  if (inherits(d, "tbl_gpu")) {
    tbl_gpu(y, lazy = identical(.subset2(d, "exec_mode"), "lazy"))
  } else {
    y
  }
}

f43_both <- function(df, p, ...) {
  expect_same_as_dplyr(df, p, ...)
  expect_same_as_dplyr_lazy(df, p, ...)
}

f43_select_nodes <- function(ast) {
  if (is.null(ast) || !is.list(ast)) return(list())
  here <- if (identical(ast$type, "select")) list(ast) else list()
  c(here, f43_select_nodes(ast$input),
    f43_select_nodes(ast$left), f43_select_nodes(ast$right))
}

test_that("F43: the brief's repro, filter + constant mutate + select", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = c(1, 2, 3))
  p <- function(d) d |> filter(x > 1) |> mutate(b = 1) |> select(b)
  f43_both(df, p)
  res <- tbl_gpu(df, lazy = TRUE) |> p() |> collect()
  expect_equal(nrow(res), 2L)
})

test_that("F43: transmute and .keep = 'none' with only constants after filter", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> filter(x > 1) |> transmute(b = 1))
  f43_both(df, function(d) d |> filter(x > 1) |> transmute(b = 1L))
  f43_both(df, function(d) d |> filter(x > 1) |> mutate(b = 1, .keep = "none"))
})

test_that("F43: constants of every type, several of them, after filter", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) {
    d |> filter(x > 1) |> mutate(a = 1L, b = 2.5, c = "k", e = TRUE) |>
      select(a, b, c, e)
  })
  f43_both(df, function(d) d |> filter(s != "q") |> transmute(b = "k"))
})

test_that("F43: chained and compound filters under a constant mutate", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> filter(x > 1) |> filter(x < 5) |> transmute(b = 1))
  f43_both(df, function(d) d |> filter(x > 1, s != "t") |> transmute(b = 1))
  f43_both(df, function(d) d |> filter(x > 1 | g == "a") |> transmute(b = 1))
  f43_both(df, function(d) d |> filter(is.na(x)) |> transmute(b = 1))
})

test_that("F43: filter on a computed or renamed column, then only constants", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> mutate(x2 = x * 2) |> filter(x2 > 2) |> transmute(b = 1))
  f43_both(df, function(d) d |> rename(w = x) |> filter(w > 1) |> transmute(b = 1))
})

test_that("F43: repeated constant mutate + select stays at the filtered row count", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) {
    d |> filter(x > 1) |> mutate(b = 1) |> select(b) |> mutate(c = 2) |> select(c)
  })
})

test_that("F43: joins around a filter, then only constants", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) {
    d |> left_join(f43_y(d), by = "g") |> filter(z > 10) |> mutate(b = 1) |> select(b)
  })
  f43_both(df, function(d) {
    d |> semi_join(f43_y(d), by = "g") |> filter(x > 1) |> transmute(b = 1)
  })
  f43_both(df, function(d) {
    d |> filter(x > 1) |> left_join(f43_y(d), by = "g") |> transmute(b = 1)
  })
})

test_that("F43: constant mutate after arrange/slice/distinct/summarise", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> arrange(x) |> mutate(b = 1) |> select(b))
  f43_both(df, function(d) d |> slice(1:3) |> transmute(b = 1))
  f43_both(df, function(d) d |> distinct(g) |> transmute(b = 1))
  f43_both(df, function(d) {
    d |> group_by(g) |> summarise(m = sum(x, na.rm = TRUE)) |> transmute(b = 1)
  })
  f43_both(df, function(d) d |> filter(x > 1) |> arrange(x) |> transmute(b = 1))
})

test_that("F43: count() and n() after a constant mutate on a filtered table", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> filter(x > 1) |> mutate(b = 1) |> count(b))
  f43_both(df, function(d) d |> filter(x > 1) |> mutate(b = 1) |> summarise(n = n()))
  f43_both(df, function(d) d |> filter(x > 1) |> mutate(b = n()) |> select(b))
})

test_that("F43: zero-row inputs and filters that keep nothing", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> filter(x > 100) |> transmute(b = 1))
  f43_both(df[0, ], function(d) d |> filter(x > 1) |> transmute(b = 1))
  f43_both(df[0, ], function(d) d |> filter(x > 1) |> mutate(b = 1) |> select(b))
})

test_that("F43: grouped input, a constant per group", {
  skip_if_no_gpu()
  df <- f43_df()
  f43_both(df, function(d) d |> group_by(g) |> filter(x > 1) |> mutate(b = 1) |> select(g, b))
  f43_both(df, function(d) {
    d |> group_by(g) |> filter(x > 1) |> mutate(b = 1) |> ungroup() |> select(b)
  })
  f43_both(df, function(d) d |> group_by(g) |> filter(x > 1) |> transmute(b = 1))
  f43_both(df, function(d) {
    d |> group_by(g) |> filter(x >= 1) |> mutate(b = n()) |> ungroup() |> select(b)
  })
})

test_that("F43: compute() then collect keeps the row count", {
  skip_if_no_gpu()
  df <- f43_df()
  p <- function(d) d |> filter(x > 1) |> mutate(b = 1) |> select(b)
  res <- tbl_gpu(df, lazy = TRUE) |> p() |> compute() |> collect()
  expect_equal(tibble::as_tibble(res), p(df))
})

test_that("F43: the optimizer never emits a zero-column select", {
  skip_if_no_gpu()
  df <- f43_df()
  lz <- tbl_gpu(df, lazy = TRUE)
  pipes <- list(
    function(d) d |> filter(x > 1) |> mutate(b = 1) |> select(b),
    function(d) d |> filter(x > 1) |> filter(s != "p") |> transmute(b = 1),
    function(d) d |> mutate(x2 = x * 2) |> filter(x2 > 2) |> transmute(b = 1)
  )
  for (p in pipes) {
    opt <- optimize_ast(.subset2(p(lz), "lazy_ops"))
    for (node in f43_select_nodes(opt)) {
      if (is.null(node$positions)) {
        expect_gt(length(node$columns), 0L)
      }
    }
  }
})
