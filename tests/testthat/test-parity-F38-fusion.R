# F38 acceptance tests, part 1: lazy mutate fusion keeps dplyr's sequential
# semantics. Every expression reads each column as it was at its position in
# the original sequence of mutate() calls, so a fused node must run the
# expressions in their original order (lower node first, then upper node).

f38_df <- function() {
  tibble::tibble(
    g = c("a", "a", "b", "b", "c"),
    x = c(1L, 2L, 3L, NA, 5L),
    y = c(10L, NA, 30L, 40L, 50L),
    z = c(0.5, 1.5, NA, 3.5, 4.5)
  )
}

f38_mutate_nodes <- function(ast) {
  if (is.null(ast)) return(list())
  here <- if (identical(ast$type, "mutate")) list(ast) else list()
  c(here, f38_mutate_nodes(ast$input))
}

f38_optimized <- function(lazy_tbl) {
  optimize_ast(.subset2(lazy_tbl, "lazy_ops"))
}

test_that("F38: the brief's repro, overwrite after read across two mutates", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  p <- function(d) d |> mutate(h = y) |> mutate(k = h, y = x)
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
  out <- collect(p(tbl_gpu(df, lazy = TRUE)))
  expect_identical(out$h, 4:6)
  expect_identical(out$k, 4:6)
  expect_identical(out$y, 1:3)
})

test_that("F38: chains of three or more mutates with overwrites", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(h = y) |> mutate(y = x) |> mutate(k = h + y),
    function(d) d |> mutate(h = y) |> mutate(y = x) |> mutate(x = h) |> mutate(s = x + y),
    function(d) d |> mutate(y = y * 2L) |> mutate(y = y + 1L) |> mutate(y = y * 3L),
    function(d) d |> mutate(tmp = x) |> mutate(x = y) |> mutate(y = tmp) |> select(-tmp),
    function(d) d |> mutate(a = x + 1L) |> mutate(b = a, a = y) |> mutate(c = a + b, b = z)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: overwrites that change the column type", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(y = z * 2) |> mutate(h = y, y = x),
    function(d) d |> mutate(h = y) |> mutate(y = z > 1) |> mutate(k = h * 2L, y = !y),
    function(d) d |> mutate(y = x * 2L) |> mutate(y = 1.5),
    function(d) d |> mutate(y = 1L, y = y + x) |> mutate(y = y / 2)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: a filter reading a column before or after it is overwritten", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(h = y) |> mutate(y = x) |> filter(h > 20L),
    function(d) d |> mutate(h = y) |> mutate(y = x) |> filter(h > 20L, y < 5L),
    function(d) d |> filter(y > 10L) |> mutate(y = x) |> filter(y < 5L),
    function(d) d |> mutate(y = x * 10L) |> filter(y > 10L) |> mutate(k = y, y = 0L),
    function(d) d |> mutate(h = y) |> filter(!is.na(x)) |> mutate(k = h, y = x) |> filter(k > 10L)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: .keep / .before / .after with fused overwrites", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(h = y) |> mutate(k = h, y = x, .keep = "used"),
    function(d) d |> mutate(h = y) |> mutate(k = h, y = x, .keep = "unused"),
    function(d) d |> mutate(h = y) |> mutate(k = h, y = x, .keep = "none"),
    function(d) d |> mutate(h = y) |> mutate(k = h, y = x, .before = 1),
    function(d) d |> mutate(h = y) |> mutate(k = h, y = x, .after = g),
    function(d) d |> mutate(h = y, .keep = "unused") |> mutate(y = x) |> mutate(k = h + y)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: grouped and .by pipelines with fused overwrites", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> group_by(g) |> mutate(h = y) |> mutate(k = h, y = x),
    function(d) d |> group_by(g) |> mutate(h = y) |> mutate(y = x, k = h - mean(h, na.rm = TRUE)),
    function(d) d |> mutate(h = y, .by = g) |> mutate(k = h, y = x, .by = g)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: dead-column pruning keeps the read-before-overwrite value", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(h = y) |> mutate(y = x) |> select(h),
    function(d) d |> mutate(y = x) |> mutate(h = y) |> select(h),
    function(d) d |> mutate(h = y) |> mutate(y = x) |> mutate(k = h) |> select(g, k)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: zero-row and all-NA inputs", {
  skip_if_no_gpu()
  empty <- f38_df()[0, ]
  all_na <- tibble::tibble(x = c(NA_integer_, NA_integer_), y = c(NA_integer_, NA_integer_))
  p <- function(d) d |> mutate(h = y) |> mutate(k = h, y = x) |> mutate(m = k + y)
  expect_same_as_dplyr(empty, p)
  expect_same_as_dplyr_lazy(empty, p)
  expect_same_as_dplyr(all_na, p)
  expect_same_as_dplyr_lazy(all_na, p)
})

test_that("F38: long chains past the fusion guards stay correct", {
  skip_if_no_gpu()
  df <- f38_df()
  p <- function(d) {
    d |>
      mutate(a = x, b = y, c = z) |>
      mutate(y = a + 1L, d = b, e = c) |>
      mutate(x = y, f = d, a = b) |>
      mutate(s = a + x + f, y = 0L)
  }
  expect_same_as_dplyr(df, p)
  expect_same_as_dplyr_lazy(df, p)
})

test_that("F38: a fused node runs its expressions in the original order", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  lz <- tbl_gpu(df, lazy = TRUE)

  # Overwrite after read: still fused into one node, original order kept.
  nodes <- f38_mutate_nodes(f38_optimized(lz |> mutate(h = y) |> mutate(k = h, y = x)))
  expect_length(nodes, 1L)
  expect_identical(
    vapply(nodes[[1]]$expressions, `[[`, character(1), "output_col"),
    c("h", "k", "y")
  )

  # Independent expressions are not reordered either.
  nodes <- f38_mutate_nodes(f38_optimized(lz |> mutate(a = x, b = y) |> mutate(c = b, d = x)))
  expect_length(nodes, 1L)
  expect_identical(
    vapply(nodes[[1]]$expressions, `[[`, character(1), "output_col"),
    c("a", "b", "c", "d")
  )

  # Three mutates with an overwrite in the middle fuse into one node.
  nodes <- f38_mutate_nodes(f38_optimized(
    lz |> mutate(h = y) |> mutate(y = x) |> mutate(k = h + y)
  ))
  expect_length(nodes, 1L)
  expect_identical(
    vapply(nodes[[1]]$expressions, `[[`, character(1), "output_col"),
    c("h", "y", "k")
  )
})

test_that("F38: chains without an overwrite dependency keep fusing", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6, z = 7:9)
  lz <- tbl_gpu(df, lazy = TRUE)
  no_dep <- lz |> mutate(a = x + 1L) |> mutate(b = y * 2L) |> mutate(c = z - 1L)
  raw_dep <- lz |> mutate(a = x + y) |> mutate(b = a * 2L) |> mutate(c = b - z)
  expect_length(f38_mutate_nodes(f38_optimized(no_dep)), 1L)
  expect_length(f38_mutate_nodes(f38_optimized(raw_dep)), 1L)
})

# Round-1 review B1: dead-column pruning drops the expression that created a
# column when a later expression overwrites it, so the column's position
# moves while compute() still names the result by the unoptimized schema.
# No trailing select() here: select() picks by name and would hide it.
test_that("F38: overwriting a column created earlier in the chain keeps names and data aligned", {
  skip_if_no_gpu()
  df <- f38_df()
  pipes <- list(
    function(d) d |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L),
    function(d) d |> mutate(a = x, b = y) |> mutate(a = b * 10L),
    function(d) d |> mutate(a = x, b = y, a = b * 10L),
    function(d) d |> mutate(flag = x > 1L) |> mutate(score = z * 2) |> mutate(flag = score > 4),
    function(d) d |> mutate(a = x) |> filter(!is.na(y)) |> mutate(b = y) |> mutate(a = b + 1L),
    function(d) d |> mutate(a = x, b = y) |> mutate(c = b, d = x),
    function(d) d |> group_by(g) |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L),
    function(d) d |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L, .keep = "unused"),
    function(d) d |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L, .before = 1)
  )
  for (p in pipes) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("F38: a computed lazy table with a pruned overwrite feeds later verbs correctly", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:3, y = 4:6)
  res <- tbl_gpu(df, lazy = TRUE) |>
    mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L) |>
    compute() |>
    mutate(s = a - b) |>
    collect()
  expect_identical(names(res), c("x", "y", "a", "b", "s"))
  expect_identical(res$a, 5:7)
  expect_identical(res$b, 4:6)
  expect_identical(res$s, rep(1L, 3))
})

# Round-2 review N4/N5: pin the pruning rule itself (the compute() safety net
# would otherwise hide a reorder), and cover joins and select/relocate between
# the creating mutate and the overwrite.
test_that("F38: the optimized plan keeps the unoptimized column order", {
  skip_if_no_gpu()
  df <- f38_df()
  y_df <- tibble::tibble(g = c("a", "b", "c"), w = c(7L, 8L, 9L))
  pipes <- list(
    function(d) d |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L),
    function(d) d |> mutate(a = x, b = y) |> mutate(a = b * 10L),
    function(d) d |> mutate(a = x) |> filter(!is.na(y)) |> mutate(b = y) |> mutate(a = b + 1L),
    function(d) d |> select(y, x) |> mutate(y = x * 2L),
    function(d) d |> mutate(a = x) |> relocate(a, .before = 1) |> mutate(b = y) |> mutate(a = b + 1L)
  )
  for (p in pipes) {
    ops <- .subset2(p(tbl_gpu(df, lazy = TRUE)), "lazy_ops")
    expect_identical(infer_schema(optimize_ast(ops))$names, infer_schema(ops)$names)
    expect_same_as_dplyr_lazy(df, p)
  }
  # y must live on the same source as d: a tbl_gpu with the same exec mode.
  y_of <- function(d) {
    if (!inherits(d, "tbl_gpu")) return(y_df)
    tbl_gpu(y_df, lazy = identical(.subset2(d, "exec_mode"), "lazy"))
  }
  join_pipes <- list(
    function(d) d |> mutate(a = x) |> mutate(b = y) |> left_join(y_of(d), by = "g") |> mutate(a = b + w),
    function(d) d |> mutate(a = x) |> mutate(b = y) |> mutate(a = b + 1L) |> inner_join(y_of(d), by = "g"),
    function(d) d |> inner_join(y_of(d) |> mutate(v = w) |> mutate(u = w) |> mutate(v = u * 2L), by = "g")
  )
  for (p in join_pipes) {
    expect_same_as_dplyr_lazy(df, p, arrange_by = c("g", "x"))
    expect_same_as_dplyr(df, p, arrange_by = c("g", "x"))
  }
})
