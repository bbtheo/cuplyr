# F35 acceptance tests: lazy projection pruning through a join must keep the
# suffix semantics. Pruning a column that is the collision partner of a kept
# column would make the kept column lose its suffix (`v.x` -> `v`).
# Helpers are in helper-parity-F35.R.

sx <- tibble::tibble(k = c(1L, 2L, 3L, NA), v = c(1, 2, 3, 4), a = c(5, 6, 7, 8))
sy <- tibble::tibble(k = c(1L, 2L, 2L, 4L), v = c(9, 8, 7, 6), b = c("p", "q", "r", "s"))

test_that("left_join then select a suffixed x column", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k") |> select(k, v.x)
  })
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k") |> select(v.x)
  })
})

test_that("inner/right/full join then select a suffixed y column", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> inner_join(y, by = "k") |> select(k, v.y)
  })
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> right_join(y, by = "k") |> select(k, b, v.y)
  })
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> full_join(y, by = "k") |> select(v.x, k)
  })
})

test_that("mutate on a suffixed column, then select", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k") |> mutate(z = v.x * 2) |> select(k, z)
  })
})

test_that("rename of a suffixed column, then select", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k") |> rename(vx = v.x) |> select(k, vx)
  })
})

test_that("custom suffix and keep = TRUE keys survive pruning", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k", suffix = c("_l", "_r")) |> select(k, v_l)
  })
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> inner_join(y, by = "k", keep = TRUE) |> select(k.x, a)
  })
})

test_that("a double suffix (v.x already in x) survives pruning", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:3, v = c(1, 2, 3), v.x = c(4, 5, 6))
  y <- tibble::tibble(k = 1:3, v = c(7, 8, 9))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> left_join(y, by = "k") |> select(k, v.x)
  })
  f35_expect_parity(list(x, y), function(x, y) {
    x |> left_join(y, by = "k") |> select(k, v.x.x)
  })
})

test_that("grouped summarise over a suffixed column", {
  skip_if_no_gpu()
  f35_expect_parity(list(sx, sy), function(x, y) {
    x |> left_join(y, by = "k") |> group_by(k) |> summarise(s = sum(v.x)) |>
      arrange(k)
  })
})

test_that("pruning through a chain of joins keeps suffixes", {
  skip_if_no_gpu()
  z <- tibble::tibble(k = c(1L, 2L, 3L), a = c(21, 22, 23), c = c(TRUE, FALSE, NA))
  f35_expect_parity(list(sx, sy, z), function(x, y, z) {
    x |> left_join(y, by = "k") |> inner_join(z, by = "k") |>
      select(k, a.x, v.y)
  })
  f35_expect_parity(list(sx, sy, z), function(x, y, z) {
    x |> left_join(y, by = "k") |> inner_join(z, by = "k") |>
      filter(a.y > 21) |> select(k, v.x, c)
  })
})

# Added after review round 1 (B1): a filter on the pruned side must not
# reorder its columns, since stacked suffixes depend on column order.
test_that("double suffix with a filter on the pruned side and out-of-order select", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:3, v = c(1, 2, 3), v.x = c(4, 5, 6), a = c(1, 1, 1))
  y <- tibble::tibble(k = 1:3, v = c(7, 8, 9))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> filter(a > 0) |> left_join(y, by = "k") |> select(k, v.x.x, v.x)
  })
  f35_expect_parity(list(x, y), function(x, y) {
    x |> mutate(b = a + 1) |> filter(b > 0) |> left_join(y, by = "k") |>
      select(v.x.x, k, v.x)
  })
})

# Added after review round 2 (B1): a filter whose input chain is cut at a
# barrier by the optimizer has no complete input schema; the select it gets
# must still keep every needed column.
test_that("filter above a barrier, after a join and on a plain table", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:3, v = c(1, 2, 3), v.x = c(4, 5, 6), a = c(1, 1, 1))
  y <- tibble::tibble(k = 1:3, v = c(7, 8, 9))
  f35_expect_parity(list(x, y), function(x, y) {
    x |> left_join(y, by = "k") |> arrange(desc(k)) |> filter(a > 0) |>
      select(k, v.x.x, v.x)
  })
  df <- tibble::tibble(x = c(3, 1, 2), a = c(10, 20, 30))
  f35_expect_parity(list(df), function(d) {
    d |> arrange(x) |> mutate(y = x * 2, z = x + 1) |> filter(y > 2) |> select(-y)
  })
})
