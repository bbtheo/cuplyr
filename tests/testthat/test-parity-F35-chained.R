# F35 acceptance tests: lazy joins whose inputs are themselves lazy joins
# (or lazy pipelines) must lower correctly, and lazy == eager == dplyr.
#
# Every case runs the same pipeline function on plain tibbles (the dplyr
# oracle), on eager tbl_gpu inputs, on lazy tbl_gpu inputs, and on a mix
# (odd-numbered inputs lazy, even-numbered eager).

# Helpers f35_inputs()/f35_expect_parity() live in helper-parity-F35.R.

f35_x <- tibble::tibble(k = c(1L, 2L, 3L, 4L, NA), a = c(10, 20, 30, 40, 50),
                        v = c(1, 2, 3, 4, 5))
f35_y <- tibble::tibble(k = c(1L, 1L, 2L, 5L, NA), b = c("p", "q", "r", "s", "t"),
                        v = c(11, 12, 13, 14, 15))
f35_z <- tibble::tibble(k = c(1L, 2L, 3L, 6L), c = c(TRUE, FALSE, NA, TRUE),
                        a = c(21, 22, 23, 24))

test_that("3-way chain: left_join then inner_join", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> left_join(y, by = "k") |> inner_join(z, by = "k")
  })
})

test_that("3-way chain: inner, full and right joins", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> inner_join(y, by = "k") |> full_join(z, by = "k")
  })
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> right_join(y, by = "k") |> left_join(z, by = "k")
  })
})

test_that("4-way chain including semi and anti joins", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> left_join(y, by = "k") |> semi_join(z, by = "k") |>
      left_join(z, by = "k")
  })
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> left_join(z, by = "k") |> anti_join(y, by = "k")
  })
})

test_that("join whose y is a lazy join", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_z, f35_x, f35_y), function(z, x, y) {
    z |> left_join(x |> inner_join(y, by = "k"), by = "k")
  })
})

test_that("join with a join on both sides", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_z, f35_y), function(x, z, y) {
    (x |> inner_join(z, by = "k")) |>
      full_join(y |> left_join(z, by = "k"), by = "k")
  })
})

test_that("join whose y is a lazy filter/mutate pipeline, chained", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |>
      left_join(y |> filter(v > 11) |> mutate(w = v * 2), by = "k") |>
      inner_join(z |> select(k, c), by = "k")
  })
})

test_that("chain followed by filter, mutate, select and arrange", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |>
      left_join(y, by = "k") |>
      inner_join(z, by = "k") |>
      filter(v.x > 1, a.y > 21) |>
      mutate(r = a.x + v.y) |>
      select(k, b, r, c) |>
      arrange(k, b)
  })
})

test_that("chain with a filter on a column from the inner join's y side", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> left_join(y, by = "k") |> inner_join(z, by = "k") |>
      filter(b %in% c("q", "r")) |> select(k, b, a.y)
  })
})

test_that("self-join chains", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x), function(x) {
    x |> left_join(x, by = "k") |> left_join(x, by = "k")
  })
})

test_that("chain on grouped x keeps x's groups", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x, f35_y, f35_z), function(x, y, z) {
    x |> group_by(k) |> left_join(y, by = "k") |> left_join(z, by = "k")
  })
})

test_that("chain with a zero-row input", {
  skip_if_no_gpu()
  f35_expect_parity(list(f35_x[0, ], f35_y, f35_z), function(x, y, z) {
    x |> left_join(y, by = "k") |> left_join(z, by = "k")
  })
  f35_expect_parity(list(f35_x, f35_y[0, ], f35_z), function(x, y, z) {
    x |> left_join(y, by = "k") |> inner_join(z, by = "k")
  })
})

test_that("a chained lazy join can be collected twice and composed after compute()", {
  skip_if_no_gpu()
  want <- f35_x |> left_join(f35_y, by = "k") |> inner_join(f35_z, by = "k")
  lz <- tbl_gpu(f35_x, lazy = TRUE) |>
    left_join(tbl_gpu(f35_y, lazy = TRUE), by = "k") |>
    inner_join(tbl_gpu(f35_z, lazy = TRUE), by = "k")
  expect_equal(tibble::as_tibble(collect(lz)), want)
  expect_equal(tibble::as_tibble(collect(lz)), want)

  done <- compute(lz)
  more <- as_lazy(done) |> left_join(tbl_gpu(f35_y, lazy = TRUE), by = "k",
                                     relationship = "many-to-many")
  expect_equal(
    tibble::as_tibble(collect(more)),
    want |> left_join(f35_y, by = "k", relationship = "many-to-many")
  )
})

test_that("show_query() renders a chained lazy join", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f35_x, lazy = TRUE) |>
    left_join(tbl_gpu(f35_y, lazy = TRUE), by = "k") |>
    inner_join(tbl_gpu(f35_z, lazy = TRUE), by = "k") |>
    filter(b == "q")
  out <- paste(capture.output(show_query(lz)), collapse = "\n")
  expect_match(out, "left join")
  expect_match(out, "inner join")
})

# F35 goal 4: `.drop = FALSE` must survive a join and the verbs after it.
test_that(".drop = FALSE survives a join and the next verbs, eager and lazy", {
  skip_if_no_gpu()
  fx <- tibble::tibble(f = factor(c("a", "a", "b"), levels = c("a", "b", "c")),
                       k = 1:3)
  fy <- tibble::tibble(k = 1:3, v = 4:6)
  f35_expect_parity(list(fx, fy), function(x, y) {
    x |> group_by(f, .drop = FALSE) |> left_join(y, by = "k") |>
      mutate(v2 = v + 1L)
  })
  f35_expect_parity(list(fx, fy), function(x, y) {
    x |> group_by(f, .drop = FALSE) |> inner_join(y, by = "k") |>
      filter(v > 4L) |> select(f, v)
  })
  f35_expect_parity(list(fx, fy, fy), function(x, y, y2) {
    x |> group_by(f, .drop = FALSE) |> left_join(y, by = "k") |>
      left_join(y2, by = "k") |> mutate(s = v.x + v.y)
  })
})

# Added after review round 1 (B2): group columns of a summarise above an inner
# join are output names; they must not be required from the join's sides.
test_that("join whose y is join |> group_by |> summarise with a filtered inner y", {
  skip_if_no_gpu()
  x <- tibble::tibble(k = 1:4, g = c("a", "a", "b", "b"))
  y <- tibble::tibble(k = 1:4, w = c(1, 2, 3, 4), ok = c(TRUE, FALSE, TRUE, TRUE))
  z <- tibble::tibble(g = c("a", "b", "c"), h = c(10, 20, 30))
  f35_expect_parity(list(z, x, y), function(z, x, y) {
    z |> left_join(
      x |> inner_join(y |> filter(ok), by = "k") |> group_by(g) |>
        summarise(s = sum(w)),
      by = "g"
    )
  })
})
