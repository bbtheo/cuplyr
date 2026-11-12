# F38 acceptance tests, part 2: a filter() that uses cur_group_id() and then
# falls back to the CPU must not leak the hidden `..cur_group_id..` helper
# column, and must work on lazy grouped / .by tables.

f38_cg_num <- function() tibble::tibble(g = c(1, 1, 2, 3), x = c(1, 2, 3, 4))
f38_cg_chr <- function() tibble::tibble(g = c("a", "a", "b", "c"), x = c(1, 2, 3, 4))

# identity() is not in the IR, so a dot using it sends the whole filter()
# call to the CPU fallback.
f38_cg_pipes <- list(
  grouped = function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() == 1, identity(x) > 0),
  grouped_and = function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() <= 2 & identity(x) > 1),
  ungrouped = function(d) d |> filter(dplyr::cur_group_id() == 1, identity(x) > 1),
  by = function(d) d |> filter(dplyr::cur_group_id() == 1, identity(x) > 0, .by = g),
  if_any = function(d) d |> group_by(g) |> filter(dplyr::if_any(.cols = x, .fns = ~ identity(.x) > 1), dplyr::cur_group_id() >= 1),
  prior_mutate = function(d) d |> mutate(x2 = x * 2) |> group_by(g) |> filter(dplyr::cur_group_id() <= 2, identity(x2) > 2),
  prior_mutate_by = function(d) d |> mutate(x2 = x * 2) |> filter(dplyr::cur_group_id() >= 2, identity(x2) > 0, .by = g)
)

test_that("F38: filter() CPU fallback with cur_group_id() matches dplyr (eager)", {
  skip_if_no_gpu()
  for (nm in names(f38_cg_pipes)) {
    p <- f38_cg_pipes[[nm]]
    expect_same_as_dplyr(f38_cg_num(), p, label = nm)
    expect_same_as_dplyr(f38_cg_chr(), p, label = nm)
  }
})

test_that("F38: filter() CPU fallback with cur_group_id() matches dplyr (lazy)", {
  skip_if_no_gpu()
  for (nm in names(f38_cg_pipes)) {
    p <- f38_cg_pipes[[nm]]
    expect_same_as_dplyr_lazy(f38_cg_num(), p, label = nm)
    expect_same_as_dplyr_lazy(f38_cg_chr(), p, label = nm)
  }
})

test_that("F38: the helper column never appears in a fallback filter() result", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    for (nm in names(f38_cg_pipes)) {
      out <- collect(f38_cg_pipes[[nm]](tbl_gpu(f38_cg_num(), lazy = lazy)))
      expect_false("..cur_group_id.." %in% names(out), label = paste(nm, lazy))
    }
  }
})

test_that("F38: fallback filter() with cur_group_id() keeps groups and continues", {
  skip_if_no_gpu()
  p <- function(d) {
    d |>
      group_by(g) |>
      filter(dplyr::cur_group_id() <= 2, identity(x) > 0) |>
      mutate(n = x + 1)
  }
  expect_same_as_dplyr(f38_cg_num(), p)
  expect_same_as_dplyr_lazy(f38_cg_num(), p)
})

test_that("F38: fallback filter() with cur_group_id() on zero rows and NA keys", {
  skip_if_no_gpu()
  empty <- f38_cg_num()[0, ]
  na_key <- tibble::tibble(g = c(1, NA, 2, NA), x = c(1, 2, 3, 4))
  p <- function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() >= 1, identity(x) > 0)
  q <- function(d) d |> filter(dplyr::cur_group_id() == 2, identity(x) > 0, .by = g)
  for (df in list(empty, na_key)) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
    expect_same_as_dplyr(df, q)
    expect_same_as_dplyr_lazy(df, q)
  }
})

test_that("F38: GPU-native cur_group_id() filter (no fallback) still matches", {
  skip_if_no_gpu()
  pipes <- list(
    function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() == 2),
    function(d) d |> filter(dplyr::cur_group_id() == 1, x > 1, .by = g),
    function(d) d |> group_by(g) |> filter(dplyr::cur_group_id() <= 2, x > mean(x))
  )
  for (p in pipes) {
    expect_same_as_dplyr(f38_cg_num(), p)
    expect_same_as_dplyr_lazy(f38_cg_num(), p)
  }
})
