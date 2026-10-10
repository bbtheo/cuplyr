# F06 extras beyond the PM's acceptance tests: key-cast plans on the less
# common paths (non-equi with a coerced equi key, cross-by multiple= on
# right/full joins, keyless-join projection pruning, lazy compute()).

like <- function(d, df) {
  if (!is_tbl_gpu(d)) {
    return(tibble::as_tibble(df))
  }
  tbl_gpu(df, lazy = identical(d$exec_mode, "lazy"))
}

both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

test_that("non-equi join with a coerced equi key (merged and keep = TRUE)", {
  skip_if_no_gpu()
  x <- data.frame(id = 1:3, a = c(1, 5, 9))
  y <- data.frame(id = c(1, 2, 3), b = c(2L, 4L, 20L))
  for (join in list(inner_join, left_join, semi_join, anti_join)) {
    both(x, function(d) join(d, like(d, y), by = dplyr::join_by(id, a < b)))
  }
  both(x, function(d) {
    inner_join(d, like(d, y), by = dplyr::join_by(id, a < b), keep = TRUE)
  })
})

test_that("factor x character keys with a group on the key", {
  skip_if_no_gpu()
  fx <- data.frame(id = factor(c("a", "b", "c")), v = 1:3)
  cy <- data.frame(id = c("a", "c", "d"), w = c(10, 30, 40))
  both(fx, function(d) group_by(d, id) |> left_join(like(d, cy), by = "id"))
  both(fx, function(d) {
    left_join(d, like(d, cy), by = "id") |> filter(id == "a") |> select(id, w)
  })
})

test_that("by = character() with multiple= on right/full joins appends unmatched y", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  x <- data.frame(x = 1:2)
  y <- data.frame(y = c(10, 20, 30))
  for (join in list(right_join, full_join)) {
    for (m in c("first", "last", "any")) {
      both(x, function(d) join(d, like(d, y), by = character(), multiple = m))
    }
  }
})

test_that("keyless joins keep a column per side when only one side is selected", {
  skip_if_no_gpu()
  withr::local_options(lifecycle_verbosity = "quiet")
  x <- data.frame(x = 1:3, s = c("a", "b", "c"))
  y <- data.frame(y = c(10, 20))
  both(x, function(d) cross_join(d, like(d, y)) |> select(x))
  both(x, function(d) left_join(d, like(d, y), by = character()) |> select(x))
  both(x, function(d) semi_join(d, like(d, y), by = character()) |> select(x))
})

test_that("an upstream mutate feeds a coerced key through the optimizer", {
  skip_if_no_gpu()
  x <- data.frame(id = 1:4, v = c(1, 2, 3, 4))
  y <- data.frame(id = c(2, 4), w = c("p", "q"))
  both(x, function(d) {
    d |> mutate(id = id + 0L) |> left_join(like(d, y), by = "id") |> select(id, w)
  })
})
