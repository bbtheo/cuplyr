# Non-equi join conditions on compatible but different key types (F06 round 2),
# plus edge cases of the temp-copy cast path that the acceptance tests don't hit.

nq_both <- function(x, y, pipeline, arrange_by = NULL) {
  oracle <- pipeline(tibble::as_tibble(x), tibble::as_tibble(y))
  for (lazy in c(FALSE, TRUE)) {
    out <- tibble::as_tibble(collect(pipeline(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy))))
    if (!is.null(arrange_by)) {
      out <- dplyr::arrange(out, dplyr::across(dplyr::all_of(arrange_by)))
      oracle <- dplyr::arrange(oracle, dplyr::across(dplyr::all_of(arrange_by)))
    }
    expect_equal(out, oracle, info = paste("lazy =", lazy))
  }
}

test_that("non-equi Date vs POSIXct, the reviewer's repro", {
  skip_if_no_gpu()
  x <- data.frame(d = as.Date(c("2020-01-01", "2020-01-03")), a = 1:2)
  y <- data.frame(t = as.POSIXct("2020-01-02 00:00:00", tz = "UTC"), b = 10)
  nq_both(x, y, function(x, y) inner_join(x, y, by = dplyr::join_by(d >= t)))
  nq_both(x, y, function(x, y) left_join(x, y, by = dplyr::join_by(d >= t)))
})

test_that("non-equi logical vs integer, the reviewer's repro", {
  skip_if_no_gpu()
  x <- data.frame(l = c(TRUE, FALSE))
  y <- data.frame(i = c(0L, 1L))
  nq_both(x, y, function(x, y) inner_join(x, y, by = dplyr::join_by(l <= i), relationship = "many-to-many"))
})

test_that("only one side needs a cast, and between() reuses a casted column", {
  skip_if_no_gpu()
  x <- data.frame(d = as.Date("2020-01-01") + 0:5, a = 1:6)
  y <- data.frame(lo = as.POSIXct("2020-01-02", tz = "UTC"),
                  hi = as.POSIXct("2020-01-04", tz = "UTC"), b = 1)
  for (join in list(inner_join, left_join, right_join, full_join, semi_join, anti_join)) {
    nq_both(x, y, function(x, y) join(x, y, by = dplyr::join_by(between(d, lo, hi))))
  }
  # two non-equi pairs, one casted (Date vs POSIXct) and one native-numeric
  x2 <- data.frame(d = as.Date("2020-01-01") + 0:5, n = 1:6)
  y2 <- data.frame(t = as.POSIXct("2020-01-03", tz = "UTC"), m = 3.5)
  nq_both(x2, y2, function(x, y) inner_join(x, y, by = dplyr::join_by(d <= t, n < m)))
})

test_that("keep = TRUE and mixed equi/non-equi with casts keep original types", {
  skip_if_no_gpu()
  x <- data.frame(k = c(1L, 2L, 3L), l = c(TRUE, FALSE, TRUE), a = 1:3)
  y <- data.frame(k = c(1, 2, 4), i = c(0L, 1L, 1L), b = c(10, 20, 30))
  for (join in list(inner_join, left_join, right_join, full_join)) {
    nq_both(x, y, function(x, y) join(x, y, by = dplyr::join_by(k, l <= i), keep = TRUE),
            arrange_by = c("a", "b"))
  }
})

test_that("a factor key keeps its levels through a lazy select + join (F32: join schemas read the children's metadata)", {
  skip_if_no_gpu()
  sx <- data.frame(id = c("a", "b", "c"), v = 1:3, f2 = factor(c("u", "v", "u")))
  sy <- data.frame(id = c("c", "a"), w = c(30, 10))
  lab <- data.frame(f2 = c("u", "v"), desc = c("U!", "V!"))
  # Used to lose the factor levels from the schema of a join over a lazy
  # select (the decode then refused with "factor column whose levels are
  # unknown"); the schema of a join's children now carries them.
  j <- tbl_gpu(sx, lazy = TRUE) |>
    select(id, v, f2) |>
    left_join(tbl_gpu(sy, lazy = TRUE), by = "id") |>
    compute()
  expect_equal(
    tibble::as_tibble(collect(left_join(j, tbl_gpu(lab, lazy = TRUE), by = "f2"))),
    tibble::as_tibble(sx |> select(id, v, f2) |> left_join(sy, by = "id") |> left_join(lab, by = "f2"))
  )
  expect_equal(
    tibble::as_tibble(collect(semi_join(j, tbl_gpu(lab, lazy = TRUE), by = "f2"))),
    tibble::as_tibble(sx |> select(id, v, f2) |> left_join(sy, by = "id") |> semi_join(lab, by = "f2"))
  )
})
