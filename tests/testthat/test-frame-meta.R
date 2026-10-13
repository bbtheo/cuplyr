# Data-frame attributes and the bare data.frame class (F31), coder checks.
# The PM's acceptance tests are in test-parity-F31-attributes.R; these cover
# the carry sites individually against dplyr.

fm_df <- function() {
  df <- data.frame(g = c(1, 1, 2, 3), x = c(4, 2, 3, 1))
  attr(df, "foo") <- "bar"
  df
}

fm_attrs <- function(x) {
  a <- attributes(x)
  a <- a[setdiff(names(a), c("names", "row.names", "class", "groups"))]
  a[order(names(a))]
}

fm_gpu_attrs <- function(x) {
  a <- attributes(x)
  a <- a[setdiff(names(a), c("names", "class"))]
  a[order(names(a))]
}

expect_meta <- function(df, pipeline, lazy = FALSE) {
  want <- pipeline(df)
  got <- pipeline(tbl_gpu(df, lazy = lazy))
  expect_identical(fm_gpu_attrs(got), fm_attrs(want))
  out <- collect(got)
  expect_identical(class(out), class(want))
  expect_identical(fm_attrs(out), fm_attrs(want))
  expect_equal(lapply(dplyr::ungroup(out), identity), lapply(dplyr::ungroup(want), identity))
}

up <- function(d, y) if (is_tbl_gpu(d)) tbl_gpu(y) else y

test_that("verbs carry attributes and the bare class like dplyr", {
  skip_if_no_gpu()
  verbs <- list(
    function(d) dplyr::filter(d, x > 1),
    function(d) dplyr::filter(d, x > mean(x), .by = g),
    function(d) dplyr::mutate(d, y = x * 2),
    function(d) dplyr::select(d, x),
    function(d) dplyr::arrange(d, x),
    function(d) dplyr::slice(d, 2:3),
    function(d) dplyr::distinct(d, g),
    function(d) dplyr::count(d, g),
    function(d) dplyr::add_count(d, g),
    function(d) dplyr::summarise(d, s = sum(x)),
    function(d) dplyr::summarise(d, s = sum(x + 1), .by = g),
    function(d) dplyr::tally(d),
    function(d) dplyr::group_by(d, g),
    function(d) dplyr::ungroup(dplyr::group_by(d, g)),
    function(d) dplyr::filter(dplyr::group_by(d, g), x > 1),
    function(d) dplyr::left_join(d, up(d, data.frame(g = 1, z = 2)), by = "g"),
    function(d) dplyr::semi_join(d, up(d, data.frame(g = 1)), by = "g"),
    function(d) dplyr::union(d, d),
    function(d) dplyr::rows_append(d, up(d, data.frame(g = 9, x = 9)))
  )
  for (lazy in c(FALSE, TRUE)) {
    for (v in verbs) {
      expect_meta(fm_df(), v, lazy = lazy)
      expect_meta(tibble::as_tibble(fm_df()), v, lazy = lazy)
    }
  }
})

test_that("binds, compute and with_groups keep the first input's meta", {
  skip_if_no_gpu()
  df <- fm_df()
  d1 <- data.frame(a = 1:4)
  out <- bind_rows(tbl_gpu(df), tbl_gpu(df))
  expect_identical(class(collect(out)), "data.frame")
  expect_identical(attr(out, "foo"), "bar")
  out <- bind_cols(tbl_gpu(d1), tbl_gpu(df))
  expect_identical(class(collect(out)), "data.frame")
  expect_null(attr(out, "foo"))
  gt <- dplyr::filter(tbl_gpu(df, lazy = TRUE), x > 1)
  expect_identical(attr(compute(gt), "foo"), "bar")
  w <- dplyr::with_groups(tbl_gpu(df), g, dplyr::mutate, m = mean(x))
  expect_identical(class(collect(w)), "data.frame")
  expect_identical(attr(w, "foo"), "bar")
})

test_that("a grouped_df uploads grouped; group_data and tbl_sum follow the class", {
  skip_if_no_gpu()
  g <- dplyr::group_by(tibble::tibble(g = c(1, 1, 2), x = 1:3), g)
  gt <- tbl_gpu(g)
  expect_identical(dplyr::group_vars(gt), "g")
  expect_identical(class(collect(gt)), class(g))
  expect_identical(class(dplyr::group_data(tbl_gpu(data.frame(x = 1:3)))), "data.frame")
  expect_identical(names(pillar::tbl_sum(tbl_gpu(data.frame(x = 1)))), "A data frame")
  expect_identical(pillar::tbl_sum(gt), pillar::tbl_sum(g))
})
