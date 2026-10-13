# F31 round 2: attributes through regrouping/ungrouping, grouped coercion,
# head()/tail() of a grouped table and vec_cast() class. Each case is compared
# with what dplyr/vctrs do on the plain data frame.

attrs_df <- function() {
  df <- data.frame(g = c(1, 1, 2), h = c(1, 2, 2), x = c(1, 2, 3))
  attr(df, "foo") <- "bar"
  df
}

user_attrs <- function(x) {
  a <- attributes(x)
  a <- a[setdiff(names(a), c("names", "row.names", "class", "groups"))]
  a[order(names(a))]
}

expect_meta_matches <- function(df, pipeline, lazy) {
  want <- pipeline(df)
  got <- pipeline(tbl_gpu(df, lazy = lazy))
  a <- attributes(got)
  a <- a[setdiff(names(a), c("names", "class"))]
  expect_identical(a[order(names(a))], user_attrs(want))
  out <- collect(got)
  expect_identical(class(out), class(want))
  expect_identical(user_attrs(out), user_attrs(want))
  expect_identical(dplyr::group_vars(out), dplyr::group_vars(want))
}

test_that("regrouping and partial ungrouping a grouped table follow dplyr", {
  skip_if_no_gpu()
  gh <- function(d) dplyr::group_by(d, g, h)
  for (lazy in c(FALSE, TRUE)) {
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d)), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), .add = TRUE), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::ungroup(gh(d), h), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::ungroup(gh(d), g, h), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::ungroup(gh(d)), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), k = x * 2, .add = TRUE), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), x, .add = TRUE), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), x), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), .data$x), lazy)
    expect_meta_matches(attrs_df(), function(d) dplyr::group_by(gh(d), dplyr::pick(x)), lazy)
  }
})

test_that("head() and tail() of a grouped table keep the attributes like dplyr", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    expect_meta_matches(attrs_df(), function(d) head(dplyr::group_by(d, g), 2), lazy)
    expect_meta_matches(attrs_df(), function(d) tail(dplyr::group_by(d, g), 2), lazy)
    expect_meta_matches(attrs_df(), function(d) head(d, 2), lazy)
  }
})

test_that("as_tibble() and as.data.frame() of a grouped table drop attributes, ungrouped keep them", {
  skip_if_no_gpu()
  df <- attrs_df()
  gt <- dplyr::group_by(tbl_gpu(df), g)
  gd <- dplyr::group_by(df, g)
  expect_identical(user_attrs(tibble::as_tibble(gt)), user_attrs(tibble::as_tibble(gd)))
  expect_identical(class(tibble::as_tibble(gt)), class(tibble::as_tibble(gd)))
  expect_identical(user_attrs(as.data.frame(gt)), user_attrs(as.data.frame(gd)))
  expect_identical(class(as.data.frame(gt)), class(as.data.frame(gd)))
  ut <- tbl_gpu(df)
  expect_identical(user_attrs(tibble::as_tibble(ut)), user_attrs(tibble::as_tibble(df)))
  expect_identical(user_attrs(as.data.frame(ut)), user_attrs(as.data.frame(df)))
  # collect() of the grouped table keeps what group_by() kept
  expect_identical(user_attrs(collect(gt)), user_attrs(gd))
})

test_that("vec_cast() to a tibble or grouped table takes the target's class", {
  skip_if_no_gpu()
  df <- data.frame(g = 1:2, v = 3:4)
  tb <- tibble::tibble(g = 1:2, v = 3:4)
  attr(df, "foo") <- "bar"
  x <- tbl_gpu(df)
  to_t <- tbl_gpu(tb)
  to_g <- dplyr::group_by(to_t, g)
  for (to in list(to_t, to_g)) {
    res <- vctrs::vec_cast(x, to)
    expect_null(.subset2(res, "bare_df"))
    expect_null(attr(res, "foo"))
  }
  expect_identical(class(collect(vctrs::vec_cast(x, to_t))), class(tb))
  out <- collect(vctrs::vec_cast(x, to_g))
  expect_identical(class(out), class(dplyr::group_by(tb, g)))
  expect_identical(dplyr::group_vars(out), "g")
  # and a bare target keeps the bare class
  to_b <- tbl_gpu(data.frame(g = 1:2, v = 3:4))
  expect_identical(class(collect(vctrs::vec_cast(tbl_gpu(tb), to_b))), "data.frame")
})

test_that("a stale bare_df flag never strips a grouped table's class", {
  skip_if_no_gpu()
  gt <- dplyr::group_by(tbl_gpu(data.frame(g = 1:2, v = 3:4)), g)
  gt <- cuplyr:::set_tbl_gpu_field(gt, "bare_df", TRUE)
  expect_s3_class(collect(gt), "grouped_df")
  expect_s3_class(tibble::as_tibble(gt), "tbl_df")
})

test_that("tbl_sum() formats the group count like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = 1:1500, v = 1)
  want <- pillar::tbl_sum(dplyr::group_by(df, g))
  expect_identical(pillar::tbl_sum(dplyr::group_by(tbl_gpu(df), g)), want)
})
