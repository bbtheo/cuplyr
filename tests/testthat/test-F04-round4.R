# F04 round 4: `.drop` once no group is left (sites beyond the join case the
# acceptance tests pin), `[[` error calls, bind_cols() groups after repair.

r4_up <- function(df, lazy) {
  g <- tbl_gpu(df)
  if (lazy) as_lazy(g) else g
}

r4_df <- function() {
  data.frame(f = factor(c("a", "b"), levels = c("a", "b", "c")), k = 1:2)
}

test_that("an ungrouped tbl_gpu reports .drop = TRUE whatever its stale flag", {
  skip_if_no_gpu()
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(r4_up(r4_df(), lazy), f, .drop = FALSE)
    # a selection that resolves to no column leaves the table ungrouped
    none <- dplyr::group_by(gt, dplyr::any_of("zz"), .drop = FALSE)
    expect_identical(dplyr::group_vars(none), character())
    expect_identical(dplyr::group_by_drop_default(none), TRUE)
    expect_identical(.subset2(none, "group_drop"), TRUE)
    # vctrs restore of a frame that lost every group column
    restored <- vctrs::vec_restore(data.frame(k = 1:2), gt)
    expect_identical(dplyr::group_vars(restored), character())
    expect_identical(dplyr::group_by_drop_default(restored), TRUE)
    expect_identical(dplyr::n_groups(dplyr::group_by(restored, k)), 2L)
  }
})

test_that("summarise() that drops the last group resets .drop", {
  skip_if_no_gpu()
  df <- data.frame(f = factor(c("a", "b"), levels = c("a", "b", "c")), x = c(1, 2))
  for (lazy in c(FALSE, TRUE)) {
    s <- dplyr::summarise(dplyr::group_by(r4_up(df, lazy), f, .drop = FALSE), n = dplyr::n())
    expect_identical(dplyr::group_vars(s), character())
    expect_identical(dplyr::group_by_drop_default(s), TRUE)
  }
})

test_that("[[ errors report the user's call, with tibble's class and message", {
  skip_if_no_gpu()
  df <- tibble::tibble(f = factor(c("a", "b"), levels = c("a", "b", "c")), k = 1:2)
  gt <- tbl_gpu(df)
  cnd <- function(expr) tryCatch(expr, error = identity)
  pairs <- list(
    list(cnd(gt[[5]]), cnd(df[[5]]), quote(gt[[5]])),
    list(cnd(gt[[NA]]), cnd(df[[NA]]), quote(gt[[NA]])),
    list(cnd(gt[[]]), cnd(df[[]]), quote(gt[[]])),
    list(cnd(gt[[5, "k"]]), cnd(df[[5, "k"]]), quote(gt[[5, "k"]])),
    list(cnd(gt[[1, 5]]), cnd(df[[1, 5]]), quote(gt[[1, 5]]))
  )
  for (p in pairs) {
    label <- deparse(p[[3]])
    expect_s3_class(p[[1]], "error")
    expect_identical(class(p[[1]]), class(p[[2]]), info = label)
    expect_identical(conditionMessage(p[[1]]), conditionMessage(p[[2]]), info = label)
    expect_identical(conditionCall(p[[1]]), p[[3]], info = label)
  }
  # an error raised while evaluating the user's own subscript is left alone
  boom <- function() rlang::abort("boom")
  e <- cnd(gt[[boom()]])
  expect_identical(conditionMessage(e), "boom")
  expect_identical(conditionCall(e), quote(boom()))
})

test_that("bind_cols()/cbind() of a grouped table with itself is ungrouped once names are repaired", {
  skip_if_no_gpu()
  g <- dplyr::group_by(tbl_gpu(r4_df()), f, .drop = FALSE)
  og <- dplyr::group_by(tibble::as_tibble(r4_df()), f, .drop = FALSE)
  b <- suppressMessages(bind_cols(g, g))
  ob <- suppressMessages(dplyr::bind_cols(og, og))
  expect_identical(dplyr::group_vars(b), dplyr::group_vars(ob))
  expect_identical(dplyr::group_by_drop_default(b), dplyr::group_by_drop_default(ob))
  expect_identical(.subset2(b, "group_drop"), TRUE)
  expect_identical(names(b), names(ob))
  cb <- suppressMessages(cbind(g, g))
  expect_identical(dplyr::group_vars(cb), character())
  expect_identical(dplyr::group_by_drop_default(cb), TRUE)
  # a group column that keeps its name stays the grouping
  other <- tbl_gpu(data.frame(z = c(TRUE, FALSE)))
  kept <- bind_cols(g, other)
  expect_identical(dplyr::group_vars(kept), "f")
  expect_identical(dplyr::group_by_drop_default(kept), FALSE)
})
