# F122 round 3: the label of a variable reference, beyond the acceptance file.

f122_unknown_df <- function() {
  tibble::tibble(`<unknown>` = c(1L, 1L, 2L), a = c("x", "y", "y"), k = 1:3)
}

expect_f122_same <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("an empty quosure is labelled and classified like dplyr's", {
  q <- rlang::new_quosure(quote(expr = ))
  expect_identical(dot_label(q), "<empty>")
  expect_true(quo_is_variable_reference(q))
  expect_false(dots_need_mutate(list(q)))
  expect_true(dots_need_mutate(list(q, rlang::quo(k + 1L))))
})

test_that("an empty dot beside a NULL dot is dplyr's error in distinct()", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = 1:3, a = c("x", "y", "y"))
  oracle <- tryCatch(dplyr::distinct(df, , NULL), error = identity)
  expect_s3_class(oracle, "error")
  for (lazy in c(FALSE, TRUE)) {
    gpu <- tryCatch(dplyr::collect(dplyr::distinct(tbl_gpu(df, lazy = lazy), , NULL)), error = identity)
    expect_identical(class(gpu)[[1L]], class(oracle)[[1L]], info = paste("lazy:", lazy))
    expect_identical(conditionMessage(gpu), conditionMessage(oracle), info = paste("lazy:", lazy))
  }
})

test_that("a pronoun as_label() can't name is computed in the mutate branch", {
  skip_if_no_gpu()
  # the real column `<unknown>` is overwritten by `a`, as in mutate()
  u <- f122_unknown_df()
  expect_f122_same(u, function(d) dplyr::group_by(d, .data$"a", k + 1L))
  v <- "a"
  raw <- rlang::new_quosure(quote(.data[[v]]), environment())
  expect_f122_same(u, function(d) dplyr::group_by(d, !!raw, k + 1L))
  # and it is still `a`'s values after an earlier dot labelled the same
  expect_f122_same(u, function(d) dplyr::group_by(d, .data$"k", .data$"a", z = k))
})

test_that("a reference that does read the real `<unknown>` column stays native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  u <- f122_unknown_df()
  expect_f122_same(u, function(d) dplyr::group_by(d, `<unknown>`, k + 1L))
  expect_f122_same(u, function(d) dplyr::group_by(d, .data$`<unknown>`, k + 1L))
  expect_f122_same(u, function(d) dplyr::group_by(d, .data[["<unknown>"]], k + 1L))
})

test_that("the no-mutate branch looks an <unknown> label up by name", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  u <- f122_unknown_df()
  expect_f122_same(u, function(d) dplyr::group_by(d, .data$"a"))
  expect_f122_same(u, function(d) dplyr::summarise(dplyr::group_by(d, .data$"a"), n = dplyr::n()))
})

test_that("count() and add_count() word several missing columns like dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = 1:3, a = c("x", "y", "y"))
  verbs <- list(
    function(d) dplyr::count(d, nope1, nope2),
    function(d) dplyr::count(d, .data$nope1, a, nope2),
    function(d) dplyr::count(d, nope1),
    function(d) dplyr::add_count(d, nope1, nope2),
    function(d) dplyr::add_count(d, nope1, nope2, nope3),
    function(d) dplyr::distinct(d, nope1, nope2)
  )
  for (verb in verbs) {
    oracle <- tryCatch(verb(df), error = identity)
    expect_s3_class(oracle, "rlang_error")
    for (lazy in c(FALSE, TRUE)) {
      gpu <- tryCatch(dplyr::collect(verb(tbl_gpu(df, lazy = lazy))), error = identity)
      expect_s3_class(gpu, "rlang_error")
      expect_identical(conditionMessage(gpu), conditionMessage(oracle), info = paste("lazy:", lazy))
    }
  }
})

test_that("a bound name after a dot that may make columns goes to dplyr", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = c(1L, 1L, 2L), a = c("x", "y", "y"))
  outside <- 3L
  shapes <- list(
    function(d) dplyr::group_by(d, k + 1L, outside),
    function(d) dplyr::group_by(d, dplyr::pick(a), outside),
    function(d) dplyr::group_by(d, dplyr::across(a), outside)
  )
  for (shape in shapes) {
    expect_f122_same(df, shape)
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_error(shape(tbl_gpu(df, lazy = FALSE)), "environment variable")
    })
  }
})

test_that("a bound name with no such earlier dot stays computed on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- tibble::tibble(k = c(1L, 1L, 2L), a = c("x", "y", "y"))
  outside <- 3L
  expect_f122_same(df, function(d) dplyr::group_by(d, outside, k + 1L))
  expect_f122_same(df, function(d) dplyr::group_by(d, outside, z = k))
  expect_f122_same(df, function(d) dplyr::group_by(d, z = k, outside))
  expect_f122_same(df, function(d) dplyr::group_by(d, NULL, outside))
})

test_that("deciding whether a name is bound forces no promise", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = c(1L, 1L, 2L), a = c("x", "y", "y"))
  by_arg <- function(d, z) dplyr::group_by(d, dplyr::across(a, .names = "z"), z)
  expect_f122_same(df, function(d) by_arg(d, stop("forced")))
})

test_that("a name in a quosure with an empty environment is not bound", {
  skip_if_no_gpu()
  df <- tibble::tibble(k = c(1L, 1L, 2L), a = c("x", "y", "y"))
  q <- rlang::new_quosure(quote(nope), rlang::empty_env())
  expect_false(quo_symbol_is_bound(q, "nope"))
  expect_error(dplyr::group_by(df, !!q, k + 1L))
  expect_error(dplyr::collect(dplyr::group_by(tbl_gpu(df), !!q, k + 1L)))
})
