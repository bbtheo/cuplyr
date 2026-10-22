# F47 round 2: error paths of the dplyr binders over a tbl_gpu

call_text <- function(e) paste(deparse(e$call), collapse = " ")

test_that("a named list of tables before the first tbl_gpu gets the clear bind error on both paths", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(x = 1:3))
  g2 <- tbl_gpu(data.frame(x = 4:6))
  for (f in list(dplyr::bind_rows, cuplyr::bind_rows)) {
    e <- rlang::catch_cnd(f(list(a = g2), g), "error")
    expect_s3_class(e, "cuplyr_error_bind_unsupported")
    expect_match(conditionMessage(e), "named list of tbl_gpu")
    expect_false(grepl("bind_row_as_df|dots\\[\\[", call_text(e)))
  }
  # a data frame that already holds a tbl_gpu column
  df <- vctrs::new_data_frame(list(a = g2), n = 3L)
  expect_error(cuplyr::bind_rows(df, g), class = "cuplyr_error_bind_unsupported")
})

test_that("a named list of tables after the first tbl_gpu errors with the call of the binder", {
  skip_if_no_gpu()
  g <- tbl_gpu(data.frame(x = 1:3))
  g2 <- tbl_gpu(data.frame(x = 4:6))
  e <- rlang::catch_cnd(dplyr::bind_rows(g, list(a = g2)), "error")
  expect_s3_class(e, "cuplyr_error_bind_unsupported")
  expect_match(call_text(e), "bind_rows", fixed = TRUE)
  expect_false(grepl("bind_row_as_df", call_text(e)))
  e <- rlang::catch_cnd(cuplyr::bind_rows(g, list(a = g2)), "error")
  expect_match(call_text(e), "bind_rows", fixed = TRUE)
  expect_false(grepl("bind_row_as_df", call_text(e)))
})

test_that("dplyr::bind_cols() with a .name_repair cuplyr doesn't take is a clear error", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(x = 1:3))
  b <- tbl_gpu(data.frame(x = 4:6))
  for (rep in list("unique_quiet", toupper, ~ toupper(.x))) {
    e <- rlang::catch_cnd(dplyr::bind_cols(a, b, .name_repair = rep), "error")
    expect_s3_class(e, "cuplyr_error_bind_unsupported")
    expect_false(inherits(e, "cuplyr_error_vctrs_bind_unsupported"))
    msg <- conditionMessage(e)
    expect_match(msg, ".name_repair", fixed = TRUE)
    for (v in c("unique", "universal", "check_unique", "minimal")) {
      expect_match(msg, v, fixed = TRUE)
    }
    expect_match(call_text(e), "bind_cols", fixed = TRUE)
  }
  # the supported strings still work
  out <- collect(dplyr::bind_cols(a, b, .name_repair = "minimal"))
  expect_identical(names(out), c("x", "x"))
  expect_identical(names(collect(dplyr::bind_cols(a, b))), c("x...1", "x...2"))
})

test_that("the direct vctrs error advises dplyr::bind_cols(list) for list_cbind()", {
  skip_if_no_gpu()
  g1 <- tbl_gpu(tibble::tibble(x = 1:3))
  g2 <- tbl_gpu(tibble::tibble(x = 4:6))
  for (f in list(vctrs::vec_rbind, vctrs::vec_cbind)) {
    msg <- conditionMessage(rlang::catch_cnd(f(g1, g2), "error"))
    expect_match(msg, "cuplyr::bind_rows()", fixed = TRUE)
    expect_match(msg, "dplyr::bind_rows()", fixed = TRUE)
    expect_match(msg, "list_rbind()", fixed = TRUE)
    expect_match(msg, "list_cbind()", fixed = TRUE)
    expect_match(msg, "dplyr::bind_cols(list)", fixed = TRUE)
  }
})
