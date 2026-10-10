# `x[i, ]` on a lazy table with pending ops resolves its row subscript when
# the plan runs; the error must still carry tibble's message and the user's
# `[` call (F03).

test_that("deferred row-subscript errors keep tibble's message and call", {
  skip_if_no_gpu()
  df <- data.frame(x = 1:4, y = letters[1:4])
  lz <- dplyr::filter(as_lazy(tbl_gpu(df)), x > 1)
  tb <- dplyr::filter(tibble::as_tibble(df), x > 1)

  for (e in list(quote(c(-3, 2)), quote(c(TRUE, FALSE)), quote(c(2.5, 9.5)))) {
    got <- tryCatch(collect(eval(bquote(lz[.(e), ]))), error = function(err) err)
    want <- tryCatch(eval(bquote(tb[.(e), ])), error = function(err) err)
    expect_s3_class(got, class(want)[[1]])
    expect_identical(conditionMessage(got), conditionMessage(want))
    expect_identical(sub("^lz", "tb", deparse(conditionCall(got))), deparse(conditionCall(want)))
  }
})

test_that("x[, TRUE] and x[, seq_len(ncol)] add no node to a lazy table", {
  skip_if_no_gpu()
  lz <- as_lazy(tbl_gpu(data.frame(x = 1:3, y = 4:6)))
  expect_false(has_pending_ops(lz[, TRUE]))
  expect_false(has_pending_ops(lz[, 1:2]))
  expect_true(has_pending_ops(lz[, 2:1]))
})
