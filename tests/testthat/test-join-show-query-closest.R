# F24: show_query() names a closest() condition as such.

test_that("show_query renders a closest() condition as closest(...)", {
  skip_if_no_gpu()

  x <- tbl_gpu(data.frame(id = 1, t = c(1, 5)), lazy = TRUE)
  y <- tbl_gpu(data.frame(id = 1, ty = c(0, 4)), lazy = TRUE)
  plan <- dplyr::left_join(x, y, by = dplyr::join_by(id, closest(t >= ty)))
  expect_output(show_query(plan), "id == id, closest(t >= ty)", fixed = TRUE)

  plain <- dplyr::left_join(x, y, by = dplyr::join_by(id, t >= ty))
  out <- paste(utils::capture.output(show_query(plain)), collapse = "\n")
  expect_match(out, "id == id, t >= ty", fixed = TRUE)
  expect_false(grepl("closest", out, fixed = TRUE))
})
