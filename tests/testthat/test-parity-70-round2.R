# #70 round 2: the key-order error on every join shape, the verb-time raise
# with pending steps, and the upload hints.

r2_y <- tibble::tibble(k1 = c("a", "b"), k2 = c("1", "2"))

r2_raises <- function(x, y, f, regexp, class = "dplyr_error_join_incompatible_type") {
  for (lazy in c(FALSE, TRUE)) {
    expect_error(f(tbl_gpu(x, lazy = lazy), tbl_gpu(y, lazy = lazy)),
                 regexp = regexp, class = class, fixed = TRUE, info = paste("lazy =", lazy))
  }
}

test_that("the first bad pair is named for keep = TRUE and filtering joins too", {
  skip_if_no_gpu()
  first <- "Can't join `x$k1` with `y$k1` due to incompatible types."
  second <- "Can't join `x$k2` with `y$k2` due to incompatible types."
  bad1 <- tibble::tibble(k1 = c(TRUE, NA), k2 = 1:2)
  bad2 <- tibble::tibble(k1 = c(NA, NA), k2 = 1:2)
  r2_raises(bad1, r2_y, function(x, y) semi_join(x, y, by = c("k1", "k2")), first)
  r2_raises(bad1, r2_y, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), first)
  r2_raises(bad2, r2_y, function(x, y) anti_join(x, y, by = c("k1", "k2")), second)
  r2_raises(bad2, r2_y, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), second)
})

test_that("the verb-time raise holds with steps pending on either side", {
  skip_if_no_gpu()
  bad <- tbl_gpu(tibble::tibble(k = c(NA, TRUE), a = 1:2), lazy = TRUE)
  chr <- tbl_gpu(tibble::tibble(k = c("u", NA), b = 2:3), lazy = TRUE)
  expect_error(left_join(filter(bad, a > 0L), chr, by = "k"),
               class = "dplyr_error_join_incompatible_type")
  expect_error(left_join(bad, mutate(chr, b = b + 1L), by = "k"),
               class = "dplyr_error_join_incompatible_type")
  ok <- tbl_gpu(tibble::tibble(k = c(NA, NA), a = 1:2), lazy = TRUE)
  out <- left_join(filter(ok, a > 0L), mutate(chr, b = b + 1L), by = "k")
  expect_equal(collect(out)$b, c(4L, 4L))
  # an unspecified() key needs no check and stays lazy
  un <- tbl_gpu(tibble::tibble(k = vctrs::unspecified(2), a = 1:2), lazy = TRUE)
  lazy_out <- left_join(un, chr, by = "k")
  expect_true(has_pending_ops(lazy_out))
})

test_that("the zone in a join error is the key's own, `local` for an empty zone", {
  skip_if_no_gpu()
  local_key <- tibble::tibble(k = as.POSIXct("2020-01-01", tz = ""), b = 2L)
  expected <- tryCatch(inner_join(tibble::tibble(k = NA, a = 1), local_key, by = "k"),
                       error = conditionMessage)
  r2_raises(tibble::tibble(k = NA, a = 1), local_key,
            function(x, y) inner_join(x, y, by = "k"), expected, class = "vctrs_error_cast")
})

test_that("the vctrs_error_cast does not report an internal call", {
  skip_if_no_gpu()
  e <- tryCatch(inner_join(tbl_gpu(tibble::tibble(k = NA, a = 1)),
                           tbl_gpu(tibble::tibble(k = "u", b = 2L)), by = "k"),
                error = identity)
  expect_s3_class(e, "vctrs_error_cast")
  expect_null(e$call)
})

test_that("complex and raw columns get the classed upload error", {
  skip_if_no_gpu()
  expect_error(tbl_gpu(tibble::tibble(z = complex(2))), "`z`", fixed = TRUE,
               class = "cuplyr_error_unsupported_class")
  expect_error(tbl_gpu(tibble::tibble(r = as.raw(1:2))), "`r`", fixed = TRUE,
               class = "cuplyr_error_unsupported_class")
})

test_that("the upload hint fits the kind of column", {
  skip_if_no_gpu()
  hint <- function(col) {
    e <- tryCatch(tbl_gpu(tibble::tibble(c = col)), error = identity)
    paste(conditionMessage(e), collapse = " ")
  }
  expect_match(hint(cbind(1:2, 3:4)), "one plain column per matrix column", fixed = TRUE)
  expect_match(hint(tibble::tibble(a = 1:2)), "tidyr::unpack()", fixed = TRUE)
  expect_match(hint(list(1, 2)), "Make a list column atomic", fixed = TRUE)
  expect_no_match(hint(list(1, 2)), "matrix column", fixed = TRUE)
})

# Round 3

test_that("failing logical keys on opposite sides name the first pair in `by` order", {
  skip_if_no_gpu()
  x <- tibble::tibble(k1 = c("a", "b"), k2 = c(TRUE, NA))
  y <- tibble::tibble(k1 = c(TRUE, NA), k2 = c("a", "b"))
  first <- "Can't join `x$k1` with `y$k1` due to incompatible types."
  r2_raises(x, y, function(x, y) semi_join(x, y, by = c("k1", "k2")), first)
  r2_raises(x, y, function(x, y) anti_join(x, y, by = c("k1", "k2")), first)
  r2_raises(x, y, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), first)
  r2_raises(x, y, function(x, y) left_join(x, y, by = c("k1", "k2")), first)
  # the y-side pair fails first when it comes first
  r2_raises(y, x, function(x, y) semi_join(x, y, by = c("k1", "k2")), first)
  r2_raises(y, x, function(x, y) left_join(x, y, by = c("k1", "k2"), keep = TRUE), first)
})

test_that("an ordered factor partner is labelled <ordered<...>> in both errors", {
  skip_if_no_gpu()
  ord <- tibble::tibble(k = factor("u", levels = c("v", "u"), ordered = TRUE), b = 2L)
  msg <- function(x) {
    tryCatch(inner_join(x, ord, by = "k"), error = conditionMessage)
  }
  cast_msg <- msg(tibble::tibble(k = NA, a = 1))
  expect_match(cast_msg, "<ordered<", fixed = TRUE)
  r2_raises(tibble::tibble(k = NA, a = 1), ord,
            function(x, y) inner_join(x, y, by = "k"), cast_msg, class = "vctrs_error_cast")
  type_msg <- tryCatch(inner_join(tibble::tibble(k = TRUE, a = 1), ord, by = "k"),
                       error = function(e) paste(conditionMessage(e), collapse = "\n"))
  expect_match(type_msg, "<ordered<", fixed = TRUE)
  label <- regmatches(type_msg, regexpr("<ordered<[0-9a-f]+>>", type_msg))
  for (lazy in c(FALSE, TRUE)) {
    e <- tryCatch(inner_join(tbl_gpu(tibble::tibble(k = TRUE, a = 1), lazy = lazy),
                             tbl_gpu(ord, lazy = lazy), by = "k"),
                  error = identity)
    expect_s3_class(e, "dplyr_error_join_incompatible_type")
    expect_match(conditionMessage(e), paste0("`y$k` is a ", label), fixed = TRUE,
                 info = paste("lazy =", lazy))
  }
})

test_that("the vctrs_error_cast is one line at any console width", {
  skip_if_no_gpu()
  withr::local_options(width = 40, cli.condition_width = 40)
  e <- tryCatch(inner_join(tbl_gpu(tibble::tibble(k = NA, a = 1)),
                           tbl_gpu(tibble::tibble(k = "u", b = 2L)), by = "k"),
                error = identity)
  expect_false(grepl("\n", conditionMessage(e), fixed = TRUE))
})

test_that("bind_cols() with a list column keeps the first input's groups", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1L, 1L, 2L))
  l <- tibble::tibble(l = list(1, "x", NULL))
  for (lazy in c(FALSE, TRUE)) {
    out <- bind_cols(group_by(tbl_gpu(df, lazy = lazy), a), l)
    expect_identical(out, dplyr::bind_cols(dplyr::group_by(df, a), l),
                     info = paste("lazy =", lazy))
  }
})

test_that("only the first input's groups survive a list-column bind_rows()", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = 1:2, b = c("u", "v"))
  l <- tibble::tibble(a = 3L, l = list(1:3))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(df, lazy = lazy)
    expect_identical(bind_rows(l, group_by(g, b)),
                     dplyr::bind_rows(l, dplyr::group_by(df, b)))
    expect_identical(bind_rows(g, l, group_by(g, b)),
                     dplyr::bind_rows(df, l, dplyr::group_by(df, b)))
  }
})

test_that("a column with no GPU form that is not a list keeps the upload error", {
  skip_if_no_gpu()
  dt <- tibble::tibble(a = 1L, d = as.difftime(1, units = "secs"))
  expect_error(bind_rows(tbl_gpu(tibble::tibble(a = 2L)), dt),
               class = "cuplyr_error_unsupported_class")
})

test_that("a filter pushes below a join on a vctrs::unspecified() key", {
  skip_if_no_gpu()
  un <- tibble::tibble(k = vctrs::unspecified(3), a = 1:3)
  chr <- tibble::tibble(k = c("u", NA), b = 2:3)
  pipe <- function(x, y) {
    left_join(x, y, by = "k", relationship = "many-to-many") |>
      filter(a > 1L, is.na(k))
  }
  out <- pipe(tbl_gpu(un, lazy = TRUE), tbl_gpu(chr, lazy = TRUE))
  plan <- optimize_ast(out$lazy_ops)
  expect_identical(plan$type, "filter") # the key predicate stays above
  expect_identical(plan$input$type, "join")
  expect_identical(plan$input$left$type, "filter")
  expect_identical(tibble::as_tibble(collect(out)), pipe(un, chr))
})
