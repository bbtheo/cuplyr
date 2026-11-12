# F49 round 3: a failed native rowwise attempt names its error in the fallback
# notification (review N2), and rowwise `[<-` keeps the vars in column order
# for a lazy table with pending ops too.

# (a body cuplyr can't inline, so mutate() falls back)
f49r3_fn <- function(x) vapply(x, function(v) v + 1L, integer(1))

test_that("a swallowed native rowwise error reaches the fallback notification", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn")
  df <- tibble::tibble(a = 1:3, b = 4:6)
  rw <- dplyr::rowwise(tbl_gpu(df))
  out <- local({
    local_mocked_bindings(mutate_run = function(...) stop("native boom"))
    expect_warning(o <- dplyr::mutate(rw, s = a + b), "native boom")
    o
  })
  expect_identical(collect(out), dplyr::mutate(dplyr::rowwise(df), s = a + b))

  # A later, unrelated fallback does not repeat the old reason.
  expect_warning(dplyr::mutate(tbl_gpu(df), s = f49r3_fn(a)), "fell back")
  w <- tryCatch(dplyr::mutate(tbl_gpu(df), s = f49r3_fn(a)), warning = conditionMessage)
  expect_false(grepl("native boom", w))
})


test_that("rowwise [<- orders the vars by column after pending ops", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:4, y = 5:8, w = 9:12)
  want <- dplyr::filter(dplyr::rowwise(df, w, x), x > 1)
  want["z"] <- 0L
  got <- dplyr::filter(dplyr::rowwise(tbl_gpu(df, lazy = TRUE), w, x), x > 1)
  got["z"] <- 0L
  expect_identical(dplyr::group_vars(got), dplyr::group_vars(want))
  expect_identical(collect(got), want)
})
