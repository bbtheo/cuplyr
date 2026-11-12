# F82 round 5: an if_any()/if_all() lambda that only READS a variable (`$`,
# `[[`, `[`, `.env`, names(), length()) stays native; one whose read can run
# code (an active binding, a classed object with a `$` method, a call in the
# subscript) goes to dplyr whole, with a notice that says why.

r5_df <- function() {
  tibble::tibble(x = c(1, 5, 10, NA, 3), y = c(4, 0, 2, 8, NA),
                 s = c("a", "b", "c", NA, "a"))
}

r5_native <- function(df, pipeline) {
  expect_same_as_dplyr(df, pipeline)
  expect_same_as_dplyr_lazy(df, pipeline)
}

test_that("F82: names(), length(), matrix and computed subscripts of plain objects stay native", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- r5_df()
  lookup <- list(a = 1, b = 2)
  vals <- c(2, 4, 6)
  m <- matrix(c(1, 5, 9, 2), 2)
  r5_native(df, function(d) filter(d, if_any(s, ~ .x %in% names(lookup))))
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > length(vals))))
  r5_native(df, function(d) mutate(d, a = dplyr::if_all(c(x, y), ~ .x < vals[length(vals)])))
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > m[2, 1])))
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > vals[length(vals) - 1])))
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > lookup[["b"]] + lookup$a)))
})

test_that("F82: a read that can run code goes to dplyr whole, with its own reason", {
  skip_if_no_gpu()
  df <- r5_df()
  withr::local_options(cuplyr.fallback = "error")
  makeActiveBinding("active_cfg", function() list(thr = 4), environment())
  classed <- structure(list(thr = 4), class = "r5_cfg")
  env_cfg <- new.env()
  env_cfg$thr <- 4
  for (f in list(
    function(d) filter(d, if_any(c(x, y), ~ .x > active_cfg$thr)),
    function(d) filter(d, if_any(c(x, y), ~ .x > classed$thr)),
    function(d) filter(d, if_any(c(x, y), ~ .x > env_cfg$thr)),
    function(d) filter(d, if_any(c(x, y), ~ .x > missing_obj$thr))
  )) {
    expect_error(collect(f(tbl_gpu(df))), class = "cuplyr_error_fallback")
  }
  withr::local_options(cuplyr.fallback = NULL, cuplyr.plan_break = NULL)
  got <- list()
  withCallingHandlers(
    collect(filter(tbl_gpu(df), if_any(c(x, y), ~ .x > classed$thr))),
    cuplyr_cpu_fallback = function(c) got[[length(got) + 1L]] <<- c,
    warning = function(w) invokeRestart("muffleWarning")
  )
  expect_length(got, 1L)
  expect_match(got[[1L]]$reason, "side effects", fixed = TRUE)
  expect_no_match(got[[1L]]$reason, "no GPU expansion")
})

test_that("F82: a read through .env is native unless the binding is active", {
  skip_if_no_gpu()
  df <- r5_df()
  thr <- 4
  withr::local_options(cuplyr.fallback = "error")
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > .env$thr)))
  makeActiveBinding("active_thr", function() 4, environment())
  expect_error(collect(filter(tbl_gpu(df), if_any(c(x, y), ~ .x > .env$active_thr))),
               class = "cuplyr_error_fallback")
})

test_that("F82: .env[[\"\"]] and .env[[k]] are not plain reads; a literal subscript is", {
  skip_if_no_gpu()
  df <- r5_df()
  withr::local_options(cuplyr.fallback = "error")
  schema <- list(names = c("x", "y"))
  e <- environment()
  pure <- function(x) TRUE
  expect_false(across_is_plain_read(quote(.env[[""]]), schema, e, pure))
  expect_false(across_is_plain_read(quote(.env[[k]]), schema, e, pure))
  expect_true(across_is_plain_read(quote(.env[["thr"]]), schema, e, pure))
  thr <- 4
  r5_native(df, function(d) filter(d, if_any(c(x, y), ~ .x > .env[["thr"]])))
})
