# F04 round 2: details the acceptance tests don't pin down (scalar assignment
# staying lazy, one-row cell reads, `.drop` through joins and renames, error
# labels of forwarded subscripts).

r2_df <- function() {
  data.frame(
    g = c("a", "b", "a", "c"),
    x = c(1.5, 2, NA, 4.5),
    i = c(1L, NA, 3L, 4L),
    f = factor(c("u", "v", "u", NA), levels = c("u", "v", "w")),
    stringsAsFactors = FALSE
  )
}

test_that("a length-one assignment extends a lazy plan instead of running it", {
  skip_if_no_gpu()
  executed <- 0L
  orig <- cuplyr:::lower_and_execute
  local_mocked_bindings(
    lower_and_execute = function(...) { executed <<- executed + 1L; orig(...) },
    .package = "cuplyr"
  )
  lz <- dplyr::filter(as_lazy(tbl_gpu(r2_df())), i > 1L | is.na(i))
  set <- function(t) { t$z <- 5L; t[["w"]] <- "k"; t }
  set <- (function(fn) { environment(fn) <- globalenv(); fn })(set)
  out <- set(lz)
  expect_identical(executed, 0L)
  expect_true(inherits(unclass(out)$lazy_ops, "ast_node"))
  expect_identical(
    tibble::as_tibble(collect(out)),
    tibble::as_tibble(dplyr::mutate(dplyr::filter(r2_df(), i > 1L | is.na(i)), z = 5L, w = "k"))
  )
})

test_that("length-one assignment keeps types, groups and .drop, and replaces in place", {
  skip_if_no_gpu()
  df <- r2_df()
  for (lazy in c(FALSE, TRUE)) {
    gt <- dplyr::group_by(if (lazy) as_lazy(tbl_gpu(df)) else tbl_gpu(df), g, .drop = FALSE)
    tb <- dplyr::group_by(tibble::as_tibble(df), g, .drop = FALSE)
    fn <- function(t) { t$x <- 7L; t$b <- TRUE; t$s <- "q"; t$f <- 1L; t$i <- 2.5; t }
    environment(fn) <- globalenv()
    o <- fn(tb)
    r <- fn(gt)
    expect_identical(tibble::as_tibble(collect(r)), tibble::as_tibble(o))
    expect_identical(dplyr::group_vars(r), "g")
    expect_identical(dplyr::group_by_drop_default(r), FALSE)
  }
})

test_that("x[[i, j]] gathers a single row on the GPU", {
  skip_if_no_gpu()
  df <- r2_df()
  gt <- tbl_gpu(df)
  rows_collected <- integer()
  orig <- cuplyr:::gpu_collect
  local_mocked_bindings(
    gpu_collect = function(ptr, ...) { rows_collected <<- c(rows_collected, gpu_dim(ptr)[1]); orig(ptr, ...) },
    .package = "cuplyr"
  )
  rd <- function(t) list(t[[2, "x"]], t[[3, "f"]], t[[3, 3]])
  environment(rd) <- globalenv()
  expect_identical(rd(gt), rd(tibble::as_tibble(df)))
  expect_identical(rd(as_lazy(dplyr::filter(gt, i > 1L | is.na(i)))),
                   rd(dplyr::filter(tibble::as_tibble(df), i > 1L | is.na(i))))
  expect_true(all(rows_collected == 1L))
})

test_that("[[ forwards exact= and extra arguments to tibble", {
  skip_if_no_gpu()
  df <- r2_df()
  gt <- tbl_gpu(df)
  fn <- function(t) t[["g", exact = FALSE]]
  environment(fn) <- globalenv()
  expect_identical(
    withCallingHandlers(fn(gt), warning = function(w) invokeRestart("muffleWarning")),
    df$g
  )
  w_gpu <- NULL
  w_tbl <- NULL
  withCallingHandlers(fn(gt), warning = function(w) { w_gpu <<- conditionMessage(w); invokeRestart("muffleWarning") })
  withCallingHandlers(fn(tibble::as_tibble(df)), warning = function(w) { w_tbl <<- conditionMessage(w); invokeRestart("muffleWarning") })
  expect_identical(w_gpu, w_tbl)
  cell <- function(t) t[[2, "x", exact = FALSE]]
  environment(cell) <- globalenv()
  n_warn <- 0L
  withCallingHandlers(cell(gt), warning = function(w) { n_warn <<- n_warn + 1L; invokeRestart("muffleWarning") })
  expect_identical(n_warn, 1L)
})

test_that(".drop = FALSE survives push_op verbs, joins and a group-column rename; resets when ungrouped", {
  skip_if_no_gpu()
  df <- data.frame(
    f = factor(c("a", "b", "a"), levels = c("a", "b", "c")),
    x = c(3L, 1L, 2L)
  )
  other <- data.frame(f = factor(c("a", "b"), levels = c("a", "b", "c")), y = c(10, 20))
  for (lazy in c(FALSE, TRUE)) {
    up <- function(d) if (lazy) as_lazy(tbl_gpu(d)) else tbl_gpu(d)
    g <- dplyr::group_by(up(df), f, .drop = FALSE)
    drop_of <- function(t) dplyr::group_by_drop_default(t)
    expect_false(drop_of(dplyr::arrange(g, x)))
    expect_false(drop_of(dplyr::rename(g, ff = f)))
    expect_false(drop_of(dplyr::left_join(g, up(other), by = "f")))
    expect_false(drop_of(dplyr::distinct(g, f, x)))
    # no grouping left: back to the default, as for a grouped_df
    expect_true(drop_of(g[, "x"]))
    expect_true(drop_of(dplyr::summarise(g, n = dplyr::n(), .groups = "drop")))
    expect_true(drop_of(dplyr::group_by(g[, "x"], x)))
    # the collected result is the same grouped_df dplyr gives
    expect_identical(
      collect(dplyr::mutate(g, y = x + 1L)),
      dplyr::mutate(dplyr::group_by(tibble::as_tibble(df), f, .drop = FALSE), y = x + 1L)
    )
  }
})

test_that("assignment on a table with ...N column names keeps them", {
  skip_if_no_gpu()
  df <- setNames(data.frame(3:4, 1:2, 5:6), c("b", "...1", "...7"))
  gt <- tbl_gpu(df)
  fn <- function(t) { t$z <- c(1, 2); t[["...7"]] <- 0L; t$b <- NULL; t }
  environment(fn) <- globalenv()
  expect_identical(tibble::as_tibble(collect(fn(gt))), fn(tibble::as_tibble(df)))
  expect_identical(
    tibble::as_tibble(collect(dplyr::dplyr_col_modify(gt, list(z = 9L)))),
    dplyr::dplyr_col_modify(tibble::as_tibble(df), list(z = 9L))
  )
})
