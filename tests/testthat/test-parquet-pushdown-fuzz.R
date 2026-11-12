# F85 (cuplyr's own checks): a seeded fuzz of the predicate pushdown. Random
# filters over every column type (doubles too, which stay above the scan), with
# literals on the edges of what converts exactly, must keep the rows dplyr keeps
# -- with the pushdown on (several row groups, so statistics prune) and off.

fuzz_frame <- function(n = 2000L) {
  set.seed(85)
  df <- tibble::tibble(
    id = seq_len(n),
    # clustered, so that row group statistics can skip groups
    i = as.integer(sort(sample(c(-50:50, NA), n, TRUE), na.last = TRUE)),
    j = sample(c(NA, -2147483647L, -1L, 0L, 1L, 7L, 2147483647L), n, TRUE),
    d = sample(c(NA, NaN, Inf, -Inf, 0, -0, 1.5, -1.5, 2, 1e300, 5e-324, 3.25), n, TRUE),
    dsort = sort(c(rnorm(n - 5L), NaN, NA, Inf, -Inf, 0), na.last = TRUE),
    s = sample(c(NA, "", "a", "b", "x", "xy", "é", "zzz", "A"), n, TRUE),
    # long strings, clustered (strings stay above the scan, so these only check the residual filter)
    ls = sort(paste0(strrep("a", 70L), sample(c(letters, "é", ""), n, TRUE)), na.last = TRUE),
    dt = as.Date("2020-01-01") + sample(c(NA, -1000:1000), n, TRUE),
    ts = as.POSIXct("2021-03-04 05:06:07", tz = "UTC") + sample(c(NA, -5000:5000, 0.5, 1.25), n, TRUE),
    l = sample(c(TRUE, FALSE, NA), n, TRUE)
  )
  if (requireNamespace("bit64", quietly = TRUE)) {
    big <- c("-9007199254740993", "-9007199254740992", "-3", "0", "1", "5", "9007199254740991",
             "9007199254740992", "9007199254740993", "1000000000000000", "4611686018427387904")
    df$k <- bit64::as.integer64(c(big, NA)[sort(sample(length(big) + 1L, n, TRUE))])
  }
  df
}

fuzz_literals <- list(
  i = c("-51L", "-50L", "0L", "1L", "50L", "51L", "0", "1.5", "7", "-3", "1e10", "NA", "NA_integer_",
        "-2147483648", "2147483647", "2147483648", "TRUE"),
  j = c("-2147483647L", "-1L", "0L", "7L", "2147483647L", "NA_integer_", "-2147483648", "2^31", "0.5", "-2147483647"),
  d = c("0", "-0", "1.5", "-1.5", "2", "1e300", "3.25", "Inf", "-Inf", "NaN", "NA_real_", "1L", "5e-324", "2.5"),
  dsort = c("0", "-1", "1", "0.5", "Inf", "-Inf", "NaN", "NA_real_", "3"),
  s = c("\"\"", "\"a\"", "\"b\"", "\"x\"", "\"é\"", "\"zzz\"", "\"A\"", "NA_character_", "\"nope\""),
  ls = c(paste0("\"", strrep("a", 70L), "\""), paste0("\"", strrep("a", 70L), "m\""),
         paste0("\"", strrep("a", 70L), "zz\""), paste0("\"", strrep("a", 64L), "\""),
         paste0("\"", strrep("a", 69L), "b\""), "NA_character_"),
  k = c("0", "5", "-3", "1.5", "2^53", "2^53 - 1", "-(2^53)", "1e15", "NA_real_", "4L",
        "bit64::as.integer64(5)", "bit64::as.integer64(\"9007199254740993\")",
        "bit64::as.integer64(\"-9007199254740993\")", "bit64::as.integer64(NA)"),
  dt = c("as.Date(\"2020-01-01\")", "as.Date(\"2019-06-15\")", "as.Date(\"2022-12-31\")", "as.Date(\"2020-01-01\") + 0.5",
         "as.Date(NA)", "as.Date(\"1900-01-01\")"),
  ts = c("as.POSIXct(\"2021-03-04 05:06:07\", tz = \"UTC\")", "as.POSIXct(\"2021-03-04 05:06:07.5\", tz = \"UTC\")",
         "as.POSIXct(\"2021-03-04 06:00:00\", tz = \"UTC\")", "as.POSIXct(\"1999-01-01\", tz = \"UTC\")",
         "as.POSIXct(NA)", "as.POSIXct(\"2021-03-04 05:06:07\", tz = \"America/New_York\")")
)
fuzz_ops <- c(">", ">=", "<", "<=", "==", "!=")

fuzz_atom <- function() {
  cols <- c(names(fuzz_literals), "l", "id")
  if (!requireNamespace("bit64", quietly = TRUE)) cols <- setdiff(cols, "k")
  col <- sample(cols, 1L)
  kind <- sample(c("cmp", "cmp", "cmp", "flip", "na", "notna", "not_cmp"), 1L)
  if (col == "l") return(sample(c("l", "!l", "is.na(l)", "l == TRUE"), 1L))
  if (col == "id") return(paste0("id ", sample(fuzz_ops, 1L), " ", sample(c("0L", "1000L", "1999L", "2000L", "17"), 1L)))
  lit <- sample(fuzz_literals[[col]], 1L)
  op <- sample(fuzz_ops, 1L)
  switch(kind,
    cmp = paste(col, op, lit),
    flip = paste(lit, op, col),
    na = paste0("is.na(", col, ")"),
    notna = paste0("!is.na(", col, ")"),
    not_cmp = paste0("!(", col, " ", op, " ", lit, ")")
  )
}

fuzz_predicate <- function() {
  n <- sample(1:3, 1L)
  atoms <- vapply(seq_len(n), function(k) fuzz_atom(), "")
  join <- sample(c("&", "&", "&", "|"), n - 1L, TRUE)
  out <- atoms[[1L]]
  for (k in seq_len(n - 1L)) out <- paste0("(", out, ") ", join[[k]], " (", atoms[[k + 1L]], ")")
  out
}

test_that("pushed predicates keep the rows dplyr keeps, row groups and statistics included", {
  skip_if_no_gpu()
  df <- fuzz_frame()
  path <- pq_write(df, row_group_size = 250L)
  eager <- tibble::as_tibble(collect(read_parquet_gpu(path, lazy = FALSE)))
  scan <- read_parquet_gpu(path, lazy = TRUE)
  set.seed(8500)
  pushed_any <- 0L
  skipped_groups <- 0L
  warned <- character()
  for (k in 1:350) {
    pred <- fuzz_predicate()
    q <- rlang::parse_expr(pred)
    want <- tryCatch(suppressWarnings(dplyr::filter(eager, !!q)), error = function(e) NULL)
    if (is.null(want)) next  # (dplyr itself refuses, e.g. a Date against a number)
    on_k <- grepl("\\bk\\b", pred)
    got <- tryCatch(
      withr::with_options(list(cuplyr.parquet_pushdown = TRUE), {
        parquet_last_read(reset = TRUE)
        r <- withCallingHandlers(collect(dplyr::filter(scan, !!q)), warning = function(w) {
          # (the planning itself, converting a literal, must not warn)
          warned <<- c(warned, conditionMessage(w))
          invokeRestart("muffleWarning")
        })
        list(rows = r, read = parquet_last_read())
      }),
      error = function(e) e
    )
    expect_false(inherits(got, "error"), info = paste(pred, if (inherits(got, "error")) conditionMessage(got)))
    if (inherits(got, "error")) next
    if (!on_k) {
      expect_identical(tibble::as_tibble(got$rows)$id, want$id, info = pred)
      expect_equal(tibble::as_tibble(got$rows), want, info = pred)
    }
    if (length(got$read$predicates) > 0L) pushed_any <- pushed_any + 1L
    if (isTRUE(got$read$row_groups_after_stats < got$read$row_groups_total)) skipped_groups <- skipped_groups + 1L
    off <- withr::with_options(list(cuplyr.parquet_pushdown = FALSE),
                               suppressWarnings(collect(dplyr::filter(scan, !!q))))
    if (!on_k) expect_identical(tibble::as_tibble(off)$id, want$id, info = paste("off:", pred))
    # (with or without the pushdown the rows are the same, integer64 included)
    expect_identical(tibble::as_tibble(got$rows)$id, tibble::as_tibble(off)$id, info = paste("on vs off:", pred))
  }
  expect_false(any(grepl("precision", warned)), info = unique(warned))
  # the fuzz must actually exercise the pushdown, and statistics must skip
  expect_gt(pushed_any, 100L)
  expect_gt(skipped_groups, 5L)
})

test_that("INT32 and INT64 values equal to R's NA are never kept by a pushed comparison", {
  skip_if_no_gpu()
  path <- pq_fixture("arrow_r_na_sentinels.parquet")
  scan <- read_parquet_gpu(path, lazy = TRUE)
  ids <- function(q) collect(q)$id
  expect_identical(ids(filter(scan, i < 0L)), 4L)
  expect_identical(ids(filter(scan, i <= 1L)), c(2L, 4L))
  expect_identical(ids(filter(scan, i != 1L)), 4L)
  expect_identical(ids(filter(scan, i > -5L)), c(2L, 4L))
  expect_identical(ids(filter(scan, !is.na(i))), c(2L, 4L))
  expect_identical(ids(filter(scan, is.na(i))), c(1L, 3L))
  expect_identical(ids(filter(scan, 0L > i)), 4L)
  r <- pq_run(filter(scan, i < 0L))
  expect_length(r$read$predicates, 1L)
  skip_if_not_installed("bit64")
  expect_identical(ids(filter(scan, b < 0L)), 4L)
  expect_identical(ids(filter(scan, b <= 1)), c(2L, 4L))
  expect_identical(ids(filter(scan, b != 1)), 4L)
  expect_identical(ids(filter(scan, !is.na(b))), c(2L, 4L))
  expect_identical(ids(filter(scan, is.na(b))), c(1L, 3L))
})

test_that("a filter column the output drops, odd column names and empty results", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    id = 1:600, `a b` = rep(1:3, 200L), `a.b` = as.numeric(rep(1:3, 200L)), `é` = rep(c("p", "q"), 300L),
    `1x` = rep(c(TRUE, FALSE), 300L), `with"quote` = as.Date("2020-01-01") + rep(0:5, 100L)
  )
  path <- pq_write(df, row_group_size = 100L)
  scan <- read_parquet_gpu(path, lazy = TRUE)
  cases <- list(
    quote(filter(scan, `a b` > 1L)),
    quote(filter(scan, `a.b` == 2)),
    quote(filter(scan, `é` == "q")),
    quote(filter(scan, `with"quote` >= as.Date("2020-01-04"))),
    quote(select(filter(scan, `a.b` > 1, `é` == "p"), id)),
    quote(select(filter(scan, `a b` < 3L), `é`, id)),
    quote(filter(scan, id > 1000L)),
    quote(filter(scan, id > 550L, `é` == "zzz")),
    quote(select(filter(scan, id > 1000L), `a b`))
  )
  for (case in cases) {
    want <- tibble::as_tibble(withr::with_options(list(cuplyr.parquet_pushdown = FALSE), collect(eval(case))))
    r <- pq_run(eval(case))
    expect_equal(tibble::as_tibble(r$result), want, info = deparse(case))
    # (a comparison on a double or string column alone leaves nothing to push)
    nothing <- list(quote(filter(scan, `a.b` == 2)), quote(filter(scan, `é` == "q")),
                    quote(select(filter(scan, `a.b` > 1, `é` == "p"), id)))
    if (any(vapply(nothing, identical, logical(1), case))) {
      expect_length(r$read$predicates, 0L)
    } else {
      expect_gt(length(r$read$predicates), 0L)
    }
  }
  # the same against dplyr
  r <- pq_run(select(filter(scan, `a.b` > 1, `é` == "p"), id))
  expect_identical(r$result$id, dplyr::filter(df, `a.b` > 1, `é` == "p")$id)
  expect_identical(dim(collect(filter(scan, id > 1000L))), c(0L, 6L))
})

test_that("a filter on a scan of several files and on a rowwise table", {
  skip_if_no_gpu()
  a <- pq_write(tibble::tibble(id = 1:3000, v = rep(c(1.5, 2.5, 3.5), 1000L)), row_group_size = 500L)
  b <- pq_write(tibble::tibble(id = 3001:6000, v = rep(c(0.5, 4.5, 3.5), 1000L)), row_group_size = 500L)
  scan <- read_parquet_gpu(c(a, b), lazy = TRUE)
  r <- pq_run(filter(scan, id > 2500L, v > 3))
  expect_identical(r$result$id, dplyr::filter(tibble::tibble(id = 1:6000,
                     v = c(rep(c(1.5, 2.5, 3.5), 1000L), rep(c(0.5, 4.5, 3.5), 1000L))), id > 2500L, v > 3)$id)
  # (`v > 3` is on a double column, so it stays above the scan)
  expect_length(r$read$predicates, 1L)
  expect_lt(r$read$row_groups_after_stats, r$read$row_groups_total)
  # rowwise filters are not pushed
  r <- pq_run(filter(rowwise(read_parquet_gpu(a, lazy = TRUE)), id > 2L))
  expect_length(r$read$predicates, 0L)
  expect_identical(nrow(r$result), 2998L)
})

test_that("a segment under a barrier is pushed like the top one, and nothing crosses the barrier", {
  skip_if_no_gpu()
  sem <- pq_sem_df()
  path <- pq_write(sem)
  scan <- read_parquet_gpu(path, lazy = TRUE)
  other <- tbl_gpu(data.frame(id = c(2L, 4L, 6L, 8L), w = c(1, 2, 3, 4)), lazy = TRUE)
  cases <- list(
    list(quote(arrange(filter(scan, i > 1L), id)), quote(arrange(filter(sem, i > 1L), id)), 1L),
    list(quote(summarise(filter(scan, i > 1L), m = n())), quote(summarise(filter(sem, i > 1L), m = dplyr::n())), 1L),
    list(quote(filter(arrange(filter(scan, i > 1L), id), d > 0)),
         quote(filter(arrange(filter(sem, i > 1L), id), d > 0)), 1L),
    list(quote(filter(slice_head(filter(scan, i > 1L), n = 5L), d > 0)),
         quote(filter(slice_head(filter(sem, i > 1L), n = 5L), d > 0)), 1L),
    list(quote(distinct(filter(scan, i > 1L), g)), quote(distinct(filter(sem, i > 1L), g)), 1L),
    list(quote(left_join(filter(scan, i > 1L), other, by = "id")),
         quote(left_join(filter(sem, i > 1L), tibble::as_tibble(collect(other)), by = "id")), 1L),
    list(quote(filter(left_join(scan, other, by = "id"), i > 1L)),
         quote(filter(left_join(sem, tibble::as_tibble(collect(other)), by = "id"), i > 1L)), NA),
    list(quote(filter(group_by(filter(scan, i > 1L), g), d > 0)),
         quote(filter(group_by(filter(sem, i > 1L), g), d > 0)), 1L),
    list(quote(filter(ungroup(group_by(scan, g)), i > 1L)), quote(filter(sem, i > 1L)), 1L)
  )
  for (case in cases) {
    r <- pq_run(eval(case[[1L]]))
    want <- eval(case[[2L]])
    expect_equal(tibble::as_tibble(r$result), tibble::as_tibble(want), ignore_attr = TRUE, info = deparse(case[[1L]]))
    if (!is.na(case[[3L]])) {
      expect_length(r$read$predicates, case[[3L]])
    }
  }
})
