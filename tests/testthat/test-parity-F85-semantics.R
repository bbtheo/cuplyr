# F85: a pipeline read from parquet equals the same pipeline on dplyr and on
# tbl_gpu(df), eager and lazy, with pushdown on and off.

run_all <- function(pipes, df = pq_sem_df(), ...) {
  for (nm in names(pipes)) {
    expect_parquet_same_as_dplyr(df, pipes[[nm]], ..., info = nm)
  }
}

test_that("a bare read equals the data frame, every column and type", {
  skip_if_no_gpu()
  expect_parquet_same_as_dplyr(pq_sem_df(), function(x) x)
  expect_parquet_same_as_dplyr(pq_sem_df(), function(x) dplyr::select(x, s, id, ts, f))
})

test_that("NA in a predicate drops the row, also inside |, & and !", {
  skip_if_no_gpu()
  run_all(list(
    gt = function(x) filter(x, i > 1L),
    or_na_true = function(x) filter(x, i > 1L | l),
    or_two_cols = function(x) filter(x, i > 3L | d > 1),
    and_na_false = function(x) filter(x, i > 1L & l),
    not_cmp = function(x) filter(x, !(i > 1L)),
    not_and = function(x) filter(x, !(i > 1L & l)),
    not_or = function(x) filter(x, !(i > 1L | l)),
    is_na = function(x) filter(x, is.na(i)),
    not_is_na = function(x) filter(x, !is.na(i)),
    is_na_or = function(x) filter(x, is.na(i) | i > 3L),
    bare_lgl = function(x) filter(x, l),
    not_lgl = function(x) filter(x, !l),
    two_dots = function(x) filter(x, i > 1L, d > 0),
    na_lit = function(x) filter(x, i > NA_integer_),
    eq_na_lit = function(x) filter(x, s == NA_character_),
    and_bare_na = function(x) filter(x, i > 1L & NA),
    na_and = function(x) filter(x, NA & i > 1L),
    and_var_na = function(x) {
      flag <- NA
      filter(x, i > 1L & flag)
    },
    bare_na = function(x) filter(x, NA),
    two_dots_na = function(x) filter(x, i > 1L, NA)
  ))
})

test_that("NaN in a parquet double reads as dplyr reads it", {
  skip_if_no_gpu()
  run_all(list(
    gt = function(x) filter(x, d > 0),
    lt = function(x) filter(x, d < 2),
    eq = function(x) filter(x, d == 1.5),
    ne = function(x) filter(x, d != 1.5),
    not_gt = function(x) filter(x, !(d > 0)),
    is_na = function(x) filter(x, is.na(d)),
    not_is_na = function(x) filter(x, !is.na(d)),
    is_nan = function(x) filter(x, is.nan(d)),
    or_ = function(x) filter(x, d > 100 | i == 1L),
    inf_lit = function(x) filter(x, d < Inf),
    neg_inf_lit = function(x) filter(x, d > -Inf),
    nan_lit = function(x) filter(x, d == NaN)
  ))
})

test_that("grouped, .by, window and reducer filters keep dplyr's rows", {
  skip_if_no_gpu()
  run_all(list(
    grouped = function(x) filter(group_by(x, g), i > 1L),
    grouped_ungroup = function(x) ungroup(filter(group_by(x, g), i > 1L)),
    by = function(x) filter(x, i > 1L, .by = g),
    reducer = function(x) filter(x, i > mean(i, na.rm = TRUE)),
    grouped_reducer = function(x) filter(group_by(x, g), i > mean(i, na.rm = TRUE)),
    window = function(x) filter(x, i == max(i, na.rm = TRUE), .by = g),
    lag_ = function(x) filter(x, id > lag(id, default = 0L)),
    row_number_ = function(x) filter(group_by(x, g), dplyr::row_number() == 1L)
  ))
})

test_that("a filter on a column an earlier verb changed keeps dplyr's rows", {
  skip_if_no_gpu()
  run_all(list(
    mutate_same = function(x) filter(mutate(x, i = i * 2L), i > 4L),
    mutate_new = function(x) filter(mutate(x, z = i + 1L), z > 2L),
    mutate_other = function(x) filter(mutate(x, z = i + 1L), d > 0),
    mutate_swap = function(x) filter(mutate(x, i = id, id = 1L), i > 6L),
    rename_ = function(x) filter(rename(x, j = i), j > 1L),
    rename_swap = function(x) filter(rename(x, i = id, id = i), i > 6L),
    select_rename = function(x) filter(select(x, j = i, id), j > 1L),
    transmute_ = function(x) filter(transmute(x, i = id), i > 6L),
    mutate_factor = function(x) filter(mutate(x, f = as.character(f)), f == "a")
  ))
})

test_that("a filter or select behind a barrier stays behind it", {
  skip_if_no_gpu()
  other <- tibble::tibble(id = c(2L, 4L, 4L, 9L), w = c(10, 20, 30, 40))
  run_all(list(
    slice_head_ = function(x) filter(slice_head(x, n = 4L), i > 1L),
    slice_tail_ = function(x) filter(slice_tail(x, n = 4L), i > 1L),
    slice_ = function(x) filter(slice(x, 2:6), i > 1L),
    slice_min_ = function(x) filter(slice_min(x, id, n = 3L), i > 1L),
    distinct_ = function(x) filter(distinct(x, g, .keep_all = TRUE), i > 1L),
    arrange_slice = function(x) filter(slice_head(arrange(x, desc(id)), n = 4L), i > 1L),
    summarise_ = function(x) filter(summarise(x, m = max(i, na.rm = TRUE), .by = g), m > 3L),
    count_ = function(x) filter(count(x, g), n > 1L),
    select_after_slice = function(x) select(filter(slice_head(x, n = 8L), i > 1L), id),
    collapse_ = function(x) filter(dplyr::collapse(x), i > 1L)
  ))
  # joins (other side is a plain table on the dplyr side and a GPU table here)
  path <- pq_write(pq_sem_df())
  for (lazy in c(TRUE, FALSE)) {
    res <- read_parquet_gpu(path, lazy = lazy) |>
      left_join(tbl_gpu(other, lazy = lazy), by = "id", relationship = "many-to-many") |>
      filter(i > 1L, !is.na(w)) |>
      collect()
    expect_equal(tibble::as_tibble(res),
                 tibble::as_tibble(filter(left_join(pq_sem_df(), other, by = "id",
                                                    relationship = "many-to-many"),
                                          i > 1L, !is.na(w))))
  }
})

test_that("a literal whose type differs from the column does not change the rows", {
  skip_if_no_gpu()
  big <- tibble::tibble(
    id = 1:6,
    i = c(1L, 2L, NA, 2147483647L, -2147483647L, 0L),
    d = c(0.5, 1, 1.5, 1e15, 1e15 + 2, NA)
  )
  run_all(list(
    int_col_dbl_whole = function(x) filter(x, i > 1),
    int_col_dbl_frac = function(x) filter(x, i > 1.5),
    int_col_frac_eq = function(x) filter(x, i == 1.5),
    int_col_huge = function(x) filter(x, i < 1e15),
    int_col_huge_eq = function(x) filter(x, i == 1e15),
    dbl_col_int = function(x) filter(x, d > 1L),
    dbl_col_big = function(x) filter(x, d == 1e15),
    dbl_col_big_ge = function(x) filter(x, d >= 1e15 + 1),
    bool_lit = function(x) filter(x, i == TRUE)
  ), df = big)
})

test_that("string comparisons match dplyr", {
  skip_if_no_gpu()
  run_all(list(
    eq = function(x) filter(x, s == "x"),
    ne = function(x) filter(x, s != "x"),
    empty = function(x) filter(x, s == ""),
    non_ascii = function(x) filter(x, s == "é"),
    latin1_lit = function(x) filter(x, s == iconv("é", "UTF-8", "latin1")),
    in_ = function(x) filter(x, s %in% c("x", "z")),
    lt = function(x) filter(x, s < "y"),
    ge = function(x) filter(x, s >= "x"),
    nchar_ = function(x) filter(x, nchar(s) > 0L)
  ))
})

test_that("Date and POSIXct comparisons match dplyr", {
  skip_if_no_gpu()
  d0 <- as.Date("2020-01-05")
  t0 <- as.POSIXct("2021-03-04 05:06:10", tz = "UTC")
  run_all(list(
    date_ge = function(x) filter(x, dt >= d0),
    date_eq = function(x) filter(x, dt == d0),
    date_frac = function(x) filter(x, dt > as.Date("2020-01-05") + 0.5),
    date_in = function(x) filter(x, dt %in% (d0 + 0:1)),
    ts_ge = function(x) filter(x, ts >= t0),
    ts_frac = function(x) filter(x, ts > t0 + 0.25),
    ts_exact_frac = function(x) filter(x, ts == as.POSIXct("2021-03-04 05:06:07", tz = "UTC") + 0.25),
    ts_other_zone = function(x) filter(x, ts >= as.POSIXct("2021-03-04 00:06:10", tz = "America/New_York"))
  ))
})

test_that("a Date literal against a POSIXct column gives dplyr's rows and condition", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  path <- pq_write(df)
  pipe <- function(x) filter(x, ts > as.Date("2021-03-04"))
  oracle <- tryCatch(suppressWarnings(tibble::as_tibble(pipe(tibble::as_tibble(df)))),
                     error = function(e) e)
  for (lazy in c(TRUE, FALSE)) {
    got <- tryCatch(suppressWarnings(tibble::as_tibble(collect(pipe(read_parquet_gpu(path, lazy = lazy))))),
                    error = function(e) e)
    if (inherits(oracle, "error")) expect_s3_class(got, "error") else expect_equal(got, oracle)
  }
})

test_that("factor comparisons match dplyr and the levels survive", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  ord <- df
  ord$f <- factor(df$f, levels = levels(df$f), ordered = TRUE)
  run_all(list(
    eq = function(x) filter(x, f == "a"),
    unused = function(x) filter(x, f == "unused"),
    absent_level = function(x) filter(x, f == "nope"),
    is_na = function(x) filter(x, is.na(f)),
    in_ = function(x) filter(x, f %in% c("a", "b")),
    as_chr = function(x) filter(x, as.character(f) == "b"),
    group_ = function(x) count(x, f, .drop = FALSE)
  ), df = df)
  run_all(list(
    ord_gt = function(x) filter(x, f > "b"),
    ord_le = function(x) filter(x, f <= "b")
  ), df = ord)
})

test_that("integer64 comparisons are exact past 2^53", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  big <- bit64::as.integer64("9007199254740993")
  df <- tibble::tibble(
    id = 1:6,
    b = c(big, big - 1L, bit64::as.integer64(NA), bit64::as.integer64(1L),
          bit64::as.integer64("-9007199254740993"), big + 1L)
  )
  run_all(list(
    eq_big = function(x) filter(x, b == big),
    gt_big = function(x) filter(x, b > big),
    int_lit = function(x) filter(x, b > 1L),
    dbl_whole = function(x) filter(x, b > 1),
    dbl_1e15 = function(x) filter(x, b > 1e15),
    dbl_frac = function(x) filter(x, b > 1.5),
    dbl_past_2_53 = function(x) filter(x, b >= 9007199254740992)
  ), df = df)
})

test_that("file row order is preserved across row groups", {
  skip_if_no_gpu()
  set.seed(8585)
  df <- tibble::tibble(id = 1:5000, v = sample.int(1000L, 5000L, replace = TRUE))
  for (pipe in list(function(x) x,
                    function(x) filter(x, v > 900L),
                    function(x) filter(x, id > 4000L),
                    function(x) select(filter(x, v %% 7L == 0L, id < 3000L), id))) {
    expect_parquet_same_as_dplyr(df, pipe, row_group_size = 100L)
  }
})

test_that("verbs on a parquet scan match dplyr over plans that read few columns", {
  skip_if_no_gpu()
  run_all(list(
    summarise_ = function(x) summarise(group_by(x, g), m = sum(i, na.rm = TRUE), n = n()),
    mutate_arrange = function(x) arrange(mutate(x, z = id * 2L), desc(z)),
    constant_mutate = function(x) select(mutate(x, k = 1L), k),
    count_all = function(x) count(x),
    filter_count = function(x) count(filter(x, i > 1L)),
    n_only = function(x) summarise(x, n = n()),
    pull_like = function(x) select(x, id)
  ), arrange_by = NULL)
})

test_that("empty and one-row files behave like dplyr on the same frames", {
  skip_if_no_gpu()
  df <- pq_sem_df()
  run_all(list(
    all_ = function(x) x,
    filter_ = function(x) filter(x, i > 1L),
    select_ = function(x) select(x, id, s)
  ), df = df[0L, ])
  run_all(list(
    all_ = function(x) x,
    filter_keep = function(x) filter(x, id == 1L),
    filter_drop = function(x) filter(x, id == 2L)
  ), df = df[1L, ])
})

test_that("reusing one lazy scan twice runs it twice with the same result", {
  skip_if_no_gpu()
  path <- pq_write(pq_sem_df())
  q <- read_parquet_gpu(path, lazy = TRUE) |> filter(i > 1L)
  a <- collect(q)
  b <- collect(q)
  expect_identical(tibble::as_tibble(a), tibble::as_tibble(b))
  c <- collect(compute(q))
  expect_identical(tibble::as_tibble(a), tibble::as_tibble(c))
})
