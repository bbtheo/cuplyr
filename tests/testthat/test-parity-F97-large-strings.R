# F97: string columns with INT64 offsets ("large strings") and sliced views.
#
# libcudf picks a strings column's offset width per result: INT64 once the
# column's character bytes reach `LIBCUDF_LARGE_STRINGS_THRESHOLD` (default
# 2^31 - 1), INT32 below. The variable is read on every call, so setting it low
# makes libcudf (and, since F97, cuplyr's upload, which follows the same rule)
# produce INT64 offsets on tiny data. That is how these tests reach the large
# strings code paths without gigabytes of data; the real > 2^31 byte column is
# in test-parity-F97-large-upload.R (opt-in, CUPLYR_TEST_LARGE=1).
#
# Internal hooks used here (src/transfer_io.cpp):
#   gpu_string_offsets_width(ptr, col_idx0) -> 32L / 64L, NA for non-strings
#   gpu_collect_slice(ptr, names, begin, end) -> gpu_collect() of the rows
#     [begin, end) of a zero-copy cudf::slice() view

local_large_strings <- function(threshold = "1", .env = parent.frame()) {
  withr::local_envvar(LIBCUDF_LARGE_STRINGS_THRESHOLD = threshold, .local_envir = .env)
}

offsets_width <- function(g, col) {
  idx <- match(col, unclass(g)$schema$names) - 1L
  gpu_string_offsets_width(unclass(g)$ptr, idx)
}

f97_df <- function() {
  tibble::tibble(
    id = 1:9,
    s = c("apple", NA, "", "Banana", "café", "日本語", "apple",
          "a longer string with spaces", NA),
    k = c("x", "y", "x", "z", "y", "x", NA, "z", "y"),
    f = factor(c("lo", "hi", "lo", "mid", "hi", "lo", "mid", NA, "hi"),
               levels = c("lo", "mid", "hi")),
    v = c(1.5, NA, 3, 4, 5, 6, 7, 8, 9)
  )
}

string_bytes <- function(x) sum(nchar(x[!is.na(x)], type = "bytes"))

# ---------------------------------------------------------------------------
# Collect: libcudf-made INT64 offsets
# ---------------------------------------------------------------------------

test_that("collect() reads string columns that libcudf builds with INT64 offsets", {
  skip_if_no_gpu()
  df <- f97_df()
  g <- tbl_gpu(df)
  local_large_strings()
  r <- mutate(g, t = paste0(s, "_x"), u = toupper(k), w = substr(s, 2, 4))
  expect_identical(offsets_width(r, "t"), 64L)
  expect_identical(offsets_width(r, "u"), 64L)
  expected <- dplyr::mutate(df, t = paste0(s, "_x"), u = toupper(k), w = substr(s, 2, 4))
  expect_equal(tibble::as_tibble(collect(r)), expected)

  lz <- as_lazy(g) |> mutate(t = paste0(s, "_x"), u = toupper(k), w = substr(s, 2, 4))
  expect_equal(tibble::as_tibble(collect(lz)), expected)
})

# ---------------------------------------------------------------------------
# Upload: the width rule
# ---------------------------------------------------------------------------

test_that("upload keeps INT32 offsets by default and uses INT64 at libcudf's threshold", {
  skip_if_no_gpu()
  df <- f97_df()
  nbytes <- string_bytes(df$s)

  g32 <- tbl_gpu(df)
  expect_identical(offsets_width(g32, "s"), 32L)
  expect_identical(offsets_width(g32, "id"), NA_integer_)
  expect_identical(offsets_width(g32, "f"), NA_integer_)

  withr::with_envvar(c(LIBCUDF_LARGE_STRINGS_THRESHOLD = as.character(nbytes)), {
    at <- tbl_gpu(df)
    expect_identical(offsets_width(at, "s"), 64L)
    expect_equal(tibble::as_tibble(collect(at)), df)
  })
  withr::with_envvar(c(LIBCUDF_LARGE_STRINGS_THRESHOLD = as.character(nbytes + 1)), {
    below <- tbl_gpu(df)
    expect_identical(offsets_width(below, "s"), 32L)
    expect_equal(tibble::as_tibble(collect(below)), df)
  })
})

test_that("with large strings disabled, an upload at the limit is a classed error naming the column", {
  skip_if_no_gpu()
  withr::local_envvar(
    LIBCUDF_LARGE_STRINGS_ENABLED = "0",
    LIBCUDF_LARGE_STRINGS_THRESHOLD = "16"
  )
  df <- tibble::tibble(a = c("x", "y"), big_col = c("abcdefghij", "klmnopqrstu"))
  expect_identical(string_bytes(df$big_col), 21L)
  err <- expect_error(tbl_gpu(df), class = "cuplyr_error_string_too_large")
  expect_match(conditionMessage(err), "big_col", fixed = TRUE)
  expect_match(conditionMessage(err), "21", fixed = TRUE)

  ok <- tbl_gpu(tibble::tibble(s = c("abc", "def", NA)))
  expect_identical(offsets_width(ok, "s"), 32L)
  expect_identical(collect(ok)$s, c("abc", "def", NA))
})

test_that("empty, all-NA and all-empty string columns round-trip under the threshold", {
  skip_if_no_gpu()
  local_large_strings()
  cases <- list(character(0), NA_character_, c(NA_character_, NA_character_),
                c("", ""), c("", NA, "x"), "é")
  for (s in cases) {
    df <- tibble::tibble(s = s)
    g <- tbl_gpu(df)
    expect_identical(collect(g)$s, s)
    expect_identical(collect(mutate(g, t = paste0(s, "y")))$t,
                     dplyr::mutate(df, t = paste0(s, "y"))$t)
    expect_identical(collect(mutate(as_lazy(g), n = nchar(s)))$n, nchar(s))
  }
})

# ---------------------------------------------------------------------------
# Downstream verbs on columns cuplyr uploaded with INT64 offsets
# ---------------------------------------------------------------------------

test_that("verbs give dplyr's result on INT64-offset string columns (eager and lazy)", {
  skip_if_no_gpu()
  df <- f97_df()
  local_large_strings()
  expect_identical(offsets_width(tbl_gpu(df), "s"), 64L)

  pipelines <- list(
    filter_eq = function(d) filter(d, s == "apple"),
    filter_in = function(d) filter(d, s %in% c("Banana", "café", "zzz")),
    filter_grepl = function(d) filter(d, grepl("an", s)),
    filter_starts = function(d) filter(d, startsWith(s, "a")),
    arrange = function(d) arrange(d, s, id),
    arrange_desc = function(d) arrange(d, dplyr::desc(k), s),
    distinct = function(d) distinct(d, k, s),
    count = function(d) dplyr::count(d, k),
    summarise = function(d) d |> group_by(k) |> summarise(n = n(), m = sum(v, na.rm = TRUE)),
    mutate_strings = function(d) mutate(d,
      n = nchar(s), t = paste0(k, "-", s), u = toupper(s), w = substr(s, 2, 4),
      g = gsub("a", "A", s), tr = trimws(paste0(" ", s, " "))
    ),
    mutate_if_else = function(d) mutate(d, t = dplyr::if_else(is.na(s), "missing", s),
                                           c = dplyr::coalesce(s, k)),
    factor_labels = function(d) mutate(d, fl = as.character(f)),
    factor_cmp = function(d) filter(d, f == "hi"),
    grouped_mutate = function(d) d |> group_by(k) |> mutate(first_s = dplyr::first(s), lag_s = dplyr::lag(s)),
    select_rename = function(d) select(d, s2 = s, k),
    slice_head = function(d) slice_head(d, n = 4),
    slice_tail = function(d) slice_tail(d, n = 4),
    slice_idx = function(d) slice(d, c(2, 5, 8))
  )
  for (nm in names(pipelines)) {
    expect_same_as_dplyr(df, pipelines[[nm]])
    expect_same_as_dplyr_lazy(df, pipelines[[nm]])
  }
})

test_that("joins on string keys work with INT64 and INT32 offsets mixed", {
  skip_if_no_gpu()
  left <- tibble::tibble(k = c("x", "y", "z", NA, "w"), a = 1:5)
  right <- tibble::tibble(k = c("y", "x", "x", "q", NA), b = c("B1", "B2", "B3", "B4", "B5"))
  gl64 <- withr::with_envvar(c(LIBCUDF_LARGE_STRINGS_THRESHOLD = "1"), tbl_gpu(left))
  gr64 <- withr::with_envvar(c(LIBCUDF_LARGE_STRINGS_THRESHOLD = "1"), tbl_gpu(right))
  gl32 <- tbl_gpu(left)
  gr32 <- tbl_gpu(right)
  expect_identical(offsets_width(gl64, "k"), 64L)
  expect_identical(offsets_width(gr32, "k"), 32L)

  for (pair in list(list(gl64, gr32), list(gl32, gr64), list(gl64, gr64))) {
    gx <- pair[[1]]
    gy <- pair[[2]]
    expect_equal(tibble::as_tibble(collect(left_join(gx, gy, by = "k"))),
                 dplyr::left_join(left, right, by = "k"))
    expect_equal(tibble::as_tibble(collect(inner_join(gx, gy, by = "k"))),
                 dplyr::inner_join(left, right, by = "k"))
    expect_equal(tibble::as_tibble(collect(semi_join(gx, gy, by = "k"))),
                 dplyr::semi_join(left, right, by = "k"))
    expect_equal(tibble::as_tibble(collect(anti_join(gx, gy, by = "k"))),
                 dplyr::anti_join(left, right, by = "k"))
    expect_equal(tibble::as_tibble(collect(right_join(gy, gx, by = "k"))),
                 dplyr::right_join(right, left, by = "k"))
  }
})

test_that("bind_rows() concatenates INT64 and INT32 string columns", {
  skip_if_no_gpu()
  a <- tibble::tibble(s = c("one", NA, "three"), x = 1:3)
  b <- tibble::tibble(s = c("", "five"), x = 4:5)
  c0 <- tibble::tibble(x = 6L)
  ga <- withr::with_envvar(c(LIBCUDF_LARGE_STRINGS_THRESHOLD = "1"), tbl_gpu(a))
  gb <- tbl_gpu(b)
  gc <- tbl_gpu(c0)
  expect_identical(offsets_width(ga, "s"), 64L)
  expect_identical(offsets_width(gb, "s"), 32L)

  expect_equal(tibble::as_tibble(collect(cuplyr::bind_rows(ga, gb, gc))),
               dplyr::bind_rows(a, b, c0))
  expect_equal(tibble::as_tibble(collect(cuplyr::bind_rows(gb, ga))),
               dplyr::bind_rows(b, a))
  local_large_strings()
  expect_equal(tibble::as_tibble(collect(cuplyr::bind_rows(gc, ga, gb))),
               dplyr::bind_rows(c0, a, b))
})

test_that("every path that brings strings back to R reads INT64 offsets", {
  skip_if_no_gpu()
  df <- f97_df()
  local_large_strings()
  g <- tbl_gpu(df)
  expect_identical(offsets_width(g, "s"), 64L)

  expect_equal(tibble::as_tibble(collect(g)), df)
  expect_identical(pull(g, s), df$s)
  expect_identical(g$s, df$s)
  expect_identical(g[["k"]], df$k)
  expect_identical(as.data.frame(g)$s, df$s)
  expect_identical(tibble::as_tibble(g)$s, df$s)

  sp <- g |> group_by(k) |> group_split()
  want <- dplyr::group_split(dplyr::group_by(df, k))
  expect_identical(length(sp), length(want))
  for (i in seq_along(want)) {
    expect_identical(collect(sp[[i]])$s, want[[i]]$s)
  }

  # print()/glimpse() previews (gpu_head())
  h <- gpu_head(unclass(g)$ptr, 4L, names(df))
  expect_identical(h$s, df$s[1:4])
  expect_identical(h$k, df$k[1:4])
  h_all <- gpu_head(unclass(g)$ptr, 100L, names(df))
  expect_identical(h_all$s, df$s)
  out <- capture.output(print(g))
  expect_true(any(grepl("Banana", out, fixed = TRUE)))
  out <- capture.output(glimpse(g))
  expect_true(any(grepl("Banana", out, fixed = TRUE)))

  # a libcudf-made INT64 column through the same paths
  r <- mutate(g, t = paste0(s, "!"))
  expect_identical(offsets_width(r, "t"), 64L)
  expect_identical(pull(r, t), paste0(df$s, "!"))
  expect_identical(gpu_head(unclass(r)$ptr, 3L, names(r))$t, paste0(df$s, "!")[1:3])
})

# ---------------------------------------------------------------------------
# Sliced views (no verb produces one today; collect must not depend on that)
# ---------------------------------------------------------------------------

f97_slice_df <- function() {
  n <- 40L
  na_at <- function(x, k) {
    x[k] <- NA
    x
  }
  tibble::tibble(
    i = na_at(seq_len(n), c(2, 9, 10, 17, 33)),
    d = na_at(seq_len(n) / 4, c(1, 11, 12, 25)),
    l = na_at(rep(c(TRUE, FALSE, TRUE), length.out = n), c(4, 13, 30)),
    s = na_at(paste0("row", seq_len(n), strrep("z", seq_len(n) %% 5)), c(3, 10, 11, 19, 40)),
    dt = na_at(as.Date("2020-01-01") + seq_len(n), c(5, 14)),
    ts = na_at(as.POSIXct("2020-01-01", tz = "UTC") + seq_len(n) * 3600, c(6, 15))
  )
}

check_slices <- function(g, df) {
  n <- nrow(df)
  bounds <- list(c(0, n), c(3, 11), c(9, 27), c(13, 13), c(37, n), c(1, 2), c(8, 16))
  for (b in bounds) {
    got <- gpu_collect_slice(unclass(g)$ptr, names(df), b[[1]], b[[2]])
    want <- df[b[[1]] + seq_len(b[[2]] - b[[1]]), ]
    expect_equal(tibble::as_tibble(got), want,
                 info = sprintf("slice [%d, %d)", b[[1]], b[[2]]))
  }
}

test_that("collect honours a sliced view's row offset for every type", {
  skip_if_no_gpu()
  df <- f97_slice_df()
  g <- tbl_gpu(df)
  expect_identical(offsets_width(g, "s"), 32L)
  check_slices(g, df)
})

test_that("collect honours a sliced view's row offset with INT64 string offsets", {
  skip_if_no_gpu()
  df <- f97_slice_df()
  local_large_strings()
  g <- tbl_gpu(df)
  expect_identical(offsets_width(g, "s"), 64L)
  check_slices(g, df)

  r <- mutate(g, s = paste0(s, "|", i))
  expect_identical(offsets_width(r, "s"), 64L)
  check_slices(r, dplyr::mutate(df, s = paste0(s, "|", i)))
})
