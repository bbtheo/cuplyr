# F53 item 1 acceptance tests: strings reach the GPU as UTF-8.
#
# Base R and vctrs compare strings after translating them to UTF-8, so a
# latin1-marked "café" equals the UTF-8 one in every dplyr verb. cuplyr must
# translate every string that crosses into cudf (column upload, factor levels,
# literals and R values in expressions, %in% sets, joins/set ops through
# upload) and mark collected non-ASCII strings as UTF-8.
#
# The parity oracle compares with expect_equal(), which does not see
# encodings, so these tests also check identical(), Encoding() and the bytes.

f53_u <- "café"                              # UTF-8 marked
f53_l <- iconv(f53_u, "UTF-8", "latin1")          # latin1 marked, same value
f53_u2 <- "été"
f53_l2 <- iconv(f53_u2, "UTF-8", "latin1")

f53_bytes_utf8 <- function(x) lapply(enc2utf8(x), charToRaw)

# Collected strings: equal to the input, non-ASCII ones marked UTF-8 and
# holding UTF-8 bytes.
f53_expect_utf8_equal <- function(got, want) {
  expect_identical(got, want)
  expect_identical(f53_bytes_utf8(got), f53_bytes_utf8(want))
  non_ascii <- !is.na(got) & grepl("[^ -~]", enc2utf8(got), useBytes = TRUE)
  expect_true(all(Encoding(got[non_ascii]) == "UTF-8"))
}

f53_both <- function(df, pipeline, ...) {
  expect_same_as_dplyr(df, pipeline, ...)
  expect_same_as_dplyr_lazy(df, pipeline, ...)
}

# ---- upload / collect round trip ---------------------------------------------

test_that("a latin1 character column round-trips as UTF-8 with the same values", {
  skip_if_no_gpu()
  d <- tibble::tibble(s = c(f53_l, "x", NA, f53_l2), n = 1:4)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(tbl_gpu(d, lazy = lazy))$s
    f53_expect_utf8_equal(got, d$s)
    expect_identical(Encoding(got), c("UTF-8", "unknown", "unknown", "UTF-8"))
  }
  # a UTF-8 column comes back unchanged (and still marked UTF-8)
  du <- tibble::tibble(s = c(f53_u, "x", NA))
  got <- collect(tbl_gpu(du))$s
  f53_expect_utf8_equal(got, du$s)
  # head() / print path
  got_head <- collect(head(tbl_gpu(d), 2))$s
  f53_expect_utf8_equal(got_head, d$s[1:2])
})

test_that("a column mixing latin1 and UTF-8 spellings of one value is one value", {
  skip_if_no_gpu()
  d <- tibble::tibble(s = c(f53_u, f53_l, "x"), v = 1:3)
  f53_both(d, function(t) count(t, s), arrange_by = "s")
  f53_both(d, function(t) distinct(t, s), arrange_by = "s")
  f53_both(d, function(t) summarise(group_by(t, s), v = sum(v)), arrange_by = "s")
  f53_both(d, function(t) filter(t, s == f53_u))
  f53_both(d, function(t) mutate(t, n = nchar(s), up = toupper(s)))
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(count(tbl_gpu(d, lazy = lazy), s))
    expect_equal(nrow(got), 2L)
    f53_expect_utf8_equal(sort(got$s), sort(c(f53_u, "x")))
  }
})

test_that("factor levels in latin1 are uploaded as UTF-8", {
  skip_if_no_gpu()
  f <- factor(c(f53_l, "x", f53_l, NA), levels = c("x", f53_l))
  d <- tibble::tibble(f = f, v = 1:4)
  for (lazy in c(FALSE, TRUE)) {
    got <- collect(tbl_gpu(d, lazy = lazy))$f
    expect_identical(as.integer(got), as.integer(f))
    f53_expect_utf8_equal(levels(got), levels(f))
  }
  f53_both(d, function(t) filter(t, f == f53_u))
  f53_both(d, function(t) filter(t, as.character(f) == f53_u))
  f53_both(d, function(t) mutate(t, s = as.character(f), up = toupper(as.character(f))))
  f53_both(d, function(t) count(t, f))
})

# ---- literals and R values in expressions ---------------------------------------

test_that("a UTF-8 column matches a latin1 literal / R value (filter, mutate, %in%)", {
  skip_if_no_gpu()
  d <- tibble::tibble(s = c(f53_u, "x", NA, f53_u2), v = 1:4)
  lat <- f53_l
  f53_both(d, function(t) filter(t, s == lat))
  f53_both(d, function(t) filter(t, s != lat))
  f53_both(d, function(t) filter(t, s %in% c(lat, f53_l2)))
  f53_both(d, function(t) filter(t, s == !!f53_l))
  f53_both(d, function(t) filter(t, s == .env$lat))
  f53_both(d, function(t) mutate(t, hit = s == lat, set = s %in% c(lat, "zz")))
  f53_both(d, function(t) mutate(t, has = grepl(lat, s, fixed = TRUE)))
  f53_both(d, function(t) summarise(t, n = sum(s == lat, na.rm = TRUE)))
  for (lazy in c(FALSE, TRUE)) {
    expect_equal(nrow(collect(filter(tbl_gpu(d, lazy = lazy), s == lat))), 1L)
  }
})

test_that("a latin1 column matches a UTF-8 literal", {
  skip_if_no_gpu()
  d <- tibble::tibble(s = c(f53_l, "x", NA, f53_l2), v = 1:4)
  f53_both(d, function(t) filter(t, s == f53_u))
  f53_both(d, function(t) filter(t, s %in% c(f53_u, f53_u2)))
  f53_both(d, function(t) mutate(t, n = nchar(s), up = toupper(s),
                                  has = grepl(f53_u, s, fixed = TRUE)))
  for (lazy in c(FALSE, TRUE)) {
    expect_equal(nrow(collect(filter(tbl_gpu(d, lazy = lazy), s == f53_u))), 1L)
  }
})

test_that("latin1 values produced by expressions come back as UTF-8", {
  skip_if_no_gpu()
  d <- tibble::tibble(s = c(f53_u, "x", "y"), v = 1:3)
  lat <- f53_l
  pipelines <- list(
    function(t) mutate(t, p = paste0(s, lat)),
    function(t) mutate(t, p = paste(s, lat, sep = "-")),
    function(t) mutate(t, r = dplyr::recode(s, x = lat)),
    function(t) mutate(t, w = dplyr::case_when(v == 2L ~ lat, .default = s)),
    function(t) mutate(t, i = dplyr::if_else(v > 1L, lat, s)),
    function(t) mutate(t, c = dplyr::coalesce(dplyr::if_else(v == 3L, NA_character_, s), lat))
  )
  for (p in pipelines) {
    f53_both(d, p)
    for (lazy in c(FALSE, TRUE)) {
      got <- collect(p(tbl_gpu(d, lazy = lazy)))
      want <- p(d)
      new_col <- setdiff(names(want), names(d))
      f53_expect_utf8_equal(got[[new_col]], want[[new_col]])
    }
  }
})

# ---- joins, set operations, binds ---------------------------------------------------

test_that("joins match latin1 and UTF-8 keys like dplyr", {
  skip_if_no_gpu()
  x <- tibble::tibble(s = c(f53_u, "x", f53_u2), a = 1:3)
  y <- tibble::tibble(s = c(f53_l, "x", "z"), b = 4:6)
  want_inner <- inner_join(x, y, by = "s")
  want_left <- left_join(x, y, by = "s")
  want_semi <- semi_join(x, y, by = "s")
  want_anti <- anti_join(x, y, by = "s")
  for (lazy in c(FALSE, TRUE)) {
    gx <- tbl_gpu(x, lazy = lazy)
    gy <- tbl_gpu(y, lazy = lazy)
    expect_equal(collect(inner_join(gx, gy, by = "s")), want_inner)
    expect_equal(collect(left_join(gx, gy, by = "s")), want_left)
    expect_equal(collect(semi_join(gx, gy, by = "s")), want_semi)
    expect_equal(collect(anti_join(gx, gy, by = "s")), want_anti)
    # host y uploaded through copy = TRUE
    expect_equal(collect(inner_join(gx, y, by = "s", copy = TRUE)), want_inner)
    # latin1 on the x side
    expect_equal(collect(inner_join(gy, gx, by = "s")), inner_join(y, x, by = "s"))
  }
})

test_that("set operations and bind_rows treat latin1 and UTF-8 as equal", {
  skip_if_no_gpu()
  a <- tibble::tibble(s = c(f53_u, "x"))
  b <- tibble::tibble(s = c(f53_l, "y"))
  for (lazy in c(FALSE, TRUE)) {
    ga <- tbl_gpu(a, lazy = lazy)
    gb <- tbl_gpu(b, lazy = lazy)
    expect_equal(collect(intersect(ga, gb)), intersect(a, b))
    expect_equal(arrange(collect(union(ga, gb)), s), arrange(union(a, b), s))
    expect_equal(collect(setdiff(ga, gb)), setdiff(a, b))
    bound <- cuplyr::bind_rows(ga, gb)
    expect_equal(collect(count(bound, s)) |> arrange(s),
                 count(dplyr::bind_rows(a, b), s) |> arrange(s))
    got <- collect(bound)$s
    f53_expect_utf8_equal(got, dplyr::bind_rows(a, b)$s)
  }
})

# ---- invalid bytes and the "bytes" encoding -----------------------------------------

test_that("invalid UTF-8 bytes in an unknown/UTF-8 string pass through like dplyr", {
  skip_if_no_gpu()
  skip_if_not(isTRUE(l10n_info()[["UTF-8"]]), "needs a UTF-8 session")
  bad <- "ab\xffcd"                       # "unknown", invalid UTF-8
  bad_u <- bad
  Encoding(bad_u) <- "UTF-8"              # marked UTF-8, still invalid
  d <- tibble::tibble(s = c(bad, "x"), v = 1:2)
  du <- tibble::tibble(s = c(bad_u, "x"), v = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    expect_identical(collect(tbl_gpu(d, lazy = lazy))$s, d$s)
    expect_identical(collect(tbl_gpu(du, lazy = lazy))$s, du$s)
    expect_equal(nrow(collect(filter(tbl_gpu(d, lazy = lazy), s == bad))), 1L)
    expect_equal(nrow(collect(filter(tbl_gpu(du, lazy = lazy), s == bad_u))), 1L)
  }
})

test_that("strings in the 'bytes' encoding are refused with a clear error", {
  skip_if_no_gpu()
  by <- "ab\xffcd"
  Encoding(by) <- "bytes"
  expect_error(tbl_gpu(tibble::tibble(s = by)), regexp = "bytes")
  g <- tbl_gpu(tibble::tibble(s = c("x", "y")))
  expect_error(collect(filter(g, s == by)), regexp = "bytes")
  expect_error(collect(mutate(g, t = paste0(s, by))), regexp = "bytes")
})

# ---- a latin1 (non-UTF-8) session ---------------------------------------------------

test_that("native strings in a latin1 session are converted to UTF-8", {
  skip_if_no_gpu()
  old <- Sys.getlocale("LC_CTYPE")
  candidates <- c("en_US.ISO-8859-1", "en_US.ISO8859-1", "en_US.iso88591", "en_US")
  ok <- FALSE
  for (loc in candidates) {
    res <- suppressWarnings(Sys.setlocale("LC_CTYPE", loc))
    if (nzchar(res) && !isTRUE(l10n_info()[["UTF-8"]]) && isTRUE(l10n_info()[["Latin-1"]])) {
      ok <- TRUE
      break
    }
  }
  if (!ok) {
    suppressWarnings(Sys.setlocale("LC_CTYPE", old))
    skip("no latin1 locale on this host")
  }
  withr::defer(Sys.setlocale("LC_CTYPE", old))
  native <- rawToChar(as.raw(c(0x63, 0x61, 0x66, 0xe9)))   # "café" in latin1, unmarked
  expect_identical(Encoding(native), "unknown")
  d <- tibble::tibble(s = c(native, "x"))
  for (lazy in c(FALSE, TRUE)) {
    g <- tbl_gpu(d, lazy = lazy)
    expect_equal(nrow(collect(filter(g, s == f53_u))), 1L)
    got <- collect(g)$s
    expect_identical(got, d$s)
    expect_identical(charToRaw(got[1]), charToRaw(f53_u))
    expect_equal(nrow(collect(filter(tbl_gpu(tibble::tibble(s = f53_u), lazy = lazy),
                                     s == native))), 1L)
  }
})
