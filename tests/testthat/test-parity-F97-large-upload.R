# F97: a string column holding more than 2^31 - 1 bytes of characters.
#
# HEAVY: about 2.2 GB of strings in R, several GB of host buffers and up to
# ~7 GB on the GPU. Runs only when CUPLYR_TEST_LARGE=1, so it never runs under
# tools/test or the merge gate. Run it by hand:
#   CUPLYR_TEST_LARGE=1 tools/test-file tests/testthat/test-parity-F97-large-upload.R
#
# Few long strings (110 x 20 MB) instead of many short ones keep R's per-string
# overhead out of the way.

skip_if_not_large <- function() {
  skip_if_not(identical(Sys.getenv("CUPLYR_TEST_LARGE"), "1"),
              "heavy test: set CUPLYR_TEST_LARGE=1 to run")
}

test_that("a string column over 2^31 - 1 bytes uploads, computes and collects", {
  skip_if_not_large()
  skip_if_no_gpu()

  n <- 110L
  len <- 20000000L
  body <- strrep("x", len - 8L)
  s <- paste0(sprintf("%08d", seq_len(n)), body)
  rm(body)
  s[50L] <- NA
  total <- sum(as.numeric(nchar(s[!is.na(s)], type = "bytes")))
  expect_gt(total, .Machine$integer.max)
  df <- tibble::tibble(id = seq_len(n), s = s)

  g <- tbl_gpu(df)

  back <- collect(g)
  expect_identical(back$id, df$id)
  expect_identical(is.na(back$s), is.na(s))
  expect_identical(nchar(back$s, type = "bytes"), nchar(s, type = "bytes"))
  expect_identical(substr(back$s, 1L, 8L), substr(s, 1L, 8L))
  expect_true(identical(back$s, s))
  rm(back)
  gc()

  # Device ops on the INT64-offset column.
  got <- g |> mutate(n = nchar(s), p = substr(s, 1L, 8L)) |> select(id, n, p) |> collect()
  want <- df |> dplyr::mutate(n = nchar(s), p = substr(s, 1L, 8L)) |> dplyr::select(id, n, p)
  expect_equal(tibble::as_tibble(got), want)

  hit <- g |> filter(substr(s, 1L, 8L) == "00000100") |> select(id) |> collect()
  expect_identical(hit$id, 100L)

  # A string result libcudf builds with INT64 offsets.
  got <- g |> mutate(t = paste0(s, "!")) |> mutate(m = nchar(t)) |> select(id, m) |> collect()
  want <- df |> dplyr::mutate(t = paste0(s, "!")) |> dplyr::mutate(m = nchar(t)) |>
    dplyr::select(id, m)
  expect_equal(tibble::as_tibble(got), want)

  # A gather of the large column, collected.
  arr <- g |> arrange(dplyr::desc(id)) |> collect()
  expect_identical(arr$id, rev(df$id))
  expect_true(identical(arr$s, rev(s)))
  rm(arr)

  # print() previews the head of the large column.
  h <- gpu_head(unclass(g)$ptr, 3L, names(df))
  expect_identical(h$s, s[1:3])

  expect_identical(gpu_string_offsets_width(unclass(g)$ptr, 1L), 64L)
})
