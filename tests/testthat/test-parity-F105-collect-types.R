# F105: collect() returns every GPU column type correctly.
#
# dplyr has no equivalent of these column types, so the oracle here is the
# contract of the F105 mapping (brief + ## Design): an explicit expected R
# vector for each libcudf type. Columns are built with the test hook
# gpu_test_make_column() (src/test_fixture_column.cpp), since nothing in cuplyr
# uploads these types.

typed_ptr <- function(type, values, scale = 0L) {
  gpu_test_make_column(type, values, scale)
}

schema_type <- function(type) if (identical(type, "DICTIONARY32_INT32")) "DICTIONARY32" else type

typed_gt <- function(type, values, scale = 0L, lazy = FALSE, name = "x") {
  new_tbl_gpu(
    ptr = typed_ptr(type, values, scale),
    schema = list(names = name, types = schema_type(type)),
    exec_mode = if (lazy) "lazy" else "eager"
  )
}

# x plus an integer column `g` (1, 2, ...), for verbs and previews.
typed_gt2 <- function(type, values, scale = 0L, lazy = FALSE) {
  g <- .subset2(tbl_gpu(tibble::tibble(g = seq_along(values))), "ptr")
  new_tbl_gpu(
    ptr = gpu_bind_cols_impl(list(typed_ptr(type, values, scale), g)),
    schema = list(names = c("x", "g"), types = c(schema_type(type), "INT32")),
    exec_mode = if (lazy) "lazy" else "eager"
  )
}

preview_line <- function(lines, col = "x") {
  hit <- grep(paste0("^\\$ ", col, " "), lines, value = TRUE)
  expect_length(hit, 1L)
  hit
}

print_lines <- function(gt) utils::capture.output(print(gt))
glimpse_lines <- function(gt) utils::capture.output(dplyr::glimpse(gt))

# `shown` are the display strings of elements 1 and 3 (element 2 is the NA in
# the preview tests).
cases <- list(
  list(type = "INT8", values = c("-128", "0", "127"),
       expected = c(-128L, 0L, 127L), shown = c("-128", "127")),
  list(type = "INT16", values = c("-32768", "1", "32767"),
       expected = c(-32768L, 1L, 32767L), shown = c("-32768", "32767")),
  list(type = "UINT8", values = c("0", "7", "255"),
       expected = c(0L, 7L, 255L), shown = c("0", "255")),
  list(type = "UINT16", values = c("0", "7", "65535"),
       expected = c(0L, 7L, 65535L), shown = c("0", "65535")),
  list(type = "UINT32", values = c("0", "7", "4294967295"),
       expected = c(0, 7, 4294967295), shown = c("0", "4294967295")),
  list(type = "UINT64", values = c("5", "9007199254740993", "9223372036854775807"),
       expected = bit64::as.integer64(c("5", "9007199254740993", "9223372036854775807")),
       shown = c("5", "9223372036854775807")),
  list(type = "TIMESTAMP_SECONDS", values = c("1700000000", "-5", "0"),
       expected = .POSIXct(c(1700000000, -5, 0), "UTC"),
       shown = c("2023-11-14 22:13:20", "1970-01-01")),
  list(type = "TIMESTAMP_MILLISECONDS", values = c("1700000000123", "-1500", "0"),
       expected = .POSIXct(c(1700000000.123, -1.5, 0), "UTC"),
       shown = c("2023-11-14 22:13:20", "1970-01-01")),
  list(type = "DURATION_DAYS", values = c("3", "-2", "0"),
       expected = as.difftime(c(3, -2, 0), units = "days"), shown = c("3", "0")),
  list(type = "DURATION_SECONDS", values = c("90", "-1", "0"),
       expected = as.difftime(c(90, -1, 0), units = "secs"), shown = c("90", "0")),
  list(type = "DURATION_MILLISECONDS", values = c("1500", "-250", "0"),
       expected = as.difftime(c(1.5, -0.25, 0), units = "secs"), shown = c("1.5", "0")),
  list(type = "DURATION_MICROSECONDS", values = c("1500000", "250", "0"),
       expected = as.difftime(c(1.5, 0.00025, 0), units = "secs"), shown = c("1.5", "0")),
  list(type = "DURATION_NANOSECONDS", values = c("1500000000", "-2500000000", "0"),
       expected = as.difftime(c(1.5, -2.5, 0), units = "secs"), shown = c("1.5", "0")),
  list(type = "DECIMAL32", scale = -2L, values = c("150", "-5", "0"),
       expected = c(1.5, -0.05, 0), shown = c("1.5", "0")),
  # the nearest double, not unscaled * 10^scale in floating point (3 * 0.1 != 0.3)
  list(type = "DECIMAL64", scale = -1L, values = c("3", "123456789012345", "0"),
       expected = c(0.3, 12345678901234.5, 0), shown = c("0.3", "0")),
  list(type = "DECIMAL64", scale = 3L, values = c("12", "-1", "0"),
       expected = c(12000, -1000, 0), shown = c("12000", "0")),
  list(type = "DECIMAL128", scale = 0L, values = c("12", "-7", "0"),
       expected = c(12, -7, 0), shown = c("12", "0")),
  list(type = "DICTIONARY32", values = c("b", "a", "b"),
       expected = factor(c("b", "a", "b"), levels = c("a", "b")), shown = c("b", "b"))
)

case_label <- function(cs) paste0(cs$type, if (!is.null(cs$scale)) paste0("(scale ", cs$scale, ")"))
with_na <- function(cs) {
  v <- cs$values
  v[2L] <- NA_character_
  v
}
# the expected vector with element 2 missing, keeping the class; a dictionary's
# keys are the values it was encoded from, so a level only element 2 held is gone
expected_na <- function(cs) {
  out <- vctrs::vec_slice(cs$expected, c(1L, NA, 3L))
  if (is.factor(out)) droplevels(out) else out
}

for (cs in cases) local({
  cs <- cs
  scale <- cs$scale %||% 0L
  lbl <- case_label(cs)

  test_that(paste0(lbl, ": an all-valid column collects with its values, never NA"), {
    skip_if_no_gpu()
    out <- collect(typed_gt(cs$type, cs$values, scale))
    expect_identical(out$x, cs$expected)
    expect_false(anyNA(out$x))
  })

  test_that(paste0(lbl, ": NA round-trips through collect, pull and $"), {
    skip_if_no_gpu()
    gt <- typed_gt(cs$type, with_na(cs), scale)
    expect_identical(collect(gt)$x, expected_na(cs))
    expect_identical(dplyr::pull(gt, x), expected_na(cs))
    expect_identical(gt$x, expected_na(cs))
  })

  test_that(paste0(lbl, ": eager and lazy verbs that do not touch it keep it"), {
    skip_if_no_gpu()
    for (lazy in c(FALSE, TRUE)) {
      gt <- typed_gt2(cs$type, with_na(cs), scale, lazy = lazy)
      expect_identical(collect(dplyr::select(gt, x))$x, expected_na(cs))
      expect_identical(
        collect(dplyr::filter(gt, g != 2L))$x,
        vctrs::vec_slice(expected_na(cs), c(1L, 3L))
      )
      expect_identical(
        collect(dplyr::arrange(gt, dplyr::desc(g)))$x,
        vctrs::vec_slice(expected_na(cs), 3:1)
      )
      grouped <- collect(dplyr::group_by(gt, g))
      expect_s3_class(grouped, "grouped_df")
      expect_identical(grouped$x, expected_na(cs))
    }
  })

  test_that(paste0(lbl, ": zero rows collect to the same R class"), {
    skip_if_no_gpu()
    out <- collect(typed_gt(cs$type, character(), scale))$x
    expect_length(out, 0L)
    expect_identical(class(out), class(cs$expected))
    expect_identical(typeof(out), typeof(cs$expected))
  })

  test_that(paste0(lbl, ": gpu_proto() has the collected R class"), {
    skip_if_no_gpu()
    proto <- gpu_proto(typed_gt(cs$type, cs$values, scale))$x
    expect_length(proto, 0L)
    expect_identical(class(proto), class(cs$expected))
    expect_identical(typeof(proto), typeof(cs$expected))
  })

  test_that(paste0(lbl, ": print and glimpse preview the values with the collected label"), {
    skip_if_no_gpu()
    abbr <- vctrs::vec_ptype_abbr(cs$expected)
    for (lazy in c(FALSE, TRUE)) {
      gt <- typed_gt2(cs$type, with_na(cs), scale, lazy = lazy)
      if (lazy) gt <- dplyr::mutate(gt, g2 = g + 1L)
      for (lines in list(print_lines(gt), glimpse_lines(gt))) {
        line <- preview_line(lines)
        expect_match(line, paste0("<", abbr, ">"), fixed = TRUE)
        expect_match(line, "\\bNA\\b")
        for (s in cs$shown) expect_match(line, s, fixed = TRUE)
        expect_false(grepl("[preview unavailable]", line, fixed = TRUE))
      }
    }
  })

  test_that(paste0(lbl, ": the schema-only label is the collected label"), {
    skip_if(identical(cs$type, "DICTIONARY32"), "a real dictionary has no labels in its schema")
    expect_identical(
      schema_type_abbr(list(names = "x", types = cs$type)),
      vctrs::vec_ptype_abbr(cs$expected)
    )
  })
})

test_that("a zero-row real dictionary collects to a factor", {
  skip_if_no_gpu()
  out <- collect(typed_gt("DICTIONARY32", character()))$x
  expect_s3_class(out, "factor")
  expect_length(out, 0L)
})

test_that("a zero-row view of a dictionary keeps its levels", {
  skip_if_no_gpu()
  ptr <- typed_ptr("DICTIONARY32", c("b", "a", "b"))
  out <- gpu_collect_slice(ptr, "x", 1L, 1L)$x
  expect_s3_class(out, "factor")
  expect_length(out, 0L)
  expect_identical(levels(out), c("a", "b"))
})

test_that("a zero-row real dictionary is labelled a factor in previews", {
  skip_if_no_gpu()
  gt <- typed_gt2("DICTIONARY32", character())
  expect_match(preview_line(print_lines(gt)), "<fct>", fixed = TRUE)
  expect_match(preview_line(glimpse_lines(gt)), "<fct>", fixed = TRUE)
})

test_that("string dictionary keys collect to a factor whatever the index type", {
  skip_if_no_gpu()
  ptr <- typed_ptr("DICTIONARY32_INT8_INDICES", c("b", NA, "a", "b"))
  gt <- new_tbl_gpu(ptr = ptr, schema = list(names = "x", types = "DICTIONARY32"),
                    exec_mode = "eager")
  expect_identical(
    collect(gt)$x,
    factor(c("b", NA, "a", "b"), levels = c("a", "b"))
  )
})

test_that("the unsupported-type error does not name an internal helper", {
  skip_if_no_gpu()
  err <- expect_error(collect(typed_gt("LIST", "1")),
                      class = "cuplyr_error_unsupported_gpu_type")
  expect_false(grepl("abort_unsupported_gpu_type", format(err), fixed = TRUE))
})

# Regression (F105 review, B1): pull()/`$`/`[[` fetch one column, and that
# column's vctrs::unspecified marker must come with it, as in dplyr.
test_that("pull(), $ and [[ keep a vctrs::unspecified column's class", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = vctrs::unspecified(2), y = 1:2)
  for (lazy in c(FALSE, TRUE)) {
    gt <- tbl_gpu(df, lazy = lazy)
    expect_identical(class(dplyr::pull(gt, x)), class(dplyr::pull(df, x)))
    expect_identical(class(gt$x), class(df$x))
    expect_identical(class(gt[["x"]]), class(df[["x"]]))
  }
})

test_that("gpu_column_type() names every type and gpu_col_types() labels every collectable one", {
  skip_if_no_gpu()
  for (cs in cases) {
    scale <- cs$scale %||% 0L
    ptr <- typed_ptr(cs$type, cs$values, scale)
    expect_identical(gpu_column_type(ptr, 0L), cs$type)
    expect_false(identical(unname(gpu_col_types(ptr)), "???"), info = case_label(cs))
  }
  expect_identical(gpu_column_type(typed_ptr("LIST", "1"), 0L), "LIST")
  expect_identical(gpu_column_type(typed_ptr("STRUCT", "1"), 0L), "STRUCT")
})

test_that("gpu_head() and a sliced view read values and nulls at their offset", {
  skip_if_no_gpu()
  head_of <- function(type, values, n, scale = 0L) {
    gpu_head(typed_ptr(type, values, scale), n, "x")$x
  }
  expect_equal(as.numeric(head_of("UINT8", c("1", NA, "3"), 2L)), c(1, NA))
  expect_equal(as.numeric(head_of("DECIMAL64", c("15", NA, "3"), 2L, -1L)), c(1.5, NA))

  slice_of <- function(type, values, begin, end, scale = 0L) {
    gpu_collect_slice(typed_ptr(type, values, scale), "x", begin, end)$x
  }
  # a null in the middle of a byte, at a non-zero offset
  expect_equal(as.numeric(slice_of("UINT16", c("9", "8", NA, "6"), 1L, 4L)), c(8, NA, 6))
  expect_equal(as.numeric(slice_of("DECIMAL64", c("15", "25", NA, "45"), 1L, 4L, -1L)), c(2.5, NA, 4.5))
  expect_equal(as.numeric(slice_of("DURATION_SECONDS", c("9", "8", NA, "6"), 1L, 4L)), c(8, NA, 6))
  expect_equal(as.numeric(slice_of("TIMESTAMP_SECONDS", c("9", "8", NA, "6"), 1L, 4L)), c(8, NA, 6))
  dict <- slice_of("DICTIONARY32", c("b", NA, "a", "b"), 1L, 3L)
  expect_s3_class(dict, "factor")
  expect_identical(as.character(dict), c(NA, "a"))
  expect_identical(levels(dict), c("a", "b"))
})

# -- Nanosecond timestamps ------------------------------------------------

test_that("TIMESTAMP_NANOSECONDS collects to POSIXct, within a microsecond", {
  skip_if_no_gpu()
  gt <- typed_gt("TIMESTAMP_NANOSECONDS", c("1700000000123456000", NA, "-1500000000"))
  expect_no_warning(out <- collect(gt))
  expect_s3_class(out$x, "POSIXct")
  expect_identical(attr(out$x, "tzone"), "UTC")
  expect_equal(as.numeric(out$x), c(1700000000.123456, NA, -1.5), tolerance = 1e-9)
  expect_identical(is.na(out$x), c(FALSE, TRUE, FALSE))
  expect_identical(class(dplyr::pull(gt, x)), c("POSIXct", "POSIXt"))
})

test_that("a sub-microsecond timestamp warns once, naming the column", {
  skip_if_no_gpu()
  gt <- typed_gt("TIMESTAMP_NANOSECONDS", c("1700000000123456789", NA, "1"), name = "tick")
  w <- expect_warning(out <- collect(gt), class = "cuplyr_warning_collect_precision")
  expect_match(conditionMessage(w), "`tick`", fixed = TRUE)
  expect_equal(as.numeric(out$tick)[[1L]], 1700000000.123456789, tolerance = 1e-9)
})

test_that("previews never raise the precision warning", {
  skip_if_no_gpu()
  gt <- typed_gt2("TIMESTAMP_NANOSECONDS", c("1700000000123456789", "2", "3"))
  expect_no_warning(print_lines(gt))
  expect_no_warning(glimpse_lines(gt))
  gt <- typed_gt2("DECIMAL64", c("12345678901234567", "2", "3"), -2L)
  expect_no_warning(print_lines(gt))
})

# -- Lossy doubles --------------------------------------------------------

test_that("a DECIMAL needing more than 15 significant digits warns, naming the column", {
  skip_if_no_gpu()
  gt <- typed_gt("DECIMAL64", c("12345678901234567", "2", NA), -2L, name = "amt")
  w <- expect_warning(out <- collect(gt), class = "cuplyr_warning_collect_precision")
  expect_match(conditionMessage(w), "`amt`", fixed = TRUE)
  expect_equal(out$amt, c(123456789012345.67, 0.02, NA))

  gt <- typed_gt("DECIMAL128", "170141183460469231731687303715884105727")
  expect_warning(out <- collect(gt), class = "cuplyr_warning_collect_precision")
  expect_equal(out$x, 1.7014118346046923e38)
})

test_that("a DECIMAL with few significant digits does not warn, even when huge", {
  skip_if_no_gpu()
  expect_no_warning(out <- collect(typed_gt("DECIMAL64", "1000000000000000000")))
  expect_identical(out$x, 1e18)
  expect_no_warning(out <- collect(typed_gt("DECIMAL128", "5", 30L)))
  expect_identical(out$x, 5e30)
  expect_no_warning(out <- collect(typed_gt("DECIMAL64", "123456789012345", -3L)))
  expect_identical(out$x, 123456789012.345)
})

test_that("a non-zero DECIMAL that underflows a double warns instead of silently becoming 0", {
  skip_if_no_gpu()
  expect_warning(out <- collect(typed_gt("DECIMAL128", "1", -400L)),
                 class = "cuplyr_warning_collect_precision")
  expect_identical(out$x, 0)
})

test_that("a duration past 2^53 ticks warns, naming the column", {
  skip_if_no_gpu()
  gt <- typed_gt("DURATION_MILLISECONDS", c("9007199254740993", "1"), name = "dur")
  w <- expect_warning(out <- collect(gt), class = "cuplyr_warning_collect_precision")
  expect_match(conditionMessage(w), "`dur`", fixed = TRUE)
  expect_s3_class(out$dur, "difftime")
  expect_equal(as.numeric(out$dur), c(9007199254740.993, 0.001))
})

test_that("several lossy columns give one warning naming all of them", {
  skip_if_no_gpu()
  a <- typed_ptr("DECIMAL64", c("12345678901234567"), -2L)
  b <- typed_ptr("TIMESTAMP_NANOSECONDS", c("1700000000123456789"))
  gt <- new_tbl_gpu(
    ptr = gpu_bind_cols_impl(list(a, b)),
    schema = list(names = c("aa", "bb"), types = c("DECIMAL64", "TIMESTAMP_NANOSECONDS")),
    exec_mode = "eager"
  )
  seen <- list()
  withCallingHandlers(
    collect(gt),
    cuplyr_warning_collect_precision = function(w) {
      seen[[length(seen) + 1L]] <<- w
      invokeRestart("muffleWarning")
    }
  )
  expect_length(seen, 1L)
  expect_match(conditionMessage(seen[[1L]]), "`aa`", fixed = TRUE)
  expect_match(conditionMessage(seen[[1L]]), "`bb`", fixed = TRUE)
})

# -- Values that cannot be held -------------------------------------------

expect_gpu_type_error <- function(expr, col, type) {
  err <- expect_error(expr, class = "cuplyr_error_unsupported_gpu_type")
  expect_match(conditionMessage(err), paste0("`", col, "`"), fixed = TRUE)
  expect_match(conditionMessage(err), type, fixed = TRUE)
}

test_that("UINT64 past INT64_MAX is a classed error, never a wrap", {
  skip_if_no_gpu()
  gt <- typed_gt("UINT64", c("5", "9223372036854775808"), name = "big")
  expect_gpu_type_error(collect(gt), "big", "UINT64")
  expect_gpu_type_error(dplyr::pull(gt, big), "big", "UINT64")
  expect_gpu_type_error(gt$big, "big", "UINT64")
  gt2 <- typed_gt("UINT64", c("18446744073709551615"), name = "big", lazy = TRUE)
  expect_gpu_type_error(collect(gt2), "big", "UINT64")
})

test_that("a DECIMAL beyond a double's range is a classed error, never Inf", {
  skip_if_no_gpu()
  expect_gpu_type_error(collect(typed_gt("DECIMAL128", "1", 400L, name = "huge")),
                        "huge", "DECIMAL128")
})

for (tp in c("LIST", "STRUCT", "DICTIONARY32_INT32")) local({
  tp <- tp
  shown <- if (tp == "DICTIONARY32_INT32") "DICTIONARY32" else tp
  test_that(paste0(tp, " is a classed error at collect, pull and $, never NA"), {
    skip_if_no_gpu()
    for (lazy in c(FALSE, TRUE)) {
      gt <- typed_gt(tp, c("1", "2"), lazy = lazy)
      expect_gpu_type_error(collect(gt), "x", shown)
      expect_gpu_type_error(dplyr::pull(gt, x), "x", shown)
      expect_gpu_type_error(gt$x, "x", shown)
    }
  })

  test_that(paste0(tp, ": the other columns still collect and preview"), {
    skip_if_no_gpu()
    for (lazy in c(FALSE, TRUE)) {
      gt <- typed_gt2(tp, c("1", "2", "3"), lazy = lazy)
      expect_gpu_type_error(collect(gt), "x", shown)
      expect_identical(dplyr::pull(gt, g), 1:3)
      expect_identical(gt$g, 1:3)
      expect_identical(collect(dplyr::select(gt, g))$g, 1:3)
      for (render in list(print_lines, glimpse_lines)) {
        expect_no_error(lines <- render(gt))
        expect_match(preview_line(lines, "g"), "1, 2, 3", fixed = TRUE)
        line <- preview_line(lines, "x")
        expect_false(grepl("\\bNA\\b", line))
        expect_false(grepl("[preview unavailable]", line, fixed = TRUE))
      }
    }
  })
})

test_that("LIST and STRUCT keep a label of their own in previews and the schema", {
  skip_if_no_gpu()
  expect_match(preview_line(print_lines(typed_gt("LIST", "1"))), "<list>", fixed = TRUE)
  expect_match(preview_line(print_lines(typed_gt("STRUCT", "1"))), "<struct>", fixed = TRUE)
  expect_identical(schema_type_abbr(list(names = "x", types = "LIST")), "list")
  expect_identical(schema_type_abbr(list(names = "x", types = "STRUCT")), "struct")
})

test_that("gpu_proto() of a column R cannot hold is the classed error", {
  skip_if_no_gpu()
  expect_gpu_type_error(gpu_proto(typed_gt("LIST", "1")), "x", "LIST")
  expect_gpu_type_error(gpu_proto(typed_gt("STRUCT", "1")), "x", "STRUCT")
})

test_that("a UINT64 preview past INT64_MAX neither errors nor wraps", {
  skip_if_no_gpu()
  gt <- typed_gt2("UINT64", c("18446744073709551615", "2", "3"))
  for (render in list(print_lines, glimpse_lines)) {
    expect_no_error(lines <- render(gt))
    expect_match(preview_line(lines, "g"), "1, 2, 3", fixed = TRUE)
    line <- preview_line(lines, "x")
    expect_false(grepl("-1", line, fixed = TRUE))
    expect_false(grepl("\\bNA\\b", line))
  }
})

# -- bit64 absent ---------------------------------------------------------

test_that("without bit64 a UINT64 is a double, warning past 2^53, erroring past INT64_MAX", {
  skip_if_no_gpu()
  local_mocked_bindings(bit64_available = function() FALSE)

  expect_no_warning(out <- collect(typed_gt("UINT64", c("5", NA))))
  expect_identical(out$x, c(5, NA))

  gt <- typed_gt("UINT64", c("9007199254740993", "1"), name = "big")
  w <- expect_warning(out <- collect(gt), class = "cuplyr_warning_int64_precision")
  expect_match(conditionMessage(w), "`big`", fixed = TRUE)
  expect_type(out$big, "double")
  expect_equal(out$big, c(9007199254740992, 1))

  expect_gpu_type_error(collect(typed_gt("UINT64", "9223372036854775808", name = "big")),
                        "big", "UINT64")

  line <- preview_line(print_lines(typed_gt2("UINT64", c("5", NA, "7"))))
  expect_match(line, "5", fixed = TRUE)
})
