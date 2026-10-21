# R gives a logical operand a numeric type the GPU evaluator used to keep or
# reject: window max()/min() and unary `-` of a logical are integers, round()
# of a logical is a double (cuDF's round rejects BOOL8), summarise()'s
# max()/min() of a logical is an integer. The parse hooks in R/ir.R
# (ir_parse_bool8_cast(), ir_parse_ts_arith("-")) cast a BOOL8 operand first, so
# the plain verbs, and everything that rewrites into them (colwise verbs,
# across()), give R's type natively. Also F19 round-1 B3 (summarise
# quantile(type != 7) goes to the CPU).

b8_df <- function() {
  data.frame(g = c(1L, 1L, 2L), b = c(FALSE, FALSE, TRUE), bn = c(TRUE, NA, FALSE),
             i = c(1L, 2L, 3L), x = c(1.5, 2, 3))
}

# `pipeline` on a tibble and on an eager and a lazy tbl_gpu, with any CPU
# fallback an error (the native path must produce it); values and column
# classes must match exactly.
b8_check <- function(df, pipeline, native = TRUE) {
  withr::local_options(cuplyr.fallback = if (native) "error" else "silent")
  oracle <- dplyr::ungroup(tibble::as_tibble(pipeline(tibble::as_tibble(df))))
  for (lazy in c(FALSE, TRUE)) {
    label <- if (lazy) "lazy" else "eager"
    gpu <- dplyr::ungroup(tibble::as_tibble(collect(pipeline(tbl_gpu(df, lazy = lazy)))))
    expect_identical(names(gpu), names(oracle), info = label)
    expect_identical(lapply(gpu, class), lapply(oracle, class), info = label)
    expect_equal(gpu, oracle, info = label)
  }
}

test_that("mutate(): max/min/unary minus of a logical are integers, round() is double", {
  skip_if_no_gpu()
  df <- b8_df()
  b8_check(df, function(d) mutate(d, m = max(b), n = min(b), k = -b, r = round(b)))
  b8_check(df, function(d) mutate(d, m = max(bn), n = min(bn), k = -bn))
  b8_check(df, function(d) mutate(dplyr::group_by(d, g), m = max(b), n = min(b), k = -b, r = round(b, 1)))
  b8_check(df, function(d) mutate(d, k = -(b & TRUE), q = -(-b), r = round(b & TRUE)))
  b8_check(df, function(d) mutate(d, s = max(b) + 1L, t = -b * 2L, u = max(b) - min(b)))
  b8_check(df, function(d) mutate(d, z = dplyr::if_else(-b < 0L, max(b), min(b))))
})

test_that("integer and double operands are untouched", {
  skip_if_no_gpu()
  df <- b8_df()
  b8_check(df, function(d) mutate(d, m = max(i), n = min(x), k = -i, l = -x, r = round(i), s = round(x, 1)))
})

test_that("filter() with the same operands", {
  skip_if_no_gpu()
  df <- b8_df()
  b8_check(df, function(d) filter(d, max(b) > 0L))
  b8_check(df, function(d) filter(d, -b < 0L))
  b8_check(df, function(d) filter(d, round(b) == 1))
})

test_that("summarise(max(<logical>)) stays native and integer", {
  skip_if_no_gpu()
  df <- b8_df()
  b8_check(df, function(d) summarise(d, m = max(b), n = min(b), mn = max(bn, na.rm = TRUE)))
  b8_check(df, function(d) summarise(dplyr::group_by(d, g), m = max(b), n = min(b)))
  out <- collect(summarise(tbl_gpu(df), m = max(b)))
  expect_type(out$m, "integer")
})

test_that("the lazy schema matches the collected column types", {
  skip_if_no_gpu()
  lz <- mutate(tbl_gpu(b8_df(), lazy = TRUE), m = max(b), k = -b, r = round(b))
  schema <- current_schema(lz)
  got <- collect(lz)
  expect_identical(unname(schema$types[match(c("m", "k", "r"), schema$names)]),
                   c("INT32", "INT32", "FLOAT64"))
  expect_identical(vapply(got[c("m", "k", "r")], typeof, ""),
                   c(m = "integer", k = "integer", r = "double"))
})

test_that("the colwise and across() forms of the same shapes (F19 round-1 B1/B2)", {
  skip_if_no_gpu()
  df <- b8_df()
  b8_check(df, function(d) dplyr::mutate_at(d, dplyr::vars(b), max))
  b8_check(df, function(d) dplyr::mutate_if(d, is.logical, max))
  b8_check(df, function(d) dplyr::mutate_at(d, dplyr::vars(b), list(mx = max, mn = min)))
  b8_check(df, function(d) dplyr::mutate_at(d, dplyr::vars(b, bn), list(mx = max, mn = min), na.rm = TRUE), native = FALSE)
  b8_check(df, function(d) dplyr::mutate_at(dplyr::group_by(d, g), dplyr::vars(b), ~ max(.)))
  b8_check(df, function(d) dplyr::mutate_at(d, dplyr::vars(b), ~ -.))
  b8_check(df, function(d) dplyr::mutate_all(dplyr::select(d, b, x), round, 1))
  b8_check(df, function(d) mutate(d, dplyr::across(c(b, bn), list(mx = max, ng = ~ -.x, rd = round))))
})

test_that("summarise quantile() with type != 7 goes to the CPU and gives dplyr's value (F19 round-1 B3)", {
  skip_if_no_gpu()
  dq <- data.frame(x = c(1, 2, 3, 4, 10), y = c(5, 1, 4, 2, 3))
  b8_check(dq, function(d) summarise(d, q = quantile(x, probs = 0.5, type = 6, names = FALSE)), native = FALSE)
  b8_check(dq, function(d) summarise(d, q = quantile(x, 0.9, type = 1, names = FALSE), r = quantile(y, 0.25, names = FALSE)), native = FALSE)
  # type = 7 and the default stay native
  b8_check(dq, function(d) summarise(d, q = quantile(x, probs = 0.5, type = 7, names = FALSE), r = quantile(y, 0.25, names = FALSE)))
  # dplyr's own errors come from dplyr
  expect_error(summarise(tbl_gpu(dq), q = quantile(x, probs = 2)))
})

test_that("an earlier colwise dot named like a placeholder shadows it (F19 round-1 B4)", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1.5, 2, NA, 4))
  b8_check(df, function(d) dplyr::mutate_at(d, dplyr::vars(x), list(.x = ~ 1, y = ~ .x + 1)), native = FALSE)
})
