# #75: a mutate() of MUTATE_HASH_MIN (24) or more dots looks names and types up
# in hashes (schema_hash_new()). Every case below is run on the hashed path and
# against dplyr, eager and lazy; the 24 filler dots make the call hashed.

p75d_fill <- function(n = 24L) {
  stats::setNames(lapply(seq_len(n), function(i) rlang::expr(x + !!i)), paste0("v", seq_len(n)))
}

p75d_same <- function(.df, ...) {
  dots <- c(p75d_fill(), rlang::enexprs(..., .named = FALSE))
  expect_same_as_dplyr(.df, function(d) mutate(d, !!!dots))
  expect_same_as_dplyr_lazy(.df, function(d) mutate(d, !!!dots))
}

p75d_df <- function() {
  data.frame(
    x = c(1, NA, 3, 4.5),
    i = c(1L, 2L, NA, 4L),
    f = factor(c("a", "b", NA, "a"), levels = c("a", "b", "c")),
    s = c("u", "v", NA, "u"),
    d = as.Date("2024-01-01") + c(0, 1, NA, 3),
    t = as.POSIXct("2024-01-01 12:00:00", tz = "UTC") + c(0, 60, NA, 7200),
    l = c(TRUE, FALSE, NA, TRUE)
  )
}

test_that("a many-dot mutate() is dplyr's on every column type", {
  skip_if_no_gpu()
  withr::local_options(dplyr.summarise.inform = FALSE)
  df <- p75d_df()
  p75d_same(df, f2 = f, s2 = s, d2 = d + 1, d3 = d, t2 = t + 60, t3 = t, l2 = !l, i2 = i * 2L)
  p75d_same(df, f = f, d = d + 2, t = t + 3600)
})

test_that("a many-dot mutate() is dplyr's on an integer64 column", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- data.frame(x = c(1, 2, NA, 4))
  df$b <- bit64::as.integer64(c(1, 2, NA, 2^40))
  p75d_same(df, b2 = b + b, b3 = b)
})

test_that("a many-dot mutate() is dplyr's on an overwrite that changes the type", {
  skip_if_no_gpu()
  df <- p75d_df()
  # x becomes logical, then character, and later dots read the new type
  p75d_same(df, x = x > 1, w = x & l, i = as.character(i), j = paste0(i, "!"))
  p75d_same(df, d = as.numeric(d), e = d + 1, f = as.integer(f))
})

test_that("a many-dot mutate() is dplyr's on NULL, then the column again", {
  skip_if_no_gpu()
  df <- p75d_df()
  p75d_same(df, i = NULL, z = x * 2)
  p75d_same(df, i = NULL, i = x + 100, k = i * 2)
  p75d_same(df, new = NULL, new = x, new2 = new + 1)
})

test_that("across() that expands to many dots is dplyr's", {
  skip_if_no_gpu()
  set.seed(7507)
  df <- as.data.frame(matrix(round(rnorm(30 * 6), 2), ncol = 30, dimnames = list(NULL, paste0("c", 1:30))))
  expect_same_as_dplyr(df, function(d) mutate(d, across(everything(), ~ .x * 2)))
  expect_same_as_dplyr_lazy(df, function(d) mutate(d, across(everything(), ~ .x * 2)))
  expect_same_as_dplyr(df, function(d) mutate(d, across(everything(), list(a = ~ .x + 1, b = abs))))
  expect_same_as_dplyr_lazy(df, function(d) mutate(d, across(c1:c26, ~ .x - c30), tot = c1 + c2))
})

test_that("an empty call argument in a many-dot mutate() is dplyr's", {
  skip_if_no_gpu()
  df <- data.frame(x = c(1, NA, 3))
  # coalesce() takes dynamic dots: a trailing comma is legal
  p75d_same(df, z = dplyr::coalesce(x, 0, ))
  p75d_same(df, z = dplyr::coalesce(x, 0, ), w = z + 1)
  expect_same_as_dplyr(df, function(d) mutate(d, z = dplyr::coalesce(x, 0, )))
  # a missing argument is dplyr's error, whatever the number of dots
  dots <- c(p75d_fill(), list(z = rlang::expr(dplyr::if_else(x > 1, , 0))))
  msg <- function(f) tryCatch(f(), error = conditionMessage)
  want <- msg(function() dplyr::mutate(df, !!!dots))
  expect_match(want, "missing")
  expect_identical(msg(function() collect(mutate(tbl_gpu(df, lazy = FALSE), !!!dots))), want)
  expect_identical(msg(function() collect(mutate(tbl_gpu(df, lazy = TRUE), !!!dots))), want)
  expect_identical(msg(function() collect(mutate(tbl_gpu(df, lazy = FALSE), !!!dots[c(1, 25)]))),
                   msg(function() dplyr::mutate(df, !!!dots[c(1, 25)])))
})

test_that("order_by() and with_order() over a computed argument in a many-dot mutate() are dplyr's", {
  skip_if_no_gpu()
  df <- data.frame(
    x = c(1, 2, 3, 4), o = c(4, 3, 2, 1),
    d = as.Date("2024-01-01") + c(3, 0, 2, 1),
    t = as.POSIXct("2024-01-01 12:00:00", tz = "America/New_York") + c(0, 60, 30, 90)
  )
  # (the computed first argument is parsed through a placeholder column that
  # only the call's own schema has)
  p75d_same(df, z = dplyr::order_by(o, cumsum(x * 2)))
  p75d_same(df, z = dplyr::with_order(o, cumsum, x * 2))
  p75d_same(df, z = dplyr::order_by(o, cumsum(x * 2)), w = z + v1)
  p75d_same(df, z = dplyr::order_by(-o, cummax(x + o)), y = dplyr::with_order(x, dplyr::lag, o * 3))
  p75d_same(df, z = dplyr::order_by(o, dplyr::lag(d + 1)), w = dplyr::order_by(x, dplyr::lead(t + 60)))
  # a bare column still is the column
  p75d_same(df, z = dplyr::order_by(o, cumsum(x)))
})

test_that("a many-dot schema's hash is not trusted on a copy it doesn't describe", {
  s <- schema_hash_new(list(names = c("a", "b"), types = c("FLOAT64", "INT32")))
  # a copy that grows without the hash (as a callee's placeholder does)
  grown <- s
  grown$names <- c(grown$names, "p")
  grown$types <- c(grown$types, "INT64")
  expect_identical(ir_schema_pos(grown, "p"), 3L)
  expect_true(ir_schema_has_type(grown, "INT64"))
  expect_identical(ir_schema_pos(s, "p"), NA_integer_)
  expect_false(ir_schema_has_type(s, "INT64"))
  # the original grows on (the hash with it) while a copy of the earlier
  # schema is kept: the copy is still read right
  kept <- s
  s$names[[3L]] <- "q"
  s$types[[3L]] <- "STRING"
  schema_hash_append(s, "q", "STRING")
  expect_identical(ir_schema_pos(s, "q"), 3L)
  expect_identical(ir_schema_pos(kept, "q"), NA_integer_)
  expect_identical(ir_schema_pos(kept, "b"), 2L)
  # two copies that grew apart, both written through the hash
  other <- kept
  other$names[[3L]] <- "r"
  other$types[[3L]] <- "BOOL8"
  schema_hash_append(other, "r", "BOOL8")
  expect_identical(ir_schema_pos(s, "q"), 3L)
  expect_identical(ir_schema_pos(s, "r"), NA_integer_)
  expect_identical(ir_schema_pos(other, "r"), 3L)
  expect_identical(ir_schema_pos(other, "q"), NA_integer_)
  # a retype never makes a type look absent that a column still has
  schema_hash_retype(s, 2L, "FLOAT64")
  s$types[[2L]] <- "FLOAT64"
  expect_true(ir_schema_has_type(s, "FLOAT64"))
  # a mutate record's schema holds no hash
  dots <- rlang::quos(!!!stats::setNames(lapply(1:30, function(i) rlang::expr(a + !!i)), paste0("v", 1:30)))
  recs <- parse_mutate_dots(dots, list(names = c("a", "b"), types = c("FLOAT64", "INT32")), "mutate")
  expect_length(recs, 30L)
  for (r in recs) expect_null(r$checked_int$schema$.hash)
  expect_identical(recs[[30]]$checked_int$schema$names, c("a", "b", paste0("v", 1:29)))
})
