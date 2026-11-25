# libcudf 26.x tightened the operand types of its AST functors (25.12 let C++
# convert an operand silently): the floating-point functions (sqrt, exp, log,
# sin, cos, tan, floor, ceil) accept floating-point operands only, abs() signed
# ones (not BOOL8), `^` on two integers is an integer power, and BITWISE_XOR is
# not defined for BOOL8. cuplyr casts the operands (and lowers xor() to `!=`),
# so every column type gives dplyr's answer. Each case below failed on libcudf
# 26.10 before that.

p94a_df <- function() {
  data.frame(
    i = c(1L, 2L, 3L, NA),
    j = c(2L, 0L, -3L, 5L),
    d = c(1.5, -2.5, 0, NA),
    b = c(TRUE, FALSE, NA, TRUE),
    c = c(FALSE, FALSE, TRUE, TRUE)
  )
}

p94a_both <- function(pipeline) {
  expect_same_as_dplyr(p94a_df(), pipeline)
  expect_same_as_dplyr_lazy(p94a_df(), pipeline)
}

test_that("floating-point math functions of an integer or logical column match dplyr", {
  skip_if_no_gpu()
  for (fn in c("sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceiling")) {
    for (col in c("i", "b")) {
      e <- rlang::parse_expr(sprintf("%s(%s)", fn, col))
      p94a_both(function(x) mutate(x, z = !!e))
    }
  }
})

test_that("abs() of a logical column is an integer, like dplyr", {
  skip_if_no_gpu()
  p94a_both(function(x) mutate(x, z = abs(b), y = abs(i), w = abs(d)))
})

test_that("^ of integer and logical operands is a double power, negative exponents included", {
  skip_if_no_gpu()
  p94a_both(function(x) mutate(x, z = i ^ j, y = j ^ j, w = b ^ c, v = j ^ 2L, u = 2L ^ j))
})

test_that("xor() of logical columns matches dplyr in filter() and mutate()", {
  skip_if_no_gpu()
  p94a_both(function(x) filter(x, xor(b, c)))
  p94a_both(function(x) mutate(x, z = xor(b, c), y = xor(b, b), w = xor(i > 1L, c)))
})

# The retyped ops also have to keep their STATIC type right when they are an
# operand of another node: the parent matches its operands' types (and builds
# unary minus's zero literal) from resolve_static_type(), so a node whose
# computed column differs from that type trips libcudf's "non-matching operand
# types". The cases below nest each retyped op in ==, &, unary -, arithmetic,
# filter() and summarise(), with NAs.

test_that("abs() of a logical column nested in ==, &, unary - and filter()", {
  skip_if_no_gpu()
  p94a_both(function(x) mutate(x, z = abs(b) == b))
  p94a_both(function(x) mutate(x, z = abs(b) & c))
  p94a_both(function(x) mutate(x, z = -abs(b)))
  p94a_both(function(x) filter(x, abs(b) == c))
  p94a_both(function(x) mutate(x, z = abs(b) + 1L, y = abs(b) * d, w = abs(b) > 0L))
  p94a_both(function(x) summarise(x, s = sum(abs(b), na.rm = TRUE), m = max(abs(b), na.rm = TRUE)))
})

test_that("float math of an integer or logical column nested in other nodes", {
  # log() and sqrt() of the negative values warn in dplyr and on the GPU alike
  both <- function(pipeline) suppressWarnings(p94a_both(pipeline))
  skip_if_no_gpu()
  for (fn in c("sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceiling", "abs")) {
    for (col in c("i", "b", "d")) {
      f <- rlang::parse_expr(sprintf("%s(%s)", fn, col))
      both(function(x) mutate(x, z = !!f == !!f, y = !!f > 0.5, w = -!!f))
      both(function(x) mutate(x, z = !!f + i, y = !!f * d, w = !!f / 2L, v = -(!!f) - 1L))
      both(function(x) mutate(x, z = (!!f > 0.5) & c, y = (!!f > 0.5) | b, w = !(!!f > 0.5)))
      both(function(x) filter(x, !!f > 0.5 | is.na(!!f)))
      both(function(x) summarise(x, s = sum(!!f, na.rm = TRUE), n = sum(!!f > 0.5, na.rm = TRUE)))
    }
  }
})

test_that("`^` nested in other nodes", {
  skip_if_no_gpu()
  p94a_both(function(x) mutate(x, z = (i ^ j) == d, y = -(i ^ j), w = (j ^ j) + 1L, v = (b ^ c) > 0))
  p94a_both(function(x) mutate(x, z = (i ^ j) > 1 & c, y = abs(j ^ j), w = sqrt(i ^ 2L), v = -(2L ^ j)))
  p94a_both(function(x) filter(x, (i ^ j) > 0.5 | is.na(i ^ j)))
  p94a_both(function(x) summarise(x, s = sum(i ^ j, na.rm = TRUE), m = max(j ^ 2L, na.rm = TRUE)))
})

test_that("xor() nested in other nodes", {
  skip_if_no_gpu()
  p94a_both(function(x) mutate(x, z = xor(b, c) == b, y = xor(b, c) & c, w = -xor(b, c)))
  p94a_both(function(x) mutate(x, z = xor(b, c) + 1L, y = !xor(b, c), w = xor(xor(b, c), b)))
  p94a_both(function(x) filter(x, xor(b, c) == c))
  p94a_both(function(x) filter(x, xor(b, xor(b, c))))
  p94a_both(function(x) summarise(x, s = sum(xor(b, c), na.rm = TRUE), a = any(xor(b, c), na.rm = TRUE)))
})
