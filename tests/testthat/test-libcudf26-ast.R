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
