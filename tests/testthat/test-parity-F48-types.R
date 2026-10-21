# F48 acceptance tests: result types of median()/ntile()/cum*() and numeric
# reducers given a STRING/BOOL8/INT32 argument (items 1, 2, 5, 7).

# dplyr is imported, not attached, in the test environment: bind the dplyr
# functions these pipelines call unqualified (file-local).
for (fn in c("if_else", "case_when", "between", "lag", "lead", "ntile",
             "cumall", "cumany", "cummean", "across")) {
  assign(fn, getExportedValue("dplyr", fn))
}
rm(fn)

# --- item 1: median() of an integer/logical ---------------------------------

test_that("median() of an integer is an integer when the middle is one value", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), i = c(1L, 5L, 3L, 4L, 9L, 2L))
  f48_check(df, function(d) summarise(d, m = median(i)))
  f48_check(df, function(d) summarise(group_by(d, g), m = median(i)))
  f48_check(df, function(d) summarise(d, m = median(i + 1L)))
  f48_check(df[1:4, ], function(d) summarise(d, m = median(i)))
})

test_that("median() of an integer averages to a double for an even count", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), i = c(1L, 2L, 3L, 4L, 9L))
  # group 1 even (double), group 2 odd (integer): the combined column is double
  f48_check(df, function(d) summarise(group_by(d, g), m = median(i)))
  f48_check(df[1:4, ], function(d) summarise(d, m = median(i)))
})

test_that("median() of an integer follows NA and na.rm like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), i = c(1L, NA, 3L, NA, NA, 8L))
  # NA without na.rm: NA_integer_
  f48_check(df, function(d) summarise(d, m = median(i)))
  f48_check(df, function(d) summarise(group_by(d, g), m = median(i)))
  # na.rm changes the counts: group 1 has 2 values (double), group 2 has 1
  f48_check(df, function(d) summarise(group_by(d, g), m = median(i, na.rm = TRUE)))
  f48_check(df[4:6, ], function(d) summarise(d, m = median(i, na.rm = TRUE)))
  # an all-NA group with na.rm = TRUE gives NA_integer_
  all_na <- data.frame(g = c(1, 1, 2), i = c(NA, NA, 5L))
  f48_check(all_na, function(d) summarise(group_by(d, g), m = median(i, na.rm = TRUE)))
})

test_that("median() of an integer on zero rows is an integer", {
  skip_if_no_gpu()
  df <- data.frame(g = numeric(0), i = integer(0))
  f48_check(df, function(d) summarise(d, m = median(i)))
  f48_check(df, function(d) summarise(group_by(d, g), m = median(i)))
})

test_that("median() of a logical keeps dplyr's type", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2), l = c(TRUE, FALSE, TRUE, TRUE, NA))
  f48_check(df, function(d) summarise(d, m = median(l)))
  f48_check(df, function(d) summarise(d, m = median(l, na.rm = TRUE)))
  f48_check(df[1:3, ], function(d) summarise(d, m = median(l)))
  f48_check(df, function(d) summarise(group_by(d, g), m = median(l, na.rm = TRUE)))
})

test_that("median() of an integer as a window keeps dplyr's type", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), i = c(1L, 5L, 3L, 4L, 9L, 2L))
  f48_check(df, function(d) mutate(d, m = median(i)))
  f48_check(df, function(d) mutate(group_by(d, g), m = median(i)))
  f48_check(df[1:4, ], function(d) mutate(d, m = median(i)))
  f48_check(df, function(d) mutate(d, m = i - median(i)))
})

# --- item 7: ntile() is an integer -------------------------------------------

test_that("ntile() returns an integer column", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2, 2), x = c(1.5, 2, NA, 4, 5, 3),
                   i = c(1L, 5L, 3L, NA, 2L, 7L))
  f48_check(df, function(d) mutate(d, n = ntile(x, 2)))
  f48_check(df, function(d) mutate(d, n = ntile(i, 3)))
  f48_check(df, function(d) mutate(group_by(d, g), n = ntile(x, 2)))
  f48_check(df, function(d) summarise(d, n = max(ntile(x, 3), na.rm = TRUE)))
  f48_check(df[0, ], function(d) mutate(d, n = ntile(x, 2)))
})

# --- item 2: cumulative functions on STRING and BOOL8 -----------------------

test_that("cummax()/cummin() of a string coerce to numeric like dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), s = c("1", "2", "x", "3", NA))
  for (fn in c("cummax", "cummin", "cumsum", "cumprod", "cummean")) {
    call <- rlang::call2(fn, quote(s))
    f48_check_warns(df, function(d) mutate(d, m = !!call), "NAs introduced by coercion")
    f48_check_warns(df, function(d) mutate(group_by(d, g), m = !!call),
                    "NAs introduced by coercion")
  }
})

test_that("cummax()/cummin() of a logical are integers", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), l = c(TRUE, FALSE, FALSE, TRUE, NA))
  f48_check(df, function(d) mutate(d, a = cummax(l), b = cummin(l)))
  f48_check(df, function(d) mutate(group_by(d, g), a = cummax(l), b = cummin(l)))
  f48_check(df, function(d) mutate(d, a = cumsum(l), b = cumprod(l), c = cummean(l)))
})

test_that("cumall()/cumany() of a non-logical match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(i = c(1L, 5L, 0L, NA, 2L), s = c("a", "b", NA, "c", "d"))
  f48_check(df, function(d) mutate(d, a = cumall(i), b = cumany(i)))
  f48_check(df, function(d) mutate(d, a = cumall(s), b = cumany(s)))
})

# --- item 5: numeric reducers given a STRING --------------------------------

test_that("mean()/sd()/var() of a string give NA with dplyr's warning", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), s = c("a", "B", "b", "A", NA))
  f48_check_warns(df, function(d) summarise(d, m = mean(s)),
                  "argument is not numeric or logical")
  f48_check_warns(df, function(d) summarise(group_by(d, g), m = mean(s, na.rm = TRUE)),
                  "argument is not numeric or logical")
  f48_check_warns(df, function(d) mutate(d, m = mean(s)),
                  "argument is not numeric or logical")
  f48_check_warns(df, function(d) summarise(d, m = sd(s, na.rm = TRUE)),
                  "NAs introduced by coercion")
  f48_check_warns(df, function(d) summarise(group_by(d, g), m = var(s)),
                  "NAs introduced by coercion")
})

test_that("median() of a string is dplyr's string for an odd count", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 1, 2, 2, 2), s = c("b", "a", "c", "d", NA, "f"))
  f48_check(df, function(d) summarise(d, m = median(s, na.rm = TRUE)))
  f48_check(df, function(d) summarise(group_by(d, g), m = median(s)))
  f48_check_warns(df[1:4, ], function(d) summarise(d, m = median(s)),
                  "argument is not numeric or logical")
})

test_that("sum()/prod()/quantile() of a string raise dplyr's error", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), s = c("a", "b", "c"))
  f48_same_error(df, function(d) summarise(d, m = sum(s)), "invalid 'type' \\(character\\)")
  f48_same_error(df, function(d) summarise(group_by(d, g), m = prod(s)),
                 "invalid 'type' \\(character\\)")
  f48_same_error(df, function(d) summarise(d, m = quantile(s, 0.5)),
                 "non-numeric argument")
  f48_same_error(df, function(d) mutate(d, m = sum(s)), "invalid 'type' \\(character\\)")
})

test_that("any()/all() of a non-logical match dplyr", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), i = c(0L, 2L, NA), s = c("a", "b", NA))
  f48_check(df, function(d) summarise(d, a = any(i, na.rm = TRUE), b = all(i)))
  f48_check(df, function(d) summarise(group_by(d, g), a = any(i), b = all(i, na.rm = TRUE)))
  f48_check_warns(df, function(d) summarise(d, a = any(s)),
                  "coercing argument of type 'character' to logical")
})

test_that("across() routes a string column's mean() like the plain verb", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2), x = c(1.5, 2, 4), s = c("a", "b", "c"))
  f48_check_warns(df, function(d) summarise(group_by(d, g), across(c(x, s), mean)),
                  "argument is not numeric or logical")
})

# --- native coverage of numeric columns must not shrink ----------------------

test_that("numeric reducers and windows stay native", {
  skip_if_no_gpu()
  df <- data.frame(g = c(1, 1, 2, 2, 2), x = c(1.5, 2, NA, 4, 5),
                   i = c(1L, 5L, 3L, NA, 2L), l = c(TRUE, FALSE, TRUE, TRUE, NA))
  f48_native(df, function(d) summarise(group_by(d, g),
    a = mean(x, na.rm = TRUE), b = sum(i, na.rm = TRUE), c = sd(x, na.rm = TRUE),
    e = var(i, na.rm = TRUE), f = median(x, na.rm = TRUE), h = min(i, na.rm = TRUE),
    k = max(x, na.rm = TRUE), m = mean(l, na.rm = TRUE), n = sum(l, na.rm = TRUE),
    o = any(l, na.rm = TRUE), p = all(l, na.rm = TRUE)))
  f48_native(df, function(d) summarise(d, a = mean(x), b = median(x), c = sum(x > 2, na.rm = TRUE)))
  f48_native(df, function(d) mutate(group_by(d, g),
    a = cumsum(i), b = cummax(x), c = cummin(i), e = cummean(x),
    f = cumall(l), h = cumany(l), k = ntile(x, 2)))
  f48_native(df, function(d) filter(d, x > 2, i != 3L))
})
