# #74: integer64 columns stay on the GPU for `%in%`, first()/last()/nth()/
# n_distinct() in mutate(), percent_rank()/cume_dist()/ntile(), and
# lag()/lead() with a non-NA default: all are exact on int64 values, including
# those past 2^53 that a double can't hold. bit64 is used through `bit64::`
# only (attaching it masks `%in%`). Helpers: helper-parity-74.R.

p74_i64_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L),
    b = p74_i64(c("9007199254740993", "2", "9007199254740993", NA, "5",
                  "-9007199254740993", "5", NA)),
    k = c(4, 1, 3, 2, 8, 7, 6, 5)
  )
}

test_that("#74: integer64 %in% a vector of integer64 or integer values", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- p74_i64_df()
  big <- p74_i64(c("9007199254740993", "5"))
  p74_native(df, function(d) filter(d, b %in% big))
  p74_native(df, function(d) filter(d, !(b %in% big)))
  p74_native(df, function(d) mutate(d, z = b %in% big))
  p74_native(df, function(d) mutate(d, z = b %in% bit64::as.integer64(c("2", "-9007199254740993"))))
  p74_native(df, function(d) mutate(d, z = b %in% c(2L, 5L)))
  # NA in the table matches an NA value
  p74_native(df, function(d) mutate(d, z = b %in% c(NA, 2L)))
  p74_native(df, function(d) mutate(d, z = b %in% p74_i64(c(NA, "5"))))
  # 2^53 + 1 is not 2^53
  p74_native(df, function(d) mutate(d, z = b %in% p74_i64("9007199254740992")))
  p74_native(df[0, ], function(d) filter(d, b %in% big))
})

test_that("#74: integer64 %in% a double vector keeps dplyr's result", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(b = p74_i64(c("100000", "2", "9007199254740993", NA)))
  p74_parity(df, function(d) mutate(d, z = b %in% c(1e5, 2, 9007199254740993)))
  p74_parity(df, function(d) mutate(d, z = b %in% c(2.5, NA)))
})

test_that("#74: first()/last()/nth()/n_distinct() of integer64 in mutate()", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- p74_i64_df()
  p74_native(df, function(d) mutate(d, z = dplyr::first(b), .by = g))
  p74_native(df, function(d) mutate(group_by(d, g), z = dplyr::last(b)))
  p74_native(df, function(d) mutate(d, z = dplyr::nth(b, 2), .by = g))
  p74_native(df, function(d) mutate(d, z = dplyr::nth(b, -1)))
  p74_native(df, function(d) mutate(d, z = dplyr::nth(b, 9), .by = g))
  p74_native(df, function(d) mutate(d, z = dplyr::first(b, order_by = k), .by = g))
  p74_native(df, function(d) mutate(d, z = n_distinct(b), .by = g))
  p74_native(df, function(d) mutate(d, z = n_distinct(b)))
  p74_native(df[0, ], function(d) mutate(d, z = dplyr::first(b), n = n_distinct(b), .by = g))
  # a later dot reads the integer64 result
  p74_native(df, function(d) mutate(d, z = dplyr::last(b), e = z == b, .by = g))
})

test_that("#74: percent_rank()/cume_dist()/ntile() of integer64", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- p74_i64_df()
  p74_native(df, function(d) mutate(d, p = dplyr::percent_rank(b), c = dplyr::cume_dist(b), t = dplyr::ntile(b, 3)))
  p74_native(df, function(d) mutate(d, p = dplyr::percent_rank(b), c = dplyr::cume_dist(b), t = dplyr::ntile(b, 2),
                                    .by = g))
  p74_native(df, function(d) mutate(group_by(d, g), t = dplyr::ntile(b, 5)))
  p74_native(df, function(d) filter(d, dplyr::cume_dist(b) <= 0.5))
  p74_native(df[0, ], function(d) mutate(d, p = dplyr::percent_rank(b), c = dplyr::cume_dist(b), t = dplyr::ntile(b, 2)))
})

test_that("#74: lag()/lead() of integer64 with a non-NA default", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- p74_i64_df()
  big <- p74_i64("-9007199254740993")
  p74_native(df, function(d) mutate(d, z = dplyr::lag(b, default = big)))
  p74_native(df, function(d) mutate(d, z = dplyr::lead(b, default = p74_i64("0")), .by = g))
  p74_native(df, function(d) mutate(group_by(d, g), z = dplyr::lag(b, 2, default = big)))
  p74_native(df, function(d) mutate(d, z = dplyr::lag(b, default = 7L), .by = g))
  p74_native(df, function(d) mutate(d, z = dplyr::lag(b, default = big, order_by = k)))
  p74_native(df[0, ], function(d) mutate(d, z = dplyr::lag(b, default = big)))
  # an NA default is the plain call
  p74_native(df, function(d) mutate(d, z = dplyr::lead(b, default = p74_i64(NA))))
  # a double default is cast by dplyr (vctrs): its result, wherever it runs
  p74_parity(df, function(d) mutate(d, z = dplyr::lag(b, default = 0)))
  p74_parity(df, function(d) mutate(d, z = dplyr::lag(b, default = 0.5)))
})
