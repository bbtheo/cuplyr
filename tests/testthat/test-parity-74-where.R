# #74: a type-based across() selection (`where(is.integer)`, ...) after a dot
# of the same mutate() that computes n(), row_number(), ntile(), a lubridate
# date part or roll_*() resolves on the GPU: cuplyr knows R's type for those
# results. lubridate's year(), month(), yday() and wday() are doubles, and so
# are they on the GPU. Helpers: helper-parity-74.R.

p74_where_df <- function() {
  tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 2L),
    dte = as.Date(c("2020-01-01", "2021-06-15", NA, "2024-02-29", "1999-12-31")),
    t = as.POSIXct(c("2020-01-01 10:11:12", "2021-06-15 01:02:03", "2022-03-03 04:05:06",
                     NA, "1969-07-20 20:17:40"), tz = "UTC"),
    v = c(1.5, 2, NA, 4, -1),
    i = c(4L, 2L, 2L, NA, 1L)
  )
}

test_that("#74: where() after n(), row_number() and ntile() in the same mutate()", {
  skip_if_no_gpu()
  df <- p74_where_df()
  p74_native(df, function(d) mutate(d, nn = n(), across(where(is.integer), ~ .x * 2L)))
  p74_native(df, function(d) mutate(d, nn = n(), across(where(is.integer), ~ .x * 2L), .by = g))
  p74_native(df, function(d) mutate(group_by(d, g), nn = n(), across(where(is.numeric), ~ .x + 1)))
  p74_native(df, function(d) mutate(d, r = row_number(), across(where(is.integer), ~ .x + 1L)))
  p74_native(df, function(d) mutate(d, r = row_number(v), across(where(is.numeric), ~ .x / 2)))
  p74_native(df, function(d) mutate(d, r = dplyr::ntile(v, 2), across(where(is.integer), ~ .x - 1L)))
  p74_native(df, function(d) mutate(group_by(d, g), r = dplyr::ntile(n = 3), across(where(is.integer), ~ -.x)))
  # the new column is picked up by name, by type and by the .names spec
  p74_native(df, function(d) {
    mutate(d, nn = n(), across(where(is.integer), ~ .x * 10L, .names = "{.col}_x"))
  })
})

test_that("#74: lubridate's year()/month()/yday()/wday() are doubles, as in lubridate", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- p74_where_df()
  p74_native(df, function(d) {
    mutate(d, y = lubridate::year(dte), m = lubridate::month(dte), yd = lubridate::yday(dte),
           wd = lubridate::wday(dte))
  })
  p74_native(df, function(d) {
    mutate(d, y = lubridate::year(t), m = lubridate::month(t), yd = lubridate::yday(t),
           wd = lubridate::wday(t))
  })
  p74_native(df, function(d) mutate(d, y = lubridate::year(dte) + 1L, k = lubridate::month(dte) %/% 4L))
  p74_native(df, function(d) filter(d, lubridate::year(dte) >= 2020L))
  p74_native(df, function(d) summarise(d, y = max(lubridate::year(dte)), .by = g))
})

test_that("#74: where() after a lubridate date part in the same mutate()", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- p74_where_df()
  p74_native(df, function(d) mutate(d, y = lubridate::year(dte), across(where(is.double), ~ .x + 1)))
  p74_native(df, function(d) mutate(d, y = lubridate::year(dte), across(where(is.integer), ~ .x + 1L)))
  p74_native(df, function(d) mutate(d, m = lubridate::month(t), across(where(is.numeric), ~ .x * 2)))
  p74_native(df, function(d) mutate(d, dd = lubridate::day(dte), q = lubridate::quarter(dte),
                                    across(where(is.integer), ~ .x + 1L)))
  p74_native(df, function(d) mutate(d, wd = lubridate::wday(dte), yd = lubridate::yday(dte),
                                    across(where(is.numeric), ~ .x * 10), .by = g))
})

test_that("#74: where() after roll_*() in the same mutate()", {
  skip_if_no_gpu()
  df <- p74_where_df()
  p74_native(df, function(d) mutate(d, r = roll_sum(v, 2), across(where(is.double), ~ .x + 1)))
  p74_native(df, function(d) mutate(d, r = roll_mean(i, 2), across(where(is.numeric), ~ .x * 2)))
  p74_native(df, function(d) mutate(d, r = roll_sd(v, 3), across(where(is.numeric), ~ .x - 1)))
  p74_native(df, function(d) mutate(d, r = roll_max(v, 2), across(where(is.numeric), ~ -.x), .by = g))
  p74_native(df, function(d) mutate(d, r = roll_min(v, 2, na.rm = TRUE), across(where(is.integer), ~ .x)))
  # roll_max() of an integer keeps the integer type on the GPU (R gives a
  # double), so a type-based selection after it is dplyr's to resolve
  p74_parity(df, function(d) mutate(d, r = roll_max(i, 2), across(where(is.double), ~ .x + 1)))
})

test_that("#74: where() in a later verb after these columns stays on the GPU", {
  skip_if_no_gpu()
  skip_if_not_installed("lubridate")
  df <- p74_where_df()
  p74_native(df, function(d) select(mutate(d, nn = n(), r = dplyr::ntile(v, 2)), where(is.integer)))
  p74_native(df, function(d) select(mutate(d, y = lubridate::year(dte), s = roll_sum(v, 2)),
                                    where(is.double)))
  p74_native(df, function(d) {
    mutate(mutate(d, nn = n(), .by = g), across(where(is.integer), ~ .x * 2L))
  })
})

test_that("#74: summarise() across(where()) after n() matches dplyr", {
  skip_if_no_gpu()
  df <- p74_where_df()
  # dplyr's across() in summarise() sees the summary columns made before it
  p74_parity(df, function(d) {
    summarise(d, nn = n(), across(where(is.integer), ~ max(.x), .names = "{.col}_m"), .by = g)
  })
  p74_parity(df, function(d) summarise(d, nn = n(), across(where(is.integer), ~ max(.x)), .by = g))
})
