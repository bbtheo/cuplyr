# #74, the coder's own cases next to the acceptance tests (test-parity-74-*.R):
# the warning of min()/max(na.rm = TRUE) for several calls of one dot, the forms
# that stay on the CPU, and the rank under an order. Helpers: helper-parity-74.R.

test_that("#74: a min()/max(na.rm = TRUE) call written twice in a dot warns for each", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    g = c(3L, 3L, 1L, 1L, 2L, 2L),
    v = c(NA, NaN, 1, NA, NA, NA),
    w = c(1, 2, NA, NA, 3, 4)
  )
  # group order: sorted keys for group_by(), first appearance for `.by`
  p74_native_warn(df, function(d) mutate(d, r = max(v, na.rm = TRUE) - max(v, na.rm = TRUE), .by = g))
  p74_native_warn(df, function(d) mutate(group_by(d, g), r = max(v, na.rm = TRUE) - max(v, na.rm = TRUE)))
  # a hit in a later call names its own group; the first warning is the first group
  p74_native_warn(df, function(d) mutate(d, r = max(w, na.rm = TRUE) + min(v, na.rm = TRUE), .by = g))
  # an expression argument
  p74_native_warn(df, function(d) mutate(d, r = max(v * 2, na.rm = TRUE), .by = g))
  # two dots: the first dot's group is the first warning
  p74_native_warn(df, function(d) mutate(d, a = min(w, na.rm = TRUE), b = max(v, na.rm = TRUE), .by = g))
})

test_that("#74: min()/max(na.rm = TRUE) shapes the GPU leaves to dplyr keep its result", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), v = c(NA, 2, NA, NaN), s = c("a", NA, NA, "b"))
  # in filter(): the warning names the predicate
  p74_parity(df, function(d) filter(d, v < max(v, na.rm = TRUE), .by = g))
  # a call whose result feeds a later window stage
  p74_parity(df, function(d) mutate(d, m = max(v, na.rm = TRUE), l = dplyr::lag(m), .by = g))
  # a string, and a call over a window result
  p74_parity(df, function(d) mutate(d, m = max(s, na.rm = TRUE), .by = g))
  p74_parity(df, function(d) mutate(d, m = max(cumsum(v), na.rm = TRUE), .by = g))
})

test_that("#74: mean(na.rm = TRUE) next to a lazy filter and an order_by rank", {
  skip_if_no_gpu()
  df <- tibble::tibble(g = c(1L, 1L, 1L, 2L, 2L), v = c(1, NA, 3, NaN, NA), k = c(2, 1, 3, 1, 2))
  p74_native(df, function(d) {
    mutate(filter(d, k < 3), m = mean(v, na.rm = TRUE), r = dplyr::order_by(k, row_number(v)), .by = g)
  })
  # ranks under desc(): ties of x broken by the order
  p74_native(df, function(d) mutate(d, r = dplyr::order_by(desc(k), row_number(desc(v))), .by = g))
})

# ---- the audit of every verb that can see a window min()/max(na.rm = TRUE) ----
# dplyr evaluates the call per group and base R warns for a group with no value
# left. mutate() plans that warning (and across() inside it); every other verb
# leaves the call to dplyr, whose warning names that verb. `u` is all NA: a call
# over it warns for the whole table.

p74_nm <- function() {
  tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), v = c(1, 2, NA, NaN, 5),
                 w = c(1, 2, 3, 4, 5), k = c(2, 1, 2, 1, 2), u = NA_real_)
}

test_that("#74: summarise() of a window min()/max(na.rm = TRUE) keeps dplyr's warning", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_parity(df, function(d) summarise(d, z = mean(v / max(v, na.rm = TRUE)), .by = g))
  p74_parity(df, function(d) summarise(group_by(d, g), z = sum(v == max(v, na.rm = TRUE), na.rm = TRUE)))
  # ungrouped, over the all-NA column
  p74_parity(df, function(d) summarise(d, z = mean(w / min(u, na.rm = TRUE))))
  # across() of the aggregate, and an aggregate of an aggregate's arithmetic
  p74_parity(df, function(d) summarise(d, across(c(v, w), ~ mean(.x / max(.x, na.rm = TRUE))), .by = g))
  p74_parity(df, function(d) summarise(d, z = max(v / max(v, na.rm = TRUE), na.rm = TRUE) + 1, .by = g))
  # the aggregate itself is native
  p74_native(df, function(d) summarise(d, z = max(w, na.rm = TRUE) + 1, .by = g))
})

test_that("#74: reframe() of a window min()/max(na.rm = TRUE) keeps dplyr's warning", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_parity(df, function(d) reframe(d, z = mean(v / max(v, na.rm = TRUE)), .by = g))
  p74_parity(df, function(d) reframe(d, z = v / max(v, na.rm = TRUE), .by = g))
  p74_parity(df, function(d) reframe(group_by(d, g), z = sum(v == max(v, na.rm = TRUE), na.rm = TRUE)))
})

test_that("#74: count()/add_count()/tally() with a min()/max(na.rm = TRUE) in wt or a key", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_parity(df, function(d) count(d, g, wt = v / max(v, na.rm = TRUE)))
  p74_parity(df, function(d) tally(group_by(d, g), wt = v / max(v, na.rm = TRUE)))
  p74_parity(df, function(d) add_count(group_by(d, g), wt = v / max(v, na.rm = TRUE)))
  p74_parity(df, function(d) add_count(d, g, wt = v / max(v, na.rm = TRUE)))
  p74_parity(df, function(d) count(d, g, z = max(u, na.rm = TRUE)))
  p74_parity(df, function(d) add_count(d, g, z = max(u, na.rm = TRUE)))
})

test_that("#74: slice_*()/arrange()/distinct()/group_by() with a min()/max(na.rm = TRUE)", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_parity(df, function(d) slice_min(d, v - max(v, na.rm = TRUE), by = g))
  p74_parity(df, function(d) slice_max(group_by(d, g), v / max(v, na.rm = TRUE), n = 1))
  p74_parity(df, function(d) slice_min(d, u - max(u, na.rm = TRUE)))
  p74_parity(df, function(d) arrange(d, v / max(v, na.rm = TRUE)))
  p74_parity(df, function(d) arrange(d, u - max(u, na.rm = TRUE)))
  p74_parity(df, function(d) arrange(group_by(d, g), u - max(u, na.rm = TRUE), .by_group = TRUE))
  p74_parity(df, function(d) distinct(d, g, z = max(v, na.rm = TRUE)))
  p74_parity(df, function(d) distinct(d, g, z = max(u, na.rm = TRUE)))
  p74_parity(df, function(d) distinct(group_by(d, g), z = max(u, na.rm = TRUE)))
  p74_parity(df, function(d) group_by(d, z = max(u, na.rm = TRUE)))
  p74_parity(df, function(d) group_by(group_by(d, g), z = max(u, na.rm = TRUE), .add = TRUE))
})

test_that("#74: transmute() and a rowwise mutate() with a min()/max(na.rm = TRUE)", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_parity(df, function(d) transmute(d, z = max(u, na.rm = TRUE) + w))
  p74_parity(df, function(d) transmute(group_by(d, g), z = max(v, na.rm = TRUE)))
  # a window call over a rowwise column (the call of one row's value) is dplyr's
  p74_parity(df, function(d) mutate(rowwise(d, g), z = max(v / max(v, na.rm = TRUE), na.rm = TRUE)))
})

test_that("#74: across() of min()/max(na.rm = TRUE) warns once, naming the first group", {
  skip_if_no_gpu()
  df <- p74_nm()
  p74_native_warn(df, function(d) mutate(d, across(c(v, w), ~ max(.x, na.rm = TRUE)), .by = g))
  p74_native_warn(df, function(d) mutate(group_by(d, g), across(c(v, w), ~ min(.x, na.rm = TRUE), .names = "lo_{.col}")))
  p74_native_warn(df, function(d) mutate(d, a = 1, across(c(w, v), list(mx = ~ max(.x, na.rm = TRUE))), .by = g))
})

test_that("#74: min()/max(na.rm = TRUE) over columns with no NA or NaN gives no warning", {
  skip_if_no_gpu()
  # (lower_window_narm_warn() ends at one whole-column sum for this input)
  df <- tibble::tibble(g = c(1L, 1L, 2L, 2L), v = c(1, 2, 3, 4), w = c(5, 6, 7, 8))
  p74_native(df, function(d) mutate(d, a = max(v, na.rm = TRUE), b = min(w, na.rm = TRUE), .by = g))
  p74_native(df, function(d) mutate(d, a = max(v, na.rm = TRUE) - min(v, na.rm = TRUE)))
  # a NaN and a NA in different columns still name their groups
  df2 <- tibble::tibble(g = c(1L, 1L, 2L, 2L, 3L), v = c(1, 2, NaN, NaN, 5), w = c(1, NA, 3, 4, NA))
  p74_native_warn(df2, function(d) mutate(d, a = max(v, na.rm = TRUE), b = min(w, na.rm = TRUE), .by = g))
  # an Inf in the column (its sum is Inf, a -Inf one makes it NaN)
  df3 <- tibble::tibble(g = c(1L, 1L, 2L, 2L), v = c(Inf, -Inf, 1, 2))
  p74_native(df3, function(d) mutate(d, a = max(v, na.rm = TRUE), .by = g))
})

test_that("#74: explain() shows the frame of a row_number(x) under order_by()", {
  skip_if_no_gpu()
  lz <- tbl_gpu(tibble::tibble(g = c(1L, 1L, 2L), v = c(3, 1, 2), w = c(1, 2, 3)), lazy = TRUE)
  q <- mutate(lz, r = dplyr::order_by(w, row_number(v)), .by = g)
  for (optimized in c(FALSE, TRUE)) {
    txt <- paste(utils::capture.output(show_query(q, optimized = optimized)), collapse = "\n")
    expect_match(txt, "row_number(v, order_by = c(v, w))", fixed = TRUE)
  }
  q2 <- mutate(lz, m = max(v, na.rm = TRUE), .by = g)
  expect_match(paste(utils::capture.output(show_query(q2)), collapse = "\n"), "max(v, na.rm = TRUE)", fixed = TRUE)
})
