# F48 acceptance tests, item 3: string ordering follows the session collation.
# Native only under a C/POSIX collation (byte order); any other collation
# gives dplyr's (locale-ordered) result through the CPU fallback.

# dplyr is imported, not attached, in the test environment: bind the dplyr
# functions these pipelines call unqualified (file-local).
for (fn in c("if_else", "case_when", "between", "lag", "lead", "ntile",
             "cumall", "cumany", "cummean", "across")) {
  assign(fn, getExportedValue("dplyr", fn))
}
rm(fn)

f48_coll_df <- function() {
  data.frame(
    g = c(1, 1, 2, 2, 2, 2),
    s = c("a", "B", "b", "A", NA, "c"),
    t = c("B", "a", "A", "b", "c", NA)
  )
}

f48_ordering_pipelines <- list(
  summarise_min = function(d) summarise(d, m = min(s, na.rm = TRUE)),
  summarise_max = function(d) summarise(d, m = max(s, na.rm = TRUE)),
  summarise_min_grouped = function(d) summarise(group_by(d, g), lo = min(s, na.rm = TRUE),
                                                hi = max(t, na.rm = TRUE)),
  window_min = function(d) mutate(group_by(d, g), m = min(s, na.rm = TRUE)),
  window_max = function(d) mutate(d, m = max(t, na.rm = TRUE)),
  filter_gt = function(d) filter(d, s > "a"),
  filter_ge = function(d) filter(d, s >= "B"),
  filter_lt_col = function(d) filter(d, s < t),
  filter_le_grouped = function(d) filter(group_by(d, g), s <= t),
  filter_eq_max = function(d) filter(group_by(d, g), s == max(s, na.rm = TRUE)),
  mutate_cmp = function(d) mutate(d, m = s > "a", n = "b" >= t),
  pmin_pmax = function(d) mutate(d, lo = pmin(s, t), hi = pmax(s, t)),
  if_else_cmp = function(d) mutate(d, m = if_else(s < t, "lt", "ge")),
  case_when_cmp = function(d) mutate(d, m = case_when(s > "a" ~ 1, .default = 2)),
  sum_cmp = function(d) summarise(group_by(d, g), n = sum(s > "a", na.rm = TRUE)),
  across_cmp = function(d) mutate(d, across(c(s, t), ~ .x < "b", .names = "{.col}_lt"))
)

test_that("string ordering under a non-C collation matches dplyr", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("en_US.UTF-8"), "en_US.UTF-8 collation not available")
  withr::local_collate("en_US.UTF-8")
  df <- f48_coll_df()
  for (nm in names(f48_ordering_pipelines)) {
    pipe <- f48_ordering_pipelines[[nm]]
    expect_same_as_dplyr(df, pipe)
    expect_same_as_dplyr_lazy(df, pipe)
  }
})

# The pipelines above that are native on dev today (window min()/max() and
# pmin()/pmax() of a string already take the CPU route); they must stay native
# under a byte-order collation.
f48_native_ordering <- c(
  "summarise_min", "summarise_max", "summarise_min_grouped", "filter_gt",
  "filter_ge", "filter_lt_col", "filter_le_grouped", "mutate_cmp",
  "if_else_cmp", "case_when_cmp", "sum_cmp", "across_cmp"
)

test_that("string ordering stays native under the C collation", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("C"), "C collation not available")
  withr::local_collate("C")
  df <- f48_coll_df()
  for (nm in f48_native_ordering) {
    f48_native(df, f48_ordering_pipelines[[nm]])
  }
})

test_that("string ordering under the POSIX collation stays native", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("POSIX"), "POSIX collation not available")
  withr::local_collate("POSIX")
  df <- f48_coll_df()
  f48_native(df, function(d) filter(d, s > "a"))
  f48_native(df, function(d) summarise(group_by(d, g), m = min(s, na.rm = TRUE)))
})

test_that("string equality and between() stay native under any collation", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("en_US.UTF-8"), "en_US.UTF-8 collation not available")
  withr::local_collate("en_US.UTF-8")
  df <- f48_coll_df()
  # == / != don't depend on the collation; dplyr's between() compares with
  # vctrs::vec_compare(), which is C order.
  f48_native(df, function(d) filter(d, s == "a" | t != "b"))
  f48_native(df, function(d) mutate(d, m = s == t))
  f48_native(df, function(d) filter(d, between(s, "B", "a")))
  f48_native(df, function(d) mutate(d, m = between(s, "A", "b")))
})

test_that("the collation is read when the verb runs", {
  skip_if_no_gpu()
  skip_if_not(f48_collate_ok("en_US.UTF-8") && f48_collate_ok("C"), "collations not available")
  df <- f48_coll_df()
  pipe <- function(d) summarise(d, m = min(s, na.rm = TRUE))
  withr::with_collate("C", {
    expect_same_as_dplyr(df, pipe)
  })
  withr::with_collate("en_US.UTF-8", {
    expect_same_as_dplyr(df, pipe)
  })
})
