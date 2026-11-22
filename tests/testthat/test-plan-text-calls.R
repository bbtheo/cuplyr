# Calls whose IR is encoded in a form of its own print as the call the user
# wrote (R/plan-text.R plan_user_call()), end to end through the parser.

pc_df <- tibble::tibble(
  s = c("ab", "cd", "ef"),
  d = as.Date("2024-01-01") + 0:2,
  ts = as.POSIXct("2024-01-01 10:30:00", tz = "UTC") + 0:2,
  x = c(1, 2, NaN),
  i = 1:3
)

pc_lines <- function(x) {
  utils::capture.output(show_query(x, optimized = FALSE))
}

pc_has <- function(x, line) {
  out <- pc_lines(x)
  expect_true(line %in% trimws(out), info = paste(c(paste0("wanted: ", line), "got:", out), collapse = "\n"))
}

test_that("date calls print as R calls", {
  skip_if_no_gpu()
  lz <- tbl_gpu(pc_df, lazy = TRUE)
  pc_has(mutate(lz, a = as.Date(ts)), "mutate(a = as.Date(ts, tz = \"UTC\"))")
  pc_has(mutate(lz, a = lubridate::floor_date(ts, "hour")), "mutate(a = floor_date(ts, \"hour\"))")
  pc_has(mutate(lz, a = lubridate::ceiling_date(ts, "day")), "mutate(a = ceiling_date(ts, \"day\"))")
})

test_that("recode() and replace_when() print as the user wrote them", {
  skip_if_no_gpu()
  lz <- tbl_gpu(pc_df, lazy = TRUE)
  pc_has(mutate(lz, a = dplyr::recode(s, ab = "A", cd = "C", .default = "z")),
         "mutate(a = recode(s, ab = \"A\", cd = \"C\", .default = \"z\"))")
  pc_has(mutate(lz, a = replace_when(x, x > 2 ~ 0, x < 1 ~ 9)),
         "mutate(a = replace_when(x, x > 2 ~ 0, x < 1 ~ 9))")
})

test_that("rowwise aggregates print as the aggregate over the row", {
  skip_if_no_gpu()
  q <- tbl_gpu(pc_df, lazy = TRUE) |> rowwise() |> mutate(m = max(x, i), t = sum(c_across(c(x, i)), na.rm = TRUE))
  out <- paste(pc_lines(q), collapse = "\n")
  expect_match(out, "m = max(c(x, i))", fixed = TRUE)
  expect_match(out, "t = sum(c(x, i), na.rm = TRUE)", fixed = TRUE)
  expect_false(grepl("rw_agg", out, fixed = TRUE))
})

test_that("a NaN literal prints as NaN, an NA as NA", {
  skip_if_no_gpu()
  lz <- tbl_gpu(pc_df, lazy = TRUE)
  pc_has(mutate(lz, a = if_else(is.nan(x), NaN, x)), "mutate(a = if_else(is.nan(x), NaN, x))")
  pc_has(mutate(lz, a = if_else(is.nan(x), NA_real_, x)), "mutate(a = if_else(is.nan(x), NA, x))")
})

test_that("the CPU history follows a group_by() whose columns come from the CPU step", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  q <- tbl_gpu(pc_df, lazy = TRUE) |> mutate(z = pnorm(x))
  grouped <- group_by(q, i)
  # the origin belongs to the GPU table it described: a copy of it on another
  # table (an edited field) is not believed
  stale <- grouped
  stale$ptr <- tbl_gpu(pc_df, lazy = TRUE)$ptr
  expect_true("CPU fallback: mutate() for pnorm(x)" %in% trimws(pc_lines(grouped)))
  expect_false("CPU fallback: mutate() for pnorm(x)" %in% trimws(pc_lines(stale)))
})
