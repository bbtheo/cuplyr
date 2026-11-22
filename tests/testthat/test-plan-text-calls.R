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
  # as.POSIXct() keeps a POSIXct's zone: no `tz`, which would change it
  ny <- tbl_gpu(tibble::tibble(t = as.POSIXct("2024-01-01 10:00", tz = "America/New_York")), lazy = TRUE)
  q <- mutate(ny, a = as.POSIXct(t))
  pc_has(q, "mutate(a = as.POSIXct(t))")
  expect_identical(attr(collect(q)$a, "tzone"), "America/New_York")
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
  pc_has(mutate(lz, a = trimws(s), b = trimws(s, "left")), "mutate(a = trimws(s), b = trimws(s, \"left\"))")
})

test_that("a long recode() lists its first 20 replacements and counts the rest", {
  skip_if_no_gpu()
  lz <- tbl_gpu(pc_df, lazy = TRUE)
  pairs <- rlang::set_names(as.list(sprintf("v%02d", 1:25)), as.character(1:25))
  q <- mutate(lz, a = dplyr::recode(i, !!!pairs, .default = "?"))
  out <- paste(pc_lines(q), collapse = "\n")
  expect_match(out, "a = recode(i, `1` = \"v01\", ", fixed = TRUE)
  expect_match(out, "`20` = \"v20\", <5 more>, .default = \"?\")", fixed = TRUE)
  expect_false(grepl("`21`|recode_lookup", out))
})

test_that("rowwise aggregates print as written, in a step marked rowwise", {
  skip_if_no_gpu()
  q <- tbl_gpu(pc_df, lazy = TRUE) |> rowwise() |>
    mutate(m = max(x, i), t = sum(c_across(c(x, i)), na.rm = TRUE), u = min(c(x, i)), v = mean(x))
  pc_has(q, "mutate(")
  out <- trimws(pc_lines(q))
  expect_true("m = max(x, i)," %in% out)
  expect_true("t = sum(c_across(c(x, i)), na.rm = TRUE)," %in% out)
  expect_true("u = min(c(x, i))," %in% out)
  expect_true("v = mean(x)" %in% out)
  expect_true(")  # rowwise" %in% out)
  expect_false(any(grepl("rw_agg", out, fixed = TRUE)))
  # a rowwise filter is marked rowwise, with or without an aggregate, and kept
  # apart from an ungrouped one
  rw <- tbl_gpu(pc_df, lazy = TRUE) |> rowwise()
  pc_has(filter(rw, x > 1), "filter(x > 1)  # rowwise")
  merged <- filter(rw, x > 1) |> filter(i > 1L) |> ungroup() |> filter(i < 3L)
  expect_identical(utils::tail(trimws(pc_lines(merged)), 3L),
                   c("filter(x > 1)  # rowwise", "filter(i > 1L)  # rowwise", "filter(i < 3L)"))
  opt <- trimws(utils::capture.output(show_query(merged, optimized = TRUE)))
  expect_identical(utils::tail(opt, 2L), c("filter(i > 1L, x > 1)  # rowwise", "filter(i < 3L)"))
  expect_equal(nrow(collect(merged)), nrow(dplyr::filter(pc_df, x > 1, i > 1L, i < 3L)))
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
