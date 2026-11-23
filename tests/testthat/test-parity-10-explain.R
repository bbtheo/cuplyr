# #10: explain() and show_query() print a readable plan: the steps as R code,
# the columns each scan reads ("5 of 41"), the steps that fell back to the CPU
# and why, and the plan as written with `optimized = FALSE`.
#
# The snapshots in _snaps/parity-10-explain.md are the contract for the text.

# 41 columns: six named ones and 35 fillers.
f10_flights <- function() {
  df <- tibble::tibble(
    year = 2013L,
    carrier = c("AA", "UA", "DL", "AA", "B6"),
    distance = c(1400L, 733L, 1089L, 2475L, 944L),
    air_time = c(227, 150, 160, 300, 140),
    dep_delay = c(2, 4, 30, 25, NA),
    arr_delay = c(11, 20, 17, 8, 3)
  )
  for (i in seq_len(35L)) {
    df[[sprintf("x%02d", i)]] <- as.double(i)
  }
  df
}

# The parquet copy of f10_flights(), always named flights.parquet.
f10_path <- function(envir = parent.frame()) {
  dir <- withr::local_tempdir(.local_envir = envir)
  path <- file.path(dir, "flights.parquet")
  write_parquet_gpu(tbl_gpu(f10_flights(), lazy = FALSE), path)
  path
}

# scan -> filter -> two mutates (fused) -> filter -> select, then a step that
# runs on the CPU (pnorm() has no GPU implementation), then more GPU steps.
f10_pipeline_head <- function(path) {
  read_parquet_gpu(path, lazy = TRUE) |>
    filter(distance > 1000L) |>
    mutate(gain = dep_delay - arr_delay) |>
    mutate(speed = distance / air_time * 60) |>
    filter(speed > 400, gain > 0) |>
    select(carrier, speed, gain)
}

f10_pipeline <- function(path) {
  withr::local_options(cuplyr.fallback = "silent")
  f10_pipeline_head(path) |>
    mutate(z = pnorm(gain / 10)) |>
    filter(z > 0.5) |>
    select(carrier, z)
}

f10_df <- function() {
  tibble::tibble(
    g = c("a", "b", "a"),
    x = c(1, NA, 3),
    y = c("p", "q", "r"),
    i = 1:3
  )
}

# expect_snapshot() skips itself unless NOT_CRAN is set, and the test runner
# doesn't set it: these snapshots are the contract, so they always run.
f10_snapshots_on <- function(envir = parent.frame()) {
  withr::local_envvar(NOT_CRAN = "true", .local_envir = envir)
}

f10_lines <- function(x, optimized = FALSE) {
  utils::capture.output(show_query(x, optimized = optimized))
}

# One line of the plan (leading/trailing spaces ignored) is exactly `line`.
expect_plan_line <- function(x, line, optimized = FALSE) {
  got <- f10_lines(x, optimized)
  expect_true(line %in% trimws(got),
              info = paste(c(paste0("wanted: ", line), "got:", got), collapse = "\n"))
}

# Does `v` hold anything that keeps an R environment or a GPU table alive?
f10_holds_ref <- function(v) {
  if (typeof(v) %in% c("externalptr", "environment", "closure", "language", "symbol", "promise")) {
    return(TRUE)
  }
  if (is.list(v)) {
    return(any(vapply(v, f10_holds_ref, logical(1))))
  }
  FALSE
}

# --- the issue's pipeline: parquet scan, fused mutate()/filter(), CPU step ---

test_that("explain() shows the optimized plan with the scan's columns and the CPU step", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- f10_pipeline(f10_path())
  expect_snapshot(explain(q))
})

test_that("explain(optimized = FALSE) shows the plan as written", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- f10_pipeline(f10_path())
  expect_snapshot(explain(q, optimized = FALSE))
})

test_that("show_query() prints the same plan as explain()", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- f10_pipeline(f10_path())
  expect_snapshot(show_query(q))
  expect_snapshot(show_query(q, optimized = FALSE))
  expect_identical(utils::capture.output(show_query(q)), utils::capture.output(explain(q)))
  expect_identical(utils::capture.output(show_query(q, optimized = FALSE)),
                   utils::capture.output(explain(q, optimized = FALSE)))
})

test_that("the plan before the CPU step is shown, fused and pruned, while it is pending", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- f10_pipeline_head(f10_path())
  expect_snapshot({
    explain(q)
    explain(q, optimized = FALSE)
  })
})

test_that("explain() and show_query() run nothing and return x invisibly", {
  skip_if_no_gpu()
  q <- f10_pipeline_head(f10_path())
  parquet_last_read(reset = TRUE)
  utils::capture.output(e <- withVisible(explain(q)))
  utils::capture.output(s <- withVisible(show_query(q, optimized = FALSE)))
  # (read before any expectation: comparing tibbles with waldo collects them)
  expect_null(parquet_last_read())
  expect_true(has_pending_ops(q))
  expect_false(e$visible)
  expect_false(s$visible)
  expect_true(identical(e$value, q))
  expect_true(identical(s$value, q))
})

test_that("printing the plan does not change the result", {
  skip_if_no_gpu()
  path <- f10_path()
  q <- f10_pipeline(path)
  utils::capture.output(explain(q), explain(q, optimized = FALSE))
  expected <- f10_flights() |>
    dplyr::filter(distance > 1000L) |>
    dplyr::mutate(gain = dep_delay - arr_delay) |>
    dplyr::mutate(speed = distance / air_time * 60) |>
    dplyr::filter(speed > 400, gain > 0) |>
    dplyr::select(carrier, speed, gain) |>
    dplyr::mutate(z = pnorm(gain / 10)) |>
    dplyr::filter(z > 0.5) |>
    dplyr::select(carrier, z)
  expect_equal(tibble::as_tibble(collect(q)), expected)
})

# --- expressions as R code -------------------------------------------------

test_that("each step prints as the dplyr call, with its expressions as R code", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- tbl_gpu(f10_df(), lazy = TRUE) |>
    filter(!is.na(x), x %in% c(1, 3) | y == "q") |>
    mutate(r = if_else(x > 1, 10, 20), l = log(x) + 1, k = -i) |>
    arrange(desc(x), y) |>
    group_by(g) |>
    summarise(n = n(), m = mean(x, na.rm = TRUE))
  expect_snapshot(show_query(q, optimized = FALSE))
})

test_that("a step longer than 80 characters is split one argument per line", {
  skip_if_no_gpu()
  f10_snapshots_on()
  q <- tbl_gpu(f10_df(), lazy = TRUE) |>
    mutate(first_long_name = x * 2 + 1, second_long_name = log(x) - 3, third_long_name = -i)
  expect_snapshot(show_query(q, optimized = FALSE))
})

test_that("parentheses, literals and dates print as R code", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f10_df(), lazy = TRUE)
  expect_plan_line(mutate(lz, p = (x + 1) * 2, q = x^2, s = !(x > 1 & i < 3L)),
                   "mutate(p = (x + 1) * 2, q = x^2, s = !(x > 1 & i < 3L))")
  expect_plan_line(filter(lz, x > -1, y != "a"), "filter(x > -1, y != \"a\")")
  expect_plan_line(filter(lz, x %in% c(1, 2, 3)), "filter(x %in% c(1, 2, 3))")
  dates <- tbl_gpu(tibble::tibble(d = as.Date(c("2024-01-01", "2024-03-01"))), lazy = TRUE)
  expect_plan_line(filter(dates, d > as.Date("2024-02-01")), "filter(d > as.Date(\"2024-02-01\"))")
})

test_that("the other verbs print as the dplyr call they stand for", {
  skip_if_no_gpu()
  lz <- tbl_gpu(f10_df(), lazy = TRUE)
  expect_plan_line(select(lz, g, x), "select(g, x)")
  expect_plan_line(rename(lz, gg = g), "rename(gg = g)")
  expect_plan_line(distinct(lz, g, .keep_all = TRUE), "distinct(g, .keep_all = TRUE)")
  expect_plan_line(arrange(lz, x), "arrange(x)")
  expect_plan_line(slice_head(lz, n = 2), "slice_head(n = 2)")
  expect_plan_line(slice_min(lz, x, n = 1), "slice_min(x, n = 1)")
  expect_plan_line(slice_max(lz, x, n = 1, with_ties = FALSE), "slice_max(x, n = 1, with_ties = FALSE)")
  expect_plan_line(collapse(filter(lz, x > 1)), "collapse()")
  expect_plan_line(summarise(lz, s = sum(x, na.rm = TRUE), .by = g),
                   "summarise(s = sum(x, na.rm = TRUE), .by = g)")

  windowed <- paste(f10_lines(mutate(group_by(lz, g), rn = dplyr::row_number())), collapse = "\n")
  expect_match(windowed, "row_number()", fixed = TRUE)
  expect_false(grepl("ast_", windowed, fixed = TRUE))
})

test_that("the optimizer's top-k slice shows where it came from", {
  skip_if_no_gpu()
  q <- head(arrange(tbl_gpu(f10_df(), lazy = TRUE), x), 2)
  expect_plan_line(q, "arrange(x)", optimized = FALSE)
  expect_plan_line(q, "slice_head(n = 2)", optimized = FALSE)
  expect_plan_line(q, "slice_min(x, n = 2, with_ties = FALSE)  # top-k: arrange() |> head()", optimized = TRUE)
})

test_that("a join shows its right side as `y` under the join", {
  skip_if_no_gpu()
  f10_snapshots_on()
  x <- tbl_gpu(tibble::tibble(k = 1:3, a = c(1, 2, 3)), lazy = TRUE)
  y <- tbl_gpu(tibble::tibble(k = 2:4, b = c("p", "q", "r")), lazy = TRUE) |>
    filter(b != "q")
  q <- left_join(x, y, by = "k") |> filter(a > 1)
  expect_snapshot(show_query(q, optimized = FALSE))
})

test_that("join conditions and arguments print as in dplyr", {
  skip_if_no_gpu()
  x <- tbl_gpu(tibble::tibble(k = 1:3, a = c(1, 2, 3)), lazy = TRUE)
  y <- tbl_gpu(tibble::tibble(k = 2:4, b = c(1, 5, 9)), lazy = TRUE)
  expect_plan_line(semi_join(x, y, by = "k"), "semi_join(y, by = join_by(k))")
  expect_plan_line(anti_join(x, y, by = "k"), "anti_join(y, by = join_by(k))")
  expect_plan_line(cross_join(x, y), "cross_join(y)")
  expect_plan_line(inner_join(x, y, by = dplyr::join_by(a >= b)), "inner_join(y, by = join_by(a >= b))")
  expect_plan_line(left_join(x, y, by = dplyr::join_by(k, closest(a >= b))),
                   "left_join(y, by = join_by(k, closest(a >= b)))")
  expect_plan_line(inner_join(x, y, by = "k", relationship = "many-to-many", suffix = c("_l", "_r")),
                   "inner_join(y, by = join_by(k), suffix = c(\"_l\", \"_r\"), relationship = \"many-to-many\")")
  # arguments that change the result are printed for every join type
  expect_plan_line(semi_join(x, y, by = "k", na_matches = "never"),
                   "semi_join(y, by = join_by(k), na_matches = \"never\")")
  expect_plan_line(anti_join(x, y, by = "k", na_matches = "never"),
                   "anti_join(y, by = join_by(k), na_matches = \"never\")")
  expect_plan_line(cross_join(x, y, suffix = c("_l", "_r")), "cross_join(y, suffix = c(\"_l\", \"_r\"))")
})

test_that("string functions print as the R call, with R's argument order", {
  skip_if_no_gpu()
  lz <- tbl_gpu(tibble::tibble(s = c("Apple", "banana", "Avocado"), t = c("x1", "y2", "x3")), lazy = TRUE)
  expect_plan_line(filter(lz, grepl("^A", s)), "filter(grepl(\"^A\", s))")
  expect_plan_line(filter(lz, grepl("a", s, fixed = TRUE)), "filter(grepl(\"a\", s, fixed = TRUE))")
  expect_plan_line(filter(lz, startsWith(t, "x")), "filter(startsWith(t, \"x\"))")
  expect_plan_line(filter(lz, endsWith(t, "2")), "filter(endsWith(t, \"2\"))")
  expect_plan_line(mutate(lz, a = substr(s, 2, 4)), "mutate(a = substr(s, 2L, 4L))")
  expect_plan_line(mutate(lz, b = gsub("a", "o", s, fixed = TRUE)), "mutate(b = gsub(\"a\", \"o\", s, fixed = TRUE))")
  expect_plan_line(mutate(lz, b = sub("^A", "a", s)), "mutate(b = sub(\"^A\", \"a\", s))")
  expect_plan_line(mutate(lz, c = paste0(s, "_", t)), "mutate(c = paste0(s, \"_\", t))")
  expect_plan_line(mutate(lz, d = paste(s, t, sep = "-")), "mutate(d = paste(s, t, sep = \"-\"))")
  expect_plan_line(mutate(lz, d = paste(s, t)), "mutate(d = paste(s, t))")
  # no internal op name ever reaches the text
  out <- paste(f10_lines(mutate(lz, a = substr(s, 2, 4), c = paste0(s, t), u = toupper(s))), collapse = "\n")
  expect_false(grepl("paste_cols|starts_with|ends_with|cast_|factor_codes|recode_lookup|rw_agg", out))
})

test_that("recode() prints as recode() for any number of replacements", {
  skip_if_no_gpu()
  lz <- tbl_gpu(tibble::tibble(d = c("Mon", "Tue", "Sun"), m = c(1L, 5L, 12L)), lazy = TRUE)
  seven <- mutate(lz, n = dplyr::recode(d, Mon = "M", Tue = "T", Wed = "W", Thu = "R", Fri = "F",
                                        Sat = "S", Sun = "U"))
  twelve <- mutate(lz, n = dplyr::recode(m, `1` = "a", `2` = "b", `3` = "c", `4` = "d", `5` = "e", `6` = "f",
                                         `7` = "g", `8` = "h", `9` = "i", `10` = "j", `11` = "k", `12` = "l",
                                         .default = "?"))
  for (q in list(seven, twelve)) {
    for (optimized in c(FALSE, TRUE)) {
      out <- paste(f10_lines(q, optimized), collapse = "\n")
      expect_false(grepl("recode_lookup", out, fixed = TRUE), info = out)
      expect_match(out, "recode(", fixed = TRUE)
    }
  }
  expect_match(paste(f10_lines(seven), collapse = "\n"), "Sun = \"U\"", fixed = TRUE)
})

test_that("a step that runs per row is marked rowwise", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  rw <- tbl_gpu(tibble::tibble(x = c(1, 5), i = 3:4), lazy = TRUE) |> rowwise()
  steps <- list(
    mutate = mutate(rw, m = max(x, i)),
    c_across = mutate(rw, s = sum(dplyr::c_across(c(x, i)))),
    filter = filter(rw, max(x, i) > 3)
  )
  for (nm in names(steps)) {
    for (optimized in c(FALSE, TRUE)) {
      out <- trimws(f10_lines(steps[[nm]], optimized))
      step <- out[grepl("^(mutate|filter)\\(", out)]
      expect_length(step, 1L)
      expect_true(endsWith(step, "  # rowwise"), info = paste(nm, step))
      expect_false(grepl("rw_agg|# grouped", step), info = paste(nm, step))
    }
  }
  # the marker survives into the history of a CPU step
  cpu <- mutate(steps$mutate, z = pnorm(m))
  out <- trimws(f10_lines(cpu, optimized = TRUE))
  expect_true(any(startsWith(out, "mutate(m = max(") & endsWith(out, "  # rowwise")))
})

# --- CPU steps ---------------------------------------------------------------

test_that("a CPU step on a table with nothing pending is still shown", {
  skip_if_no_gpu()
  f10_snapshots_on()
  withr::local_options(cuplyr.fallback = "silent")
  q <- tbl_gpu(f10_df(), lazy = TRUE) |> mutate(z = pnorm(x))
  expect_false(has_pending_ops(q))
  expect_snapshot(explain(q))
})

test_that("CPU steps nest: the input of a CPU step can hold an earlier one", {
  skip_if_no_gpu()
  f10_snapshots_on()
  withr::local_options(cuplyr.fallback = "silent")
  q <- tbl_gpu(f10_df(), lazy = TRUE) |>
    mutate(z = pnorm(x)) |>
    filter(z > 0) |>
    mutate(w = qnorm(z))
  expect_snapshot(explain(q))
})

test_that("only the newest three CPU steps are kept", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  q <- tbl_gpu(f10_df(), lazy = TRUE) |>
    mutate(a1 = pnorm(x)) |>
    mutate(a2 = pnorm(a1)) |>
    mutate(a3 = pnorm(a2)) |>
    mutate(a4 = pnorm(a3)) |>
    filter(a4 > 0)
  out <- trimws(f10_lines(q, optimized = TRUE))
  expect_identical(sum(startsWith(out, "CPU fallback: ")), 3L)
  expect_identical(out[startsWith(out, "CPU fallback: ")],
                   c("CPU fallback: mutate() for pnorm(a3)",
                     "CPU fallback: mutate() for pnorm(a2)",
                     "CPU fallback: mutate() for pnorm(a1)"))
  expect_true("input: not shown (older CPU steps)" %in% out)
  expect_false("table on the GPU" %in% out)
})

test_that("the CPU history holds text only, no GPU table or R environment", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  x <- tbl_gpu(tibble::tibble(k = 1:3, a = c(1, 2, 3)), lazy = TRUE)
  y <- tbl_gpu(tibble::tibble(k = 2:4, b = c(1, 5, 9)), lazy = TRUE)
  res <- inner_join(x, y, by = "k") |> mutate(z = pnorm(a))
  expect_false(is.null(res$origin))
  expect_false(f10_holds_ref(res$origin))
  nxt <- filter(res, z > 0)
  expect_false(is.null(nxt$lazy_ops$input$origin))
  expect_false(f10_holds_ref(nxt$lazy_ops$input$origin))
  out <- trimws(f10_lines(nxt, optimized = TRUE))
  expect_true("inner_join(y, by = join_by(k))" %in% out)
})

test_that("verbs that only change the grouping keep the CPU step", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  df <- tibble::tibble(g = c("a", "b", "a"), x = c(1, 2, 3), i = 1:3)
  q <- tbl_gpu(df, lazy = TRUE) |> filter(x > 1) |> mutate(z = pnorm(x))
  plans <- list(
    group_by = group_by(q, g),
    group_by_summarise = q |> group_by(g) |> summarise(m = mean(z)),
    ungroup = q |> group_by(g) |> ungroup() |> filter(z > 0),
    count = count(q, g),
    add_count = add_count(q, g),
    rowwise = rowwise(q)
  )
  for (nm in names(plans)) {
    out <- trimws(f10_lines(plans[[nm]], optimized = TRUE))
    expect_true("CPU fallback: mutate() for pnorm(x)" %in% out, info = nm)
    expect_true("filter(x > 1)" %in% out, info = nm)
  }

  dir <- withr::local_tempdir()
  path <- file.path(dir, "small.parquet")
  write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path)
  pq <- read_parquet_gpu(path, lazy = TRUE) |>
    filter(i > 1L) |>
    mutate(z = pnorm(x)) |>
    group_by(g) |>
    summarise(m = mean(z))
  out <- trimws(f10_lines(pq, optimized = TRUE))
  expect_true("CPU fallback: mutate() for pnorm(x)" %in% out)
  expect_true("parquet scan: small.parquet" %in% out)
  expect_true("reader filter: i > 1L" %in% out)
  expect_identical(utils::tail(out, 2), c("group_by(g)", "summarise(m = mean(z))"))
})

test_that("the reader filter is R code", {
  skip_if_no_gpu()
  dir <- withr::local_tempdir()
  path <- file.path(dir, "dates.parquet")
  df <- tibble::tibble(d = as.Date(c("2024-01-01", "2024-03-01", "2024-05-01")), i = 1:3)
  write_parquet_gpu(tbl_gpu(df, lazy = FALSE), path)
  q <- read_parquet_gpu(path, lazy = TRUE) |> filter(d > as.Date("2024-02-01"), i < 5L)
  expect_plan_line(q, "filter(d > as.Date(\"2024-02-01\"), i < 5L)", optimized = FALSE)
  expect_plan_line(q, "reader filter: d > as.Date(\"2024-02-01\"), i < 5L", optimized = TRUE)
})

test_that("compute() drops the history", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "silent")
  q <- tbl_gpu(f10_df(), lazy = TRUE) |> mutate(z = pnorm(x)) |> filter(z > 0)
  out <- f10_lines(compute(q), optimized = TRUE)
  expect_false(any(grepl("CPU fallback", out, fixed = TRUE)))
  expect_identical(trimws(out[-1]), c("table on the GPU", "columns read: all 5"))
})

# --- tables with nothing pending, arguments ----------------------------------

test_that("an eager table and a lazy table with nothing pending", {
  skip_if_no_gpu()
  f10_snapshots_on()
  df <- f10_df()
  expect_snapshot({
    explain(tbl_gpu(df, lazy = FALSE))
    explain(tbl_gpu(df, lazy = TRUE))
    show_query(tbl_gpu(df, lazy = TRUE), optimized = FALSE)
  })
})

test_that("`optimized` must be TRUE or FALSE, and other arguments are refused", {
  skip_if_no_gpu()
  q <- tbl_gpu(f10_df(), lazy = TRUE) |> filter(x > 1)
  for (bad in list(NA, "yes", c(TRUE, FALSE), 1)) {
    expect_error(explain(q, optimized = bad), "`optimized` must be `TRUE` or `FALSE`.", fixed = TRUE)
    expect_error(show_query(q, optimized = bad), "`optimized` must be `TRUE` or `FALSE`.", fixed = TRUE)
  }
  expect_error(explain(q, FALSE), class = "rlib_error_dots_nonempty")
  expect_error(show_query(q, cte = TRUE), class = "rlib_error_dots_nonempty")
})
