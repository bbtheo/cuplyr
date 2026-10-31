# F91: print() and glimpse() show R types, factor labels and a lazy hint.
#
# Contract (scratchpad/parity/features/F91.md):
# - The `<type>` label of every column, in print() and glimpse(), eager and
#   lazy, is `vctrs::vec_ptype_abbr()` of that column of `collect(x)`
#   (fct, ord, date, dttm, ...), derived from the schema without collecting.
# - Factor previews show level labels, not codes; a missing factor or string
#   value shows as a bare `NA` (a real string "NA" stays quoted).
# - A lazy print with pending operations ends with one footer line pointing
#   to collect() for the data and explain() for the plan; so do the
#   plan-failed and not-materialized views, which also list `name <type>`.
#   An eager print, or a lazy table with nothing pending, has no footer.
# - Eager print does no GPU work beyond its gpu_head() preview.

f91_df <- function() {
  tibble::tibble(
    d = c(1.5, NA, 3),
    i = c(1L, NA, 3L),
    s = c("a", NA, "NA"),
    l = c(TRUE, NA, FALSE),
    dt = as.Date(c("2024-01-01", NA, "2024-03-01")),
    ts = as.POSIXct(c("2024-01-01 10:00:00", NA, "2024-01-02 00:00:00"),
                    tz = "America/New_York"),
    # level order differs from appearance: codes (2, NA, 1) != labels
    f = factor(c("x", NA, "y"), levels = c("y", "x")),
    o = factor(c("lo", "hi", NA), levels = c("lo", "hi"), ordered = TRUE)
  )
}

f91_print <- function(x, ...) utils::capture.output(print(x, ...))
f91_glimpse <- function(x, ...) utils::capture.output(dplyr::glimpse(x, ...))

# The "$ name <type> values" body lines as a data frame (name, type, vals).
f91_body <- function(out) {
  body <- out[grepl("^\\$ ", out)]
  m <- regmatches(body, regexec("^\\$ (\\S+)\\s+<([^>]+)> ?(.*)$", body))
  data.frame(
    name = vapply(m, `[`, "", 2L),
    type = vapply(m, `[`, "", 3L),
    vals = trimws(vapply(m, `[`, "", 4L)),
    stringsAsFactors = FALSE
  )
}

# What pillar calls each collected column.
f91_abbr <- function(tbl) {
  vapply(tibble::as_tibble(collect(tbl)), vctrs::vec_ptype_abbr, character(1))
}

f91_footer_re <- "^# .*collect\\(\\).*explain\\(\\)"

f91_expect_labels <- function(x, label) {
  want <- f91_abbr(x)
  for (fun in c("print", "glimpse")) {
    out <- if (fun == "print") f91_print(x) else f91_glimpse(x)
    body <- f91_body(out)
    expect_identical(body$name, names(want), info = paste(label, fun))
    expect_identical(body$type, unname(want), info = paste(label, fun))
  }
}

test_that("eager print() and glimpse() type labels are pillar's labels of the collected columns", {
  skip_if_no_gpu()
  x <- tbl_gpu(f91_df(), lazy = FALSE)
  f91_expect_labels(x, "eager")
  body <- f91_body(f91_print(x))
  expect_identical(body$type[body$name == "f"], "fct")
  expect_identical(body$type[body$name == "o"], "ord")
  expect_identical(body$type[body$name == "ts"], "dttm")
  expect_identical(body$type[body$name == "dt"], "date")
})

test_that("lazy print() and glimpse() type labels match the collected columns", {
  skip_if_no_gpu()
  df <- f91_df()
  # nothing pending
  f91_expect_labels(tbl_gpu(df, lazy = TRUE), "lazy source")
  # pending operations
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(!is.na(d) | is.na(d)) |>
    dplyr::mutate(z = d + 1, f2 = f, c2 = as.character(f))
  expect_true(has_pending_ops(lz))
  f91_expect_labels(lz, "lazy pending")
})

test_that("labels follow factors through verbs (summarise, count, select, join), eager and lazy", {
  skip_if_no_gpu()
  df <- f91_df()
  keys <- tibble::tibble(f = factor(c("x", "y"), levels = c("y", "x")), k = c(10L, 20L))
  pipelines <- list(
    summarise = function(t) t |> dplyr::group_by(f) |>
      dplyr::summarise(n = dplyr::n(), m = mean(d, na.rm = TRUE), fo = dplyr::first(o)),
    count = function(t) dplyr::count(t, o),
    select = function(t) dplyr::select(t, o, f, s),
    as_int = function(t) dplyr::mutate(t, fi = as.integer(f)),
    arrange = function(t) dplyr::arrange(t, f)
  )
  for (nm in names(pipelines)) {
    f91_expect_labels(pipelines[[nm]](tbl_gpu(df, lazy = FALSE)), paste(nm, "eager"))
    f91_expect_labels(pipelines[[nm]](tbl_gpu(df, lazy = TRUE)), paste(nm, "lazy"))
  }
  for (lazy in c(FALSE, TRUE)) {
    j <- dplyr::left_join(tbl_gpu(df, lazy = lazy), tbl_gpu(keys, lazy = lazy), by = "f")
    f91_expect_labels(j, paste("join lazy =", lazy))
  }
})

test_that("factor previews show labels, and missing factors and strings show a bare NA", {
  skip_if_no_gpu()
  df <- f91_df()
  ref <- f91_body(utils::capture.output(dplyr::glimpse(df)))
  expect_identical(ref$vals[ref$name == "f"], "x, NA, y")
  expect_identical(ref$vals[ref$name == "s"], "\"a\", NA, \"NA\"")
  views <- list(
    eager = tbl_gpu(df, lazy = FALSE),
    lazy_source = tbl_gpu(df, lazy = TRUE),
    lazy_pending = dplyr::mutate(tbl_gpu(df, lazy = TRUE), z = d)
  )
  for (nm in names(views)) {
    for (fun in c("print", "glimpse")) {
      out <- if (fun == "print") f91_print(views[[nm]]) else f91_glimpse(views[[nm]])
      body <- f91_body(out)
      for (col in c("f", "o", "s")) {
        expect_identical(body$vals[body$name == col], ref$vals[ref$name == col],
                         info = paste(nm, fun, col))
      }
      expect_false(any(grepl("\"NA\", ", body$vals[body$name == "s"], fixed = TRUE)),
                   info = paste(nm, fun))
    }
  }
})

test_that("an explicit NA level, an all-NA factor and a factor key after summarise show labels", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    a = addNA(factor(c("p", NA, "q"))),
    b = factor(c(NA, NA, NA), levels = c("u", "v")),
    s = c(NA_character_, NA_character_, NA_character_)
  )
  ref <- f91_body(utils::capture.output(dplyr::glimpse(df)))
  for (lazy in c(FALSE, TRUE)) {
    x <- tbl_gpu(df, lazy = lazy)
    for (fun in c("print", "glimpse")) {
      body <- f91_body(if (fun == "print") f91_print(x) else f91_glimpse(x))
      expect_identical(body$type, ref$type, info = paste(lazy, fun))
      expect_identical(body$vals, ref$vals, info = paste(lazy, fun))
    }
  }
  # summarised factor keys keep their labels
  sm <- tbl_gpu(f91_df(), lazy = FALSE) |>
    dplyr::group_by(o) |>
    dplyr::summarise(n = dplyr::n())
  body <- f91_body(f91_print(sm))
  expect_identical(body$type[body$name == "o"], "ord")
  expect_identical(body$vals[body$name == "o"], "lo, hi, NA")
})

test_that("integer64 columns are labelled like their collected column", {
  skip_if_no_gpu()
  skip_if_not_installed("bit64")
  df <- tibble::tibble(k = bit64::as.integer64(c(1, 2, 3)), x = c(1, 2, 3))
  x <- suppressWarnings(tbl_gpu(df, lazy = FALSE))
  f91_expect_labels(x, "integer64")
})

test_that("a lazy print with pending operations ends with one collect()/explain() footer", {
  skip_if_no_gpu()
  df <- f91_df()
  pipelines <- list(
    mutate = function(t) dplyr::mutate(t, z = d + 1),
    grouped = function(t) t |> dplyr::mutate(z = d + 1) |> dplyr::group_by(f),
    rowwise = function(t) t |> dplyr::filter(!is.na(i)) |> dplyr::rowwise(i),
    zero_rows = function(t) dplyr::filter(t, d > 100),
    summarise = function(t) t |> dplyr::group_by(o) |> dplyr::summarise(n = dplyr::n())
  )
  for (nm in names(pipelines)) {
    lz <- pipelines[[nm]](tbl_gpu(df, lazy = TRUE))
    expect_true(has_pending_ops(lz), info = nm)
    out <- f91_print(lz)
    expect_identical(sum(grepl(f91_footer_re, out)), 1L, info = nm)
    expect_match(out[length(out)], f91_footer_re, info = nm)
    # eager: the same print without the footer and without "[lazy]"
    eg <- f91_print(pipelines[[nm]](tbl_gpu(df, lazy = FALSE)))
    expect_false(any(grepl("explain()", eg, fixed = TRUE)), info = nm)
    expected <- eg
    expected[1] <- sub("^# A gibble: ", "# A gibble [lazy]: ", expected[1])
    expect_identical(out[-length(out)], expected, info = nm)
  }
  # nothing pending: no footer
  expect_false(any(grepl("explain()", f91_print(tbl_gpu(df, lazy = TRUE)), fixed = TRUE)))
  # glimpse keeps its Lazy: line and gets no footer
  g <- f91_glimpse(dplyr::mutate(tbl_gpu(df, lazy = TRUE), z = d + 1))
  expect_identical(sum(grepl("^Lazy: ", g)), 1L)
  expect_false(any(grepl(f91_footer_re, g)))
})

test_that("grouped and rowwise headers are unchanged", {
  skip_if_no_gpu()
  df <- f91_df()
  out <- f91_print(dplyr::group_by(tbl_gpu(df, lazy = FALSE), f))
  expect_match(out[1], "^# A gibble: 3 . 8$")
  expect_match(out[2], "^# Groups:\\s+f \\[3\\]$")
  expect_match(out[3], "^\\$ ")
  out <- f91_print(dplyr::rowwise(tbl_gpu(df, lazy = FALSE), i))
  expect_match(out[1], "^# A gibble: 3 . 8$")
  expect_match(out[2], "^# Rowwise:\\s+i$")
  expect_match(out[3], "^\\$ ")
  out <- f91_print(dplyr::group_by(dplyr::mutate(tbl_gpu(df, lazy = TRUE), z = d), o))
  expect_match(out[1], "^# A gibble \\[lazy\\]: 3 . 9$")
  expect_match(out[2], "^# Groups:\\s+o \\[3\\]$")
})

test_that("a failed plan lists every column with its type and ends with the footer", {
  skip_if_no_gpu()
  df <- f91_df()
  lz <- tbl_gpu(df, lazy = TRUE) |>
    dplyr::filter(!is.na(d) | is.na(d)) |>
    dplyr::mutate(z = d + 1L, fi = as.integer(f))
  want <- f91_abbr(lz)
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("f91: plan not run"),
    .package = "cuplyr"
  )
  out <- NULL
  expect_no_error(out <- f91_print(lz))
  expect_match(out[1], "^# A gibble \\[lazy\\]: \\?\\? . 10$")
  expect_true(any(grepl("f91: plan not run", out, fixed = TRUE)))
  txt <- paste(out, collapse = "\n")
  for (nm in names(want)) {
    expect_true(grepl(paste0("\\b", nm, " <", want[[nm]], ">"), txt), info = nm)
  }
  expect_true(all(startsWith(out, "#")))
  expect_match(out[length(out)], f91_footer_re)
  expect_identical(sum(grepl(f91_footer_re, out)), 1L)

  g <- f91_glimpse(lz)
  body <- f91_body(g)
  expect_identical(body$name, names(want))
  expect_identical(body$type, unname(want))
  expect_true(any(grepl("f91: plan not run", g, fixed = TRUE)))
})

test_that("a long failed-plan schema wraps into # lines at the print width", {
  skip_if_no_gpu()
  wide <- tibble::as_tibble(stats::setNames(
    as.list(as.double(1:30)), paste0("column_", 1:30)
  ))
  lz <- dplyr::mutate(tbl_gpu(wide, lazy = TRUE), extra = column_1 + 1)
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("f91: plan not run"),
    .package = "cuplyr"
  )
  out <- f91_print(lz, width = 60)
  expect_true(all(startsWith(out, "#")))
  # the schema lines wrap; the one-line footer is exempt
  expect_true(all(nchar(out[!grepl(f91_footer_re, out)]) <= 60))
  txt <- paste(out, collapse = " ")
  for (nm in c(paste0("column_", 1:30), "extra")) {
    expect_true(grepl(paste0("\\b", nm, " <dbl>"), txt), info = nm)
  }
})

test_that("a lazy table with no data and no plan lists name <type> and the footer", {
  skip_if_no_gpu()
  df <- f91_df()
  x <- tbl_gpu(df, lazy = TRUE)
  y <- unclass(x)
  y$ptr <- NULL
  class(y) <- class(x)
  out <- f91_print(y)
  expect_match(out[1], "not materialized")
  txt <- paste(out, collapse = "\n")
  want <- vapply(df, vctrs::vec_ptype_abbr, character(1))
  for (nm in names(want)) {
    expect_true(grepl(paste0("\\b", nm, " <", want[[nm]], ">"), txt), info = nm)
  }
  expect_match(out[length(out)], f91_footer_re)
})

test_that("eager print does no GPU work beyond its gpu_head() preview", {
  skip_if_no_gpu()
  x <- tbl_gpu(f91_df(), lazy = FALSE)
  heads <- 0L
  real_head <- cuplyr:::gpu_head
  testthat::local_mocked_bindings(
    gpu_head = function(...) {
      heads <<- heads + 1L
      real_head(...)
    },
    lower_and_execute = function(...) stop("f91: eager print ran a plan"),
    gpu_collect = function(...) stop("f91: eager print collected"),
    .package = "cuplyr"
  )
  out <- NULL
  expect_no_error(out <- f91_print(x))
  expect_identical(heads, 1L)
  expect_identical(f91_body(out)$type[7:8], c("fct", "ord"))
})

# Round-1 review (B1, B2, N1, N2): pillar's glimpse quotes every factor label
# when a level contains a comma, escapes quoted strings with encodeString(),
# labels a vctrs::unspecified() column <???> (what collect() returns), and a
# NaN is not a missing value.
test_that("comma levels are quoted, strings escaped and unspecified columns labelled like pillar", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("A", "B, C", NA)),
    s = c("say \"hi\"", "a\tb", NA),
    u = vctrs::unspecified(3),
    y = c(1L, 2L, 3L)
  )
  ref <- f91_body(utils::capture.output(dplyr::glimpse(df)))
  expect_identical(ref$vals[ref$name == "f"], "\"A\", \"B, C\", NA")
  expect_identical(ref$type[ref$name == "u"], "???")
  views <- list(
    eager = tbl_gpu(df, lazy = FALSE),
    lazy_source = tbl_gpu(df, lazy = TRUE),
    lazy_pending = dplyr::filter(tbl_gpu(df, lazy = TRUE), y > 0L)
  )
  for (nm in names(views)) {
    want <- f91_abbr(views[[nm]])
    expect_identical(unname(want[["u"]]), "???", info = nm)
    for (fun in c("print", "glimpse")) {
      out <- if (fun == "print") f91_print(views[[nm]]) else f91_glimpse(views[[nm]])
      body <- f91_body(out)
      expect_identical(body$type, unname(want), info = paste(nm, fun))
      for (col in c("f", "s", "u")) {
        expect_identical(body$vals[body$name == col], ref$vals[ref$name == col],
                         info = paste(nm, fun, col))
      }
    }
  }
  # a failed plan labels the unspecified column like collect() would
  lz <- views$lazy_pending
  testthat::local_mocked_bindings(
    lower_and_execute = function(...) stop("f91: plan not run"),
    .package = "cuplyr"
  )
  txt <- paste(f91_print(lz), collapse = "\n")
  expect_true(grepl("\\bu <\\?\\?\\?>", txt))
  body <- f91_body(f91_glimpse(lz))
  expect_identical(body$type[body$name == "u"], "???")
})

test_that("a NaN previews as NaN, a missing double as NA", {
  skip_if_no_gpu()
  df <- tibble::tibble(n = c(NaN, 1, NA))
  for (lazy in c(FALSE, TRUE)) {
    x <- dplyr::mutate(tbl_gpu(df, lazy = lazy), m = n)
    for (fun in c("print", "glimpse")) {
      body <- f91_body(if (fun == "print") f91_print(x) else f91_glimpse(x))
      expect_identical(body$vals[body$name == "m"], "NaN, 1, NA", info = paste(lazy, fun))
    }
  }
})

# Round-2 review N2: pillar decides factor quoting from levels(x), not from
# the shown values; a plain all-NA logical column is <lgl>, not <???>.
test_that("an unused comma level still quotes, and an all-NA logical stays <lgl>", {
  skip_if_no_gpu()
  df <- tibble::tibble(
    f = factor(c("a", NA, "b"), levels = c("a", "b", "x, y")),
    l = c(NA, NA, NA)
  )
  ref <- f91_body(utils::capture.output(dplyr::glimpse(df)))
  expect_identical(ref$vals[ref$name == "f"], "\"a\", NA, \"b\"")
  for (lazy in c(FALSE, TRUE)) {
    x <- dplyr::filter(tbl_gpu(df, lazy = lazy), is.na(l))
    want <- f91_abbr(x)
    expect_identical(unname(want[["l"]]), "lgl")
    for (fun in c("print", "glimpse")) {
      body <- f91_body(if (fun == "print") f91_print(x) else f91_glimpse(x))
      expect_identical(body$type, unname(want), info = paste(lazy, fun))
      expect_identical(body$vals, ref$vals, info = paste(lazy, fun))
    }
  }
})
