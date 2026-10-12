# F32 (POSIXct time zones): paths beyond the acceptance file -- metadata
# carriers (names<-, vctrs, `$<-`, duplicated names), the UTC fast paths that
# must stay native, non-equi Date keys, window ops, and the zone helpers.

f32x_instants <- function(tz) {
  x <- as.POSIXct(c("2024-03-10 06:59:59", "2024-03-10 07:00:00",
                    "2024-11-03 05:30:00", "2024-11-03 06:30:00",
                    "2024-11-04 03:00:00", NA), tz = "UTC")
  attr(x, "tzone") <- tz
  x
}

f32x_zones <- function(x) {
  vapply(as.list(x), function(col) {
    z <- attr(col, "tzone")
    if (is.null(z)) "<none>" else z
  }, character(1))
}

test_that("tz_is_utc() resolves local time through the session zone", {
  expect_true(tz_is_utc("UTC"))
  expect_true(tz_is_utc("Etc/GMT+0"))
  expect_true(tz_is_utc("GMT"))
  expect_false(tz_is_utc("Etc/GMT+5"))
  expect_false(tz_is_utc("Europe/London"))
  expect_false(tz_is_utc(NULL))
  withr::local_timezone("UTC")
  expect_true(tz_is_utc(""))
  expect_true(tz_is_utc(NA_character_))
  withr::local_timezone("America/Chicago")
  expect_false(tz_is_utc(""))
  expect_false(tz_is_utc(NA_character_))
})

test_that("tz_fold() is vctrs' common tzone", {
  expect_identical(tz_fold("America/New_York", "UTC"), "America/New_York")
  expect_identical(tz_fold("", "UTC"), "UTC")
  expect_identical(tz_fold(NA_character_, "Asia/Kolkata"), "Asia/Kolkata")
  expect_identical(tz_fold(NULL, "UTC"), "UTC")
  expect_null(tz_fold(NULL, NULL))
  # the same rule as vctrs
  a <- f32x_instants("")
  b <- f32x_instants("Asia/Kolkata")
  expect_identical(attr(vctrs::vec_ptype2(a, b), "tzone"), "Asia/Kolkata")
})

test_that("ir_infer_tz() follows the operands", {
  schema <- list(
    names = c("a", "b", "d", "n"),
    types = c("TIMESTAMP_MICROSECONDS", "TIMESTAMP_MICROSECONDS", "TIMESTAMP_DAYS", "FLOAT64"),
    tz = c(a = "America/New_York", b = "")
  )
  col <- ir_col
  expect_identical(ir_infer_tz(col("a"), schema), "America/New_York")
  expect_identical(ir_infer_tz(col("n"), schema), "UTC")
  expect_identical(ir_infer_tz(ir_call("coalesce", list(col("b"), col("a"))), schema),
                   "America/New_York")
  expect_identical(ir_infer_tz(ir_call("coalesce", list(col("a"), col("b"))), schema),
                   "America/New_York")
  expect_identical(ir_infer_tz(ir_call("as_posixct", list(col("d"))), schema), "UTC")
  lit <- ir_lit_from_r(f32x_instants("Asia/Kolkata")[1])
  expect_identical(ir_infer_tz(lit, schema), "Asia/Kolkata")
})

test_that("zones follow names<-, vctrs slicing and column assignment", {
  skip_if_no_gpu()
  df <- data.frame(t = f32x_instants("America/New_York"), u = f32x_instants("UTC"), v = 1:6)
  gt <- tbl_gpu(df)
  renamed <- gt
  names(renamed) <- c("x", "y", "z")
  expect_identical(f32x_zones(collect(renamed))[1:2], c(x = "America/New_York", y = "UTC"))
  expect_identical(attr(collect(vctrs::vec_slice(gt, 1:2))$t, "tzone"), "America/New_York")

  gt$s <- f32x_instants("Asia/Kolkata")
  gt[["u"]] <- f32x_instants("Europe/Paris")
  expect_identical(
    f32x_zones(collect(gt)),
    c(t = "America/New_York", u = "Europe/Paris", v = "<none>", s = "Asia/Kolkata")
  )
})

test_that("zones of duplicated column names follow their position", {
  skip_if_no_gpu()
  df <- data.frame(t = f32x_instants("America/New_York"), u = f32x_instants("UTC"),
                   t2 = f32x_instants("Asia/Kolkata"), u2 = f32x_instants("Etc/GMT+5"))
  gt <- tbl_gpu(df)
  names(gt) <- c("a", "a", "b", "b")
  out <- collect(gt)
  expect_identical(unname(f32x_zones(out)),
                   c("America/New_York", "UTC", "Asia/Kolkata", "Etc/GMT+5"))
  out <- collect(head(gt, 2))
  expect_identical(unname(f32x_zones(out)),
                   c("America/New_York", "UTC", "Asia/Kolkata", "Etc/GMT+5"))
  out <- collect(dplyr::distinct(gt))
  expect_identical(unname(f32x_zones(out)),
                   c("America/New_York", "UTC", "Asia/Kolkata", "Etc/GMT+5"))
})

test_that("UTC and UTC-equivalent paths stay on the GPU", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "error")
  df <- data.frame(u = f32x_instants("UTC"), e = f32x_instants("Etc/UTC"), v = 1:6)
  gt <- tbl_gpu(df)
  expect_no_error(collect(dplyr::mutate(
    gt, h = lubridate::hour(u), h2 = lubridate::hour(e), D = as.Date(u),
    E = lubridate::as_date(u), later = dplyr::if_else(v > 2L, u, e)
  )))
  d <- data.frame(k = as.Date("2024-03-10"), a = 1L)
  y <- data.frame(k = as.POSIXct("2024-03-10", tz = "UTC"), b = 1L)
  expect_no_error(collect(dplyr::left_join(tbl_gpu(d), tbl_gpu(y), by = "k")))
  expect_no_error(collect(cuplyr::bind_rows(tbl_gpu(d), dplyr::rename(tbl_gpu(y), a = b))))
  # a non-UTC zone is a (notified) CPU fallback
  gt_ny <- tbl_gpu(data.frame(t = f32x_instants("America/New_York")))
  expect_error(collect(dplyr::mutate(gt_ny, h = lubridate::hour(t))), "fell back to CPU")
})

test_that("a tz-less column is native when the session zone is UTC", {
  skip_if_no_gpu()
  withr::local_timezone("UTC")
  withr::local_options(cuplyr.fallback = "error")
  gt <- tbl_gpu(data.frame(t = f32x_instants("")))
  expect_no_error(collect(dplyr::mutate(gt, h = lubridate::hour(t))))
})

test_that("non-equi Date x zoned POSIXct keys cast to local midnight", {
  skip_if_no_gpu()
  d <- data.frame(k = as.Date(c("2024-03-10", "2024-11-03")), a = 1:2)
  y <- data.frame(k = f32x_instants("America/New_York")[1:5], b = 1:5)
  expect_same_as_dplyr(d, function(x) {
    dplyr::inner_join(x, if (is_tbl_gpu(x)) tbl_gpu(y) else y, by = dplyr::join_by(k <= k))
  }, arrange_by = c("a", "b"))
})

test_that("a Date set-op with a zoned POSIXct goes to the CPU and keeps the zone", {
  skip_if_no_gpu()
  d <- data.frame(t = as.Date(c("2024-03-10", "2024-11-03")), a = 1:2)
  p <- data.frame(t = f32x_instants("America/New_York")[1:2], a = 7:8)
  expect_same_as_dplyr(d, function(x) {
    dplyr::union(x, if (is_tbl_gpu(x)) tbl_gpu(p) else p)
  }, arrange_by = c("t", "a"))
})

test_that("window ops over POSIXct keep the zone and feed the accessors", {
  skip_if_no_gpu()
  df <- data.frame(t = f32x_instants("America/New_York"), g = c(1, 1, 2, 2, 2, 2), v = 1:6)
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, h = lubridate::hour(dplyr::lag(t))),
                       ignore_col_types = TRUE)
  expect_same_as_dplyr(df, function(d) {
    d |> dplyr::group_by(g) |> dplyr::mutate(first_t = dplyr::first(t), last_t = dplyr::last(t))
  })
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, z = dplyr::if_else(v > 2L, t, NA)))
})

test_that("t + n works with a column or a double, and as.POSIXct() of a Date is UTC", {
  skip_if_no_gpu()
  df <- data.frame(t = f32x_instants("America/New_York"), v = 1:6, n = c(0.5, 1, 2, 3, 4, 5))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, a = t + v, b = t + n, c = t - n))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, D = as.Date(t), P = as.POSIXct(D)))
})

test_that("print() and glimpse() show NA dates and zoned times", {
  skip_if_no_gpu()
  gt <- tbl_gpu(data.frame(d = as.Date(c("2020-01-01", NA)),
                           t = f32x_instants("Asia/Kolkata")[c(1, 6)]))
  printed <- paste(utils::capture.output(print(gt)), collapse = "\n")
  expect_match(printed, "2020-01-01, NA", fixed = TRUE)
  expect_match(printed, "2024-03-10 12:29:59, NA", fixed = TRUE)
  glimpsed <- paste(utils::capture.output(dplyr::glimpse(gt)), collapse = "\n")
  expect_match(glimpsed, "2024-03-10 12:29:59", fixed = TRUE)
})

# ---- round 2 ----------------------------------------------------------------

test_that("a timestamp mix signals the timestamp CPU route; other mixes stay plain errors", {
  # F37: a timestamp next to any other value type goes to the CPU, so dplyr
  # combines it (Date + POSIXct) or raises vctrs' own error (Date + number)
  for (types in list(c("TIMESTAMP_DAYS", "TIMESTAMP_MICROSECONDS"),
                     c("TIMESTAMP_DAYS", "INT32"),
                     c("TIMESTAMP_MICROSECONDS", "FLOAT64"),
                     c("TIMESTAMP_MICROSECONDS", "STRING"))) {
    err <- tryCatch(ir_unify_types(types, "coalesce"), error = function(e) e)
    expect_s3_class(err, "cuplyr_timestamp_needs_cpu")
    expect_s3_class(err, "cuplyr_factor_needs_cpu")
  }
  err <- tryCatch(ir_unify_types(c("FLOAT64", "STRING"), "coalesce"), error = function(e) e)
  expect_s3_class(err, "error")
  expect_false(inherits(err, "cuplyr_factor_needs_cpu"))
})

test_that("ir_parse_quo() sends timestamp +/- numeric to the CPU, leaves the rest alone", {
  schema <- list(names = c("t", "t2", "d", "f", "n"),
                 types = c("TIMESTAMP_MICROSECONDS", "TIMESTAMP_MICROSECONDS",
                           "TIMESTAMP_DAYS", "DICTIONARY32", "FLOAT64"))
  expect_null(ir_parse_quo(rlang::quo(t + 3600), schema))
  expect_null(ir_parse_quo(rlang::quo(n - t), schema))
  # a Date shifted by a whole number of days is native (F37); other shifts
  # (a fractional literal, a double column) go to the CPU
  expect_identical(ir_parse_quo(rlang::quo(d + 1), schema)$op, "date_add_days")
  expect_null(ir_parse_quo(rlang::quo(d + 0.5), schema))
  expect_null(ir_parse_quo(rlang::quo(d + n), schema))
  # timestamp - timestamp is a difftime (F37): the CPU computes it, and the
  # re-upload raises F18's classed error unless the result drops it
  for (q in list(rlang::quo(t - t2), rlang::quo(d - d))) {
    expect_null(ir_parse_quo(q, schema))
  }
  # a factor expression is still returned as IR for ir_factor_rule()
  ir <- ir_parse_quo(rlang::quo(is.na(f - 1)), schema)
  expect_false(is.null(ir))
  expect_identical(ir_factor_rule(ir, schema), "cpu")
})

test_that("summarise() over a window op on a POSIXct column keeps the value's zone", {
  skip_if_no_gpu()
  for (zone in c("UTC", "America/New_York")) {
    df <- data.frame(t = f32x_instants(zone)[1:5], g = c(1, 1, 2, 2, 2))
    pipelines <- list(
      function(d) dplyr::summarise(d, m = max(dplyr::lag(t), na.rm = TRUE)),
      function(d) dplyr::summarise(d, m = dplyr::first(dplyr::lead(t))),
      function(d) dplyr::summarise(dplyr::group_by(d, g), m = max(dplyr::lag(t), na.rm = TRUE))
    )
    for (p in pipelines) {
      expect_same_as_dplyr(df, p, arrange_by = NULL)
      expect_same_as_dplyr_lazy(df, p, arrange_by = NULL)
    }
  }
})

test_that("intersect()/setdiff() keep the common zone through a type promotion", {
  skip_if_no_gpu()
  x <- data.frame(t = f32x_instants("America/New_York")[1:2], v = 1:2)
  y <- x
  y$v <- as.numeric(y$v)
  xl <- x
  attr(xl$t, "tzone") <- ""
  other <- function(d, tbl) if (is_tbl_gpu(d)) tbl_gpu(tbl) else tbl
  for (check in list(expect_same_as_dplyr, expect_same_as_dplyr_lazy)) {
    check(x, function(d) dplyr::intersect(d, other(d, y)), arrange_by = "v")
    check(x, function(d) dplyr::setdiff(d, other(d, y[1, ])), arrange_by = "v")
    check(xl, function(d) dplyr::intersect(d, other(d, x)), arrange_by = "v")
    check(x, function(d) dplyr::intersect(d, other(d, xl)), arrange_by = "v")
    check(xl, function(d) dplyr::setdiff(d, other(d, x[1, ])), arrange_by = "v")
    check(x, function(d) dplyr::semi_join(d, other(d, y), by = "v"), arrange_by = "v")
    check(x, function(d) dplyr::anti_join(d, other(d, y[1, ]), by = "v"), arrange_by = "v")
  }
})

test_that("an absent tzone becomes \"\" exactly where dplyr/vctrs make it so", {
  skip_if_no_gpu()
  t0 <- as.POSIXct(c("2024-01-01 10:00:00", "2024-01-02 11:00:00", "2024-01-03 12:00:00"),
                   tz = "UTC")
  attr(t0, "tzone") <- NULL
  df <- data.frame(t = t0, g = c("a", "b", "a"), v = 1:3)
  other <- function(d) if (is_tbl_gpu(d)) tbl_gpu(df) else df
  pipelines <- list(
    # keep the attribute absent
    select = function(d) dplyr::select(d, t, v),
    rename = function(d) dplyr::rename(d, u = t),
    relocate = function(d) dplyr::relocate(d, v),
    copy = function(d) dplyr::mutate(d, t2 = t),
    plus = function(d) dplyr::mutate(d, t2 = t + 1),
    minus = function(d) dplyr::mutate(d, t2 = t - 60),
    other_col = function(d) dplyr::mutate(d, w = v + 1L),
    # become ""
    filter = function(d) dplyr::filter(d, v > 1L),
    arrange = function(d) dplyr::arrange(d, dplyr::desc(v)),
    slice = function(d) dplyr::slice(d, 1:2),
    head = function(d) head(d, 2),
    distinct = function(d) dplyr::distinct(d, t, g),
    summarise_key = function(d) dplyr::summarise(dplyr::group_by(d, t), n = dplyr::n()),
    summarise_max = function(d) dplyr::summarise(d, m = max(t)),
    lag = function(d) dplyr::mutate(d, l = dplyr::lag(t)),
    if_else = function(d) dplyr::mutate(d, l = dplyr::if_else(v > 1L, t, t)),
    coalesce = function(d) dplyr::mutate(d, l = dplyr::coalesce(t, t)),
    window_max = function(d) dplyr::mutate(d, m = max(t)),
    left_join = function(d) dplyr::left_join(d, other(d), by = "g", relationship = "many-to-many"),
    join_key = function(d) dplyr::inner_join(d, other(d), by = "t"),
    bind_rows = function(d) {
      if (is_tbl_gpu(d)) cuplyr::bind_rows(d, other(d)) else dplyr::bind_rows(d, other(d))
    },
    union_all = function(d) dplyr::union_all(d, other(d)),
    intersect = function(d) dplyr::intersect(d, other(d)),
    semi_join = function(d) dplyr::semi_join(d, other(d), by = "v"),
    anti_join = function(d) dplyr::anti_join(d, dplyr::filter(other(d), v > 1L), by = "v")
  )
  for (nm in names(pipelines)) {
    p <- pipelines[[nm]]
    info <- paste("pipeline:", nm)
    withCallingHandlers(
      {
        expect_same_as_dplyr(df, p)
        expect_same_as_dplyr_lazy(df, p)
      },
      error = function(e) message(info)
    )
  }
})

# ---- round 3 ----------------------------------------------------------------
# Each POSIXct-valued op has an explicit zone rule (ir_tz_of()): ops whose
# result is `x` cast back to x's type take x's zone, vctrs-common ops fold
# their value operands in order, and an op without a rule goes to the CPU.

f32x_local <- function() {
  .POSIXct(c(1710054000, 1710057600, NA, 1704103200), tz = "")
}

test_that("x-typed ops take x's zone, not the other operand's (round-2 B1)", {
  skip_if_no_gpu()
  withr::local_timezone("Europe/Helsinki")
  df <- data.frame(t = f32x_local(), v = c(1, NA, 3, 4))
  attr(df$t, "tzone") <- ""
  absent <- df
  attr(absent$t, "tzone") <- NULL
  u0 <- .POSIXct(1710054000, tz = "UTC")
  ny0 <- .POSIXct(1710054000, tz = "America/New_York")
  ny1 <- .POSIXct(1710061200, tz = "America/New_York")
  pipelines <- list(
    na_if_utc = function(d) dplyr::mutate(d, u = dplyr::na_if(t, u0)),
    na_if_ny = function(d) dplyr::mutate(d, u = dplyr::na_if(t, ny0)),
    replace_when_ny = function(d) dplyr::mutate(d, u = dplyr::replace_when(t, is.na(v) ~ ny1)),
    replace_values_ny = function(d) dplyr::mutate(d, u = dplyr::replace_values(t, NA ~ ny1)),
    lag_default_ny = function(d) dplyr::mutate(d, u = dplyr::lag(t, default = ny1)),
    lead_ny_cond = function(d) dplyr::mutate(d, u = dplyr::lead(dplyr::na_if(t, ny0)))
  )
  for (nm in names(pipelines)) {
    for (input in list(df, absent)) {
      withCallingHandlers({
        expect_same_as_dplyr(input, pipelines[[nm]])
        expect_same_as_dplyr_lazy(input, pipelines[[nm]])
      }, error = function(e) message("pipeline: ", nm))
    }
  }
})

test_that("an accessor over an x-typed op on a local column uses the local wall clock", {
  skip_if_no_gpu()
  withr::local_timezone("Europe/Helsinki")
  df <- data.frame(t = .POSIXct(c(1704103200, 1704106800, 0), tz = ""), v = 1:3)
  u0 <- .POSIXct(0, tz = "UTC")
  pipelines <- list(
    function(d) dplyr::mutate(d, h = lubridate::hour(dplyr::na_if(t, u0))),
    function(d) dplyr::mutate(d, h = lubridate::hour(dplyr::replace_when(t, v > 2L ~ u0))),
    function(d) dplyr::mutate(d, h = lubridate::hour(dplyr::lag(t, default = u0))),
    function(d) dplyr::mutate(d, h = lubridate::hour(dplyr::coalesce(dplyr::na_if(t, u0), t)))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
  # and it is not computed natively (the native hour() is the UTC hour)
  withr::local_options(cuplyr.fallback = "error")
  expect_error(collect(pipelines[[1]](tbl_gpu(df))), "fell back to CPU")
  expect_error(collect(pipelines[[1]](as_lazy(tbl_gpu(df)))), "fell back to CPU")
})

test_that("vctrs-common ops fold the value operands in order", {
  skip_if_no_gpu()
  withr::local_timezone("Europe/Helsinki")
  df <- data.frame(l = f32x_local(), v = c(1, NA, 3, 4))
  df$u <- df$l
  attr(df$u, "tzone") <- "UTC"
  df$n <- df$l
  attr(df$n, "tzone") <- "America/New_York"
  pipelines <- list(
    function(d) dplyr::mutate(d, o = dplyr::if_else(v > 1, l, n)),
    function(d) dplyr::mutate(d, o = dplyr::if_else(v > 1, l, l, missing = u)),
    function(d) dplyr::mutate(d, o = dplyr::case_when(v > 1 ~ u, .default = n)),
    function(d) dplyr::mutate(d, o = dplyr::case_when(v > 1 ~ l, v > 3 ~ n, .default = u)),
    function(d) dplyr::mutate(d, o = dplyr::coalesce(l, n, u)),
    function(d) dplyr::mutate(d, o = dplyr::coalesce(dplyr::na_if(l, u), u))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
})

test_that("POSIXct ops dplyr rejects (unary -, abs, cummax, cummin) go to the CPU", {
  skip_if_no_gpu()
  df <- data.frame(t = f32x_instants("America/New_York"), v = 1:6)
  for (e in c("-t", "abs(t)", "cummax(t)", "cummin(t)", "lubridate::hour(cummax(t))")) {
    ex <- rlang::parse_expr(e)
    expect_error(dplyr::mutate(df, o = !!ex), info = e)
    withr::with_options(list(cuplyr.fallback = "error"), {
      expect_error(collect(dplyr::mutate(tbl_gpu(df), o = !!ex)), "fell back to CPU", info = e)
    })
  }
})

test_that("a Date mixed into an x-typed op on a POSIXct goes to the CPU", {
  skip_if_no_gpu()
  df <- data.frame(t = .POSIXct(c(1710043200, 1710057600, NA), tz = "America/New_York"),
                   v = c(1, NA, 3))
  d0 <- as.Date("2024-03-10")
  pipelines <- list(
    function(d) dplyr::mutate(d, o = dplyr::na_if(t, d0)),
    function(d) dplyr::mutate(d, o = dplyr::replace_when(t, is.na(v) ~ d0)),
    function(d) dplyr::mutate(d, o = dplyr::lag(t, default = d0)),
    function(d) dplyr::mutate(d, o = dplyr::lead(t, default = d0))
  )
  for (p in pipelines) {
    expect_same_as_dplyr(df, p)
    expect_same_as_dplyr_lazy(df, p)
  }
  # a Date x with a POSIXct replacement is a vctrs error in dplyr, not garbage
  dd <- data.frame(d = as.Date(c("2024-03-10", "2024-03-11", NA)), v = c(1, NA, 3))
  p0 <- .POSIXct(1710043200, tz = "America/New_York")
  expect_error(collect(dplyr::mutate(tbl_gpu(dd), o = dplyr::replace_when(d, is.na(v) ~ p0))))
  expect_error(collect(dplyr::mutate(tbl_gpu(dd), o = dplyr::na_if(d, p0))))
})

test_that("every registry op with a POSIXct result has a zone rule", {
  withr::local_timezone("Europe/Helsinki")
  schema <- list(names = c("l", "n", "b"),
                 types = c("TIMESTAMP_MICROSECONDS", "TIMESTAMP_MICROSECONDS", "BOOL8"),
                 tz = c(l = "", n = "America/New_York"))
  x_typed <- c("na_if", "replace_when", "lag", "lead", "first", "last", "nth",
               "min", "max", "roll_min", "roll_max")
  common <- c("if_else", "case_when", "coalesce")
  for (op in names(ir_call_registry)) {
    for (k in 1:3) {
      # x local, every other operand New York; a leading BOOL8 condition for
      # the ops that take one
      args <- c(list(ir_col("l")), rep(list(ir_col("n")), k - 1L))
      if (op %in% c("if_else", "case_when") && k >= 2L) {
        args <- c(list(ir_col("b")), args)
      }
      ir <- ir_call(op, args)
      type <- tryCatch(unname(ir_infer_type(ir, schema)), error = function(e) NULL)
      if (!identical(type, "TIMESTAMP_MICROSECONDS")) next
      zone <- tryCatch(ir_tz_of(ir, schema), cuplyr_timestamp_needs_cpu = function(e) "<cpu>")
      info <- paste0(op, "/", k)
      if (op %in% x_typed) {
        expect_identical(zone, "", info = info)
      } else if (op %in% common) {
        expect_identical(zone, if (k == 1L) "" else "America/New_York", info = info)
      } else if (op == "as_posixct") {
        expect_identical(zone, "", info = info)
      } else {
        # no rule: never claims a zone, the expression goes to the CPU
        expect_identical(zone, "<cpu>", info = info)
      }
      # never UTC with a local x and a non-UTC other operand
      expect_false(identical(zone, "UTC"), info = info)
    }
  }
})

test_that("a zone-only difference in bind_rows() is metadata, no column copy", {
  skip_if_no_gpu()
  a <- tbl_gpu(data.frame(t = f32x_instants("America/New_York"), v = 1:6))
  b <- tbl_gpu(data.frame(t = f32x_instants("UTC"), v = 1:6))
  target <- a$schema
  aligned <- align_to_schema(b, target)
  expect_identical(aligned$schema$tz, target$tz)
  expect_true(identical(aligned$ptr, b$ptr))  # the same device table
  expect_same_as_dplyr(data.frame(t = f32x_instants("America/New_York"), v = 1:6), function(d) {
    other <- data.frame(t = f32x_instants("UTC"), v = 1:6)
    if (is_tbl_gpu(d)) cuplyr::bind_rows(d, tbl_gpu(other)) else dplyr::bind_rows(d, other)
  })
})
