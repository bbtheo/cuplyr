# count_groups() (print header count, n_groups()) must agree with
# nrow(compute_group_data()) and with dplyr for tables carrying fixed group
# keys, including a `.drop = FALSE` factor key, where the fixed keys win over
# the unused-level expansion. And print_width() follows pillar's option order.

f51_cg_df <- function() {
  tibble::tibble(
    f = factor(c("a", "a", "b", NA, "b"), levels = c("a", "b", "c")),
    id = c(1, 2, 2, NA, 3),
    v = c(0.1, 0.9, 0.2, 0.7, 0.8)
  )
}

f51_cg_check <- function(gpu, ref) {
  expect_identical(count_groups(gpu), nrow(compute_group_data(gpu)))
  expect_identical(dplyr::n_groups(gpu), dplyr::n_groups(ref))
  expect_identical(nrow(dplyr::group_data(gpu)), dplyr::n_groups(ref))
}

test_that("count_groups() matches group_data() and dplyr with fixed keys", {
  skip_if_no_gpu()
  df <- f51_cg_df()
  g <- tbl_gpu(df)
  cases <- list(
    # .drop = FALSE factor key, then filter(): fixed keys, empty levels kept
    list(
      dplyr::filter(dplyr::group_by(g, f, .drop = FALSE), v > 0.5),
      dplyr::filter(dplyr::group_by(df, f, .drop = FALSE), v > 0.5)
    ),
    # the same with every row filtered out
    list(
      dplyr::filter(dplyr::group_by(g, f, .drop = FALSE), v > 5),
      dplyr::filter(dplyr::group_by(df, f, .drop = FALSE), v > 5)
    ),
    # factor and double keys, .drop = FALSE, filter()
    list(
      dplyr::filter(dplyr::group_by(g, f, id, .drop = FALSE), v > 0.5),
      dplyr::filter(dplyr::group_by(df, f, id, .drop = FALSE), v > 0.5)
    ),
    # slice(.preserve = TRUE) on a .drop = FALSE factor key
    list(
      dplyr::slice(dplyr::group_by(g, f, .drop = FALSE), 2, .preserve = TRUE),
      dplyr::slice(dplyr::group_by(df, f, .drop = FALSE), 2, .preserve = TRUE)
    ),
    # filter(.preserve = TRUE) on a factor key with .drop = TRUE
    list(
      dplyr::filter(dplyr::group_by(g, f), v > 0.5, .preserve = TRUE),
      dplyr::filter(dplyr::group_by(df, f), v > 0.5, .preserve = TRUE)
    ),
    # .drop = FALSE factor key without fixed keys: still the expanded count
    list(dplyr::group_by(g, f, .drop = FALSE), dplyr::group_by(df, f, .drop = FALSE))
  )
  for (case in cases) {
    f51_cg_check(case[[1]], case[[2]])
  }
})

test_that("the fixed-key count is read without a GPU groupby", {
  skip_if_no_gpu()
  df <- f51_cg_df()
  fp <- dplyr::filter(dplyr::group_by(tbl_gpu(df), id), v > 0.5, .preserve = TRUE)
  want <- dplyr::n_groups(dplyr::filter(dplyr::group_by(df, id), v > 0.5, .preserve = TRUE))
  testthat::local_mocked_bindings(
    gpu_group_count = function(...) stop("counted on the GPU"),
    gpu_group_info = function(...) stop("built group info on the GPU"),
    .package = "cuplyr"
  )
  expect_identical(dplyr::n_groups(fp), want)
})

test_that("print_width() follows pillar's option order", {
  withr::local_options(
    pillar.width = NULL, tibble.width = NULL, dplyr.width = NULL, width = 70
  )
  expect_identical(print_width(), 70L)
  withr::local_options(dplyr.width = 60)
  expect_identical(print_width(), 60)
  withr::local_options(tibble.width = 50)
  expect_identical(print_width(), 50)
  withr::local_options(pillar.width = 40)
  expect_identical(print_width(), 40)
  expect_identical(print_width(90), 90)
})

test_that("body_width() follows glimpse's rule: an infinite option falls back to the console", {
  withr::local_options(
    pillar.width = NULL, tibble.width = NULL, dplyr.width = NULL, width = 70
  )
  expect_identical(body_width(), 70L)
  withr::local_options(tibble.width = Inf)
  expect_identical(body_width(), 70L)
  withr::local_options(tibble.width = 50)
  expect_identical(body_width(), 50)
  # pillar.width wins, and an infinite pillar.width is not overridden by a
  # finite tibble.width (pillar's get_width_glimpse())
  withr::local_options(pillar.width = Inf)
  expect_identical(body_width(), 70L)
  withr::local_options(pillar.width = 40)
  expect_identical(body_width(), 40)
  # an explicit width is taken as given
  expect_identical(body_width(90), 90)
  expect_identical(body_width(Inf), Inf)
})

test_that("an infinite width option prints the header like tibble and keeps the body in the console", {
  skip_if_no_gpu()
  df <- tibble::tibble(customer_identifier = 1, transaction_category = strrep("x", 15))
  df <- df[rep(1, 10), ]
  gg <- dplyr::group_by(tbl_gpu(df), customer_identifier, transaction_category)
  tg <- dplyr::group_by(df, customer_identifier, transaction_category)
  for (opt in c("pillar.width", "tibble.width")) {
    withr::with_options(stats::setNames(list(80, Inf), c("width", opt)), {
      out <- utils::capture.output(print(gg))
      want <- utils::capture.output(print(tg))
      k <- match(FALSE, startsWith(want, "#")) - 1L
      expect_identical(out[seq_len(k)], sub("^# A tibble:", "# A gibble:", want[seq_len(k)]))
      body <- out[startsWith(out, "$ ")]
      expect_length(body, 2L)
      expect_true(all(nchar(body) <= 80L), info = opt)
      expect_true(endsWith(body[2], "..."), info = opt)
    })
  }
  # a finite option narrows the body like glimpse() (50, not 40: with these
  # 20-character names a 40 budget leaves under 10 characters for values,
  # where the body never truncates)
  withr::with_options(list(width = 80, pillar.width = 50), {
    body <- utils::capture.output(print(gg))
    body <- body[startsWith(body, "$ ")]
    expect_true(all(nchar(body) <= 50L))
    expect_true(endsWith(body[2], "..."))
    want <- utils::capture.output(dplyr::glimpse(tg))
    want <- want[startsWith(want, "$ ")]
    expect_true(all(nchar(want) <= 50L))
  })
})
