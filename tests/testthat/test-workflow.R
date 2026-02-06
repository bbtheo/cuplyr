# Full workflow tests comparing to dplyr

test_that("workflow: filter -> mutate -> select -> arrange matches dplyr", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- df |>
    dplyr::filter(mpg > 15, cyl != 8) |>
    dplyr::mutate(kpl = mpg * 0.425, ratio = hp / wt) |>
    dplyr::select(mpg, kpl, ratio, cyl) |>
    dplyr::arrange(dplyr::desc(kpl), mpg)

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::filter(mpg > 15) |>
      dplyr::filter(cyl != 8) |>
      dplyr::mutate(kpl = mpg * 0.425) |>
      dplyr::mutate(ratio = hp / wt) |>
      dplyr::select(mpg, kpl, ratio, cyl) |>
      dplyr::arrange(dplyr::desc(kpl), mpg) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})

test_that("workflow: group_by -> summarise -> arrange matches dplyr", {
  skip_if_no_gpu()

  df <- mtcars
  expected <- df |>
    dplyr::group_by(cyl, gear) |>
    dplyr::summarise(avg_mpg = mean(mpg), total_hp = sum(hp), .groups = "drop") |>
    dplyr::arrange(cyl, dplyr::desc(avg_mpg))

  results <- with_exec_modes(df, function(tbl, mode) {
    tbl |>
      dplyr::group_by(cyl, gear) |>
      dplyr::summarise(avg_mpg = mean(mpg), total_hp = sum(hp)) |>
      dplyr::arrange(cyl, dplyr::desc(avg_mpg)) |>
      collect()
  })

  expect_equal(tibble::as_tibble(results$eager), tibble::as_tibble(expected))
  expect_equal(tibble::as_tibble(results$lazy), tibble::as_tibble(expected))
})
