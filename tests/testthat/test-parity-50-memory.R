# #50: summarise(.by =) memory and speed parity with
# group_by() |> summarise(.groups = "drop"), and a cheap ungrouped row_number().
#
# Memory is measured with gpu_alloc_tracking() (F56): an RMM statistics adaptor
# counts every byte allocated while the call runs (`allocated_bytes`) and the
# most held at once (`peak_bytes`), excluding the resident input. These counts
# are exact and do not depend on other processes, so the bounds are tight.
#
# Before #50 the `.by` form added a whole-table `row_number()` column through
# the window machinery: its peak was the group_by form's plus one INT32 column,
# but it allocated about six INT32 columns more in total, and ran four verb
# calls (2.6x the group_by form's time at 1e3-1e6 rows). Ordering the groups
# by first appearance needs at most one transient INT32 position per row, so:
#   peak(.by)      <= peak(group_by)      + 1.1 x 4 bytes/row
#   allocated(.by) <= allocated(group_by) + 1.5 x 4 bytes/row
# and an ungrouped bare row_number() allocates about its own INT32 column.

mem50_n <- 2e6
mem50_slack <- 1024^2 # bytes, for the k-row results and cudf's small scratch

mem50_df <- function(n) {
  set.seed(5001)
  data.frame(
    k = sample.int(5000L, n, replace = TRUE),
    s = sprintf("g%02d", sample.int(20L, n, replace = TRUE)),
    x = runif(n),
    y = runif(n),
    stringsAsFactors = FALSE
  )
}

mem50_measure <- function(gt, pipeline) {
  run <- function() {
    res <- pipeline(gt)
    if (identical(gt$exec_mode, "lazy")) res <- compute(res)
    gpu_sync()
    res
  }
  invisible(run()) # first-use state (module loading, cudf caches) is not counted
  gc(verbose = FALSE)
  gpu_gc(verbose = FALSE)
  tr <- gpu_alloc_tracking(run())
  list(peak = tr$peak_bytes, allocated = tr$allocated_bytes)
}

test_that("summarise(.by) needs no more memory than group_by() |> summarise() plus one INT32 per row", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(512 * 1024^2)

  n <- mem50_n
  df <- mem50_df(n)
  per_row <- 4 * n
  cases <- list(
    int_key = list(
      gb = function(d) d |> dplyr::group_by(k) |>
        dplyr::summarise(m = mean(x), s = sum(y), n = dplyr::n(), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(m = mean(x), s = sum(y), n = dplyr::n(), .by = k)
    ),
    string_key = list(
      gb = function(d) d |> dplyr::group_by(s) |>
        dplyr::summarise(m = mean(x), n = dplyr::n(), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(m = mean(x), n = dplyr::n(), .by = s)
    ),
    two_keys = list(
      gb = function(d) d |> dplyr::group_by(s, k) |>
        dplyr::summarise(s_y = sum(y), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(s_y = sum(y), .by = c(s, k))
    )
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    for (nm in names(cases)) {
      label <- sprintf("%s (%s)", nm, if (lazy) "lazy" else "eager")
      gb <- mem50_measure(gt, cases[[nm]]$gb)
      by <- mem50_measure(gt, cases[[nm]]$by)
      expect_lte(
        by$peak, gb$peak + 1.1 * per_row + mem50_slack,
        label = sprintf("%s .by peak %.1f MiB (group_by peak %.1f MiB)", label, by$peak / 2^20, gb$peak / 2^20)
      )
      expect_lte(
        by$allocated, gb$allocated + 1.5 * per_row + mem50_slack,
        label = sprintf("%s .by allocated %.1f MiB (group_by allocated %.1f MiB)",
                        label, by$allocated / 2^20, gb$allocated / 2^20)
      )
    }
    rm(gt)
  }
})

test_that("an ungrouped row_number() allocates about one INT32 column", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(512 * 1024^2)

  n <- mem50_n
  per_row <- 4 * n
  df <- data.frame(x = runif(n))
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    label <- if (lazy) "lazy" else "eager"
    m <- mem50_measure(gt, function(d) dplyr::mutate(d, r = dplyr::row_number()))
    expect_lte(m$peak, 1.25 * per_row + mem50_slack,
               label = sprintf("%s row_number() peak %.1f MiB", label, m$peak / 2^20))
    expect_lte(m$allocated, 1.5 * per_row + mem50_slack,
               label = sprintf("%s row_number() allocated %.1f MiB", label, m$allocated / 2^20))
    rm(gt)
  }
})

test_that("summarise(.by) costs about what group_by() |> summarise() costs", {
  skip_if_no_gpu()
  testthat::skip_on_ci()

  for (n in c(1e3, 1e6)) {
    set.seed(5002)
    df <- data.frame(
      id = sample.int(5000L, n, replace = TRUE),
      x = runif(n), y = runif(n),
      grp = sample(paste0("g", seq_len(20)), n, replace = TRUE),
      stringsAsFactors = FALSE
    )
    for (lazy in c(FALSE, TRUE)) {
      gt <- compute(tbl_gpu(df, lazy = lazy))
      fns <- list(
        gb = function() collect(gt |> dplyr::group_by(grp) |>
          dplyr::summarise(m = mean(x), s = sum(y), n = dplyr::n(), .groups = "drop")),
        by = function() collect(gt |> dplyr::summarise(m = mean(x), s = sum(y), n = dplyr::n(), .by = grp))
      )
      attempts <- timing_retry(
        measure = function() {
          m <- timing_medians(fns)
          m[["by"]] / m[["gb"]]
        },
        ok = function(v) v <= 1.3,
        retry_if = function(v) v <= 1.6
      )
      ratio <- attempts[[length(attempts)]]
      expect_lte(
        ratio, 1.3,
        label = sprintf("n=%g %s: .by / group_by time ratio (attempts: %s)",
                        n, if (lazy) "lazy" else "eager",
                        paste(sprintf("%.2f", unlist(attempts)), collapse = ", "))
      )
    }
  }
})
