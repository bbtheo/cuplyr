# #50: summarise(.by =) memory and speed parity with
# group_by() |> summarise(.groups = "drop"), and a cheap ungrouped row_number().
#
# Memory is measured with gpu_alloc_tracking() (F56): an RMM statistics adaptor
# counts every byte allocated while the call runs (`allocated_bytes`) and the
# most held at once (`peak_bytes`), excluding the resident input. These counts
# are exact and do not depend on other processes, so the bounds are tight.
#
# Before #50 the `.by` form added a whole-table `row_number()` column through
# the window machinery: its peak was the group_by form's plus one or two INT32
# columns (4-8 bytes/row), it allocated about six INT32 columns more in total,
# and it ran four verb calls (2.6x the group_by form's time at 1e3-1e6 rows).
# A row-position MIN() inside the groupby still costs 8 bytes/row at the
# aggregation peak (the position column plus libcudf's per-row intermediate),
# while finding each group's first row in a pass of its own (a hash
# distinct) peaks below the groupby itself. So the contract is:
#   peak(.by) <= peak(group_by) + 1 byte/row
#   time(.by) <= 1.3 x time(group_by)
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
    su = sprintf("key-%09d", sample.int(n)),
    # n/2 string groups, short and long, and n/2 - 1000 pairs plus 2000 singletons
    s2 = sprintf("key-%09d", sample(rep(seq_len(n / 2), 2))),
    l2 = sprintf("customer-account-identifier-%012d", sample(rep(seq_len(n / 2), 2))),
    s2p = sprintf("key-%09d", sample(c(rep(seq_len(n / 2 - 1000), 2), n / 2 - 1000 + seq_len(2000)))),
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

test_that("summarise(.by) peaks no higher than group_by() |> summarise()", {
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
    ),
    # every row its own string group: the reorder of the k-row result is the peak
    unique_strings = list(
      gb = function(d) d |> dplyr::group_by(su) |>
        dplyr::summarise(s_x = sum(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(s_x = sum(x), .by = su)
    ),
    half_strings = list(
      gb = function(d) d |> dplyr::group_by(s2) |>
        dplyr::summarise(s_x = sum(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(s_x = sum(x), .by = s2)
    ),
    half_long_strings = list(
      gb = function(d) d |> dplyr::group_by(l2) |>
        dplyr::summarise(s_x = sum(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(s_x = sum(x), .by = l2)
    ),
    half_plus_strings = list(
      gb = function(d) d |> dplyr::group_by(s2p) |>
        dplyr::summarise(s_x = sum(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(s_x = sum(x), .by = s2p)
    )
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    for (nm in names(cases)) {
      label <- sprintf("%s (%s)", nm, if (lazy) "lazy" else "eager")
      gb <- mem50_measure(gt, cases[[nm]]$gb)
      by <- mem50_measure(gt, cases[[nm]]$by)
      expect_lte(
        by$peak, gb$peak + 0.25 * per_row + mem50_slack,
        label = sprintf("%s .by peak %.1f MiB (group_by peak %.1f MiB; allocated %.1f vs %.1f MiB)",
                        label, by$peak / 2^20, gb$peak / 2^20, by$allocated / 2^20, gb$allocated / 2^20)
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

test_that("summarise(.by) on input sorted by the key costs no more than it used to", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(1024^3)

  # Before #50 the `.by` form took 1.6-1.8x the group_by() form's time on
  # key-sorted input (a row-number window plus the summarise). Every group
  # starts where the previous one ends, so the first rows are the k run
  # starts: finding them must not cost several hash passes over the table.
  n <- 1e7
  set.seed(5003)
  df <- data.frame(
    s = rep(sprintf("k%02d", 1:10), each = n / 10),
    k = rep(seq_len(1000L), each = n / 1000),
    x = runif(n), y = runif(n),
    stringsAsFactors = FALSE
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    pipes <- list(
      s = list(
        gb = function() gt |> dplyr::group_by(s) |>
          dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .groups = "drop"),
        by = function() gt |> dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .by = s)
      ),
      k = list(
        gb = function() gt |> dplyr::group_by(k) |>
          dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .groups = "drop"),
        by = function() gt |> dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .by = k)
      )
    )
    # today: 1.68-2.07 for the string key, 1.40-1.57 for the integer key
    bound <- c(s = 1.6, k = 1.35)
    for (key in names(pipes)) {
      fns <- lapply(pipes[[key]], function(p) {
        force(p)
        function() {
          r <- compute(p())
          gpu_sync()
          r
        }
      })
      attempts <- timing_retry(
        measure = function() {
          m <- timing_medians(fns, samples = 9L)
          m[["by"]] / m[["gb"]]
        },
        ok = function(v) v <= bound[[key]],
        retry_if = function(v) v <= 2
      )
      ratio <- attempts[[length(attempts)]]
      expect_lte(
        ratio, bound[[key]],
        label = sprintf("sorted key %s (%s): .by / group_by time ratio (attempts: %s)",
                        key, if (lazy) "lazy" else "eager",
                        paste(sprintf("%.2f", unlist(attempts)), collapse = ", "))
      )
    }
    rm(gt)
  }
})

test_that("summarise(.by) with high-cardinality string keys costs less than group_by()", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(1024^3)

  # The group_by() form sorts its k-row result by the string keys (F33), which
  # dominates its time at high cardinality; the `.by` order is by first
  # appearance, and before #50 `.by` never sorted the keys (0.17-0.25x of the
  # group_by() form at 1e7 rows). A `.by` that pays that sort, and more, is a
  # regression.
  n <- 4e6
  set.seed(5004)
  df <- data.frame(
    su = sprintf("key-%09d", sample.int(n)),
    s2 = sprintf("key-%09d", sample(rep(seq_len(n / 2), 2))),
    x = runif(n), y = runif(n),
    stringsAsFactors = FALSE
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    pipes <- list(
      su = list(
        gb = function() gt |> dplyr::group_by(su) |>
          dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .groups = "drop"),
        by = function() gt |> dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .by = su)
      ),
      s2 = list(
        gb = function() gt |> dplyr::group_by(s2) |>
          dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .groups = "drop"),
        by = function() gt |> dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .by = s2)
      )
    )
    for (key in names(pipes)) {
      fns <- lapply(pipes[[key]], function(p) {
        force(p)
        function() {
          r <- compute(p())
          gpu_sync()
          r
        }
      })
      attempts <- timing_retry(
        measure = function() {
          m <- timing_medians(fns, samples = 7L)
          m[["by"]] / m[["gb"]]
        },
        ok = function(v) v <= 0.8,
        retry_if = function(v) v <= 1.0
      )
      ratio <- attempts[[length(attempts)]]
      expect_lte(
        ratio, 0.8,
        label = sprintf("high-cardinality key %s (%s): .by / group_by time ratio (attempts: %s)",
                        key, if (lazy) "lazy" else "eager",
                        paste(sprintf("%.2f", unlist(attempts)), collapse = ", "))
      )
    }
    rm(gt)
  }
})
