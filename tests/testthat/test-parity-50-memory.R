# #50: summarise(.by =) memory and speed against
# group_by() |> summarise(.groups = "drop"), and a cheap ungrouped row_number().
#
# Memory is measured with gpu_alloc_tracking() (F56): an RMM statistics adaptor
# counts every byte allocated while the call runs (`allocated_bytes`) and the
# most held at once (`peak_bytes`), excluding the resident input. These counts
# are exact and do not depend on other processes, so the bounds are tight.
#
# Before #50 the `.by` form added a whole-table `row_number()` column through
# the window machinery, then ran a grouped summarise with a `min()` of it, an
# arrange() and a select(): four verb calls, 2-3x the group_by form's time up to
# 1e6 rows and 1.2-2.3x at 1e7 with numeric keys. Its peak was the group_by
# form's plus 4 bytes/row on the issue's pipelines (1.14-1.20x) and up to about
# 10 bytes/row on narrow summaries.
#
# The contract (decided for #50): the first-appearance position is a MIN over a
# row position inside the one groupby. At libcudf's hash-groupby peak that is
# at most three INT32 columns: the n-row position, the n-row sparse MIN
# intermediate and its k-row dense result. Measured over group_by: about 4
# bytes/row on the issue's pipelines, about 8 on a one-aggregation summary with
# many groups, up to 12 with all-unique keys and logical key columns; never
# more than the .by peak before #50 on any of these shapes:
#   peak(.by) <= 1.2 x peak(group_by)         on the issue's pipelines
#   peak(.by) <= peak(group_by) + 12 bytes/row on any summary
#   time(.by) no worse than before #50, and <= 1.3 x group_by at 1e3-1e6 rows
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
    # narrow keys: all-unique INT32, and all-unique (INT32, small INT32) and
    # (INT32, logical) pairs
    iu = sample.int(n),
    ib = sample.int(3L, n, replace = TRUE),
    lb = sample(c(TRUE, FALSE), n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}

# the issue's table: 8 columns (4 integer, 2 double, 2 character), z 5% NA
mem50_task_df <- function(n) {
  set.seed(20261114L)
  z <- sample.int(100L, n, replace = TRUE)
  z[stats::runif(n) < 0.05] <- NA_integer_
  data.frame(
    id = seq_len(n),
    g_low = sprintf("k%02d", sample.int(10L, n, replace = TRUE)),
    g_high = sample.int(max(1L, as.integer(n) %/% 100L), n, replace = TRUE),
    g_skew = sample.int(10000L, n, replace = TRUE, prob = 1 / seq_len(10000L)^1.1),
    x = stats::rnorm(n), y = stats::runif(n), z = z,
    s = sample(sprintf("s%04d", seq_len(1000L) * 7L), n, replace = TRUE),
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

test_that("summarise(.by) on the issue's pipelines peaks within 1.2x of group_by()", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(512 * 1024^2)

  df <- mem50_task_df(mem50_n)
  cases <- list(
    lowcard = list(
      gb = function(d) d |> dplyr::group_by(g_low) |>
        dplyr::summarise(n = dplyr::n(), mean_x = mean(x), sum_y = sum(y), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), mean_x = mean(x), sum_y = sum(y), .by = g_low)
    ),
    highcard = list(
      gb = function(d) d |> dplyr::group_by(g_high) |>
        dplyr::summarise(n = dplyr::n(), mean_x = mean(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), mean_x = mean(x), .by = g_high)
    ),
    skew = list(
      gb = function(d) d |> dplyr::group_by(g_skew) |>
        dplyr::summarise(n = dplyr::n(), sum_x = sum(x), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), sum_x = sum(x), .by = g_skew)
    ),
    across = list(
      gb = function(d) d |> dplyr::group_by(g_low) |>
        dplyr::summarise(dplyr::across(c(x, y), list(mean = mean, sd = sd)), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(dplyr::across(c(x, y), list(mean = mean, sd = sd)), .by = g_low)
    ),
    na = list(
      gb = function(d) d |> dplyr::group_by(g_low) |>
        dplyr::summarise(mean_z = mean(z, na.rm = TRUE), n_na = sum(is.na(z)), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(mean_z = mean(z, na.rm = TRUE), n_na = sum(is.na(z)), .by = g_low)
    )
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    for (nm in names(cases)) {
      gb <- mem50_measure(gt, cases[[nm]]$gb)
      by <- mem50_measure(gt, cases[[nm]]$by)
      expect_lte(
        by$peak, 1.2 * gb$peak + mem50_slack,
        label = sprintf("%s (%s) .by peak %.1f MiB (group_by peak %.1f MiB, ratio %.3f)",
                        nm, if (lazy) "lazy" else "eager", by$peak / 2^20, gb$peak / 2^20, by$peak / gb$peak)
      )
    }
    rm(gt)
  }
})

test_that("summarise(.by) peaks at most three INT32 columns above group_by()", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(512 * 1024^2)

  n <- mem50_n
  df <- mem50_df(n)
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
    # every row its own group: the reorder of the k-row result is the peak
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
    ),
    # a count over narrow keys keeps the groupby's own peak low
    unique_pairs_count = list(
      gb = function(d) d |> dplyr::group_by(iu, ib) |>
        dplyr::summarise(n = dplyr::n(), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), .by = c(iu, ib))
    ),
    unique_int_count = list(
      gb = function(d) d |> dplyr::group_by(iu) |>
        dplyr::summarise(n = dplyr::n(), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), .by = iu)
    ),
    unique_logical_pairs_count = list(
      gb = function(d) d |> dplyr::group_by(iu, lb) |>
        dplyr::summarise(n = dplyr::n(), .groups = "drop"),
      by = function(d) d |> dplyr::summarise(n = dplyr::n(), .by = c(iu, lb))
    )
  )
  for (lazy in c(FALSE, TRUE)) {
    gt <- compute(tbl_gpu(df, lazy = lazy))
    for (nm in names(cases)) {
      label <- sprintf("%s (%s)", nm, if (lazy) "lazy" else "eager")
      gb <- mem50_measure(gt, cases[[nm]]$gb)
      by <- mem50_measure(gt, cases[[nm]]$by)
      expect_lte(
        by$peak, gb$peak + 12 * n + mem50_slack,
        label = sprintf("%s .by peak %.1f MiB (group_by peak %.1f MiB, +%.2f B/row)",
                        label, by$peak / 2^20, gb$peak / 2^20, (by$peak - gb$peak) / n)
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

test_that("summarise(.by) with shuffled keys is no slower than before #50", {
  skip_if_no_gpu()
  testthat::skip_on_ci()
  skip_if_insufficient_gpu_memory(1024^3)

  # `bound` is the .by / group_by time ratio of the row-number desugar before
  # #50, measured on main (af9714b) at 1e7 rows with these keys, eager, plus 5%
  # for timing noise. The group_by() form is the same code on both builds, so
  # a ratio above it means `.by` got slower. A first-appearance pass that hashes
  # every key a second time shows up as string keys over this bound.
  n <- 1e7
  specs <- list(
    i4 = list(make = function(n) sample.int(n / 4, n, replace = TRUE), bound = 1.42),
    iu = list(make = function(n) sample.int(n), bound = 1.19),
    d4 = list(make = function(n) sample.int(n / 4, n, replace = TRUE) / 7, bound = 0.88),
    s4 = list(make = function(n) sprintf("key-%08d", sample.int(n / 4, n, replace = TRUE)), bound = 0.25),
    L4 = list(make = function(n) sprintf("customer-account-identifier-%012d", sample.int(n / 4, n, replace = TRUE)),
              bound = 0.09)
  )
  for (key in names(specs)) {
    set.seed(5600 + match(key, names(specs)))
    df <- data.frame(k = specs[[key]]$make(n), x = runif(n), y = runif(n), stringsAsFactors = FALSE)
    gt <- tbl_gpu(df, lazy = FALSE)
    rm(df)
    invisible(gc())
    fns <- list(
      gb = function() {
        r <- gt |> dplyr::group_by(k) |>
          dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .groups = "drop")
        gpu_sync()
        r
      },
      by = function() {
        r <- gt |> dplyr::summarise(n = dplyr::n(), m = mean(x), sy = sum(y), .by = k)
        gpu_sync()
        r
      }
    )
    bound <- 1.05 * specs[[key]]$bound
    attempts <- timing_retry(
      measure = function() {
        m <- timing_medians(fns, samples = 7L)
        m[["by"]] / m[["gb"]]
      },
      ok = function(v) v <= bound,
      retry_if = function(v) v <= 2 * bound
    )
    ratio <- attempts[[length(attempts)]]
    expect_lte(
      ratio, bound,
      label = sprintf("shuffled key %s: .by / group_by time ratio (attempts: %s)",
                      key, paste(sprintf("%.3f", unlist(attempts)), collapse = ", "))
    )
    rm(gt)
    invisible(gc())
    gpu_gc(verbose = FALSE)
  }
})
