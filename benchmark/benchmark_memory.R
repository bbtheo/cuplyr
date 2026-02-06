# Benchmark lazy vs eager execution memory and timing

devtools::load_all()
library(dplyr)
library(data.table)

if (!has_gpu()) {
  stop("GPU not available; cannot run benchmark.", call. = FALSE)
}

set.seed(42)

memory_gb <- function() {
  gpu_memory_state()$used_gb
}

format_gb <- function(x) sprintf("%.3f", x)

# Synthetic data generation (mirrors benchmark.R)
generate_taxi_data <- function(n_rows = 1000000) {
  cat(sprintf("Generating %s synthetic rows...\n", format(n_rows, big.mark = ",")))

  dt <- data.table(
    VendorID = sample(1:4, n_rows, replace = TRUE),
    payment_type = sample(1:5, n_rows, replace = TRUE),
    fare_amount = pmax(2.5, rnorm(n_rows, mean = 13, sd = 10)),
    tip_amount = pmax(0, rnorm(n_rows, mean = 2.5, sd = 3)),
    tolls_amount = sample(c(rep(0, 3), runif(n_rows / 10, 1, 15)), n_rows, replace = TRUE),
    trip_distance = pmax(0.1, rexp(n_rows, rate = 0.3))
  )

  cat(sprintf("Generated %s rows x %d columns\n", format(nrow(dt), big.mark = ","), ncol(dt)))
  as.data.frame(dt)
}

run_pipeline <- function(tbl) {
  tbl |>
    dplyr::mutate(tip_pct = (tip_amount / fare_amount)*100) |>
    dplyr::mutate(total = fare_amount + tip_amount + tolls_amount) |> 
    dplyr::filter(fare_amount > 10) |>
    dplyr::filter(VendorID == 1) |> 
    summarise(
      n = n(),
      max_tot=max(total),
      mean_tip_pct = mean(tip_pct)
    )
}

run_benchmark <- function(df, mode = c("eager", "lazy")) {
  mode <- match.arg(mode)
  lazy_flag <- identical(mode, "lazy")

  gpu_gc(verbose = FALSE)
  baseline <- memory_gb()

  tbl <- tbl_gpu(df, lazy = lazy_flag)
  mem_after_load <- memory_gb()

  timings <- list()
  mem_steps <- list()

  timings$build <- system.time({
    pipeline <- run_pipeline(tbl)
  })
  mem_steps$after_build <- memory_gb()

  timings$collect <- system.time({
    result <- pipeline |> collect()
  })
  mem_steps$after_collect <- memory_gb()

  list(
    mode = mode,
    baseline_gb = baseline,
    after_load_gb = mem_after_load,
    after_build_gb = mem_steps$after_build,
    after_collect_gb = mem_steps$after_collect,
    build_time_s = timings$build[["elapsed"]],
    collect_time_s = timings$collect[["elapsed"]],
    result_rows = nrow(result)
  )
}

N_ROWS <- 10000000
data <- generate_taxi_data(n_rows = N_ROWS)

eager <- run_benchmark(data, "eager")
lazy <- run_benchmark(data, "lazy")

summary <- rbind(
  data.frame(eager, stringsAsFactors = FALSE),
  data.frame(lazy, stringsAsFactors = FALSE)
)

summary$load_delta_gb <- summary$after_load_gb - summary$baseline_gb
summary$build_delta_gb <- summary$after_build_gb - summary$after_load_gb
summary$collect_delta_gb <- summary$after_collect_gb - summary$after_build_gb
summary$peak_delta_gb <- pmax(
  summary$after_load_gb,
  summary$after_build_gb,
  summary$after_collect_gb
) - summary$baseline_gb

summary_fmt <- summary
summary_fmt$baseline_gb <- format_gb(summary_fmt$baseline_gb)
summary_fmt$after_load_gb <- format_gb(summary_fmt$after_load_gb)
summary_fmt$after_build_gb <- format_gb(summary_fmt$after_build_gb)
summary_fmt$after_collect_gb <- format_gb(summary_fmt$after_collect_gb)
summary_fmt$load_delta_gb <- format_gb(summary_fmt$load_delta_gb)
summary_fmt$build_delta_gb <- format_gb(summary_fmt$build_delta_gb)
summary_fmt$collect_delta_gb <- format_gb(summary_fmt$collect_delta_gb)
summary_fmt$peak_delta_gb <- format_gb(summary_fmt$peak_delta_gb)

print(summary_fmt, row.names = FALSE)
