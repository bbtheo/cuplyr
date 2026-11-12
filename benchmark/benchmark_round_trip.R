library(dplyr)
devtools::load_all()

generate_taxi_data <- function(n_rows = 1000000) {
  dt <- data.table::data.table(
    VendorID = sample(1:4, n_rows, replace = TRUE),
    payment_type = sample(1:5, n_rows, replace = TRUE),
    fare_amount = pmax(2.5, rnorm(n_rows, mean = 13, sd = 10)),
    tip_amount = pmax(0, rnorm(n_rows, mean = 2.5, sd = 3)),
    tolls_amount = sample(
      c(rep(0, 3), runif(n_rows / 10, 1, 15)),
      n_rows,
      replace = TRUE
    ),
    trip_distance = pmax(0.1, rexp(n_rows, rate = 0.3))
  )

  cat(sprintf(
    "Generated %s rows x %d columns\n",
    format(nrow(dt), big.mark = ","),
    ncol(dt)
  ))

  return(dt)
}

n_obs <- 500000

df <- generate_taxi_data(n_obs) |> tibble::as_tibble()

system.time(gpu_df <- tbl_gpu(df))

analyse <- function(data, low_bound = 1) {
  data |>
    filter(fare_amount >= !!low_bound) |>
    group_by(VendorID, payment_type) |>
    summarise(
      n = n(),
      n_zero_tips = sum(tip_amount == 0)
    ) |>
    collect()
}

analyse_cpu <- function(data, low_bound = 1) {
  data |>
    filter(fare_amount >= !!low_bound) |>
    summarise(
      n = n(),
      n_zero_tips = sum(tip_amount == 0),
      .by = c(VendorID, payment_type)
    ) |>
    collect()
}


system.time({
  analyse_cpu(df)
})
system.time({
  analyse(gpu_df)
})

low_bounds <- pmax(5, rnorm(100000, 10, 5))

gpu_time <- system.time({
  gpu_df <- tbl_gpu(df, lazy = F)
  for (low_bound in low_bounds) {
    analyse(gpu_df, low_bound)
  }
})

cpu_time <- system.time({
  for (low_bound in low_bounds) {
    analyse_cpu(df, low_bound)
  }
})

gpu_gc(T)
cat(glue::glue(
  "Speedup {round(cpu_time[['elapsed']]/gpu_time[['elapsed']],2)}x"
))
