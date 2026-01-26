# Synthetic data generation for benchmarking
# comparison: dplyr vs cuplr vs data.table

library(dplyr)
library(data.table)
devtools::load_all()

set.seed(42)
# Generate synthetic taxi trip data as data.table directly
generate_taxi_data <- function(n_rows = 1000000) {
  cat(sprintf("Generating %s synthetic rows...\n", format(n_rows, big.mark = ",")))
  
  dt <- data.table(
    VendorID = sample(1:4, n_rows, replace = TRUE),
    payment_type = sample(1:5, n_rows, replace = TRUE),
    fare_amount = pmax(2.5, rnorm(n_rows, mean = 13, sd = 10)),
    tip_amount = pmax(0, rnorm(n_rows, mean = 2.5, sd = 3)),
    tolls_amount = sample(c(rep(0, 3), runif(n_rows/10, 1, 15)), n_rows, replace = TRUE),
    trip_distance = pmax(0.1, rexp(n_rows, rate = 0.3))
  )
  
  cat(sprintf("Generated %s rows x %d columns\n", format(nrow(dt), big.mark = ","), ncol(dt)))
  
  return(dt)
}

# Generate data as data.table first
data_dt <- generate_taxi_data(n_rows = 50000000)
data <- as.data.frame(data_dt)  # Convert to data.frame for dplyr

N_ITER <- 3

run_benchmark <- function(name, dplyr_expr, cuplr_expr, dt_expr, data, data_dt, 
                          dt_modifies = FALSE, n_iter = N_ITER) {
  cat(sprintf("\n=== %s ===\n", name))
  
  # Warm-up runs (not timed)
  invisible(dplyr_expr(data))
  if (dt_modifies) {
    invisible(dt_expr(copy(data_dt)))
  } else {
    invisible(dt_expr(data_dt))
  }
  gc()

  # Benchmark dplyr
  dplyr_times <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    dplyr_times[i] <- system.time({ result <- dplyr_expr(data) })["elapsed"]
  }
  rm(result)

  # Benchmark data.table
  dt_times <- numeric(n_iter)
  if (dt_modifies) {
    for (i in seq_len(n_iter)) {
      dt_copy <- copy(data_dt)
      dt_times[i] <- system.time({ result <- dt_expr(dt_copy) })["elapsed"]
      rm(dt_copy)
    }
  } else {
    for (i in seq_len(n_iter)) {
      dt_times[i] <- system.time({ result <- dt_expr(data_dt) })["elapsed"]
    }
  }
  rm(result)

  # Fresh GPU data for cuplr
  data_gpu <- as_tbl_gpu(data)

  # Benchmark cuplr
  cuplr_times <- numeric(n_iter)
  for (i in seq_len(n_iter)) {
    cuplr_times[i] <- system.time({ result <- cuplr_expr(data_gpu) })["elapsed"]
  }
  rm(result)

  # Cleanup GPU
  rm(data_gpu)
  gpu_gc()

  cat(sprintf("  dplyr:      median %.1f ms (range: %.1f - %.1f)\n",
              median(dplyr_times) * 1000,
              min(dplyr_times) * 1000,
              max(dplyr_times) * 1000))
  cat(sprintf("  data.table: median %.1f ms (range: %.1f - %.1f)\n",
              median(dt_times) * 1000,
              min(dt_times) * 1000,
              max(dt_times) * 1000))
  cat(sprintf("  cuplr:      median %.1f ms (range: %.1f - %.1f)\n",
              median(cuplr_times) * 1000,
              min(cuplr_times) * 1000,
              max(cuplr_times) * 1000))
  cat(sprintf("  speedup vs dplyr:      data.table %.1fx | cuplr %.1fx\n", 
              median(dplyr_times) / median(dt_times),
              median(dplyr_times) / median(cuplr_times)))
  cat(sprintf("  speedup vs data.table: cuplr %.1fx\n",
              median(dt_times) / median(cuplr_times)))

  invisible(list(dplyr = dplyr_times, dt = dt_times, cuplr = cuplr_times))
}

# Group & Summarise (read-only, no copy needed)
run_benchmark(
  "Group & Summarise",
  dplyr_expr = function(d) {
    d |>
      group_by(VendorID) |>
      summarise(mean_fare = mean(fare_amount, na.rm = TRUE), total_trips = n())
  },
  dt_expr = function(d) {
    d[, .(mean_fare = mean(fare_amount, na.rm = TRUE), total_trips = .N), by = VendorID]
  },
  cuplr_expr = function(d) {
    d |>
      group_by(VendorID) |>
      summarise(mean_fare = mean(fare_amount), total_trips = n()) |>
      collect()
  },
  data = data,
  data_dt = data_dt,
  dt_modifies = FALSE
)

gpu_gc(T)

# Mutate (uses :=, needs copy)
run_benchmark(
  "Mutate",
  dplyr_expr = function(d) {
    d |>
      mutate(tip_pct = tip_amount / fare_amount * 100,
             total = fare_amount + tip_amount + tolls_amount)
  },
  dt_expr = function(d) {
    d[, `:=`(tip_pct = tip_amount / fare_amount * 100,
             total = fare_amount + tip_amount + tolls_amount)]
  },
  cuplr_expr = function(d) {
    d |>
      mutate(tip_pct = tip_amount / fare_amount) |>
      mutate(tip_pct = tip_pct * 100) |>
      mutate(total = fare_amount + tip_amount) |>
      mutate(total = total + tolls_amount) |>
      collect()
  },
  data = data,
  data_dt = data_dt,
  dt_modifies = TRUE
)

# Filter (read-only, no copy needed)
run_benchmark(
  "Filter",
  dplyr_expr = function(d) {
    d |>
      filter(fare_amount > 10, tip_amount > 0, trip_distance > 1)
  },
  dt_expr = function(d) {
    d[fare_amount > 10 & tip_amount > 0 & trip_distance > 1]
  },
  cuplr_expr = function(d) {
    d |>
      filter(fare_amount > 10, tip_amount > 0, trip_distance > 1) |>
      collect()
  },
  data = data,
  data_dt = data_dt,
  dt_modifies = FALSE
)

# Complete Workflow (read-only until final result)
run_benchmark(
  "Complete Workflow",
  dplyr_expr = function(d) {
    d |>
      filter(fare_amount > 0, trip_distance > 0) |>
      mutate(fare_per_mile = fare_amount / trip_distance) |>
      group_by(VendorID, payment_type) |>
      summarise(avg_fare_per_mile = mean(fare_per_mile, na.rm = TRUE),
                avg_tip = mean(tip_amount, na.rm = TRUE),
                trips = n(),
                .groups = "drop")
  },
  dt_expr = function(d) {
    d[fare_amount > 0 & trip_distance > 0
      ][, fare_per_mile := fare_amount / trip_distance
        ][, .(avg_fare_per_mile = mean(fare_per_mile, na.rm = TRUE),
              avg_tip = mean(tip_amount, na.rm = TRUE),
              trips = .N),
          by = .(VendorID, payment_type)]
  },
  cuplr_expr = function(d) {
    d |>
      filter(fare_amount > 0, trip_distance > 0) |>
      mutate(fare_per_mile = fare_amount / trip_distance) |>
      group_by(VendorID, payment_type) |>
      summarise(avg_fare_per_mile = mean(fare_per_mile),
                avg_tip = mean(tip_amount),
                trips = n()) |>
      collect()
  },
  data = data,
  data_dt = data_dt,
  dt_modifies = TRUE  # Uses := for fare_per_mile
)