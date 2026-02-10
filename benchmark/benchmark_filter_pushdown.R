# Benchmark: filter pushdown effectiveness in lazy mode

library(dplyr)
devtools::load_all()

set.seed(42)

generate_taxi_data <- function(n_rows = 10000000) {
  data.frame(
    VendorID = sample(1:4, n_rows, replace = TRUE),
    payment_type = sample(1:5, n_rows, replace = TRUE),
    fare_amount = pmax(2.5, rnorm(n_rows, mean = 13, sd = 10)),
    tip_amount = pmax(0, rnorm(n_rows, mean = 2.5, sd = 3)),
    tolls_amount = sample(c(rep(0, 3), runif(n_rows / 10, 1, 15)), n_rows, replace = TRUE),
    trip_distance = pmax(0.1, rexp(n_rows, rate = 0.3))
  )
}

run_case <- function(label, expr_fn, n_iter = 3) {
  cat(sprintf("\n=== %s ===\n", label))
  times <- numeric(n_iter)
  mem_deltas <- numeric(n_iter)

  for (i in seq_len(n_iter)) {
    gpu_gc()
    before <- gpu_memory_state()
    times[i] <- system.time({
      result <- expr_fn()
    })["elapsed"]
    after <- gpu_memory_state()
    mem_deltas[i] <- after$used_bytes - before$used_bytes
    rm(result)
  }

  cat(sprintf("  median %.1f ms (range: %.1f - %.1f)\n",
              median(times) * 1000,
              min(times) * 1000,
              max(times) * 1000))
  cat(sprintf("  median GPU used delta: %.2f GB (range: %.2f - %.2f)\n",
              median(mem_deltas) / 1e9,
              min(mem_deltas) / 1e9,
              max(mem_deltas) / 1e9))
  invisible(list(times = times, mem_deltas = mem_deltas))
}

if (!has_gpu()) {
  stop("GPU not available")
}

df <- generate_taxi_data()

auto_pushdown <- function() {
  tbl_gpu(df, lazy = TRUE) |>
    mutate(
      tip_pct = (tip_amount / fare_amount) * 100,
      total = fare_amount + tip_amount + tolls_amount,
      fare_per_mile = fare_amount / trip_distance
    ) |>
    filter(fare_amount > 10, VendorID == 1) |>
    summarise(
      n = n(),
      max_tot = max(total),
      mean_tip_pct = mean(tip_pct)
    ) |>
    collect()
}

manual_pushdown <- function() {
  tbl_gpu(df, lazy = TRUE) |>
    filter(fare_amount > 10, VendorID == 1) |>
    mutate(
      tip_pct = (tip_amount / fare_amount) * 100,
      total = fare_amount + tip_amount + tolls_amount,
      fare_per_mile = fare_amount / trip_distance
    ) |>
    summarise(
      n = n(),
      max_tot = max(total),
      mean_tip_pct = mean(tip_pct)
    ) |>
    collect()
}

cat(sprintf("Rows: %s\n", format(nrow(df), big.mark = ",")))

auto_results <- run_case("Auto pushdown (mutate -> filter)", auto_pushdown)
manual_results <- run_case("Manual pushdown (filter -> mutate)", manual_pushdown)

auto_result <- auto_pushdown()
manual_result <- manual_pushdown()

cat("\nResult equivalence check:\n")
print(all.equal(auto_result, manual_result, tolerance = 1e-8))

cat(sprintf("\nSpeedup (manual / auto): %.2fx\n",
            median(manual_results$times) / median(auto_results$times)))
cat(sprintf("Memory delta ratio (manual / auto): %.2fx\n",
            median(manual_results$mem_deltas) / median(auto_results$mem_deltas)))
