
devtools::load_all()
library(arrow)
library(dplyr)

parquet_files <- list.files("benchmark_data", pattern = "\\.parquet$", full.names = TRUE)

benchmark_cols <- c(
  "VendorID", "payment_type", "fare_amount",
  "tip_amount", "tolls_amount", "trip_distance"
)
data <- open_dataset(parquet_files) |>
  select(all_of(benchmark_cols)) |>
  collect()
gpu_data <- tbl_gpu(data)
gpu_memory_state()$used_gb
gpu_data <- gpu_data |> 
  mutate(tip_pct = tip_amount / fare_amount) 
gpu_memory_state()$used_gb
gpu_data <- gpu_data |>
  mutate(tip_pct = tip_pct * 100)
gpu_memory_state()$used_gb
gpu_data <- gpu_data |>
  mutate(total = fare_amount + tip_amount) 
gpu_memory_state()$used_gb
gpu_data |>
  mutate(total = total + tolls_amount) |>
  collect()
