#!/usr/bin/env Rscript

# Download NYC Yellow Taxi trip data as parquet files
# Data source: NYC Taxi and Limousine Commission (TLC)
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

library(arrow)

BASE_URL <- "https://d37ci6vzurychx.cloudfront.net/trip-data"
DATA_DIR <- "benchmark_data"

download_taxi_data <- function(years = 2023, months = 1:12, type = "yellow",
                                output_dir = DATA_DIR, overwrite = FALSE) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  files <- expand.grid(year = years, month = months)
  files$month_str <- sprintf("%02d", files$month)
  files$filename <- sprintf("%s_tripdata_%d-%s.parquet", type, files$year, files$month_str)
  files$url <- file.path(BASE_URL, files$filename)
  files$output_path <- file.path(output_dir, files$filename)

  for (i in seq_len(nrow(files))) {
    output_path <- files$output_path[i]
    url <- files$url[i]
    filename <- files$filename[i]

    if (file.exists(output_path) && !overwrite) {
      message(sprintf("Skipping %s (already exists)", filename))
      next
    }

    message(sprintf("Downloading %s...", filename))

    tryCatch({
      download.file(url, output_path, mode = "wb", quiet = TRUE)
      message(sprintf("  Saved to %s", output_path))
    }, error = function(e) {
      warning(sprintf("  Failed to download %s: %s", filename, e$message))
      if (file.exists(output_path)) file.remove(output_path)
    })
  }

  existing_files <- list.files(output_dir, pattern = "\\.parquet$", full.names = TRUE)
  message(sprintf("\nDone. %d parquet files in %s", length(existing_files), output_dir))
  invisible(existing_files)
}

if (sys.nframe() == 0 || identical(environment(), globalenv())) {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) >= 1) {
    years <- as.integer(strsplit(args[1], ",")[[1]])
  } else {
    years <- 2023
  }

  if (length(args) >= 2) {
    months <- as.integer(strsplit(args[2], ",")[[1]])
  } else {
    months <- 1:12
  }

  download_taxi_data(years = years, months = months)
}
