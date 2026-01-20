# GPU information and utilities

#' Check if GPU is available
#'
#' @return Logical indicating GPU availability
#' @export
#' @examples
#' has_gpu()
has_gpu <- function() {
  tryCatch(gpu_is_available(), error = function(e) FALSE)
}

#' Get GPU information
#'
#' Returns details about the available GPU including name, memory, and compute capability.
#'
#' @return A list with GPU information, or a list with `available = FALSE` if no GPU
#' @export
#' @examples
#' info <- gpu_details()
#' if (info$available) {
#'   cat("GPU:", info$name, "\n")
#'   cat("Memory:", round(info$total_memory / 1e9, 1), "GB\n")
#' }
gpu_details <- function() {
  tryCatch(gpu_info(), error = function(e) list(available = FALSE))
}

#' Print GPU information
#'
#' Prints formatted GPU information to the console.
#'
#' @return Invisibly returns the GPU info list
#' @export
#' @examples
#' show_gpu()
show_gpu <- function() {
  info <- gpu_details()

  if (isTRUE(info$available)) {
    total_gb <- round(info$total_memory / 1e9, 1)
    free_gb <- round(info$free_memory / 1e9, 1)
    used_gb <- round((info$total_memory - info$free_memory) / 1e9, 1)

    cat("GPU Information\n")
    cat(strrep("-", 40), "\n")
    cat("Device:      ", info$name, "\n")
    cat("Compute:     ", info$compute_capability, "\n")
    cat("Memory:      ", total_gb, "GB total\n")
    cat("             ", free_gb, "GB free\n")
    cat("             ", used_gb, "GB used\n")
    cat("SMs:         ", info$multiprocessors, "\n")
  } else {
    cat("No GPU available\n")
  }

  invisible(info)
}
