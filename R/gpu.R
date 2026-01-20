#' Check GPU availability
#'
#' Tests whether a CUDA-capable GPU is available and accessible.
#' This function is useful for conditional code that should only
#' run when GPU acceleration is possible.
#'
#' @return `TRUE` if a GPU is available and CUDA is properly configured,
#'   `FALSE` otherwise. Returns `FALSE` (not an error) if CUDA libraries
#'   are not found.
#'
#' @details
#' This function checks:
#' \enumerate{
#'   \item CUDA driver is loaded
#'   \item At least one CUDA device is present
#'   \item Device is accessible (not in exclusive mode by another process)
#' }
#'
#' @seealso
#' \code{\link{gpu_details}} for detailed GPU information,
#' \code{\link{show_gpu}} for formatted GPU info display
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   message("GPU acceleration available!")
#'   gpu_df <- tbl_gpu(mtcars)
#' } else {
#'   message("No GPU found, using CPU")
#' }
has_gpu <- function() {
  tryCatch(gpu_is_available(), error = function(e) FALSE)
}

#' Get detailed GPU information
#'
#' Retrieves comprehensive information about the available GPU including
#' device name, compute capability, memory capacity, and multiprocessor count.
#'
#' @return A named list with GPU details:
#' \describe{
#'   \item{available}{Logical: TRUE if GPU is available}
#'   \item{device_count}{Number of CUDA devices}
#'   \item{device_id}{ID of the current device (0-indexed)}
#'   \item{name}{GPU model name (e.g., "NVIDIA GeForce RTX 4090")}
#'   \item{compute_capability}{CUDA compute capability (e.g., "8.9")}
#'   \item{total_memory}{Total GPU memory in bytes}
#'   \item{free_memory}{Currently available GPU memory in bytes}
#'   \item{multiprocessors}{Number of streaming multiprocessors (SMs)}
#' }
#'
#' If no GPU is available, returns `list(available = FALSE, device_count = 0)`.
#'
#' @details
#' ## Compute capability
#' The compute capability indicates the GPU architecture and supported features:
#' \itemize{
#'   \item 7.x - Volta/Turing (V100, RTX 20 series)
#'   \item 8.x - Ampere (A100, RTX 30 series)
#'   \item 8.9 - Ada Lovelace (RTX 40 series)
#'   \item 9.x - Hopper (H100)
#'   \item 10.x+ - Blackwell and newer
#' }
#'
#' ## Memory
#' The `free_memory` value reflects memory available at the time of the call.
#' Other applications or CUDA contexts may be using GPU memory.
#'
#' @seealso
#' \code{\link{has_gpu}} for simple availability check,
#' \code{\link{show_gpu}} for formatted display
#'
#' @export
#' @examples
#' info <- gpu_details()
#' if (info$available) {
#'   cat("GPU:", info$name, "\n")
#'   cat("Memory:", round(info$total_memory / 1e9, 1), "GB\n")
#'   cat("Compute:", info$compute_capability, "\n")
#' }
gpu_details <- function() {
  tryCatch(gpu_info(), error = function(e) list(available = FALSE))
}

#' Display GPU information
#'
#' Prints formatted information about the available GPU to the console.
#' Useful for verifying GPU setup and checking available resources.
#'
#' @return Invisibly returns the GPU info list (same as [gpu_details()]).
#'
#' @details
#' Output includes:
#' \itemize{
#'   \item Device name and compute capability
#'   \item Total, free, and used memory
#'   \item Number of streaming multiprocessors
#' }
#'
#' @seealso
#' \code{\link{has_gpu}} for availability check,
#' \code{\link{gpu_details}} for programmatic access
#'
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
    cat("\nPossible reasons:\n")
    cat("- No NVIDIA GPU installed\n")
    cat("- CUDA drivers not installed\n")
    cat("- GPU in use by another process (exclusive mode)\n")
  }

  invisible(info)
}
