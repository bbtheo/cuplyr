#' Verify cuplyr installation
#'
#' Performs a post-install runtime verification by running a small GPU
#' round-trip: creates a GPU table, performs a filter, and collects back
#' to R. This confirms that the package loads, GPU is accessible, and
#' basic operations work.
#'
#' @return Invisibly returns a list with:
#' \describe{
#'   \item{ok}{Logical: `TRUE` if all checks pass}
#'   \item{package_loaded}{Logical: package loads without error}
#'   \item{gpu_available}{Logical: GPU detected and accessible}
#'   \item{gpu_name}{Character: GPU model name, or `NA`}
#'   \item{roundtrip_ok}{Logical: data round-trip succeeded}
#'   \item{time_seconds}{Numeric: time for the round-trip test}
#' }
#'
#' @details
#' This function runs three checks in sequence:
#' \enumerate{
#'   \item **Package load**: `library(cuplyr)` succeeds
#'   \item **GPU detection**: `gpu_details()` shows an available GPU
#'   \item **Round-trip**: `tbl_gpu(mtcars) |> filter(mpg > 20) |> collect()`
#'     returns correct results
#' }
#'
#' @seealso [check_deps()] for pre-install dependency checks,
#'   [diagnostics()] for full system diagnostics
#'
#' @export
#' @examples
#' \dontrun{
#' result <- verify_installation()
#' if (result$ok) {
#'   message("cuplyr is working!")
#' }
#' }
verify_installation <- function() {
  result <- list(
    ok = FALSE,
    package_loaded = FALSE,
    gpu_available = FALSE,
    gpu_name = NA_character_,
    roundtrip_ok = FALSE,
    time_seconds = NA_real_
  )

  # 0. Auto-configure cloud library paths if needed
  env <- detect_environment()
  if (env %in% c("colab", "cloud_gpu")) {
    driver_lib <- find_real_driver_lib()
    if (!is.null(driver_lib)) {
      conda_prefix <- Sys.getenv("CONDA_PREFIX", "/opt/rapids")
      configure_cloud_library_paths(driver_lib, conda_prefix)
    }
  }

  # 1. Package load
  cat("Checking package load... ")
  loaded <- tryCatch({
    requireNamespace("cuplyr", quietly = TRUE)
  }, error = function(e) FALSE)

  if (!loaded) {
    cat(cli::col_red("FAILED"), "\n")
    cat("cuplyr package could not be loaded.\n")
    cat("Run: check_deps() to diagnose.\n")
    return(invisible(result))
  }
  result$package_loaded <- TRUE
  cat(cli::col_green("OK"), "\n")

  # 2. GPU detection
  cat("Checking GPU access... ")
  gpu_info <- tryCatch(gpu_details(), error = function(e) list(available = FALSE))

  if (!isTRUE(gpu_info$available)) {
    cat(cli::col_red("FAILED"), "\n")
    cat("No GPU detected. Possible causes:\n")
    cat("  - NVIDIA driver not installed or not loaded\n")
    cat("  - CUDA stub libraries masking real driver (cloud environments)\n")
    cat("  - GPU in exclusive mode\n")
    cat("\nRun: diagnostics() for details.\n")
    return(invisible(result))
  }
  result$gpu_available <- TRUE
  result$gpu_name <- gpu_info$name %||% "unknown"
  cat(cli::col_green("OK"), " (", result$gpu_name, ")\n", sep = "")

  # 3. Round-trip test
  cat("Running round-trip test... ")
  start <- proc.time()

  rt_ok <- tryCatch({
    test_df <- data.frame(
      x = c(1.0, 2.0, 3.0, 4.0, 5.0),
      y = c("a", "b", "c", "d", "e"),
      stringsAsFactors = FALSE
    )
    gpu_df <- tbl_gpu(test_df)
    result_df <- gpu_df |>
      dplyr::filter(x > 2) |>
      dplyr::collect()

    # Verify results
    identical(nrow(result_df), 3L) &&
      all(result_df$x > 2) &&
      identical(result_df$y, c("c", "d", "e"))
  }, error = function(e) {
    cat("\n  Error: ", conditionMessage(e), "\n", sep = "")
    FALSE
  })

  elapsed <- (proc.time() - start)[["elapsed"]]
  result$time_seconds <- elapsed

  if (!rt_ok) {
    cat(cli::col_red("FAILED"), "\n")
    cat("GPU round-trip returned incorrect results.\n")
    cat("Run: diagnostics() for details.\n")
    return(invisible(result))
  }

  result$roundtrip_ok <- TRUE
  result$ok <- TRUE
  cat(cli::col_green("OK"), sprintf(" (%.2fs)\n", elapsed), sep = "")

  cat(strrep("-", 50), "\n")
  cat(cli::col_green("cuplyr is working correctly!"), "\n")
  cat("GPU: ", result$gpu_name, "\n", sep = "")

  invisible(result)
}
