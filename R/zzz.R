#' @useDynLib cuplr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom rlang %||%
NULL
.onLoad <- function(libname, pkgname) {
  # Check GPU availability
  gpu_ok <- tryCatch(
    gpu_is_available(),
    error = function(e) FALSE
  )

  # Set package options
  op <- options()
  op.cuplr <- list(
    cuplr.verbose = FALSE,
    cuplr.lazy = TRUE,
    cuplr.gpu_available = gpu_ok
  )
  toset <- !(names(op.cuplr) %in% names(op))
  if (any(toset)) options(op.cuplr[toset])

  invisible()
}

.onAttach <- function(libname, pkgname) {
  info <- tryCatch(gpu_info(), error = function(e) list(available = FALSE))

  if (isTRUE(info$available)) {
    # Format memory in GB
    total_gb <- round(info$total_memory / 1e9, 1)
    free_gb <- round(info$free_memory / 1e9, 1)

    msg <- paste0(
      "cuplr: GPU-accelerated data manipulation\n",
      "GPU: ", info$name, " (", info$compute_capability, ")\n",
      "Memory: ", free_gb, " GB free / ", total_gb, " GB total"
    )
  } else {
    msg <- paste0(
      "cuplr: GPU-accelerated data manipulation\n",
      "WARNING: No GPU detected. Package will not function correctly."
    )
  }

  packageStartupMessage(msg)
}
