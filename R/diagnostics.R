#' System diagnostics for cuplyr
#'
#' Collects comprehensive system information for debugging installation
#' issues or filing bug reports. Includes cuplyr version, R/Rcpp versions,
#' CUDA/cuDF paths, GPU info, and environment variables.
#'
#' @param redact If `TRUE` (default), redact potentially sensitive paths
#'   (home directory, usernames). Set to `FALSE` for local debugging.
#'
#' @return Invisibly returns a list with all diagnostic information.
#'   Also prints formatted output to the console.
#'
#' @details
#' The output includes:
#' \itemize{
#'   \item **Package versions**: cuplyr, R, Rcpp, dplyr
#'   \item **CUDA info**: toolkit version, driver version, CUDA_HOME path
#'   \item **libcudf info**: CONDA_PREFIX, CUDF_HOME, detected paths
#'   \item **GPU info**: device name, compute capability, memory
#'   \item **Library paths**: LD_LIBRARY_PATH, RPATH of cuplyr.so
#'   \item **Environment**: OS, distro, key environment variables
#' }
#'
#' @seealso [check_deps()] for pre-install checks,
#'   [verify_installation()] for post-install verification
#'
#' @export
#' @examples
#' # Full diagnostics
#' diagnostics()
#'
#' # Redacted for sharing in bug reports
#' diagnostics(redact = TRUE)
#'
#' # Unredacted for local debugging
#' diagnostics(redact = FALSE)
diagnostics <- function(redact = TRUE) {
  info <- list()

  # -- Package versions --
  info$cuplyr_version <- tryCatch(
    as.character(utils::packageVersion("cuplyr")),
    error = function(e) "not installed"
  )
  info$r_version <- paste0(R.version$major, ".", R.version$minor)
  info$r_platform <- R.version$platform
  info$rcpp_version <- tryCatch(
    as.character(utils::packageVersion("Rcpp")),
    error = function(e) "not installed"
  )
  info$dplyr_version <- tryCatch(
    as.character(utils::packageVersion("dplyr")),
    error = function(e) "not installed"
  )

  # -- OS info --
  info$os <- Sys.info()[["sysname"]]
  info$os_release <- tryCatch({
    if (file.exists("/etc/os-release")) {
      lines <- readLines("/etc/os-release", warn = FALSE)
      pretty <- grep("^PRETTY_NAME=", lines, value = TRUE)
      if (length(pretty) > 0) gsub('^PRETTY_NAME="|"$', "", pretty[1])
      else "unknown"
    } else {
      "unknown"
    }
  }, error = function(e) "unknown")

  # -- CUDA info --
  info$cuda_home <- Sys.getenv("CUDA_HOME", "/usr/local/cuda")
  info$cuda_version <- tryCatch({
    nvcc <- file.path(info$cuda_home, "bin", "nvcc")
    if (!file.exists(nvcc)) nvcc <- Sys.which("nvcc")
    if (nzchar(nvcc)) {
      out <- system2(nvcc, "--version", stdout = TRUE, stderr = TRUE)
      line <- grep("release", out, value = TRUE)
      if (length(line) > 0) {
        m <- regmatches(line, regexpr("[0-9]+\\.[0-9]+", line))
        if (length(m) > 0) m[1] else "unknown"
      } else "unknown"
    } else "nvcc not found"
  }, error = function(e) "error")

  info$driver_version <- tryCatch({
    out <- system2("nvidia-smi", c("--query-gpu=driver_version",
                                    "--format=csv,noheader"),
                   stdout = TRUE, stderr = TRUE)
    if (!is.null(attr(out, "status"))) "nvidia-smi failed"
    else trimws(out[1])
  }, error = function(e) "nvidia-smi not found")

  # -- cuDF/RAPIDS info --
  info$conda_prefix <- Sys.getenv("CONDA_PREFIX", "")
  info$cudf_home <- Sys.getenv("CUDF_HOME", "")

  # -- GPU info --
  info$gpu <- tryCatch({
    if (requireNamespace("cuplyr", quietly = TRUE)) {
      gpu_details()
    } else {
      list(available = FALSE, note = "cuplyr not loaded")
    }
  }, error = function(e) list(available = FALSE, error = conditionMessage(e)))

  # -- Library paths --
  info$ld_library_path <- Sys.getenv("LD_LIBRARY_PATH", "")

  info$cuplyr_so_path <- tryCatch({
    pkg_path <- find.package("cuplyr", quiet = TRUE)
    if (length(pkg_path) > 0) {
      so <- file.path(pkg_path, "libs", "cuplyr.so")
      if (file.exists(so)) so else "not found"
    } else "package not installed"
  }, error = function(e) "error")

  info$cuplyr_runpath <- tryCatch({
    if (info$cuplyr_so_path != "not found" &&
        info$cuplyr_so_path != "package not installed" &&
        info$cuplyr_so_path != "error") {
      out <- system2("readelf", c("-d", info$cuplyr_so_path),
                     stdout = TRUE, stderr = TRUE)
      rp <- grep("RUNPATH|RPATH", out, value = TRUE)
      if (length(rp) > 0) trimws(rp[1]) else "no RPATH/RUNPATH"
    } else NA_character_
  }, error = function(e) "readelf not available")

  # -- Redaction --
  if (redact) {
    home <- Sys.getenv("HOME", "~")
    user <- Sys.info()[["user"]]
    redact_path <- function(x) {
      if (is.na(x) || !is.character(x)) return(x)
      x <- gsub(home, "~", x, fixed = TRUE)
      if (nzchar(user)) x <- gsub(user, "<user>", x, fixed = TRUE)
      x
    }
    info$cuda_home <- redact_path(info$cuda_home)
    info$conda_prefix <- redact_path(info$conda_prefix)
    info$cudf_home <- redact_path(info$cudf_home)
    info$ld_library_path <- redact_path(info$ld_library_path)
    info$cuplyr_so_path <- redact_path(info$cuplyr_so_path)
    info$cuplyr_runpath <- redact_path(info$cuplyr_runpath)
  }

  # -- Print --
  cat("cuplyr diagnostics\n")
  cat(strrep("=", 50), "\n\n")

  cat("Package Versions\n")
  cat("  cuplyr:    ", info$cuplyr_version, "\n")
  cat("  R:         ", info$r_version, " (", info$r_platform, ")\n", sep = "")
  cat("  Rcpp:      ", info$rcpp_version, "\n")
  cat("  dplyr:     ", info$dplyr_version, "\n")
  cat("\n")

  cat("System\n")
  cat("  OS:        ", info$os, "\n")
  cat("  Distro:    ", info$os_release, "\n")
  cat("\n")

  cat("CUDA\n")
  cat("  CUDA_HOME: ", info$cuda_home, "\n")
  cat("  CUDA ver:  ", info$cuda_version, "\n")
  cat("  Driver:    ", info$driver_version, "\n")
  cat("\n")

  cat("RAPIDS / libcudf\n")
  cat("  CONDA_PREFIX: ", if (nzchar(info$conda_prefix)) info$conda_prefix else "(not set)", "\n")
  cat("  CUDF_HOME:    ", if (nzchar(info$cudf_home)) info$cudf_home else "(not set)", "\n")
  cat("\n")

  cat("GPU\n")
  if (isTRUE(info$gpu$available)) {
    cat("  Available: ", "YES\n")
    cat("  Name:      ", info$gpu$name %||% "unknown", "\n")
    cat("  Compute:   ", info$gpu$compute_capability %||% "unknown", "\n")
    total_gb <- round((info$gpu$total_memory %||% 0) / 1e9, 1)
    free_gb <- round((info$gpu$free_memory %||% 0) / 1e9, 1)
    cat("  Memory:    ", total_gb, "GB total, ", free_gb, "GB free\n")
  } else {
    cat("  Available:  NO\n")
    if (!is.null(info$gpu$error)) {
      cat("  Error:     ", info$gpu$error, "\n")
    }
  }
  cat("\n")

  cat("Library Paths\n")
  cat("  LD_LIBRARY_PATH: ", if (nzchar(info$ld_library_path)) info$ld_library_path else "(not set)", "\n")
  cat("  cuplyr.so:       ", info$cuplyr_so_path, "\n")
  if (!is.na(info$cuplyr_runpath)) {
    cat("  RUNPATH:         ", info$cuplyr_runpath, "\n")
  }
  cat("\n")
  cat(strrep("=", 50), "\n")
  cat("Paste this output in bug reports: https://github.com/bbtheo/cuplyr/issues\n")

  invisible(info)
}
