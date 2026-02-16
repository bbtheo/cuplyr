#' Check cuplyr dependencies
#'
#' Performs a pre-flight check of all dependencies required to install and
#' run cuplyr. Reports status of NVIDIA driver, CUDA toolkit, libcudf,
#' R version, and required R packages.
#'
#' @param format Output format: `"text"` for human-readable console output,
#'   `"json"` for machine-readable JSON string.
#' @param verbose If `TRUE`, print additional details about each check.
#'
#' @return Invisibly returns a list with:
#' \describe{
#'   \item{ok}{Logical: `TRUE` if all required checks pass}
#'   \item{checks}{Named list of individual check results, each with
#'     `ok`, `value`, and `message` fields}
#' }
#'
#' @details
#' The following checks are performed:
#' \enumerate{
#'   \item **NVIDIA Driver**: Verifies `nvidia-smi` is available and reports
#'     driver version
#'   \item **CUDA Toolkit**: Checks for `nvcc` and reports CUDA version
#'   \item **libcudf**: Looks for cudf headers in standard paths
#'   \item **R Version**: Checks R >= 4.3.0
#'   \item **R Packages**: Verifies required R packages are installed
#'   \item **GPU Access**: Tests if a GPU device is accessible (if cuplyr
#'     is already installed)
#' }
#'
#' @seealso [verify_installation()] for post-install verification,
#'   [diagnostics()] for full system diagnostics
#'
#' @importFrom cli col_green col_red col_yellow symbol
#' @export
#' @examples
#' # Human-readable output
#' check_deps()
#'
#' # Machine-readable JSON
#' check_deps(format = "json")
#'
#' # Programmatic use
#' result <- check_deps()
#' if (!result$ok) {
#'   message("Some checks failed. See result$checks for details.")
#' }
check_deps <- function(format = c("text", "json"), verbose = FALSE) {
  format <- match.arg(format)

  checks <- list()

  # 1. NVIDIA Driver
  checks$driver <- check_nvidia_driver()

  # 2. CUDA Toolkit
  checks$cuda <- check_cuda_toolkit()

  # 3. libcudf
  checks$cudf <- check_libcudf()

  # 4. R version
  checks$r_version <- check_r_version()

  # 5. Required R packages
  checks$r_packages <- check_r_packages()

  # 6. GPU access (only if cuplyr is loaded)
  checks$gpu <- check_gpu_access()

  # Treat NA checks as "not evaluated" (non-fatal), but fail on explicit FALSE.
  all_ok <- !any(vapply(checks, function(c) identical(c$ok, FALSE), logical(1)))

  result <- list(ok = all_ok, checks = checks)

  if (format == "json") {
    cat(deps_to_json(result), "\n")
  } else {
    print_deps_text(result, verbose = verbose)
  }

  invisible(result)
}


# --- Individual check functions ---

check_nvidia_driver <- function() {
  # Try simple nvidia-smi first (most reliable across versions)
  smi <- tryCatch(
    system2("nvidia-smi", stdout = TRUE, stderr = TRUE),
    error = function(e) NULL
  )

  if (is.null(smi) || !is.null(attr(smi, "status"))) {
    return(list(
      ok = FALSE,
      name = "NVIDIA Driver",
      value = NA_character_,
      message = "nvidia-smi not found or failed. Install NVIDIA driver."
    ))
  }

  # Parse driver version from nvidia-smi output header
  version <- NA_character_
  driver_line <- grep("Driver Version", smi, value = TRUE)
  if (length(driver_line) > 0) {
    m <- regmatches(driver_line, regexpr("Driver Version:\\s*([0-9.]+)", driver_line))
    if (length(m) > 0) {
      version <- sub("Driver Version:\\s*", "", m[1])
    }
  }

  list(
    ok = TRUE,
    name = "NVIDIA Driver",
    value = version,
    message = if (is.na(version)) "nvidia-smi found (version unknown)" else paste0("Driver version ", version)
  )
}

check_cuda_toolkit <- function() {
  # Try CUDA_HOME/bin/nvcc first, then PATH
  cuda_home <- Sys.getenv("CUDA_HOME", "/usr/local/cuda")
  nvcc <- file.path(cuda_home, "bin", "nvcc")

  if (!file.exists(nvcc)) {
    nvcc <- Sys.which("nvcc")
  }

  if (!nzchar(nvcc) || !file.exists(nvcc)) {
    return(list(
      ok = FALSE,
      name = "CUDA Toolkit",
      value = NA_character_,
      message = "nvcc not found. Install CUDA toolkit or set CUDA_HOME."
    ))
  }

  version_out <- tryCatch(
    system2(nvcc, "--version", stdout = TRUE, stderr = TRUE),
    error = function(e) NULL
  )

  version <- NA_character_
  if (!is.null(version_out)) {
    release_line <- grep("release", version_out, value = TRUE)
    if (length(release_line) > 0) {
      m <- regmatches(release_line, regexpr("[0-9]+\\.[0-9]+", release_line))
      if (length(m) > 0) version <- m[1]
    }
  }

  list(
    ok = TRUE,
    name = "CUDA Toolkit",
    value = version,
    message = paste0("CUDA ", version, " at ", dirname(dirname(nvcc)))
  )
}

check_libcudf <- function(search_paths = NULL) {
  if (is.null(search_paths)) {
    conda_prefix <- Sys.getenv("CONDA_PREFIX", "")
    cudf_home <- Sys.getenv("CUDF_HOME", "")

    search_paths <- c(
      if (nzchar(cudf_home)) cudf_home,
      if (nzchar(conda_prefix)) conda_prefix,
      "/usr/local", "/opt/rapids", "/opt/conda", "/usr"
    )
  }

  search_paths <- unique(search_paths[nzchar(search_paths)])
  header_only_paths <- character()

  for (path in search_paths) {
    header <- file.path(path, "include", "cudf", "types.hpp")
    libcudf <- c(
      file.path(path, "lib", "libcudf.so"),
      file.path(path, "lib64", "libcudf.so")
    )

    has_header <- file.exists(header)
    has_library <- any(file.exists(libcudf))

    if (has_header && has_library) {
      return(list(
        ok = TRUE,
        name = "libcudf",
        value = path,
        message = paste0("Found headers and libcudf.so at ", path)
      ))
    }

    if (has_header && !has_library) {
      header_only_paths <- c(header_only_paths, path)
    }
  }

  if (length(header_only_paths) > 0) {
    return(list(
      ok = FALSE,
      name = "libcudf",
      value = header_only_paths[1],
      message = paste0(
        "Found cuDF headers but libcudf.so is missing under: ",
        paste(unique(header_only_paths), collapse = ", "),
        ". Check RAPIDS runtime libraries and CONDA_PREFIX/CUDF_HOME."
      )
    ))
  }

  list(
    ok = FALSE,
    name = "libcudf",
    value = NA_character_,
    message = "libcudf headers/libraries not found. Install RAPIDS or set CUDF_HOME/CONDA_PREFIX."
  )
}

check_r_version <- function() {
  version <- getRversion()
  ok <- version >= "4.3.0"

  list(
    ok = ok,
    name = "R Version",
    value = as.character(version),
    message = if (ok) {
      paste0("R ", version)
    } else {
      paste0("R ", version, " (need >= 4.3.0)")
    }
  )
}

check_r_packages <- function() {
  required <- c("Rcpp", "dplyr", "rlang", "vctrs", "cli", "tidyselect", "tibble")

  missing <- character()
  for (pkg in required) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      missing <- c(missing, pkg)
    }
  }

  if (length(missing) > 0) {
    return(list(
      ok = FALSE,
      name = "R Packages",
      value = paste(missing, collapse = ", "),
      message = paste0("Missing: ", paste(missing, collapse = ", "),
                       ". Run: install.packages(c(",
                       paste0('"', missing, '"', collapse = ", "), "))")
    ))
  }

  list(
    ok = TRUE,
    name = "R Packages",
    value = paste(required, collapse = ", "),
    message = paste0("All ", length(required), " required packages installed")
  )
}

check_gpu_access <- function() {
  if (!requireNamespace("cuplyr", quietly = TRUE)) {
    return(list(
      ok = NA,
      name = "GPU Access",
      value = NA_character_,
      message = "cuplyr not yet installed (skipped)"
    ))
  }

  available <- tryCatch(cuplyr::has_gpu(), error = function(e) FALSE)

  if (available) {
    info <- tryCatch(cuplyr::gpu_details(), error = function(e) list())
    gpu_name <- info$name %||% "unknown"
    return(list(
      ok = TRUE,
      name = "GPU Access",
      value = gpu_name,
      message = paste0("GPU accessible: ", gpu_name)
    ))
  }

  list(
    ok = FALSE,
    name = "GPU Access",
    value = NA_character_,
    message = "GPU not accessible. Check driver and library paths."
  )
}


# --- Output formatters ---

print_deps_text <- function(result, verbose = FALSE) {
  cat("cuplyr dependency check\n")
  cat(strrep("-", 50), "\n")

  for (check in result$checks) {
    if (isTRUE(check$ok)) {
      status <- cli::col_green(cli::symbol$tick)
    } else if (is.na(check$ok)) {
      status <- cli::col_yellow("-")
    } else {
      status <- cli::col_red(cli::symbol$cross)
    }

    cat(status, " ", check$name, ": ", check$message, "\n", sep = "")
  }

  cat(strrep("-", 50), "\n")
  if (result$ok) {
    cat(cli::col_green("All checks passed."), "\n")
    cat("Next step: ./configure && R CMD INSTALL .\n")
  } else {
    cat(cli::col_red("Some checks failed."), " Fix the issues above.\n", sep = "")
  }
}

deps_to_json <- function(result) {
  # Properly escape JSON strings
  escape_json <- function(s) {
    # Order matters: backslash first, then others
    s <- gsub("\\", "\\\\", s, fixed = TRUE)   # \ -> \\
    s <- gsub('"', '\\"', s, fixed = TRUE)     # " -> \"
    s <- gsub("\n", "\\n", s, fixed = TRUE)    # newline -> \n
    s <- gsub("\r", "\\r", s, fixed = TRUE)    # CR -> \r
    s <- gsub("\t", "\\t", s, fixed = TRUE)    # tab -> \t
    s
  }

  checks_json <- vapply(result$checks, function(c) {
    val <- if (is.na(c$value)) "null" else paste0('"', escape_json(c$value), '"')
    ok_str <- if (is.na(c$ok)) "null" else if (c$ok) "true" else "false"
    sprintf('    "%s": {"ok": %s, "value": %s, "message": "%s"}',
            gsub(" ", "_", tolower(c$name)),
            ok_str, val, escape_json(c$message))
  }, character(1))

  paste0('{\n  "ok": ', if (result$ok) "true" else "false",
         ',\n  "checks": {\n',
         paste(checks_json, collapse = ",\n"),
         '\n  }\n}')
}
