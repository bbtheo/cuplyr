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
#'     the driver version, read with `nvidia-smi --query-gpu=driver_version`
#'     and, failing that, from the `Driver Version:` or `KMD Version:` header
#'   \item **CUDA Toolkit**: Checks for `nvcc`, validates CUDA >= 12.2, and
#'     reports CUDA version
#'   \item **libcudf**: Reports the libcudf that cuplyr has loaded, if any.
#'     Otherwise looks for cudf headers and `libcudf.so` under `CUDF_HOME`,
#'     `CONDA_PREFIX`, the pixi environment (`PIXI_PROJECT_ROOT`, or `.pixi`
#'     in the working directory) and standard paths. A `CUDF_HOME` or
#'     `CONDA_PREFIX` that points to a missing directory is named in the
#'     message but does not fail the check by itself.
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

# The one place that shells out to nvidia-smi (tests mock it). NULL when the
# command is missing, errors or exits non-zero.
run_nvidia_smi <- function(args = character()) {
  out <- tryCatch(
    suppressWarnings(system2("nvidia-smi", args, stdout = TRUE, stderr = TRUE)),
    error = function(e) NULL
  )
  if (is.null(out) || !is.null(attr(out, "status"))) {
    return(NULL)
  }
  out
}

# First driver version in nvidia-smi output: a bare version line (from
# --query-gpu csv output), `Driver Version: X` (old header) or
# `KMD Version: X` (new header). NA_character_ when none.
parse_nvidia_smi_driver_version <- function(lines) {
  version <- "[0-9]+(\\.[0-9]+)+"
  bare <- paste0("^\\s*(", version, ")\\s*$")
  labelled <- paste0("(Driver|KMD) Version:\\s*(", version, ")")

  for (line in lines) {
    if (is.na(line)) next
    if (grepl(bare, line)) {
      return(trimws(line))
    }
    m <- regmatches(line, regexpr(labelled, line))
    if (length(m) > 0) {
      return(sub("^.*Version:\\s*", "", m))
    }
  }
  NA_character_
}

check_nvidia_driver <- function() {
  # The query interface is stable across versions; the header format is not.
  queried <- run_nvidia_smi(c("--query-gpu=driver_version", "--format=csv,noheader"))
  version <- if (is.null(queried)) NA_character_ else parse_nvidia_smi_driver_version(queried)

  header <- NULL
  if (is.na(version)) {
    header <- run_nvidia_smi()
    if (!is.null(header)) {
      version <- parse_nvidia_smi_driver_version(header)
    }
  }

  if (is.null(queried) && is.null(header)) {
    return(list(
      ok = FALSE,
      name = "NVIDIA Driver",
      value = NA_character_,
      message = "nvidia-smi not found or failed. Install NVIDIA driver."
    ))
  }

  list(
    ok = TRUE,
    name = "NVIDIA Driver",
    value = version,
    message = if (is.na(version)) "nvidia-smi found (version unknown)" else paste0("Driver version ", version)
  )
}

parse_cuda_release_version <- function(version_out) {
  release_line <- grep("release", version_out, value = TRUE)
  if (length(release_line) == 0) {
    return(NA_character_)
  }

  m <- regmatches(release_line, regexpr("[0-9]+\\.[0-9]+", release_line))
  if (length(m) == 0) {
    return(NA_character_)
  }

  m[1]
}

is_cuda_version_supported <- function(version, required = "12.2") {
  if (is.na(version) || !nzchar(version)) {
    return(FALSE)
  }
  utils::compareVersion(version, required) >= 0
}

check_cuda_toolkit <- function() {
  required <- "12.2"

  # Same order as configure: an explicit CUDA_HOME, the conda/pixi
  # environment, the system toolkit paths, then PATH.
  homes <- c(
    Sys.getenv("CUDA_HOME", ""),
    Sys.getenv("CONDA_PREFIX", ""),
    "/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"
  )
  candidates <- file.path(homes[nzchar(homes)], "bin", "nvcc")
  nvcc <- c(candidates[file.exists(candidates)], unname(Sys.which("nvcc")))[1]

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

  version <- if (!is.null(version_out)) parse_cuda_release_version(version_out) else NA_character_

  if (is.na(version)) {
    return(list(
      ok = FALSE,
      name = "CUDA Toolkit",
      value = NA_character_,
      message = paste0("Could not parse CUDA version from nvcc output (need >= ", required, ").")
    ))
  }

  ok <- is_cuda_version_supported(version, required = required)

  list(
    ok = ok,
    name = "CUDA Toolkit",
    value = version,
    message = if (ok) {
      paste0("CUDA ", version, " at ", dirname(dirname(nvcc)))
    } else {
      paste0("CUDA ", version, " at ", dirname(dirname(nvcc)),
             " (need >= ", required, ")")
    }
  )
}

# The libcudf shared library mapped into this R process (cuplyr.so links it,
# so once the package DLL is loaded this is the library cuplyr really uses).
# NA_character_ when none is mapped or the maps file can't be read.
loaded_libcudf_path <- function(maps_file = "/proc/self/maps") {
  lines <- tryCatch(
    suppressWarnings(readLines(maps_file, warn = FALSE)),
    error = function(e) character()
  )
  # address perms offset dev inode  pathname (the path may contain spaces)
  has_path <- grepl("^\\S+\\s+\\S+\\s+\\S+\\s+\\S+\\s+\\S+\\s+/", lines)
  paths <- sub("^\\S+\\s+\\S+\\s+\\S+\\s+\\S+\\s+\\S+\\s+", "", lines[has_path])
  hits <- paths[grepl("^libcudf\\.so(\\.[0-9]+)*$", basename(paths))]
  if (length(hits) == 0) NA_character_ else hits[1]
}

default_cudf_search_paths <- function() {
  pixi_root <- Sys.getenv("PIXI_PROJECT_ROOT", "")
  pixi_env <- Sys.getenv("PIXI_ENVIRONMENT_NAME", "")
  if (!nzchar(pixi_env)) pixi_env <- "default"

  paths <- c(
    Sys.getenv("CUDF_HOME", ""),
    Sys.getenv("CONDA_PREFIX", ""),
    if (nzchar(pixi_root)) file.path(pixi_root, ".pixi", "envs", pixi_env),
    file.path(getwd(), ".pixi", "envs", "default"),
    "/usr/local", "/opt/rapids", "/opt/conda", "/usr"
  )
  unique(paths[nzchar(paths)])
}

check_libcudf <- function(search_paths = NULL) {
  if (!is.null(search_paths)) {
    return(find_libcudf_in(search_paths))
  }

  notes <- character()
  for (var in c("CUDF_HOME", "CONDA_PREFIX")) {
    value <- Sys.getenv(var, "")
    if (nzchar(value) && !dir.exists(value)) {
      notes <- c(notes, paste0(var, " is set to '", value, "', which does not exist"))
    }
  }

  lib <- loaded_libcudf_path()
  res <- if (!is.na(lib)) {
    list(
      ok = TRUE,
      name = "libcudf",
      value = dirname(dirname(lib)),
      message = paste0("libcudf loaded by cuplyr from ", lib)
    )
  } else {
    find_libcudf_in(default_cudf_search_paths())
  }

  if (length(notes) > 0) {
    res$message <- paste0(res$message, ". Note: ", paste(notes, collapse = "; "), ".")
  }
  res
}

find_libcudf_in <- function(search_paths) {
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
