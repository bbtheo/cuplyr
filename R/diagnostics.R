#' System diagnostics for cuplyr
#'
#' Collects comprehensive system information for debugging installation
#' issues or filing bug reports. Includes cuplyr version, R/Rcpp versions,
#' CUDA/cuDF paths, libcudf versions, GPU info, the GPU memory resource, the
#' cuplyr options in effect and environment variables.
#'
#' @param redact If `TRUE` (default), redact potentially sensitive paths
#'   (home directory, usernames). Set to `FALSE` for local debugging.
#' @param format `"text"` (default) prints a readable report. `"json"` prints
#'   one JSON object instead (and nothing else), for tools and scripts; its
#'   keys are the names of the returned list.
#'
#' @return Invisibly returns a list with all diagnostic information, whichever
#'   `format` is used. Also prints the report (or the JSON object) to the
#'   console. Besides the package, system, CUDA and GPU fields, the list has:
#'   * `libcudf_version` and `rmm_version`: the versions cuplyr was built
#'     against, as `"25.12.0"`.
#'   * `libcudf_path`: the libcudf library loaded into this R session (`NA` if
#'     none), and `libcudf_loaded_version`, its version (`NA` when its headers
#'     can't be found). A difference from `libcudf_version` means cuplyr runs
#'     against another libcudf than it was built with.
#'   * `memory`: a list with `available`, the memory `resource` (`"async"`,
#'     `"pool"` or `"cuda"`), `pool_bytes` and `pool_used_bytes`. See
#'     [gpu_memory_state()].
#'   * `options`: a data frame with one row per cuplyr option, user and
#'     internal: `option`, `value`, `default`, `source`, `env_var`, `internal`
#'     and `description`. `default` is `NA` both for a default computed at run
#'     time (the initial pool size) and for no value at all (no release
#'     threshold); the text report says "computed" or "unset". See
#'     [cuplyr_options()].
#'
#'   A lookup that fails gives `NA` for its field; `diagnostics()` itself does
#'   not fail.
#'
#' @details
#' The output includes:
#' \itemize{
#'   \item **Package versions**: cuplyr, R, Rcpp, dplyr
#'   \item **CUDA info**: toolkit version, driver version, CUDA_HOME path
#'   \item **libcudf info**: CONDA_PREFIX, CUDF_HOME, the libcudf and RMM
#'     versions cuplyr was built against, the libcudf loaded and its version
#'   \item **GPU info**: device name, compute capability, memory
#'   \item **GPU memory resource**: the active resource and its pool size
#'   \item **Options**: every cuplyr option with its value, source and default
#'   \item **Library paths**: LD_LIBRARY_PATH, RPATH of cuplyr.so
#'   \item **Environment**: OS, distro, key environment variables
#' }
#'
#' @seealso [check_deps()] for pre-install checks,
#'   [verify_installation()] for post-install verification,
#'   [cuplyr_options()] for the options
#'
#' @importFrom cli col_green
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
#'
#' # As JSON, for scripts
#' diagnostics(format = "json")
diagnostics <- function(redact = TRUE, format = c("text", "json")) {
  format <- match.arg(format)
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

  # The versions cuplyr was compiled against (no GPU needed). Every lookup
  # below is wrapped: diagnostics() is for broken installs, so it must not fail.
  built <- tryCatch(cuplyr_build_versions(), error = function(e) NULL)
  info$libcudf_version <- diag_chr(built[["libcudf"]])
  info$rmm_version <- diag_chr(built[["rmm"]])

  # The libcudf actually loaded, and the version its headers say it is. A
  # libcudf from another environment (LD_LIBRARY_PATH) shows up as a mismatch.
  info$libcudf_path <- diag_chr(tryCatch(loaded_libcudf_path(), error = function(e) NULL))
  info$libcudf_loaded_version <- tryCatch({
    if (is.na(info$libcudf_path)) {
      NA_character_
    } else {
      prefix <- dirname(dirname(info$libcudf_path))
      read_cudf_version_header(
        file.path(prefix, "include", "cudf", "version_config.hpp")
      )
    }
  }, error = function(e) NA_character_)

  # -- GPU info --
  info$gpu <- tryCatch({
    if (requireNamespace("cuplyr", quietly = TRUE)) {
      gpu_details()
    } else {
      list(available = FALSE, note = "cuplyr not loaded")
    }
  }, error = function(e) list(available = FALSE, error = conditionMessage(e)))

  # -- GPU memory resource --
  info$memory <- diag_memory()

  # -- cuplyr options (user and internal) --
  info$options <- diag_options()

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
      # useBytes: a path with non-UTF-8 bytes must not make gsub() fail.
      # An empty HOME is skipped (gsub() refuses an empty pattern), and so is
      # HOME = "/" (it would turn every "/" into "~").
      if (nzchar(home) && home != "/") {
        x <- gsub(home, "~", x, fixed = TRUE, useBytes = TRUE)
      }
      if (nzchar(user)) x <- gsub(user, "<user>", x, fixed = TRUE, useBytes = TRUE)
      x
    }
    info$cuda_home <- redact_path(info$cuda_home)
    info$conda_prefix <- redact_path(info$conda_prefix)
    info$cudf_home <- redact_path(info$cudf_home)
    info$ld_library_path <- redact_path(info$ld_library_path)
    info$cuplyr_so_path <- redact_path(info$cuplyr_so_path)
    info$cuplyr_runpath <- redact_path(info$cuplyr_runpath)
    info$libcudf_path <- redact_path(info$libcudf_path)
  }

  if (format == "json") {
    cat(to_json(info), "\n", sep = "")
    return(invisible(info))
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
  cat("  libcudf (built against): ", diag_show(info$libcudf_version), "\n", sep = "")
  cat("  libcudf (loaded):        ", diag_show(info$libcudf_loaded_version),
      "  ", diag_show(info$libcudf_path, "(no libcudf loaded)"), "\n", sep = "")
  cat("  RMM (built against):     ", diag_show(info$rmm_version), "\n", sep = "")
  if (!is.na(info$libcudf_version) && !is.na(info$libcudf_loaded_version) &&
      info$libcudf_version != info$libcudf_loaded_version) {
    cat("  WARNING: the loaded libcudf differs from the one cuplyr was built against\n")
  }
  cat("\n")

  cat("GPU\n")
  if (isTRUE(info$gpu$available)) {
    cat("  Available: ", "YES\n")
    cat("  Name:      ", info$gpu$name %||% "unknown", "\n")
    cat("  Compute:   ", info$gpu$compute_capability %||% "unknown", "\n")
    # Use GiB (binary) not GB (SI) to match GPU spec sheets
    total_gib <- round((info$gpu$total_memory %||% 0) / (1024^3), 1)
    free_gib <- round((info$gpu$free_memory %||% 0) / (1024^3), 1)
    cat("  Memory:    ", total_gib, "GiB total, ", free_gib, "GiB free\n")
  } else {
    cat("  Available:  NO\n")
    if (!is.null(info$gpu$error)) {
      cat("  Error:     ", info$gpu$error, "\n")
    }
  }
  cat("\n")

  cat("Memory resource\n")
  mem <- info$memory
  if (isTRUE(mem$available)) {
    cat("  Memory resource: ", diag_show(mem$resource), "\n", sep = "")
    cat("  Pool held:       ", diag_gib(mem$pool_bytes), "\n", sep = "")
    cat("  Pool in use:     ", diag_gib(mem$pool_used_bytes), "\n", sep = "")
  } else {
    cat("  Memory resource: unknown (no GPU memory information)\n")
  }
  cat("\n")

  cat("Options\n")
  opts <- info$options
  for (i in seq_len(nrow(opts))) {
    # An NA default is "computed" when the registry says cuplyr works it out
    # at run time, and "unset" otherwise (no value, e.g. no automatic release).
    entry <- cuplyr_option_registry[[sub("^cuplyr\\.", "", opts$option[i])]]
    none <- if (isTRUE(entry$computed)) "computed" else "unset"
    cat(sprintf("  %s = %s (%s; default %s)%s\n", opts$option[i],
                diag_show(opts$value[i], "<unset>"), opts$source[i],
                diag_show(opts$default[i], none),
                if (isTRUE(opts$internal[i])) " [internal]" else ""))
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

# A character scalar from a lookup result: NA when the lookup gave nothing
# usable.
diag_chr <- function(x) {
  if (is.character(x) && length(x) == 1L) unname(x) else NA_character_
}

# A value for the text report: `none` for NA.
diag_show <- function(x, none = "unknown") {
  if (length(x) != 1L || is.na(x)) none else as.character(x)
}

# Bytes as GiB for the text report, "n/a" when there is no pool.
diag_gib <- function(bytes) {
  if (length(bytes) != 1L || is.na(bytes)) "n/a" else paste(round(bytes / 1024^3, 2), "GiB")
}

# The GPU memory resource, from gpu_memory_state(). When it can't be read
# (`available` is FALSE) the resource is unknown, so NA: gpu_memory_state()
# then reports "cuda", which only means that nothing is installed.
diag_memory <- function() {
  state <- tryCatch(gpu_memory_state(), error = function(e) NULL)
  num <- function(x) if (is.numeric(x) && length(x) == 1L) as.numeric(x) else NA_real_
  available <- isTRUE(state$available)
  list(
    available = available,
    resource = if (available) diag_chr(state$resource) else NA_character_,
    pool_bytes = num(state$pool_bytes),
    pool_used_bytes = num(state$pool_used_bytes)
  )
}

# One row per cuplyr option, user and internal: the value in use and where it
# came from (cuplyr_options()), plus the registry's default and whether the
# option is internal. The default is NA both when it is computed at run time
# (the registry's `computed`) and when it is no value at all.
diag_options <- function() {
  # cuplyr_options() warns about an invalid option and reports its default.
  # With options(warn = 2) that warning would become an error and empty the
  # table. diagnostics() is a report that must not fail, so here the warning is
  # printed (warn = 1) instead of raised.
  #
  # But a warning that is printed counts as given, and cuplyr gives each one
  # only once per session: under warn = 2 every later read would then stay
  # quiet, while ?cuplyr-options promises that each one raises it again until
  # the value is fixed. So, when the caller has warn = 2 or more, the invalid-
  # option warnings seen here are forgotten again on the way out
  # (opt_forget_warned(), as memory_resource_install() and opt_reraise() do).
  # Under warn 0 or 1 nothing is forgotten: the user saw the warning, as with
  # any other read. Only warnings signalled here are collected, so one given
  # before diagnostics() stays given.
  strict <- getOption("warn", 0) >= 2
  seen <- list()
  old <- options(warn = min(getOption("warn", 0), 1))
  on.exit({
    options(old)
    if (strict) opt_forget_warned(seen)
  })
  tryCatch({
    opts <- withCallingHandlers(
      cuplyr_options(internal = TRUE),
      # (collected only, not muffled: the warning still reaches the user)
      cuplyr_warning_invalid_option = function(w) seen[[length(seen) + 1L]] <<- w
    )
    reg <- cuplyr_option_registry[sub("^cuplyr\\.", "", opts$option)]
    data.frame(
      option = opts$option,
      value = opts$value,
      default = vapply(reg, function(e) opt_render(e$default), ""),
      source = opts$source,
      env_var = opts$env_var,
      internal = vapply(reg, function(e) isTRUE(e$internal), NA),
      description = opts$description,
      stringsAsFactors = FALSE, row.names = NULL
    )
  }, error = function(e) {
    data.frame(option = character(), value = character(), default = character(),
               source = character(), env_var = character(), internal = logical(),
               description = character(), stringsAsFactors = FALSE)
  })
}

# The libcudf version in a version_config.hpp, formatted as RAPIDS (and
# configure) write it: "MAJOR.MM.PATCH", e.g. "26.02.0". NA when the file is
# missing or has no CUDF_VERSION_MAJOR and CUDF_VERSION_MINOR. A missing PATCH
# means 0.
read_cudf_version_header <- function(file) {
  if (length(file) != 1L || is.na(file) || !file.exists(file)) {
    return(NA_character_)
  }
  lines <- tryCatch(readLines(file, warn = FALSE), error = function(e) character())
  # "#define CUDF_VERSION_<what> <number>", spaces and tabs allowed anywhere
  find <- function(what) {
    re <- paste0("^\\s*#\\s*define\\s+CUDF_VERSION_", what, "\\s+([0-9]+)")
    hit <- regmatches(lines, regexec(re, lines))
    hit <- Filter(length, hit)
    if (length(hit) == 0L) NA_integer_ else as.integer(hit[[1L]][2L])
  }
  major <- find("MAJOR")
  minor <- find("MINOR")
  patch <- find("PATCH")
  if (is.na(major) || is.na(minor)) {
    return(NA_character_)
  }
  if (is.na(patch)) patch <- 0L
  sprintf("%d.%02d.%d", major, minor, patch)
}

# A small JSON writer, so that diagnostics() needs no JSON package (it has to
# work on a broken install). It handles what diagnostics() holds:
#   NULL, NA, NaN, Inf  -> null
#   logical             -> true / false
#   number              -> up to 15 significant digits
#   string              -> quoted and escaped
#   named list          -> object
#   unnamed list, or a vector that is not length 1 -> array
#   data frame          -> array of one object per row
to_json <- function(x) {
  if (is.null(x)) {
    return("null")
  }
  if (is.data.frame(x)) {
    rows <- lapply(seq_len(nrow(x)), function(i) {
      row <- lapply(x, function(col) col[[i]])
      to_json(row)
    })
    return(paste0("[", paste(unlist(rows), collapse = ","), "]"))
  }
  if (is.list(x)) {
    if (length(x) > 0L && !is.null(names(x)) && all(nzchar(names(x)))) {
      keys <- paste0('"', json_escape(names(x)), '"')
      vals <- vapply(x, to_json, "")
      return(paste0("{", paste0(keys, ":", vals, collapse = ","), "}"))
    }
    return(paste0("[", paste(vapply(x, to_json, ""), collapse = ","), "]"))
  }
  if (is.factor(x)) {
    x <- as.character(x)
  }
  if (length(x) != 1L) {
    return(paste0("[", paste(vapply(as.list(x), to_json, ""), collapse = ","), "]"))
  }
  # one atomic value
  if (is.na(x) || (is.numeric(x) && !is.finite(x))) {
    "null"
  } else if (is.logical(x)) {
    if (x) "true" else "false"
  } else if (is.numeric(x)) {
    sprintf("%.15g", as.numeric(x))
  } else {
    paste0('"', json_escape(as.character(x)), '"')
  }
}
