#' Install cuplyr from source
#'
#' Installs cuplyr by setting up build dependencies (if needed) and
#' running `./configure && R CMD INSTALL .`. Supports multiple methods
#' for providing CUDA/cuDF dependencies.
#'
#' @param method How to obtain build dependencies:
#'   \describe{
#'     \item{`"auto"`}{Try methods in order: pixi, conda, system}
#'     \item{`"pixi"`}{Use pixi to manage a reproducible environment}
#'     \item{`"conda"`}{Use conda/mamba to install RAPIDS into a prefix}
#'     \item{`"system"`}{Assume CUDA and cuDF are already installed on the system}
#'   }
#' @param repo GitHub repository URL or local path to cuplyr source.
#'   If `NULL`, assumes current directory is the cuplyr source tree.
#' @param ref Git branch or tag to install. Only used when `repo` is a URL.
#' @param conda_prefix Path for the conda environment when `method = "conda"`.
#'   Defaults to `"/opt/rapids"` or a temp directory.
#' @param configure_args Extra arguments passed to `./configure`.
#' @param dry_run If `TRUE`, print what would be done without executing.
#' @param verbose If `TRUE`, print detailed progress.
#'
#' @return Invisibly returns `TRUE` on success.
#'
#' @details
#' ## Method details
#'
#' **auto** (default): Tries methods in order of preference:
#' 1. If `pixi` is on PATH and `pixi.toml` exists, use pixi
#' 2. If `mamba` or `conda` is on PATH, use conda
#' 3. If CUDA and cuDF are already available, use system
#'
#' **pixi**: Runs `pixi run install` which handles everything.
#' Requires [pixi](https://pixi.sh) to be installed.
#'
#' **conda**: Creates a conda environment with RAPIDS packages, then
#' configures and builds cuplyr against it. Requires `mamba` or `conda`.
#'
#' **system**: Assumes `CUDA_HOME` and `CONDA_PREFIX`/`CUDF_HOME` are
#' already set. Runs `./configure && R CMD INSTALL .` directly.
#'
#' ## Pre-flight
#'
#' Before building, `install_cuplyr()` runs [check_deps()] and reports
#' any missing dependencies. With `method = "conda"`, missing RAPIDS
#' packages are installed automatically.
#'
#' @seealso [check_deps()] for dependency checking,
#'   [verify_installation()] for post-install verification,
#'   [diagnostics()] for troubleshooting
#'
#' @export
#' @examples
#' \dontrun{
#' # Auto-detect best method
#' install_cuplyr()
#'
#' # Use conda explicitly
#' install_cuplyr(method = "conda")
#'
#' # System install (CUDA/cuDF already available)
#' install_cuplyr(method = "system")
#'
#' # Preview what would happen
#' install_cuplyr(dry_run = TRUE)
#'
#' # Install from GitHub
#' install_cuplyr(repo = "https://github.com/bbtheo/cuplyr.git")
#' }
install_cuplyr <- function(
  method = c("auto", "pixi", "conda", "system"),
  repo = NULL,
  ref = "master",
  conda_prefix = NULL,
  configure_args = character(),
  dry_run = FALSE,
  verbose = FALSE
) {
  method <- match.arg(method)

  if (method == "auto") {
    method <- detect_install_method(verbose = verbose)
    message("Selected install method: ", method)
  }

  # If repo is a URL, clone it first
  src_dir <- get_source_dir(repo, ref, dry_run = dry_run, verbose = verbose)

  switch(method,
    pixi = install_via_pixi(src_dir, dry_run = dry_run, verbose = verbose),
    conda = install_via_conda(src_dir, conda_prefix = conda_prefix,
                               configure_args = configure_args,
                               dry_run = dry_run, verbose = verbose),
    system = install_via_system(src_dir, configure_args = configure_args,
                                 dry_run = dry_run, verbose = verbose)
  )

  if (!dry_run) {
    message("")
    message("Install complete. Run cuplyr::verify_installation() to verify.")
  }

  invisible(TRUE)
}


# =============================================================================
# Method detection
# =============================================================================

detect_install_method <- function(verbose = FALSE) {
  # 1. pixi available + pixi.toml exists?
  if (has_command("pixi") && file.exists("pixi.toml")) {
    if (verbose) message("Found pixi + pixi.toml")
    return("pixi")
  }

  # 2. conda/mamba available?
  if (has_command("mamba") || has_command("conda")) {
    if (verbose) message("Found mamba/conda")
    return("conda")
  }

  # 3. System dependencies present?
  # Suppress console output from check_deps (we only need the result)
  deps <- suppressMessages(capture.output(
    check_deps(format = "text", verbose = FALSE),
    type = "output"
  ))
  deps <- check_deps(format = "text", verbose = FALSE)
  if (isTRUE(deps$checks$cuda$ok) && isTRUE(deps$checks$cudf$ok)) {
    if (verbose) message("CUDA and cuDF found on system")
    return("system")
  }

  stop(
    "Could not auto-detect install method.\n",
    "No pixi, conda/mamba, or system CUDA+cuDF found.\n\n",
    "Options:\n",
    "  1. Install pixi: curl -fsSL https://pixi.sh/install.sh | bash\n",
    "  2. Install mamba: https://mamba.readthedocs.io/\n",
    "  3. Install CUDA + cuDF manually, then: install_cuplyr(method = 'system')\n",
    call. = FALSE
  )
}


# =============================================================================
# Source directory handling
# =============================================================================

get_source_dir <- function(repo, ref, dry_run, verbose) {
  if (is.null(repo)) {
    is_cuplyr_source <- function(path) {
      desc <- file.path(path, "DESCRIPTION")
      if (!file.exists(desc)) {
        return(FALSE)
      }

      pkg <- tryCatch(unname(as.character(read.dcf(desc, fields = "Package")[1, 1])),
                      error = function(e) NA_character_)
      if (!identical(pkg, "cuplyr")) {
        return(FALSE)
      }

      file.exists(file.path(path, "configure"))
    }

    # Search current dir, direct children (common in Colab), then parents.
    child_dirs <- tryCatch(
      list.dirs(".", recursive = FALSE, full.names = FALSE),
      error = function(e) character()
    )
    candidates <- unique(c(".", "cuplyr", child_dirs, "..", "../.."))

    for (candidate in candidates) {
      if (is_cuplyr_source(candidate)) {
        return(normalizePath(candidate))
      }
    }

    stop(
      "No cuplyr source tree found in current directory.\n",
      "Either cd to the cuplyr directory or pass repo = 'https://github.com/bbtheo/cuplyr.git'",
      call. = FALSE
    )
  }

  # Clone from URL
  if (grepl("^https?://|^git@", repo)) {
    tmp_dir <- file.path(tempdir(), "cuplyr-install")
    if (dry_run) {
      message("[dry-run] Would clone ", repo, " (ref: ", ref, ") to ", tmp_dir)
      return(tmp_dir)
    }
    if (dir.exists(tmp_dir)) unlink(tmp_dir, recursive = TRUE)
    message("Cloning ", repo, " ...")
    status <- system2("git", c("clone", "--depth", "1", "-b", ref, repo, tmp_dir),
                       stdout = if (verbose) "" else FALSE,
                       stderr = if (verbose) "" else FALSE)
    if (status != 0) stop("git clone failed", call. = FALSE)
    return(tmp_dir)
  }

  # Local path
  normalizePath(repo, mustWork = TRUE)
}


# =============================================================================
# Install methods
# =============================================================================

install_via_pixi <- function(src_dir, dry_run, verbose) {
  message("Installing via pixi...")

  if (dry_run) {
    message("[dry-run] Would run: pixi run install  (in ", src_dir, ")")
    return(invisible(TRUE))
  }

  status <- run_in_dir(src_dir, "pixi", c("run", "install"),
                        verbose = verbose)
  if (status != 0) {
    stop(
      "pixi run install failed (exit code ", status, ").\n",
      "Run 'pixi run configure' for details, or try: install_cuplyr(method = 'conda')",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

install_via_conda <- function(src_dir, conda_prefix, configure_args,
                               dry_run, verbose) {
  env <- detect_environment()
  message("Installing via conda... (environment: ", env, ")")

  conda_cmd <- if (has_command("mamba")) "mamba" else "conda"

  # Determine prefix
  if (is.null(conda_prefix)) {
    conda_prefix <- Sys.getenv("CONDA_PREFIX", "")
    if (!nzchar(conda_prefix)) {
      conda_prefix <- if (env %in% c("colab", "cloud_gpu")) "/opt/rapids"
                      else file.path(tempdir(), "cuplyr-rapids")
    }
  }

  # Cloud: disable CUDA stubs before anything else
  if (env %in% c("colab", "cloud_gpu")) {
    message("Disabling CUDA stubs...")
    disable_cuda_stubs(file.path(conda_prefix, "lib"))
  }

  prefix_has_cudf <- function(prefix) {
    header <- file.path(prefix, "include", "cudf", "types.hpp")
    lib <- c(
      file.path(prefix, "lib", "libcudf.so"),
      file.path(prefix, "lib64", "libcudf.so")
    )
    file.exists(header) && any(file.exists(lib))
  }

  # Check if RAPIDS already in this prefix
  has_cudf <- prefix_has_cudf(conda_prefix)

  if (!has_cudf) {
    message("Installing RAPIDS packages into ", conda_prefix, " ...")
    base_create_args <- c(
      "create", "-y", "-p", conda_prefix,
      "-c", "rapidsai", "-c", "conda-forge", "-c", "nvidia"
    )
    create_arg_sets <- list(
      c(base_create_args, "libcudf=25.12", "librmm=25.12", "libkvikio=25.12", "spdlog", "fmt"),
      c(base_create_args, "libcudf", "librmm", "libkvikio", "spdlog", "fmt")
    )

    base_install_args <- c(
      "install", "-y", "-p", conda_prefix,
      "-c", "rapidsai", "-c", "conda-forge", "-c", "nvidia"
    )
    install_arg_sets <- list(
      c(base_install_args, "libcudf-dev=25.12", "librmm-dev=25.12", "libkvikio-dev=25.12"),
      c(base_install_args, "libcudf-dev", "librmm-dev", "libkvikio-dev")
    )

    if (dry_run) {
      message("[dry-run] Would run: ", conda_cmd, " ", paste(create_arg_sets[[1]], collapse = " "))
      message("[dry-run] Fallback if solve fails: ", conda_cmd, " ",
              paste(create_arg_sets[[2]], collapse = " "))
      message("[dry-run] If headers are missing, would also run: ", conda_cmd, " ",
              paste(install_arg_sets[[1]], collapse = " "))
    } else {
      solved <- FALSE
      last_status <- NA_integer_
      last_args <- create_arg_sets[[length(create_arg_sets)]]

      for (i in seq_along(create_arg_sets)) {
        conda_args <- create_arg_sets[[i]]
        if (i > 1) {
          message("Retrying conda solve with relaxed RAPIDS constraints...")
        }

        status <- system2(conda_cmd, conda_args,
                          stdout = if (verbose) "" else FALSE,
                          stderr = if (verbose) "" else FALSE)
        last_status <- as.integer(status)
        last_args <- conda_args

        if (!isTRUE(last_status == 0L)) {
          next
        }

        if (prefix_has_cudf(conda_prefix)) {
          solved <- TRUE
          break
        }

        message("Conda solve succeeded but libcudf headers/libraries were not found; trying dev packages...")
        for (install_args in install_arg_sets) {
          status <- system2(conda_cmd, install_args,
                            stdout = if (verbose) "" else FALSE,
                            stderr = if (verbose) "" else FALSE)
          last_status <- as.integer(status)
          last_args <- install_args

          if (isTRUE(last_status == 0L) && prefix_has_cudf(conda_prefix)) {
            solved <- TRUE
            break
          }
        }
        if (solved) break
      }

      if (!solved) {
        stop(
          conda_cmd, " dependency setup did not produce usable libcudf headers/libraries",
          " under ", conda_prefix, " (last exit code ", last_status, ").\n",
          "Expected files:\n",
          "  ", file.path(conda_prefix, "include", "cudf", "types.hpp"), "\n",
          "  ", file.path(conda_prefix, "lib", "libcudf.so"), " (or lib64)\n",
          "Last command:\n  ",
          conda_cmd, " ", paste(last_args, collapse = " "),
          call. = FALSE
        )
      }

      # Disable stubs again (conda may have re-created them)
      if (env %in% c("colab", "cloud_gpu")) {
        disable_cuda_stubs(file.path(conda_prefix, "lib"))
      }
    }
  } else {
    message("RAPIDS packages already found at ", conda_prefix)
  }

  # Configure and build
  message("Configuring and building cuplyr...")

  if (dry_run) {
    message("[dry-run] CONDA_PREFIX=", conda_prefix, " ./configure")
    message("[dry-run] R CMD INSTALL .")
    return(invisible(TRUE))
  }

  # Set env vars for configure
  old_conda <- Sys.getenv("CONDA_PREFIX", NA)
  Sys.setenv(CONDA_PREFIX = conda_prefix)
  on.exit({
    if (is.na(old_conda)) Sys.unsetenv("CONDA_PREFIX")
    else Sys.setenv(CONDA_PREFIX = old_conda)
  }, add = TRUE)

  # Cloud: set up driver paths before configure
  driver_lib <- NULL
  if (env %in% c("colab", "cloud_gpu")) {
    driver_lib <- find_real_driver_lib()
    if (is.null(driver_lib)) {
      stop(
        "Could not find real NVIDIA driver (libcuda.so.1).\n",
        "Ensure the GPU runtime is attached.",
        call. = FALSE
      )
    }
    message("Real driver found at: ", driver_lib)
    configure_cloud_library_paths(driver_lib, conda_prefix)

    # Register paths via ldconfig so library() can find them after process startup
    register_library_paths(c(
      file.path(conda_prefix, "lib"),
      driver_lib
    ))
  }

  # Configure
  status <- run_in_dir(src_dir, "./configure", configure_args, verbose = verbose)
  if (status != 0) {
    stop(
      "./configure failed (exit code ", status, ").\n",
      "Check that CUDA and cuDF are visible. Run: check_deps()",
      call. = FALSE
    )
  }

  # Cloud: patch Makevars to add conda lib and driver to RUNPATH
  if (env %in% c("colab", "cloud_gpu") && !is.null(driver_lib)) {
    patch_makevars_for_cloud(src_dir, driver_lib)
    # Clean stale build artifacts to force relink with new RUNPATH
    src_files <- list.files(file.path(src_dir, "src"),
                            pattern = "\\.(o|so)$", full.names = TRUE)
    if (length(src_files) > 0) {
      unlink(src_files)
      message("Cleaned ", length(src_files), " stale build artifacts")
    }
  }

  # Build
  status <- run_in_dir(src_dir, "R", c("CMD", "INSTALL", "."), verbose = verbose)
  if (status != 0) {
    stop(
      "R CMD INSTALL failed (exit code ", status, ").\n",
      "Check the build log above for errors.",
      call. = FALSE
    )
  }

  if (env %in% c("colab", "cloud_gpu")) {
    validate_cloud_loader_resolution(conda_prefix, verbose = verbose)
  }

  invisible(TRUE)
}

install_via_system <- function(src_dir, configure_args, dry_run, verbose) {
  env <- detect_environment()
  conda_prefix <- Sys.getenv("CONDA_PREFIX", "/opt/rapids")
  message("Installing from system dependencies... (environment: ", env, ")")

  # Pre-flight check
  deps <- check_deps(format = "text", verbose = FALSE)
  if (!isTRUE(deps$checks$cuda$ok)) {
    stop(
      "CUDA toolkit not found on system.\n",
      "Set CUDA_HOME or install CUDA, then retry.\n",
      "Or use: install_cuplyr(method = 'conda') to auto-install dependencies.",
      call. = FALSE
    )
  }
  if (!isTRUE(deps$checks$cudf$ok)) {
    stop(
      "libcudf not found on system.\n",
      "Set CONDA_PREFIX or CUDF_HOME, then retry.\n",
      "Or use: install_cuplyr(method = 'conda') to auto-install dependencies.",
      call. = FALSE
    )
  }

  # Cloud: disable stubs and configure paths
  driver_lib <- NULL
  if (env %in% c("colab", "cloud_gpu")) {
    disable_cuda_stubs(file.path(conda_prefix, "lib"))
    driver_lib <- find_real_driver_lib()
    if (!is.null(driver_lib)) {
      configure_cloud_library_paths(driver_lib, conda_prefix)

      # Register paths via ldconfig so library() can find them after process startup
      register_library_paths(c(
        file.path(conda_prefix, "lib"),
        driver_lib
      ))
    }
  }

  if (dry_run) {
    message("[dry-run] Would run: ./configure ", paste(configure_args, collapse = " "))
    message("[dry-run] Would run: R CMD INSTALL .")
    return(invisible(TRUE))
  }

  # Configure
  status <- run_in_dir(src_dir, "./configure", configure_args, verbose = verbose)
  if (status != 0) {
    stop("./configure failed (exit code ", status, ").", call. = FALSE)
  }

  # Cloud: patch Makevars
  if (env %in% c("colab", "cloud_gpu") && !is.null(driver_lib)) {
    patch_makevars_for_cloud(src_dir, driver_lib)
    # Clean stale build artifacts to force relink with new RUNPATH
    src_files <- list.files(file.path(src_dir, "src"),
                            pattern = "\\.(o|so)$", full.names = TRUE)
    if (length(src_files) > 0) {
      unlink(src_files)
      message("Cleaned ", length(src_files), " stale build artifacts")
    }
  }

  # Build
  status <- run_in_dir(src_dir, "R", c("CMD", "INSTALL", "."), verbose = verbose)
  if (status != 0) {
    stop("R CMD INSTALL failed (exit code ", status, ").", call. = FALSE)
  }

  if (env %in% c("colab", "cloud_gpu")) {
    validate_cloud_loader_resolution(conda_prefix, verbose = verbose)
  }

  invisible(TRUE)
}


# =============================================================================
# Environment detection
# =============================================================================

#' Detect the runtime environment
#'
#' Identifies whether we're running on Colab, in a container, on a cloud GPU
#' instance, or locally. Used by [install_cuplyr()] to apply environment-specific
#' fixes (e.g. CUDA stub disabling on Colab).
#'
#' @param override Explicit override, e.g. from `CUPLYR_ENV` env var.
#' @return One of `"local"`, `"colab"`, `"container"`, `"cloud_gpu"`.
#'
#' @export
detect_environment <- function(override = Sys.getenv("CUPLYR_ENV")) {
  if (nzchar(override)) {
    valid <- c("local", "colab", "container", "cloud_gpu")
    if (override %in% valid) return(override)
    warning("Invalid CUPLYR_ENV='", override, "', ignoring. Valid: ",
            paste(valid, collapse = ", "))
  }

  # Colab (require 2+ signals to avoid false positives). Check before generic
  # container markers because Colab itself runs in containers.
  colab_signals <- c(
    file.exists("/content"),
    nzchar(Sys.getenv("COLAB_RELEASE_TAG")),
    nzchar(Sys.getenv("COLAB_GPU")),
    file.exists("/usr/local/share/jupyter/kernels/ir")
  )
  if (sum(colab_signals) >= 2) {
    return("colab")
  }

  # Container
  if (file.exists("/.dockerenv") || nzchar(Sys.getenv("KUBERNETES_SERVICE_HOST"))) {
    return("container")
  }

  # Generic cloud GPU: nvidia-smi works but driver in non-standard path
  if (has_command("nvidia-smi")) {
    nonstandard <- any(file.exists(c(
      "/usr/lib64-nvidia/libcuda.so.1",
      "/usr/lib/nvidia/libcuda.so.1"
    )))
    standard <- file.exists("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if (nonstandard && !standard) {
      return("cloud_gpu")
    }
  }

  "local"
}


# =============================================================================
# Cloud / Colab helpers
# =============================================================================

#' Find the real NVIDIA driver library path (not stubs)
#' @return Path to directory containing real libcuda.so.1, or NULL.
#' @keywords internal
find_real_driver_lib <- function() {
  # Preferred known locations
  for (p in c("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu")) {
    so <- file.path(p, "libcuda.so.1")
    if (file.exists(so)) {
      size <- file.info(so)$size
      # Real driver is 30MB+, stubs are <1MB
      if (!is.na(size) && size >= 1000000) return(p)
    }
  }

  # Try ldconfig
  ld_lines <- tryCatch(
    system2("ldconfig", "-p", stdout = TRUE, stderr = TRUE),
    error = function(e) character()
  )
  hits <- grep("libcuda\\.so\\.1", ld_lines, value = TRUE)
  hits <- sub(".* => ", "", hits)
  hits <- hits[nzchar(hits) & file.exists(hits)]
  hits <- hits[!grepl("/compat|/stubs|/opt/rapids|/opt/conda", hits)]
  if (length(hits) > 0) {
    size <- file.info(hits[1])$size
    if (!is.na(size) && size >= 1000000) return(dirname(hits[1]))
  }

  NULL
}

#' Disable CUDA stub libraries in a directory
#' @param lib_dir Path to search for stubs.
#' @return Character vector of disabled stub paths (invisible).
#' @keywords internal
disable_cuda_stubs <- function(lib_dir) {
  disabled <- character()
  for (stub_name in c("libcuda.so", "libcuda.so.1")) {
    for (loc in c(file.path(lib_dir, "stubs"), lib_dir)) {
      stub_path <- file.path(loc, stub_name)
      disabled_path <- paste0(stub_path, ".disabled")
      if (!file.exists(stub_path) || file.exists(disabled_path)) next
      info <- file.info(stub_path)
      if (is.na(info$size) || info$size >= 1000000) next
      ok <- tryCatch(file.rename(stub_path, disabled_path), error = function(e) FALSE)
      if (isTRUE(ok)) {
        message("  Disabled stub: ", stub_path)
        disabled <- c(disabled, stub_path)
      }
    }
  }
  invisible(disabled)
}

#' Configure LD_LIBRARY_PATH for cloud environments
#'
#' Sets LD_LIBRARY_PATH with conda lib FIRST (for newer libstdc++),
#' then driver path (for libcuda.so.1), then CUDA runtime.
#'
#' @param driver_lib Path to directory with real libcuda.so.1.
#' @param conda_prefix RAPIDS/conda prefix.
#' @keywords internal
configure_cloud_library_paths <- function(driver_lib, conda_prefix) {
  cuda_home <- Sys.getenv("CUDA_HOME", "/usr/local/cuda")
  current <- Sys.getenv("LD_LIBRARY_PATH")
  # CRITICAL ORDER: conda lib FIRST to find newer libstdc++
  parts <- unique(c(
    file.path(conda_prefix, "lib"),
    driver_lib,
    file.path(cuda_home, "lib64"),
    strsplit(current, ":", fixed = TRUE)[[1]]
  ))
  parts <- parts[nzchar(parts)]
  lib_path <- paste(parts, collapse = ":")
  Sys.setenv(LD_LIBRARY_PATH = lib_path)
  # R CMD INSTALL test-load subprocess respects R_LD_LIBRARY_PATH
  Sys.setenv(R_LD_LIBRARY_PATH = lib_path)
}

#' Register library paths via ldconfig for runtime dlopen()
#'
#' Writes library paths to /etc/ld.so.conf.d/00-cuplyr-rapids.conf and runs ldconfig.
#' This ensures the dynamic linker finds conda's newer libstdc++ even when
#' LD_LIBRARY_PATH changes after process startup don't affect dlopen().
#'
#' @param paths Character vector of library directories to register.
#' @return Logical: TRUE if ldconfig succeeded, FALSE otherwise.
#' @keywords internal
register_library_paths <- function(paths) {
  conf <- "/etc/ld.so.conf.d/00-cuplyr-rapids.conf"
  paths <- paths[nzchar(paths) & dir.exists(paths)]
  if (length(paths) == 0) return(invisible(FALSE))

  ok <- tryCatch({
    writeLines(paths, conf)
    status <- system2("ldconfig", stdout = FALSE, stderr = FALSE)
    if (is.null(status)) status <- 0L
    status == 0L
  }, error = function(e) {
    message("Warning: Could not run ldconfig (may need sudo): ", conditionMessage(e))
    FALSE
  })

  if (ok) {
    message("Registered ", length(paths), " library path(s) via ldconfig")
  }
  invisible(ok)
}

#' Patch Makevars for Colab/cloud: add conda lib and driver to RUNPATH
#' @param src_dir Source directory containing src/Makevars.
#' @param driver_lib Real driver library path.
#' @keywords internal
patch_makevars_for_cloud <- function(src_dir, driver_lib) {
  makevars <- file.path(src_dir, "src", "Makevars")
  if (!file.exists(makevars)) return(invisible(FALSE))

  mk <- readLines(makevars, warn = FALSE)

  # Add RMM compat flags if missing
  if (!any(grepl("DRMM_ENABLE_LEGACY", mk))) {
    mk <- sub(
      "^PKG_CPPFLAGS=(.*)$",
      "PKG_CPPFLAGS=-DRMM_ENABLE_LEGACY_MR_INTERFACE -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \\1",
      mk
    )
  }

  # Prepend conda lib (for libstdc++) and driver lib to RUNPATH
  pkg_libs_idx <- grep("^PKG_LIBS=", mk)
  if (length(pkg_libs_idx) > 0) {
    conda_prefix <- Sys.getenv("CONDA_PREFIX", "")
    conda_lib <- if (nzchar(conda_prefix)) file.path(conda_prefix, "lib") else ""

    # Extract existing flags (everything after PKG_LIBS=)
    existing <- sub("^PKG_LIBS=", "", mk[pkg_libs_idx[1]])

    # Prepend conda lib FIRST (libstdc++), then driver lib (libcuda.so.1)
    if (nzchar(conda_lib)) {
      mk[pkg_libs_idx[1]] <- sprintf(
        "PKG_LIBS=-Wl,--enable-new-dtags -Wl,-rpath,%s -Wl,-rpath,%s %s",
        conda_lib, driver_lib, existing
      )
    } else {
      mk[pkg_libs_idx[1]] <- sprintf(
        "PKG_LIBS=-Wl,--enable-new-dtags -Wl,-rpath,%s %s",
        driver_lib, existing
      )
    }
  }

  writeLines(mk, makevars)
  message("Patched src/Makevars: RUNPATH prepended with conda lib and driver")
  invisible(TRUE)
}

#' Find installed cuplyr shared object path
#' @return Path to installed cuplyr shared object, or NULL if not found.
#' @keywords internal
find_installed_cuplyr_so <- function() {
  ext <- .Platform$dynlib.ext
  pkg_dir <- tryCatch(system.file(package = "cuplyr"), error = function(e) "")
  if (nzchar(pkg_dir)) {
    so <- file.path(pkg_dir, "libs", paste0("cuplyr", ext))
    if (file.exists(so)) {
      return(so)
    }
  }

  candidates <- file.path(.libPaths(), "cuplyr", "libs", paste0("cuplyr", ext))
  candidates <- candidates[file.exists(candidates)]
  if (length(candidates) > 0) {
    return(candidates[[1]])
  }

  NULL
}

#' Validate cloud runtime loader resolution after install
#'
#' Confirms cuplyr.so RUNPATH includes conda lib and that ldd resolves
#' libstdc++.so.6 from the conda prefix (not system path).
#'
#' @param conda_prefix RAPIDS/conda prefix.
#' @param verbose Logical; print pass details.
#' @keywords internal
validate_cloud_loader_resolution <- function(conda_prefix, verbose = FALSE) {
  conda_lib <- file.path(conda_prefix, "lib")
  if (!dir.exists(conda_lib)) {
    stop(
      "Post-build loader validation failed: conda lib directory not found: ",
      conda_lib,
      call. = FALSE
    )
  }

  if (!has_command("readelf") || !has_command("ldd")) {
    stop(
      "Post-build loader validation failed: missing required tools (`readelf`, `ldd`).\n",
      "Install binutils/libc-bin and retry installation.",
      call. = FALSE
    )
  }

  so_path <- find_installed_cuplyr_so()
  if (is.null(so_path)) {
    stop(
      "Post-build loader validation failed: could not locate installed cuplyr shared library.\n",
      "Expected under: ", paste(.libPaths(), collapse = ", "),
      call. = FALSE
    )
  }

  readelf_out <- system2("readelf", c("-d", so_path), stdout = TRUE, stderr = TRUE)
  readelf_status <- attr(readelf_out, "status")
  if (is.null(readelf_status)) readelf_status <- 0L
  if (readelf_status != 0L) {
    stop(
      "Post-build loader validation failed: `readelf -d` errored for ", so_path, ".\n",
      paste(utils::tail(readelf_out, 20), collapse = "\n"),
      call. = FALSE
    )
  }

  runpath_lines <- grep("RPATH|RUNPATH", readelf_out, value = TRUE)
  if (length(runpath_lines) == 0L) {
    stop(
      "Post-build loader validation failed: no RPATH/RUNPATH found in ", so_path, ".\n",
      paste(utils::tail(readelf_out, 20), collapse = "\n"),
      call. = FALSE
    )
  }
  if (!any(grepl(conda_lib, runpath_lines, fixed = TRUE))) {
    stop(
      "Post-build loader validation failed: cuplyr RUNPATH does not include conda lib path.\n",
      "Expected to include: ", conda_lib, "\n",
      "Found:\n", paste(runpath_lines, collapse = "\n"),
      call. = FALSE
    )
  }

  ldd_out <- system2("ldd", so_path, stdout = TRUE, stderr = TRUE)
  ldd_status <- attr(ldd_out, "status")
  if (is.null(ldd_status)) ldd_status <- 0L
  if (ldd_status != 0L) {
    stop(
      "Post-build loader validation failed: `ldd` errored for ", so_path, ".\n",
      paste(utils::tail(ldd_out, 20), collapse = "\n"),
      call. = FALSE
    )
  }

  stdc_line <- grep("libstdc\\+\\+\\.so\\.6", ldd_out, value = TRUE)
  if (length(stdc_line) == 0L) {
    stop(
      "Post-build loader validation failed: libstdc++.so.6 not listed by ldd.\n",
      paste(ldd_out, collapse = "\n"),
      call. = FALSE
    )
  }
  stdc_line <- stdc_line[[1]]
  if (grepl("not found", stdc_line, fixed = TRUE)) {
    stop(
      "Post-build loader validation failed: libstdc++.so.6 unresolved.\n",
      stdc_line,
      call. = FALSE
    )
  }

  stdc_path <- sub(".*=>\\s*([^[:space:]]+).*", "\\1", stdc_line, perl = TRUE)
  if (!nzchar(stdc_path) || identical(stdc_path, stdc_line)) {
    fields <- strsplit(trimws(stdc_line), "\\s+")[[1]]
    stdc_path <- if (length(fields) >= 1) fields[[1]] else ""
  }

  stdc_path_norm <- normalizePath(stdc_path, winslash = "/", mustWork = FALSE)
  conda_prefix_norm <- normalizePath(conda_prefix, winslash = "/", mustWork = FALSE)
  if (!startsWith(stdc_path_norm, paste0(conda_prefix_norm, "/"))) {
    stop(
      "Post-build loader validation failed: libstdc++.so.6 resolves outside conda prefix.\n",
      "Resolved: ", stdc_path_norm, "\n",
      "Expected prefix: ", conda_prefix_norm, "\n",
      "ldd line: ", stdc_line,
      call. = FALSE
    )
  }

  if (verbose) {
    message("Post-build loader validation passed")
    message("  cuplyr.so: ", so_path)
    message("  libstdc++.so.6: ", stdc_path_norm)
    message("  RUNPATH entries:\n", paste(runpath_lines, collapse = "\n"))
  }

  invisible(TRUE)
}


# =============================================================================
# Diagnostic helpers
# =============================================================================

#' Diagnose dynamic linker configuration for cuplyr
#'
#' Prints diagnostic information about how the cuplyr shared library will
#' resolve its dependencies (libstdc++, libcudf, libcuda). Useful when
#' `library(cuplyr)` fails with missing symbol errors.
#'
#' @param verbose If TRUE, print full ldd output.
#' @return Invisibly returns a list with diagnostic results.
#'
#' @export
#' @examples
#' \dontrun{
#' diagnose_loader()
#' }
diagnose_loader <- function(verbose = FALSE) {
  cat("=== cuplyr Dynamic Linker Diagnostics ===\n\n")

  # Find cuplyr.so
  so_path <- find_installed_cuplyr_so()
  if (is.null(so_path)) {
    cat("✗ cuplyr shared library not found\n")
    cat("  Expected under: ", paste(.libPaths(), collapse = ", "), "\n")
    cat("  Run install_cuplyr() first.\n")
    return(invisible(list(found = FALSE)))
  }
  cat("✓ cuplyr.so found:\n")
  cat("  ", so_path, "\n\n")

  # Check RUNPATH/RPATH
  if (has_command("readelf")) {
    cat("RUNPATH/RPATH:\n")
    runpath <- system2("readelf", c("-d", so_path), stdout = TRUE, stderr = TRUE)
    runpath_lines <- grep("RPATH|RUNPATH", runpath, value = TRUE)
    if (length(runpath_lines) > 0) {
      cat("  ", paste(runpath_lines, collapse = "\n  "), "\n")
    } else {
      cat("  (none set)\n")
    }
    cat("\n")
  }

  # Check ldd resolution
  if (has_command("ldd")) {
    cat("Dependency resolution (ldd):\n")
    ldd_out <- system2("ldd", so_path, stdout = TRUE, stderr = TRUE)

    # Key libraries
    for (lib in c("libstdc\\+\\+\\.so\\.6", "libcudf\\.so", "libcuda\\.so\\.1", "libcudart\\.so")) {
      lib_line <- grep(lib, ldd_out, value = TRUE)
      if (length(lib_line) > 0) {
        cat("  ", lib_line[1], "\n")
      }
    }

    if (verbose) {
      cat("\nFull ldd output:\n")
      cat("  ", paste(ldd_out, collapse = "\n  "), "\n")
    }
    cat("\n")

    # Check for issues
    unresolved <- grep("not found", ldd_out, value = TRUE)
    if (length(unresolved) > 0) {
      cat("✗ Unresolved dependencies:\n")
      cat("  ", paste(unresolved, collapse = "\n  "), "\n\n")
    }
  }

  # Check conda libstdc++ for GLIBCXX_3.4.31
  conda_prefix <- Sys.getenv("CONDA_PREFIX", "/opt/rapids")
  conda_stdc <- file.path(conda_prefix, "lib", "libstdc++.so.6")
  if (file.exists(conda_stdc) && has_command("strings")) {
    cat("RAPIDS libstdc++ version check:\n")
    cat("  Path: ", conda_stdc, "\n")
    glibcxx <- system2("strings", conda_stdc, stdout = TRUE, stderr = TRUE)
    has_3_4_31 <- any(grepl("GLIBCXX_3\\.4\\.31", glibcxx))
    if (has_3_4_31) {
      cat("  ✓ Contains GLIBCXX_3.4.31\n")
    } else {
      cat("  ✗ GLIBCXX_3.4.31 NOT found (RAPIDS may be outdated)\n")
    }
    cat("\n")
  }

  # Check ldconfig cache
  if (has_command("ldconfig")) {
    cat("System linker cache (ldconfig -p | grep libstdc++):\n")
    cache <- system2("ldconfig", "-p", stdout = TRUE, stderr = TRUE)
    stdc_lines <- grep("libstdc\\+\\+\\.so\\.6", cache, value = TRUE)
    if (length(stdc_lines) > 0) {
      cat("  ", paste(head(stdc_lines, 5), collapse = "\n  "), "\n")
    } else {
      cat("  (not found in cache)\n")
    }
    cat("\n")
  }

  # Current environment
  cat("Current environment:\n")
  cat("  LD_LIBRARY_PATH (first 3 entries):\n")
  ld_path <- Sys.getenv("LD_LIBRARY_PATH")
  if (nzchar(ld_path)) {
    ld_parts <- strsplit(ld_path, ":", fixed = TRUE)[[1]]
    cat("    ", paste(head(ld_parts, 3), collapse = "\n    "), "\n")
  } else {
    cat("    (not set)\n")
  }
  cat("\n")

  cat("=== End Diagnostics ===\n")

  invisible(list(
    found = TRUE,
    so_path = so_path
  ))
}


# =============================================================================
# Helpers
# =============================================================================

has_command <- function(cmd) {
  nzchar(Sys.which(cmd))
}

run_in_dir <- function(dir, cmd, args = character(), verbose = FALSE) {
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(dir)

  if (verbose) {
    system2(cmd, args, stdout = "", stderr = "")
  } else {
    # Capture output to show on failure
    out <- system2(cmd, args, stdout = TRUE, stderr = TRUE)
    status <- attr(out, "status")
    if (is.null(status)) status <- 0L
    if (status != 0L) {
      # Show tail of output for debugging
      n <- length(out)
      start <- max(1L, n - 80L)
      message("\n--- Last ", n - start + 1L, " lines of output ---")
      message(paste(out[start:n], collapse = "\n"))
    }
    status
  }
}
