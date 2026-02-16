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
    # Assume current directory or find it
    if (file.exists("DESCRIPTION") && file.exists("configure")) {
      return(getwd())
    }
    # Check if we're in a subdirectory
    for (parent in c(".", "..", "../..")) {
      desc <- file.path(parent, "DESCRIPTION")
      if (file.exists(desc)) {
        pkg <- read.dcf(desc, fields = "Package")[1, 1]
        if (identical(pkg, "cuplyr")) {
          return(normalizePath(parent))
        }
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

  # Check if RAPIDS already in this prefix
  has_cudf <- file.exists(file.path(conda_prefix, "include", "cudf", "types.hpp"))

  if (!has_cudf) {
    message("Installing RAPIDS packages into ", conda_prefix, " ...")
    conda_args <- c(
      "create", "-y", "-p", conda_prefix,
      "-c", "rapidsai", "-c", "conda-forge", "-c", "nvidia",
      "libcudf>=25.12", "librmm>=25.12", "libkvikio>=25.12",
      "spdlog", "fmt",
      "cuda-toolkit>=12.0,<13"
    )

    if (dry_run) {
      message("[dry-run] Would run: ", conda_cmd, " ", paste(conda_args, collapse = " "))
    } else {
      status <- system2(conda_cmd, conda_args,
                          stdout = if (verbose) "" else FALSE,
                          stderr = if (verbose) "" else FALSE)
      if (status != 0) {
        stop(
          conda_cmd, " create failed (exit code ", status, ").\n",
          "Try running manually:\n  ",
          conda_cmd, " ", paste(conda_args, collapse = " "),
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

  # Cloud: patch Makevars to put real driver FIRST in RUNPATH
  if (env %in% c("colab", "cloud_gpu")) {
    driver_lib <- driver_lib %||% find_real_driver_lib()
    if (!is.null(driver_lib)) {
      patch_makevars_for_cloud(src_dir, driver_lib)
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

  invisible(TRUE)
}

install_via_system <- function(src_dir, configure_args, dry_run, verbose) {
  env <- detect_environment()
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
  if (env %in% c("colab", "cloud_gpu")) {
    conda_prefix <- Sys.getenv("CONDA_PREFIX", "/opt/rapids")
    disable_cuda_stubs(file.path(conda_prefix, "lib"))
    driver_lib <- find_real_driver_lib()
    if (!is.null(driver_lib)) {
      configure_cloud_library_paths(driver_lib, conda_prefix)
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
  if (env %in% c("colab", "cloud_gpu")) {
    driver_lib <- driver_lib %||% find_real_driver_lib()
    if (!is.null(driver_lib)) {
      patch_makevars_for_cloud(src_dir, driver_lib)
    }
  }

  # Build
  status <- run_in_dir(src_dir, "R", c("CMD", "INSTALL", "."), verbose = verbose)
  if (status != 0) {
    stop("R CMD INSTALL failed (exit code ", status, ").", call. = FALSE)
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

  # Container (check first — containers may run on cloud)
  if (file.exists("/.dockerenv") || nzchar(Sys.getenv("KUBERNETES_SERVICE_HOST"))) {
    return("container")
  }

  # Colab (require 2+ signals to avoid false positives)
  colab_signals <- c(
    file.exists("/content"),
    nzchar(Sys.getenv("COLAB_RELEASE_TAG")),
    nzchar(Sys.getenv("COLAB_GPU")),
    file.exists("/usr/local/share/jupyter/kernels/ir")
  )
  if (sum(colab_signals) >= 2) {
    return("colab")
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
#' Sets LD_LIBRARY_PATH with the real driver path first, so it takes
#' priority over any RAPIDS stubs.
#'
#' @param driver_lib Path to directory with real libcuda.so.1.
#' @param conda_prefix RAPIDS/conda prefix.
#' @keywords internal
configure_cloud_library_paths <- function(driver_lib, conda_prefix) {
  cuda_home <- Sys.getenv("CUDA_HOME", "/usr/local/cuda")
  current <- Sys.getenv("LD_LIBRARY_PATH")
  parts <- unique(c(
    driver_lib,
    file.path(cuda_home, "lib64"),
    file.path(conda_prefix, "lib"),
    strsplit(current, ":", fixed = TRUE)[[1]]
  ))
  parts <- parts[nzchar(parts)]
  Sys.setenv(LD_LIBRARY_PATH = paste(parts, collapse = ":"))
}

#' Patch Makevars for Colab/cloud: put driver FIRST in RUNPATH
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

  # Rewrite PKG_LIBS: driver_lib FIRST in RUNPATH
  mk <- sub(
    "^PKG_LIBS=(.*)$",
    sprintf(
      paste0(
        "PKG_LIBS=-Wl,--enable-new-dtags ",
        "-Wl,-rpath,%s ",
        "-L$(CUDF_LIB) -Wl,-rpath,$(CUDF_LIB) -Wl,-rpath,$(RAPIDS_RPATHS) -lcudf ",
        "-L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart"
      ),
      driver_lib
    ),
    mk
  )

  writeLines(mk, makevars)
  message("Patched src/Makevars: driver RUNPATH = ", driver_lib)
  invisible(TRUE)
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

  system2(cmd, args,
          stdout = if (verbose) "" else FALSE,
          stderr = if (verbose) "" else FALSE)
}
