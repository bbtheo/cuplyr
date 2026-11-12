# Install the third-party packages of the rolling-window benchmark (F54) into
# a separate library. Base install.packages() only; never touches the default
# library and never installs cuplyr. knitr and rmarkdown are there because the
# benchmark is a Quarto document (bench/rolling-window.qmd, F63) and Quarto's
# knitr engine needs them; packages already visible on the library path are
# skipped, so a machine that has them installs nothing extra.
#
# polars and DuckDB (F73) are benchmark entries too. duckdb comes from CRAN as a
# C++ source build (several minutes; it compiles at R's Makeconf flags, not
# DuckDB's own release flags). polars (r-polars) is not on CRAN: it comes from
# R-multiverse, the repository its install vignette recommends, with
# NOT_CRAN=true so that the source package downloads the project's prebuilt Rust
# library instead of building it with cargo (LIBR_POLARS_BUILD=true forces the
# cargo build). callr runs the polars cells in their own processes and DBI is
# DuckDB's interface. arrow and nanoparquet are the two parquet readers of the
# parquet-to-summary benchmark (F121, bench/parquet-pipeline.R). The summary table gives the elapsed `secs` of each install.
#
#   tools/r 'LIB <- "/path/to/rlib"; source("bench/install-deps.R")'
#
# The library comes from a pre-set `LIB` variable, else the first command-line
# argument, else the default below.

if (!exists("LIB", inherits = FALSE)) {
  args <- commandArgs(trailingOnly = TRUE)
  LIB <- if (length(args) >= 1L && nzchar(args[[1]])) args[[1]] else
    "/home/theo/cuplr/scratchpad/benchmarks/rlib"
}
dir.create(LIB, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(LIB, .libPaths()))
repos <- "https://cloud.r-project.org"

# The pixi environment puts a conda gcc first on PATH, whose sysroot cannot
# link against the distro R's Makeconf (-latomic_asneeded, old crt1 symbols):
# every package that links Fortran/BLAS or configures TBB fails with it.
# Compile with the system toolchain instead (set BENCH_PIXI_TOOLCHAIN=1 to
# keep the inherited PATH).
if (!nzchar(Sys.getenv("BENCH_PIXI_TOOLCHAIN")) && file.exists("/usr/bin/gcc")) {
  Sys.setenv(PATH = paste("/usr/bin", Sys.getenv("PATH"), sep = ":"),
             CC = "/usr/bin/gcc", CXX = "/usr/bin/g++", FC = "/usr/bin/gfortran")
}
options(repos = c(CRAN = repos), Ncpus = 4L)

cran_pkgs <- c("roll", "slider", "RcppRoll", "zoo", "runner", "microbenchmark",
               "ggdist", "paletteer", "ggtext", "data.table", "ggplot2",
               "scales", "patchwork", "knitr", "rmarkdown", "DBI", "callr", "duckdb", "arrow",
               "nanoparquet")
polars_repo <- "https://community.r-multiverse.org"

# RcppParallel's bundled TBB build does not configure under the pixi
# toolchain; use the distro's TBB when it is there.
if (file.exists("/usr/include/tbb/tbb.h") && file.exists("/usr/lib/libtbb.so")) {
  Sys.setenv(TBB_INC = "/usr/include", TBB_LIB = "/usr/lib")
}

results <- list()
record <- function(pkg, ok, msg = "", secs = NA_real_) {
  results[[pkg]] <<- list(ok = ok, msg = msg, secs = secs)
}

try_install <- function(pkg, ...) {
  warns <- character()
  t0 <- proc.time()[["elapsed"]]
  tryCatch({
    withCallingHandlers(
      install.packages(pkg, lib = LIB, ...),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    ok <- requireNamespace(pkg, lib.loc = LIB, quietly = TRUE)
    record(pkg, ok, if (ok) "" else paste(c("not loadable", warns), collapse = "; "),
           secs = proc.time()[["elapsed"]] - t0)
  }, error = function(e) record(pkg, FALSE, conditionMessage(e), secs = proc.time()[["elapsed"]] - t0))
}

# RcppParallel is installed explicitly so roll's build sees it (and its error
# is attributed to it rather than to roll).
if (!requireNamespace("RcppParallel", quietly = TRUE)) {
  message("== installing RcppParallel")
  try_install("RcppParallel", repos = repos)
}

# Packages already available anywhere on the library path are not rebuilt;
# install.packages() resolves missing dependencies into LIB itself.
for (pkg in cran_pkgs) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    record(pkg, TRUE, "already installed")
  } else {
    message("== installing ", pkg)
    try_install(pkg, repos = repos)
  }
}

# polars: R-multiverse, with NOT_CRAN=true set for this install only.
if (!requireNamespace("polars", quietly = TRUE)) {
  message("== installing polars (R-multiverse, prebuilt Rust library)")
  old_not_cran <- Sys.getenv("NOT_CRAN", unset = NA)
  Sys.setenv(NOT_CRAN = "true")
  try_install("polars", repos = c(polars_repo, repos))
  if (is.na(old_not_cran)) Sys.unsetenv("NOT_CRAN") else Sys.setenv(NOT_CRAN = old_not_cran)
} else record("polars", TRUE, "already installed")

# runstats is archived on CRAN: newest tarball in the Archive directory.
if (!requireNamespace("runstats", quietly = TRUE)) {
  message("== installing runstats (CRAN Archive)")
  tryCatch({
    base <- paste0(repos, "/src/contrib/Archive/runstats/")
    idx <- readLines(base, warn = FALSE)
    files <- unique(regmatches(idx, regexpr("runstats_[0-9.]+\\.tar\\.gz", idx)))
    vers <- numeric_version(sub("runstats_(.*)\\.tar\\.gz", "\\1", files))
    newest <- files[order(vers, decreasing = TRUE)[1]]
    # install.packages() of a tarball does not resolve dependencies: fetch
    # them from CRAN first (fftwtools needs the system fftw3).
    for (d in c("Rcpp", "RcppArmadillo", "fftwtools")) {
      if (!requireNamespace(d, quietly = TRUE)) install.packages(d, lib = LIB, repos = repos)
    }
    install.packages(paste0(base, newest), lib = LIB, repos = NULL, type = "source")
    ok <- requireNamespace("runstats", lib.loc = LIB, quietly = TRUE)
    record("runstats", ok, if (ok) newest else "installed but not loadable")
  }, error = function(e) record("runstats", FALSE, conditionMessage(e)))
} else record("runstats", TRUE, "already installed")

# RollingWindow is GitHub-only.
if (!requireNamespace("RollingWindow", quietly = TRUE)) {
  message("== installing RollingWindow (GitHub master tarball)")
  tryCatch({
    # Its generated RcppExports.cpp trips -Werror=format-security on current
    # Rcpp (PKG_CXXFLAGS comes before the Makeconf flags, so it cannot undo
    # it): build this one package with a user Makevars that drops the flag.
    mk <- tempfile("Makevars")
    writeLines(c("LTO =", "CXXFLAGS = -O2 -g -Wno-format-security"), mk)
    Sys.setenv(R_MAKEVARS_USER = mk)
    install.packages("https://github.com/andrewuhl/RollingWindow/archive/refs/heads/master.tar.gz",
                     lib = LIB, repos = NULL, type = "source")
    ok <- requireNamespace("RollingWindow", lib.loc = LIB, quietly = TRUE)
    record("RollingWindow", ok, if (ok) "" else "installed but not loadable")
  }, error = function(e) record("RollingWindow", FALSE, conditionMessage(e)),
  finally = Sys.unsetenv("R_MAKEVARS_USER"))
} else record("RollingWindow", TRUE, "already installed")

ver_table <- do.call(rbind, lapply(names(results), function(p) {
  data.frame(
    package = p,
    version = if (isTRUE(results[[p]]$ok)) as.character(utils::packageVersion(p)) else NA_character_,
    status = if (isTRUE(results[[p]]$ok)) "ok" else "not run: install failed",
    secs = results[[p]]$secs,
    note = results[[p]]$msg,
    stringsAsFactors = FALSE
  )
}))
cat("\n==== install summary (LIB = ", LIB, ") ====\n", sep = "")
print(ver_table, row.names = FALSE)
