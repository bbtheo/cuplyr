# `@importFrom dplyr n` is load-bearing: dplyr evaluates `n()` inside
# group_by(df, n()) from the calling (package/test) namespace; without the
# import that lookup fails with `could not find function "n"`. `group_cols` is
# the same for selections such as ungroup(df, group_cols()): tidyselect brings
# its own helpers (starts_with(), ...) but not dplyr's group_cols().
# `n_distinct` is the same for summarise(d, k = n_distinct(x)) evaluated by
# dplyr itself (a pipeline run on a plain data frame from a test namespace).
#' @useDynLib cuplyr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom rlang %||% obj_address is_reference
#' @importFrom dplyr n group_cols n_distinct
NULL

# Opt this package's namespace into {cli}-formatted rlang conditions
# (bullet glyphs AND width-aware wrapping) -- rlang's `use_cli()`
# (`env_get(..., ".__rlang_use_cli__.", inherit = TRUE)`) looks for this
# exact binding name in the topenv of whichever frame calls
# `rlang::abort()`/`rlang::warn()`. Phase 7 J5's `check_join_cardinality()`
# (R/join.R) needs its conditions to wrap identically to dplyr's own
# (dplyr sets this same `c(format = TRUE, inline = FALSE)` value in its own
# namespace) -- without this, rlang defaults to `c(format = FALSE, inline =
# FALSE)` and never wraps long bullets, which would otherwise make
# `expect_snapshot()`'s dplyr-vs-cuplyr message comparisons diverge purely
# on line-wrapping, not any real content difference.
.__rlang_use_cli__. <- c(format = TRUE, inline = FALSE)
.onLoad <- function(libname, pkgname) {
  # Check GPU availability
  gpu_ok <- tryCatch(
    gpu_is_available(),
    error = function(e) FALSE
  )

  # Set package options
  op <- options()
  op.cuplyr <- list(
    cuplyr.verbose = FALSE,
    cuplyr.lazy = TRUE,
    cuplyr.gpu_available = gpu_ok
  )
  toset <- !(names(op.cuplyr) %in% names(op))
  if (any(toset)) options(op.cuplyr[toset])

  if (gpu_ok) memory_resource_install()

  invisible()
}

.onAttach <- function(libname, pkgname) {
  info <- tryCatch(gpu_info(), error = function(e) list(available = FALSE))

  if (isTRUE(info$available)) {
    # Format memory in GB
    total_gb <- round(info$total_memory / 1e9, 1)
    # What cuplyr can allocate (the free memory plus the pool's unused pages)
    free_gb <- round(tryCatch(
      gpu_allocatable_bytes(),
      error = function(e) info$free_memory
    ) / 1e9, 1)

    msg <- paste0(
      "cuplyr: GPU-accelerated data manipulation\n",
      "GPU: ", info$name, " (", info$compute_capability, ")\n",
      "Memory: ", free_gb, " GB free / ", total_gb, " GB total"
    )
  } else {
    msg <- paste0(
      "cuplyr: GPU-accelerated data manipulation\n",
      "WARNING: No GPU detected. Package will not function correctly."
    )
  }

  packageStartupMessage(msg)
}
