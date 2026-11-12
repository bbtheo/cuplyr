#' Estimate GPU memory usage of a tbl_gpu object
#'
#' Calculates the estimated GPU memory footprint of a GPU table based on
#' its dimensions and column types. This is useful for understanding memory
#' requirements before working with large datasets.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return The estimated memory usage in bytes (numeric), or `NA` if the
#'   object is not a valid `tbl_gpu` or has no data on GPU.
#'
#' @details
#' The estimate includes:
#' \itemize{
#'   \item Column data (varies by type: 8 bytes for FLOAT64, 4 bytes for INT32, etc.)
#'   \item Validity bitmasks for NA handling (1 bit per row per column)
#' }
#'
#' String columns use an average estimate of 32 bytes per element, which may
#' vary significantly based on actual string lengths.
#'
#' Columns are shared between tables: a verb that leaves a column untouched
#' (`select()`, a bare copy in `mutate()`, ...) returns a table that holds the
#' same device memory as its input. The estimates of such tables therefore
#' overlap, and their sum can exceed what the GPU holds.
#'
#' This is an estimate and actual GPU memory usage may be higher due to:
#' \itemize{
#'   \item Memory alignment requirements
#'   \item RMM memory pool overhead
#'   \item Temporary allocations
#' }
#'
#' @seealso
#' \code{\link{gpu_details}} for overall GPU memory info,
#' \code{\link{gpu_object_info}} for detailed object information
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   size <- gpu_memory_usage(gpu_mtcars)
#'   cat("Estimated GPU memory:", round(size / 1024, 1), "KB\n")
#' }
gpu_memory_usage <- function(x) {
  if (!is_tbl_gpu(x) || is.null(x$ptr)) {
    return(NA_real_)
  }

  dims <- tryCatch(base_dim(x), error = function(e) NULL)
  if (is.null(dims)) {
    return(NA_real_)
  }

  nrow <- dims[1]
  types <- x$schema$types

  # Bytes per element for each GPU type
  bytes_per_type <- gpu_type_bytes(types)

  # Data bytes
  data_bytes <- sum(nrow * bytes_per_type)

  # Validity mask overhead (1 bit per row per column, rounded up to bytes)
  mask_bytes <- ceiling(nrow / 8) * length(types)

  data_bytes + mask_bytes
}

#' Get detailed information about a GPU table object
#'
#' Returns comprehensive information about a `tbl_gpu` object including
#' its dimensions, column types, estimated memory usage, and verification
#' that data resides on the GPU.
#'
#' @param x A `tbl_gpu` object.
#'
#' @details
#' The memory estimate is the one of [gpu_memory_usage()]: columns shared with
#' other tables are counted for each table, so estimates of related tables
#' overlap.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{valid}{Logical: TRUE if the object is a valid tbl_gpu with GPU data}
#'   \item{nrow}{Number of rows}
#'   \item{ncol}{Number of columns}
#'   \item{column_names}{Character vector of column names}
#'   \item{column_types}{Character vector of GPU column types}
#'   \item{estimated_gpu_bytes}{Estimated GPU memory usage in bytes}
#'   \item{estimated_gpu_mb}{Estimated GPU memory usage in megabytes}
#'   \item{r_object_bytes}{Size of the R object (should be small)}
#'   \item{data_on_gpu}{Logical: TRUE if data is verified to be on GPU}
#'   \item{pointer_valid}{Logical: TRUE if the external pointer is valid}
#' }
#'
#' @seealso
#' \code{\link{gpu_memory_usage}} for just the memory estimate,
#' \code{\link{verify_gpu_data}} to check if data is on GPU
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'   info <- gpu_object_info(gpu_mtcars)
#'   cat("Rows:", info$nrow, "\n")
#'   cat("GPU memory:", round(info$estimated_gpu_mb, 2), "MB\n")
#'   cat("Data on GPU:", info$data_on_gpu, "\n")
#' }
gpu_object_info <- function(x) {
  result <- list(
    valid = FALSE,
    nrow = NA_integer_,
    ncol = NA_integer_,
    column_names = character(0),
    column_types = character(0),
    estimated_gpu_bytes = NA_real_,
    estimated_gpu_mb = NA_real_,
    r_object_bytes = as.numeric(utils::object.size(x)),
    data_on_gpu = FALSE,
    pointer_valid = FALSE
  )

  if (!is_tbl_gpu(x)) {
    return(result)
  }

  result$valid <- TRUE

  if (!is.null(x$ptr) && inherits(x$ptr, "externalptr")) {
    result$pointer_valid <- TRUE

    # base_dim(): a zero-column table's rows live on the R side.
    dims <- tryCatch(base_dim(x), error = function(e) NULL)
    if (!is.null(dims)) {
      result$nrow <- dims[1]
      result$ncol <- dims[2]
      result$data_on_gpu <- TRUE
    }
  }

  result$column_names <- x$schema$names
  result$column_types <- x$schema$types

  gpu_bytes <- gpu_memory_usage(x)
  result$estimated_gpu_bytes <- gpu_bytes
  result$estimated_gpu_mb <- gpu_bytes / (1024 * 1024)

  result
}

#' Verify that data resides on GPU
#'
#' Performs checks to confirm that a `tbl_gpu` object has its data
#' stored on the GPU, not in R memory. This is useful for debugging
#' and ensuring GPU operations are working correctly.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return `TRUE` if all checks pass and data is verified to be on GPU,
#'   `FALSE` otherwise.
#'
#' @details
#' This function performs multiple verification steps:
#' \enumerate{
#'   \item Object has the `tbl_gpu` class
#'   \item Object has a valid external pointer
#'   \item GPU operations (dim, types) work on the pointer
#'   \item R object is small (no data copy in R memory)
#' }
#'
#' @seealso
#' \code{\link{gpu_object_info}} for detailed object information,
#' \code{\link{tbl_gpu}} for creating GPU tables
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   gpu_mtcars <- tbl_gpu(mtcars)
#'
#'   # Should return TRUE
#'   verify_gpu_data(gpu_mtcars)
#'
#'   # Regular data frames return FALSE
#'   verify_gpu_data(mtcars)
#' }
verify_gpu_data <- function(x) {
  # Check 1: Must be tbl_gpu
  if (!is_tbl_gpu(x)) {
    return(FALSE)
  }

  # Check 2: Must have a valid pointer
  if (is.null(x$ptr)) {
    return(FALSE)
  }

  # Check 3: Pointer must be external pointer
  if (!inherits(x$ptr, "externalptr")) {
    return(FALSE)
  }

  # Check 4: GPU operations should work
  dims <- tryCatch(gpu_dim(x$ptr), error = function(e) NULL)
  if (is.null(dims) || !is.integer(dims) || length(dims) != 2) {
    return(FALSE)
  }

  # Check 5: R object should be lightweight (no data copy)
  r_size <- as.numeric(utils::object.size(x))
  ncol <- length(x$schema$names)
  max_expected <- 10000 + ncol * 200  # Base + per-column metadata

  if (r_size > max_expected) {
    return(FALSE)
  }

  TRUE
}

#' Compare R object size vs GPU data size
#'
#' Computes the ratio of GPU memory usage to R object size for a `tbl_gpu`
#' object. A high ratio confirms that data is stored on GPU, not in R.
#'
#' @param x A `tbl_gpu` object.
#'
#' @return A list with:
#' \describe{
#'   \item{r_bytes}{Size of the R object in bytes}
#'   \item{gpu_bytes}{Estimated GPU memory in bytes}
#'   \item{ratio}{GPU size divided by R size (should be > 1 if data is on GPU)}
#' }
#'
#' @seealso
#' \code{\link{verify_gpu_data}} for boolean verification,
#' \code{\link{gpu_object_info}} for detailed information
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Create a larger dataset
#'   df <- data.frame(matrix(runif(10000), ncol = 10))
#'   gpu_df <- tbl_gpu(df)
#'
#'   comparison <- gpu_size_comparison(gpu_df)
#'   cat("R object:", round(comparison$r_bytes / 1024, 1), "KB\n")
#'   cat("GPU data:", round(comparison$gpu_bytes / 1024, 1), "KB\n")
#'   cat("Ratio:", round(comparison$ratio, 1), "x\n")
#' }
gpu_size_comparison <- function(x) {
  r_bytes <- as.numeric(utils::object.size(x))
  gpu_bytes <- gpu_memory_usage(x)

  list(
    r_bytes = r_bytes,
    gpu_bytes = gpu_bytes,
    ratio = if (!is.na(gpu_bytes) && gpu_bytes > 0) gpu_bytes / r_bytes else NA_real_
  )
}

#' Get GPU memory snapshot
#'
#' Returns the current GPU memory state including total, free, and used memory,
#' and what cuplyr's memory resource holds. Useful for monitoring GPU memory
#' usage during operations.
#'
#' @return A list with:
#' \describe{
#'   \item{available}{Logical: TRUE if GPU is available}
#'   \item{total_bytes}{Total GPU memory in bytes}
#'   \item{free_bytes}{Free GPU memory in bytes}
#'   \item{used_bytes}{Used GPU memory in bytes: device level, `total - free`,
#'     for every process on the card. It includes everything the memory resource
#'     holds from the device, so freeing a table does not lower it while the
#'     resource keeps the memory for reuse (always under `"pool"`; under
#'     `"async"` until [gpu_gc()] or an automatic release hands it back, see
#'     below). Use `pool_used_bytes` for what cuplyr's tables occupy.}
#'   \item{total_gb}{Total GPU memory in gigabytes}
#'   \item{free_gb}{Free GPU memory in gigabytes}
#'   \item{used_gb}{Used GPU memory in gigabytes}
#'   \item{resource}{`"async"` (the default), `"pool"` or `"cuda"`: the RMM
#'     memory resource cuplyr installed at load time (see below). `"cuda"` means
#'     none: every allocation is a plain `cudaMalloc`.}
#'   \item{pool_bytes}{Bytes the resource currently holds from the device. Under
#'     `"pool"` that is the pool's size; under `"async"` it is the async pool's
#'     reserved memory (`cudaMemPoolAttrReservedMemCurrent`), which shrinks when
#'     memory is returned. `NA` when `resource` is `"cuda"`.}
#'   \item{pool_used_bytes}{Bytes currently allocated from the resource to live
#'     columns and scratch space: what cuplyr is using, as opposed to what it
#'     reserved. It rises when a table is created and falls when the table is
#'     garbage collected. Exact under both resources. `NA` when `resource` is
#'     `"cuda"`.}
#' }
#'
#' @section The memory resource:
#' Loading cuplyr installs an RMM memory resource as the current device
#' resource, so the many short-lived columns a pipeline allocates are carved out
#' of memory reserved once instead of going through `cudaMalloc`/`cudaFree`
#' each time. There are three resources:
#' \describe{
#'   \item{`"async"` (default)}{RMM's `cuda_async_memory_resource`, CUDA's
#'     stream-ordered allocator (`cudaMallocAsync`). By default it keeps freed
#'     memory cached for reuse, as the pool does, and runs at the same speed
#'     (measured, see the vignette). Unlike the pool it can give the memory back:
#'     [gpu_gc()] synchronises the device and returns everything the pool holds
#'     unused, so other processes on the card can use it. With the opt-in
#'     `cuplyr.release_threshold` (below) CUDA also returns unused memory by
#'     itself, at the next synchronisation, while the pool's whole reservation
#'     is above that threshold. The reservation counts tables in use, so a
#'     session that holds more than the threshold keeps nothing cached and
#'     allocates from the driver again, at about plain `cudaMalloc` speed.}
#'   \item{`"pool"`}{RMM's `pool_memory_resource`. Same speed as async by
#'     default, but it keeps the high-water mark of its size until the R session
#'     ends, so the card's free memory (`free_bytes`) stays lower than the data
#'     you hold; use `pool_used_bytes` for the latter. [gpu_gc()] cannot shrink
#'     it (`released_bytes` is 0).}
#'   \item{`"cuda"`}{No resource (`cuplyr.memory_pool = FALSE`, or
#'     `cuplyr.memory_resource = "cuda"`): plain `cudaMalloc`/`cudaFree` for
#'     every allocation. Slowest per allocation; device free memory is always
#'     the truth, and `pool_bytes` and `pool_used_bytes` are `NA`.}
#' }
#' Synchronisations happen at `collect()`, `print()` and [gpu_sync()], and also
#' inside many libcudf calls (for example whenever one copies a size or count
#' back to the host), so an automatic release can happen in the middle of a
#' pipeline. See `vignette("memory", package = "cuplyr")` for measurements and
#' advice on choosing.
#'
#' The resource is chosen when the package loads, from options (which win) or
#' environment variables. There is one resource per R process: the switches are
#' read by the first load that creates it, and loading the package again in the
#' same session (`pkgload::load_all()`, or a reload after unloading) reuses that
#' resource, whatever the switches say then, and reports it; such a load does
#' not read the switches, so it does not warn about them either. Restart R to
#' change them.
#'
#' Where the GPU or driver does not support `cudaMallocAsync`, the async
#' resource cannot be created. If async was the default (nothing chosen, or an
#' unknown kind), cuplyr then installs `"pool"` and gives a warning saying so.
#' If `"async"` was asked for by name, the warning says the resource could not
#' be installed and nothing is (`resource` is `"cuda"`).
#' \describe{
#'   \item{`cuplyr.memory_pool` / `CUPLYR_MEMORY_POOL`}{`FALSE`, or `"0"`,
#'     `"false"`, `"no"`, `"off"`, installs nothing (`resource` is `"cuda"`,
#'     the behaviour before the pool existed) unless an earlier load in the
#'     session already created the resource. Anything else, or unset (a
#'     zero-length or `NA` option counts as unset), installs the resource.}
#'   \item{`cuplyr.memory_resource` / `CUPLYR_MEMORY_RESOURCE`}{`"async"`
#'     (default) or `"pool"` to pick the kind of resource; `"cuda"` or `"none"`
#'     install nothing, like `cuplyr.memory_pool = FALSE`. Case and surrounding
#'     blanks do not matter. Unset or empty means async. Any other value gives a
#'     warning naming the option or variable and the value as typed, and async is
#'     used; when `cuplyr.memory_pool` switches the resource off, the warning
#'     says the value is ignored and no resource (`"cuda"`) is used.}
#'   \item{`cuplyr.pool_initial` / `CUPLYR_POOL_INITIAL`}{How much the resource
#'     reserves at load, in bytes, so the first calls need no driver
#'     allocation. Default: a quarter of the free device memory, at most 1 GiB.
#'     A larger request is clamped to 90 percent of the free memory. `0`
#'     reserves nothing up front. Both `"pool"` and `"async"` grow by
#'     themselves when needed; this is not a limit and not a release threshold.}
#'   \item{`cuplyr.release_threshold` / `CUPLYR_RELEASE_THRESHOLD`}{Async only,
#'     opt-in, in bytes; unset (the default) means no automatic release. It is
#'     CUDA's release threshold for the async pool: at a synchronisation CUDA
#'     returns unused memory to the device until the pool's whole reservation,
#'     memory in use included, is at or below it. `0` keeps nothing, the "share
#'     the card" setting. The price is speed: while the session holds more than
#'     the threshold, every allocation goes to the driver. A zero-length, `NA`
#'     or empty option counts as unset (the environment variable decides). Any
#'     other value that is not a non-negative number gives a warning naming the
#'     option or variable, and means unset. Ignored under `"pool"` and
#'     `"cuda"`.}
#' }
#'
#' All cuplyr options, and how an invalid value is handled, are listed on
#' [cuplyr-options].
#'
#' @seealso
#' \code{\link{gpu_details}} for device information (device level only),
#' \code{\link{gpu_memory_usage}} for per-object memory estimates,
#' [cuplyr-options] for the options
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Check memory before allocation
#'   before <- gpu_memory_state()
#'
#'   # Allocate some GPU data
#'   gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))
#'
#'   # Check memory after allocation
#'   after <- gpu_memory_state()
#'
#'   cat("Memory used:", after$used_gb - before$used_gb, "GB\n")
#'   cat("Allocated from the resource:",
#'       after$pool_used_bytes - before$pool_used_bytes, "bytes\n")
#' }
gpu_memory_state <- function() {
  # (only the memory figures: gpu_details() also reads the device properties,
  # which costs most of a millisecond, and a join calls this on every verb)
  mem <- tryCatch(gpu_mem_info(), error = function(e) c(NA_real_, NA_real_))
  info <- list(available = !anyNA(mem), free_memory = mem[[1]], total_memory = mem[[2]])

  if (!isTRUE(info$available)) {
    return(list(
      available = FALSE,
      total_bytes = NA_real_,
      free_bytes = NA_real_,
      used_bytes = NA_real_,
      total_gb = NA_real_,
      free_gb = NA_real_,
      used_gb = NA_real_,
      resource = "cuda",
      pool_bytes = NA_real_,
      pool_used_bytes = NA_real_
    ))
  }

  mr <- gpu_mr_state()

  list(
    available = TRUE,
    total_bytes = info$total_memory,
    free_bytes = info$free_memory,
    used_bytes = info$total_memory - info$free_memory,
    total_gb = info$total_memory / 1e9,
    free_gb = info$free_memory / 1e9,
    used_gb = (info$total_memory - info$free_memory) / 1e9,
    resource = gpu_mr_kind(),
    pool_bytes = mr[[1]],
    pool_used_bytes = mr[[2]]
  )
}

# Bytes cuplyr can allocate right now (F57, internal): the device's free memory
# plus the part of its memory resource's pages that is not in use. Under a pool
# the device-level free memory understates this, since the pool keeps what it
# grew to; allocations are served from those pages first. Callers that decide
# whether something will fit (the join-size warning, tests that skip for lack of
# memory, the attach message) use this, not `free_bytes`. `state` is
# gpu_memory_state(); a state without the resource fields counts as no resource.
gpu_allocatable_bytes <- function(state = gpu_memory_state()) {
  free <- state$free_bytes
  if (is.null(free) || is.na(free)) return(NA_real_)
  resource <- state$resource
  held <- state$pool_bytes
  used <- state$pool_used_bytes
  if (is.null(resource) || identical(resource, "cuda") ||
      is.null(held) || is.null(used) || is.na(held) || is.na(used)) {
    return(free)
  }
  free + max(held - used, 0)
}

# The memory resource (F57, internal). The switches are read at load; only the
# first load that creates the resource uses them.
memory_resource_config <- function() {
  enabled <- cuplyr_opt("memory_pool")

  initial <- cuplyr_opt("pool_initial")
  if (is.na(initial)) {
    free <- tryCatch(gpu_info()$free_memory, error = function(e) 0)
    initial <- min(2^30, 0.25 * free)
  }

  # Default "async"; "cuda"/"none" mean no resource (what gpu_memory_state()
  # reports for none); an unknown kind warns, naming where it came from and
  # quoting the value as typed. `explicit` is TRUE when the kind was asked for
  # by name ("async" or "pool"): only a default async falls back to the pool
  # when it can't be created (memory_resource_install()).
  invalid_kind <- function(shown, r) {
    typed <- opt_typed(r)
    source <- opt_source_label(r)
    valid <- "Valid values are \"async\", \"pool\" and \"cuda\" (or \"none\", no resource)."
    if (enabled) {
      sprintf("Unknown memory resource '%s' in %s; using \"async\". %s",
              typed, source, valid)
    } else {
      # The off switch decides: no resource, whatever the kind says.
      sprintf(paste0(
        "Unknown memory resource '%s' in %s, ignored: cuplyr.memory_pool / ",
        "CUPLYR_MEMORY_POOL switches the resource off, so none is used (\"cuda\"). %s"),
        typed, source, valid)
    }
  }
  res <- cuplyr_opt_resolve("memory_resource", invalid_kind)
  kind <- res$value
  explicit <- !identical(res$source, "default") && kind %in% c("pool", "async")
  if (kind %in% c("cuda", "none")) {
    enabled <- FALSE
    kind <- "async"
  }

  list(enabled = enabled, initial = initial, kind = kind, explicit = explicit,
       release_threshold = memory_resource_release_threshold())
}

# The opt-in release threshold in bytes (async only): cuplyr.release_threshold,
# else CUPLYR_RELEASE_THRESHOLD; NA_real_ when unset (the default: no automatic
# release). A zero-length, NA or "" option counts as unset (the env var
# decides), as for the other switches. Any other value that is not a
# non-negative number warns, naming where it came from, and means unset.
memory_resource_release_threshold <- function() {
  cuplyr_opt("release_threshold")
}

# Make the process's resource current for this load, creating it when this is
# the first load with the switch on (src/memory_resource.cpp). A resource an
# earlier load created is reused whatever the switches say now: it can't be
# freed while anything may point into it, and it is never stacked. The switches
# are then not read at all, so they don't warn either.
#
# When the default async resource can't be created (cudaMallocAsync is
# unsupported: RMM's constructor throws), the pool is used instead, with a
# warning. An async that was asked for by name, a pool that fails, or an error
# before anything was created (say an option memory_resource_config() can't
# read) leaves plain cudaMalloc ("cuda"), with a warning naming every error.
#
# Every warning (the switches' invalid-value warnings included) is held and
# signalled once the resource is settled, outside the tryCatch()es: when
# warnings are errors (options(warn = 2)) a converted warning must neither be
# taken for a failure to create a resource nor stop the one that was chosen.
memory_resource_install <- function() {
  called <- FALSE
  held <- list()
  hold <- function(w) {
    held[[length(held) + 1L]] <<- w
    invokeRestart("muffleWarning")
  }
  say <- function(...) {
    held[[length(held) + 1L]] <<- simpleWarning(paste0(...), call = NULL)
  }
  install <- function(enabled, kind, initial = 0, release_threshold = NA_real_) {
    called <<- TRUE
    gpu_mr_install(enabled, kind, initial, release_threshold)
  }
  fail <- function(...) {
    # gpu_mr_install() pins this copy of the library; if it was never called,
    # call it switched off, which creates nothing (it reuses a resource only if
    # one exists, else returns "cuda") but still pins.
    kind <- if (called) "cuda" else tryCatch(install(FALSE, "async"), error = function(e) "cuda")
    say("cuplyr: no memory resource, using plain cudaMalloc (",
        paste(vapply(list(...), conditionMessage, ""), collapse = "; "), ").")
    kind
  }
  kind <- tryCatch({
    if (gpu_mr_exists()) {
      # Reuse only: with `enabled = FALSE` the other arguments are ignored.
      install(FALSE, "async")
    } else {
      cfg <- withCallingHandlers(memory_resource_config(), warning = hold)
      create <- function(kind) {
        install(cfg$enabled, kind, cfg$initial, cfg$release_threshold)
      }
      tryCatch(create(cfg$kind), error = function(e) {
        if (!identical(cfg$kind, "async") || isTRUE(cfg$explicit)) return(fail(e))
        tryCatch({
          kind <- create("pool")
          say("cuplyr: default \"async\" unavailable, using \"pool\" instead (",
              conditionMessage(e), ").")
          kind
        }, error = function(e2) fail(e, e2))
      })
    }
  }, error = fail)
  # (when re-raising one does not return -- options(warn = 2) -- the invalid-
  # option warnings from it on stay un-given: the next read reports them)
  raised <- 0L
  on.exit(opt_forget_warned(held[seq_along(held) > raised]))
  for (w in held) {
    warning(w)
    raised <- raised + 1L
  }
  invisible(kind)
}

#' Wait for the GPU to finish
#'
#' Blocks until all GPU work queued by cuplyr and libcudf on the device has
#' completed. GPU calls return as soon as the work is queued, so a timing taken
#' without a wait can measure only the launch. Call `gpu_sync()` before reading
#' the clock, as one would use `torch.cuda.synchronize()`.
#'
#' Results that are collected to R (`collect()`, `print()`) are already waited
#' for; this is for timing code that leaves its result on the GPU.
#'
#' @return `NULL`, invisibly.
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   g <- tbl_gpu(data.frame(x = runif(1e6)), lazy = FALSE)
#'   t0 <- Sys.time()
#'   r <- dplyr::mutate(g, y = cumsum(x))
#'   gpu_sync()
#'   Sys.time() - t0
#' }
gpu_sync <- function() {
  gpu_sync_device()
  invisible(NULL)
}

#' Force GPU memory cleanup
#'
#' Triggers R garbage collection to free GPU memory held by unreferenced
#' `tbl_gpu` objects. Use this between operations when GPU memory is limited
#' or before large allocations.
#'
#' @param verbose Logical. If TRUE, prints memory freed. Default FALSE.
#' @param aggressive Logical. If TRUE (default), runs multiple GC passes with
#'   short delays to more aggressively trigger finalizers. If FALSE, runs a
#'   lighter cleanup pass.
#'
#' @return Invisibly returns a list with memory state before and after cleanup,
#'   and the amount freed in bytes and gigabytes. `freed_bytes` is the drop in
#'   `pool_used_bytes` (allocator level) when cuplyr's memory resource is
#'   installed, since the device-level free memory does not move while the
#'   resource keeps its pages; with `resource == "cuda"` it is the rise in
#'   device free memory. `released_bytes` is what the resource gave back to the
#'   device: the drop in `pool_bytes` across the call. Under `"async"` the call
#'   ends by trimming the async pool, so it is the memory the pool held unused
#'   (0 or more); under `"pool"` it is 0 (RMM's pool never shrinks); under
#'   `"cuda"` it is `NA`.
#'
#' @details
#' GPU memory is automatically freed when `tbl_gpu` objects are garbage
#' collected by R. However, R's garbage collector doesn't know about GPU
#' memory pressure and may not run immediately. This function forces
#' garbage collection and allows time for GPU cleanup.
#'
#' A cuplyr call you make (an eager verb, `compute()`, `collect()`, a bind)
#' frees the temporary GPU tables its own steps built when it returns. What a
#' call returns is yours and is never freed early, including a table that your
#' code creates while a verb evaluates its arguments or expressions. So this
#' function is for tables you created and dropped without R having collected
#' them yet: the `group_by()` result of a one-line pipeline lives until R's
#' next collection. So does the inner result of a dplyr verb that dplyr itself
#' builds from several verb calls, such as `add_tally(sort = TRUE)`. When a GPU step runs out of device memory, cuplyr runs
#' `gc()` and `gpu_gc(aggressive = FALSE)` itself and retries that step once;
#' a second failure raises an error of class `cuplyr_error_gpu_oom`.
#'
#' Under the default `"async"` resource, this function also synchronises the
#' device and returns all unused memory of cuplyr's async pool to the device, so
#' other processes can use it (see [gpu_memory_state()] and
#' `vignette("memory", package = "cuplyr")`). The pool is then smaller than the
#' session's recent peak, so the next calls grow it again from the driver; that
#' costs the first call after `gpu_gc()` a few percent. If the device is in a
#' CUDA error state, the return step cannot run: `gpu_gc()` then gives a warning
#' instead of an error and leaves the pool as it is, so it stays safe in cleanup
#' code.
#'
#' Call this function:
#' \itemize{
#'   \item After removing large GPU objects with `rm()`, when other processes
#'     need the memory (under `"async"` it returns what the pool holds unused)
#'   \item When you see out-of-memory errors
#'   \item Before allocating large new GPU tables
#' }
#' It is not meant for the loop of a benchmark: under `"async"` the trim makes
#' the next call grow the pool again.
#'
#' @seealso
#' \code{\link{gpu_memory_state}} for checking current memory usage
#'
#' @export
#' @examples
#' if (has_gpu()) {
#'   # Create and discard a GPU table
#'   gpu_df <- tbl_gpu(data.frame(x = runif(1000000)))
#'   rm(gpu_df)
#'
#'   # Force cleanup
#'   gpu_gc(verbose = TRUE)
#' }
gpu_gc <- function(verbose = FALSE, aggressive = TRUE) {
  before <- gpu_memory_state()

  if (aggressive) {
    # Multiple GC passes with delays to ensure finalizers run
    for (i in 1:5) {
      gc(verbose = FALSE, full = TRUE)
      Sys.sleep(0.2)
    }
  } else {
    gc(verbose = FALSE, full = TRUE)
    Sys.sleep(0.1)
    gc(verbose = FALSE, full = TRUE)
  }

  # Async: hand back everything the pool holds unused (no-op otherwise). A CUDA
  # error here (sticky device error) must not make gpu_gc() fail: it also runs in
  # the OOM retry and in cleanup code.
  tryCatch(
    gpu_mr_release(),
    error = function(e) {
      warning("gpu_gc() could not return memory to the device: ", conditionMessage(e),
              call. = FALSE)
    }
  )

  after <- gpu_memory_state()

  # Under a memory resource the device keeps the pages it reserved, so the
  # device-level free memory doesn't move when a table is freed: report the drop
  # in what is allocated from the resource instead (allocator level).
  pooled <- !is.na(before$pool_used_bytes) && !is.na(after$pool_used_bytes)
  freed_bytes <- if (pooled) {
    before$pool_used_bytes - after$pool_used_bytes
  } else {
    after$free_bytes - before$free_bytes
  }
  freed_gb <- freed_bytes / 1e9
  # What the resource gave back to the device: the drop in what it holds.
  released_bytes <- if (pooled) before$pool_bytes - after$pool_bytes else NA_real_
  level <- if (!pooled) {
    ""
  } else if (identical(after$resource, "async")) {
    " (allocator level)"
  } else {
    " (allocator level; the pool keeps the pages)"
  }

  if (verbose && !is.na(freed_bytes)) {
    if (freed_gb >= 0.01) {
      message(sprintf("GPU memory freed: %.2f GB%s", freed_gb, level))
    } else if (freed_bytes > 0) {
      message(sprintf("GPU memory freed: %.0f bytes%s", freed_bytes, level))
    } else {
      message("No GPU memory freed", level)
    }
    if (pooled && identical(after$resource, "async")) {
      message(sprintf("Returned to the device: %.2f GB", released_bytes / 1e9))
    }
  }

  invisible(list(
    before = before,
    after = after,
    freed_bytes = freed_bytes,
    freed_gb = freed_gb,
    released_bytes = released_bytes
  ))
}

# Allocation tracking (F56, internal). Evaluates `expr` with an RMM statistics
# adaptor as the current device resource and reports what was allocated through
# RMM meanwhile. Unlike device-level free memory (cudaMemGetInfo), this cannot
# be hidden by pages the CUDA driver already holds. Windows nest; an outer
# window also counts the inner window's allocations.
#
# Returns list(value, allocated_bytes, held_bytes, peak_bytes): every byte
# allocated while `expr` ran, the part not yet freed when it returned, and the
# maximum held amount during it. All doubles.
gpu_alloc_tracking <- function(expr) {
  id <- gpu_alloc_window_push()
  open <- TRUE
  on.exit(if (open) gpu_alloc_window_pop(id), add = TRUE)
  value <- expr
  open <- FALSE
  counts <- gpu_alloc_window_pop(id)
  list(
    value = value,
    allocated_bytes = counts[[1]],
    held_bytes = counts[[2]],
    peak_bytes = counts[[3]]
  )
}

# Number of open tracking windows.
gpu_alloc_tracking_active <- function() {
  gpu_alloc_window_count()
}
