# Utility functions for cuplyr

# Type mapping from R to GPU types
gpu_type_from_r <- function(x) {
  if (is.logical(x)) return("BOOL8")
  if (is.integer(x)) return("INT32")
  if (is.double(x)) {
    if (inherits(x, "Date")) return("TIMESTAMP_DAYS")
    if (inherits(x, "POSIXct")) return("TIMESTAMP_MICROSECONDS")
    # integer64 is stored as REALSXP in R but represents int64
    # Currently we ingest it as FLOAT64 (may lose precision for large values)
    if (inherits(x, "integer64")) {
      warning("integer64 columns are currently stored as FLOAT64 on GPU. ",
              "Values exceeding 2^53 may lose precision.",
              call. = FALSE)
      return("FLOAT64")
    }
    return("FLOAT64")
  }
  if (is.character(x)) return("STRING")
  if (is.factor(x)) return("DICTIONARY32")
  "UNKNOWN"
}

# Column index lookup (0-based for C++)
col_index <- function(x, name) {
  idx <- match(name, x$schema$names)
  if (is.na(idx)) stop("Column '", name, "' not found")
  idx - 1L
}

# Wrap GPU calls with a clearer error message for allocation failures
#
# `dplyr_error_join` (Phase 7 J5, `check_join_cardinality()`, R/join.R) is a
# deliberately-raised, well-formed dplyr-parity condition -- not a GPU
# allocation/kernel failure -- signaled from INSIDE `lower_join()`, which
# runs underneath this wrapper (`push_join()`, R/execute.R, wraps its own
# `lower_and_execute()` call the same way `push_op()` does for every other
# verb). Re-throwing it unchanged (rather than re-wrapping into the generic
# "GPU operation ... failed" message) preserves its class/message exactly,
# which every `expect_snapshot()`/`cnd_message()` comparison against
# dplyr's own condition in tests/testthat/test-dplyr-join-rows.R depends
# on. No other current caller of `push_op()`/`push_join()` raises a
# classed condition from inside `lower_and_execute()`, so this is a no-op
# for every other verb.
wrap_gpu_call <- function(op_name, expr) {
  tryCatch(
    expr,
    error = function(e) {
      if (inherits(e, "dplyr_error_join")) {
        stop(e)
      }
      msg <- conditionMessage(e)
      stop(
        "GPU operation '", op_name, "' failed. This is often caused by insufficient device memory. ",
        "Try filtering inputs, calling gpu_gc(), or reducing the workload.\n",
        "Original error: ", msg,
        call. = FALSE
      )
    }
  )
}
