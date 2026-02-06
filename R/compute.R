# Computation and Execution Control for Lazy Evaluation
#
# This module provides functions to control when GPU operations are executed:
# - compute(): Execute pending operations, keep result on GPU
# - collapse(): Insert optimization barrier without execution
# - as_eager()/as_lazy(): Switch execution modes
# - is_lazy(): Check execution mode

#' Force computation of pending GPU operations
#'
#' Executes any pending lazy operations and stores the result in a new GPU
#' table. Data remains on the GPU (unlike `collect()` which transfers to R).
#'
#' @param x A `tbl_gpu` object.
#' @param ... Additional arguments (unused, for compatibility).
#' @param name Ignored (included for dplyr compatibility).
#'
#' @return A `tbl_gpu` with all operations materialized and `lazy_ops` cleared.
#'
#' @details
#' Use `compute()` when you want to:
#' \itemize{
#'   \item Force optimization and execution of a lazy pipeline
#'   \item Create a checkpoint before branching operations
#'   \item Free memory from intermediate tables
#'   \item Prepare data for non-cuplyr functions that need a materialized table
#' }
#'
#' In eager mode, `compute()` is a no-op since operations execute immediately.
#'
#' @seealso
#' \code{\link{collect.tbl_gpu}} to bring data back to R,
#' \code{\link{collapse.tbl_gpu}} to add optimization barrier without executing
#'
#' @export
#' @importFrom dplyr compute
#'
#' @examples
#' if (has_gpu()) {
#'   # Lazy pipeline
#'   lazy_result <- tbl_gpu(mtcars, lazy = TRUE) |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425)
#'
#'   # Force execution, keep on GPU
#'   gpu_result <- lazy_result |> compute()
#'
#'   # Now branch into two different analyses
#'   analysis1 <- gpu_result |> filter(cyl == 4) |> collect()
#'   analysis2 <- gpu_result |> filter(cyl == 6) |> collect()
#' }
compute.tbl_gpu <- function(x, ..., name = NULL) {
  if (!is_tbl_gpu(x)) {
    stop("compute() requires a tbl_gpu object", call. = FALSE)
  }

  # Nothing to do if no pending ops or in eager mode
  if (is.null(x$lazy_ops) || x$exec_mode == "eager") {
    return(x)
  }

  # Run optimizer and execute
  optimized <- optimize_ast(x$lazy_ops)
  new_ptr <- lower_and_execute(optimized, x$ptr)
  final_schema <- infer_schema(x$lazy_ops)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = final_schema,
    lazy_ops = NULL,
    groups = x$groups,
    exec_mode = x$exec_mode
  )
}

#' Create a subquery barrier without executing
#'
#' Marks the current point in the pipeline as an optimization barrier.
#' Operations before and after the barrier are optimized separately.
#' Does not execute anything until `collect()` or `compute()` is called.
#'
#' @param x A `tbl_gpu` object.
#' @param ... Additional arguments (unused).
#'
#' @return A `tbl_gpu` with a barrier marker in its lazy_ops.
#'
#' @details
#' Use `collapse()` when you want to prevent certain optimizations from
#' crossing a boundary (e.g., prevent filter pushdown past a certain point).
#'
#' In eager mode, `collapse()` is a no-op.
#'
#' @export
#' @importFrom dplyr collapse
collapse.tbl_gpu <- function(x, ...) {
  if (!is_tbl_gpu(x)) {
    stop("collapse() requires a tbl_gpu object", call. = FALSE)
  }

  # No-op in eager mode
  if (x$exec_mode == "eager") {
    return(x)
  }

  # Add barrier node to lazy_ops
  if (is.null(x$lazy_ops)) {
    # Create source node first
    x$lazy_ops <- ast_source(x$schema)
  }

  x$lazy_ops <- ast_barrier(x$lazy_ops)
  x
}

#' Switch to eager execution mode
#'
#' Computes any pending operations and returns a `tbl_gpu` that will
#' execute all subsequent operations immediately.
#'
#' @param .data A `tbl_gpu` object.
#' @return A `tbl_gpu` in eager execution mode.
#'
#' @export
#'
#' @examples
#' if (has_gpu()) {
#'   # Start lazy, switch to eager mid-pipeline
#'   result <- tbl_gpu(mtcars, lazy = TRUE) |>
#'     filter(mpg > 20) |>
#'     as_eager() |>
#'     mutate(x = hp * 2) |>  # executes immediately
#'     collect()
#' }
as_eager <- function(.data) {
  if (!is_tbl_gpu(.data)) {
    stop("as_eager() requires a tbl_gpu object", call. = FALSE)
  }

  # Compute pending ops first
  if (!is.null(.data$lazy_ops)) {
    .data <- compute(.data)
  }

  .data$exec_mode <- "eager"
  .data
}

#' Switch to lazy execution mode
#'
#' Returns a `tbl_gpu` that will defer operations until `collect()` or
#' `compute()` is called.
#'
#' @param .data A `tbl_gpu` object.
#' @return A `tbl_gpu` in lazy execution mode.
#'
#' @export
#'
#' @examples
#' if (has_gpu()) {
#'   # Create in eager mode, switch to lazy
#'   gpu_data <- tbl_gpu(mtcars) |>
#'     as_lazy() |>
#'     filter(mpg > 20) |>
#'     mutate(kpl = mpg * 0.425)  # not yet executed
#'
#'   # Execute with collect
#'   result <- gpu_data |> collect()
#' }
as_lazy <- function(.data) {
  if (!is_tbl_gpu(.data)) {
    stop("as_lazy() requires a tbl_gpu object", call. = FALSE)
  }

  .data$exec_mode <- "lazy"
  .data
}

#' Check if a tbl_gpu uses lazy execution
#'
#' @param .data A `tbl_gpu` object.
#' @return Logical. `TRUE` if the table is in lazy execution mode.
#'
#' @export
#'
#' @examples
#' if (has_gpu()) {
#'   eager_tbl <- tbl_gpu(mtcars)
#'   is_lazy(eager_tbl)  # FALSE
#'
#'   lazy_tbl <- tbl_gpu(mtcars, lazy = TRUE)
#'   is_lazy(lazy_tbl)   # TRUE
#' }
is_lazy <- function(.data) {
  is_tbl_gpu(.data) && identical(.data$exec_mode, "lazy")
}

#' Check if a tbl_gpu has pending lazy operations
#'
#' @param .data A `tbl_gpu` object.
#' @return Logical. `TRUE` if there are pending operations.
#'
#' @export
has_pending_ops <- function(.data) {
  is_tbl_gpu(.data) && !is.null(.data$lazy_ops)
}

#' Show pending operations for a lazy tbl_gpu
#'
#' @param .data A `tbl_gpu` object.
#' @return Invisibly returns the AST, prints it for inspection.
#'
#' @export
show_query <- function(.data) {
  if (!is_tbl_gpu(.data)) {
    stop("show_query() requires a tbl_gpu object", call. = FALSE)
  }

  if (is.null(.data$lazy_ops)) {
    cat("No pending operations (eager mode or already computed)\n")
  } else {
    cat("Pending operations:\n")
    print(.data$lazy_ops)
    cat("\nAST depth:", ast_depth(.data$lazy_ops), "\n")
    cat("Node count:", ast_count(.data$lazy_ops), "\n")
  }

  invisible(.data$lazy_ops)
}
