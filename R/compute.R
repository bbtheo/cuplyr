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
#' Tables are lazy by default, so `compute()` is how you run a pipeline and keep
#' the result on the GPU. In eager mode (`tbl_gpu(df, lazy = FALSE)`),
#' `compute()` is a no-op since operations execute immediately. Printing a lazy
#' table also runs its plan, but keeps nothing; `compute()` first if you want to
#' look at the result repeatedly.
#'
#' @seealso
#' \code{\link{collect.tbl_gpu}} to bring data back to R,
#' \code{\link{collapse.tbl_gpu}} to add optimization barrier without executing
#'
#' @export
#' @importFrom dplyr compute
#'
#' @examples
#' library(dplyr)
#' if (has_gpu()) {
#'   # Lazy pipeline
#'   lazy_result <- tbl_gpu(mtcars) |>  # lazy by default
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

  # Backward compatibility: older objects may store no pending ops as list().
  ops <- .subset2(x, "lazy_ops")
  if (is.list(ops) && length(ops) == 0) {
    x$lazy_ops <- NULL
    ops <- NULL
  }

  if (!is.null(ops) && !inherits(ops, "ast_node")) {
    stop("Invalid lazy_ops: expected an AST node or NULL.", call. = FALSE)
  }

  # Nothing to do if no pending ops or in eager mode
  if (is.null(ops) || .subset2(x, "exec_mode") == "eager") {
    return(x)
  }

  plan <- compute_plan(x, ops)

  copy_frame_meta(new_tbl_gpu(
    ptr = plan$ptr,
    schema = plan$schema,
    lazy_ops = NULL,
    groups = .subset2(x, "groups"),
    exec_mode = .subset2(x, "exec_mode"),
    group_drop = .subset2(x, "group_drop") %||% TRUE,
    nrow = .subset2(x, "nrow"),
    fixed_keys = .subset2(x, "fixed_keys"),
    unspecified = .subset2(x, "unspecified"),
    rowwise = is_rowwise_gpu(x)
  ), x)
}

# Optimize and run the pending plan of a lazy `x`: the GPU pointer of the result
# and its schema, which is all `collect()` needs (compute() wraps them in a
# tbl_gpu).
# @param ops `x$lazy_ops`, an AST node
# @keywords internal
compute_plan <- function(x, ops = .subset2(x, "lazy_ops")) {
  # One schema memo for the optimizer and the lowering (R/ast.R): each node's
  # schema is inferred once, so both are linear in the plan's depth.
  memo_prev <- .schema_memo$plain
  on.exit(.schema_memo$plain <- memo_prev, add = TRUE)
  schema_memo_open()

  # Run optimizer and execute
  optimized <- optimize_ast(ops)
  # The stored schema (R/execute.R `current_schema()`) is the plan's output
  # schema with its factor levels and POSIXct zones: every writer of
  # `lazy_ops` stores what `infer_schema_full(lazy_ops)` would give.
  final_schema <- current_schema(x)

  # Safety net: the result is named by the unoptimized schema but holds the
  # optimized plan's columns by position. Passes keep the output columns and
  # their order by contract; if one ever reorders them, gather back by name
  # (rather than return values under the wrong names). A different column set
  # is a bug in a pass, so fail loudly.
  optimized_names <- infer_schema_memo(optimized)$names
  reorder <- NULL
  if (!identical(optimized_names, final_schema$names)) {
    if (anyDuplicated(optimized_names) > 0 ||
        !setequal(optimized_names, final_schema$names)) {
      stop("Internal error: the optimized plan's columns (",
           paste(optimized_names, collapse = ", "), ") differ from the plan's ",
           "schema (", paste(final_schema$names, collapse = ", "), ").",
           call. = FALSE)
    }
    reorder <- match(final_schema$names, optimized_names) - 1L
  }

  # One GPU step (F58): an out-of-memory failure is retried once. Other
  # errors pass through as lowering raised them (`reword = FALSE`), as they
  # did before F58.
  new_ptr <- wrap_gpu_call("compute", reword = FALSE, {
    ptr <- lower_and_execute(optimized, .subset2(x, "ptr"))
    if (is.null(reorder)) ptr else gpu_select(ptr, reorder)
  })

  list(ptr = new_ptr, schema = final_schema)
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
  # A barrier passes its input's columns through; the stored schema is the plan's
  # (current_schema()), which also normalizes the empty metadata slots a table
  # fresh from tbl_gpu() carries as explicit NULLs.
  x$schema <- infer_schema_full(x$lazy_ops)
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
#' library(dplyr)
#' if (has_gpu()) {
#'   # Tables are lazy by default; switch to eager mid-pipeline
#'   result <- tbl_gpu(mtcars) |>
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
  if (has_pending_ops(.data)) {
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
#' library(dplyr)
#' if (has_gpu()) {
#'   # Create in eager mode, switch to lazy
#'   gpu_data <- tbl_gpu(mtcars, lazy = FALSE) |>
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
#'   lazy_tbl <- tbl_gpu(mtcars)
#'   is_lazy(lazy_tbl)   # TRUE
#'
#'   eager_tbl <- tbl_gpu(mtcars, lazy = FALSE)
#'   is_lazy(eager_tbl)  # FALSE
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
  # (is_tbl_gpu() inline: verbs ask this several times per call)
  inherits(.data, "tbl_gpu") && inherits(.subset2(.data, "lazy_ops"), "ast_node")
}

#' Show pending operations for a lazy tbl_gpu
#'
#' `tbl_gpu` method for dplyr's [dplyr::show_query()] generic. Prints the
#' pending AST (nothing is executed).
#'
#' @param x A `tbl_gpu` object.
#' @param ... Not used.
#' @return `x`, invisibly.
#'
#' @export
#' @importFrom dplyr show_query
show_query.tbl_gpu <- function(x, ...) {
  if (is.null(x$lazy_ops)) {
    cat("No pending operations (eager mode or already computed)\n")
  } else if (!inherits(x$lazy_ops, "ast_node")) {
    cat("Invalid pending operations object (expected AST node)\n")
  } else {
    cat("Pending operations:\n")
    print(x$lazy_ops)
    cat("\nAST depth:", ast_depth(x$lazy_ops), "\n")
    cat("Node count:", ast_count(x$lazy_ops), "\n")
  }

  invisible(x)
}

#' @importFrom dplyr show_query
#' @export
dplyr::show_query

#' Explain a tbl_gpu query
#'
#' `tbl_gpu` method for dplyr's [dplyr::explain()] generic. Prints
#' [show_query()]'s output and, when operations are pending, the optimized
#' plan `compute()` would run. Nothing is executed.
#'
#' @param x A `tbl_gpu` object.
#' @param ... Not used.
#' @return `x`, invisibly.
#'
#' @export
#' @importFrom dplyr explain
explain.tbl_gpu <- function(x, ...) {
  show_query(x)
  # Same condition compute() uses to decide whether it has a plan to run.
  if (is_lazy(x) && has_pending_ops(x)) {
    cat("\nOptimized plan:\n")
    print(optimize_ast(x$lazy_ops))
  }
  invisible(x)
}

#' Source compatibility and upload for tbl_gpu
#'
#' `tbl_gpu` methods for dplyr's [dplyr::same_src()] and [dplyr::auto_copy()]
#' generics, used by the join verbs' `copy =` handling.
#'
#' @param x A `tbl_gpu`.
#' @param y Object to test or upload.
#' @param copy Passed by dplyr; `auto_copy()` only reaches this method when
#'   copying is allowed.
#' @param ... Not used.
#' @return `same_src()`: `TRUE` when `y` is a `tbl_gpu`. `auto_copy()`: `y`
#'   as a `tbl_gpu`.
#'
#' @export
#' @importFrom dplyr same_src
same_src.tbl_gpu <- function(x, y) {
  is_tbl_gpu(y)
}

#' @rdname same_src.tbl_gpu
#' @export
#' @importFrom dplyr auto_copy
auto_copy.tbl_gpu <- function(x, y, copy = FALSE, ...) {
  # dplyr's data frame method coerces with as.data.frame() first, so a list
  # (or matrix) `y` is accepted the same way here.
  if (!is.data.frame(y)) {
    y <- as.data.frame(y)
  }
  # in `x`'s mode: a lazy y would make an eager x's join result lazy
  upload_other_side(y, lazy = identical(.subset2(x, "exec_mode"), "lazy"))
}
