# Utility functions for cuplyr

# Is bit64 installed? (a function so a test can mock it)
bit64_available <- function() requireNamespace("bit64", quietly = TRUE)

# Type mapping from R to GPU types
gpu_type_from_r <- function(x) {
  if (is.logical(x)) return("BOOL8")
  # Date/POSIXct first: seq.Date() and friends can have integer storage.
  if (inherits(x, "Date")) return("TIMESTAMP_DAYS")
  if (inherits(x, "POSIXct")) return("TIMESTAMP_MICROSECONDS")
  if (is.integer(x)) return("INT32")
  if (is.double(x)) {
    # integer64 is a REALSXP holding int64 bits: an exact INT64 column
    if (inherits(x, "integer64")) return("INT64")
    return("FLOAT64")
  }
  if (is.character(x)) return("STRING")
  if (is.factor(x)) return("DICTIONARY32")
  "UNKNOWN"
}

#' Physical GPU storage type for a logical schema type
#'
#' Phase 11 L2 design decision: cuplyr keeps factor columns as plain INT32
#' codes physically on the GPU rather than adopting cudf's real
#' `DICTIONARY32` column type (a dictionary-encoded keys+indices structure).
#' `"DICTIONARY32"` is therefore a LOGICAL type only, in `gpu_type_from_r()`
#' and every schema's `$types` -- confirmed by `df_to_gpu()`: a factor's
#' `TYPEOF()` is `INTSXP`, so it uploads through the exact same
#' `integer_to_gpu()` path as a real integer column (`src/transfer_io.cpp`).
#'
#' Call sites split into two camps:
#' \itemize{
#'   \item \strong{Physical-column sites} (talk to the actual `cudf::column`,
#'     which is INT32 for a factor): C++ casting primitives
#'     (`gpu_cast_column()`, `gpu_make_null_column()`), transfer-layer
#'     assertions. These must route the type string through this helper
#'     first, or hand the C++ side a `"DICTIONARY32"` it has no case for.
#'   \item \strong{Logical-semantics sites} (care whether a column IS a
#'     factor, not what bytes back it): the mutate arithmetic guard
#'     (`R/ast.R::infer_mutate_output_type()`), join key type-compatibility
#'     checks (`resolve_join_key_types()`, `R/join.R`), memory accounting
#'     (`R/gpu-memory.R`). These must keep seeing `"DICTIONARY32"` verbatim
#'     -- routing them through this helper would make a factor column
#'     silently indistinguishable from a real integer column (e.g. letting
#'     arithmetic on a factor slip through, or joining a factor key against
#'     a plain integer key as if they were the same domain).
#' }
#'
#' @param type A GPU type string (e.g. from `schema$types`), or a vector of them
#' @return `"INT32"` for `"DICTIONARY32"`, `type` unchanged otherwise
#' @keywords internal
gpu_physical_type <- function(type) {
  if (is.character(type) && length(type) > 0L && !anyNA(type)) {
    # (ifelse() on a character vector, without its general machinery)
    type[which(type == "DICTIONARY32")] <- "INT32"
    return(type)
  }
  ifelse(type == "DICTIONARY32", "INT32", type)
}

# Does the session order strings as the GPU does (bytes)? Only the C and
# POSIX collations; R sorts "a" before "B" under en_US.UTF-8 and C.UTF-8.
# @keywords internal
session_collation_is_c <- function() {
  Sys.getlocale("LC_COLLATE") %in% c("C", "POSIX")
}

# Column index lookup (0-based for C++)
col_index <- function(x, name) {
  idx <- match(name, x$schema$names)
  if (is.na(idx)) stop("Column '", name, "' not found")
  idx - 1L
}

# ---- Release scopes and the OOM retry (F58) ---------------------------------
#
# Every GPU step runs inside wrap_gpu_call(). It gives the step a release scope
# (src/gpu_scope.cpp), rewords what the GPU layer raised, and retries an
# out-of-memory failure once.
#
# Who owns a table. A table belongs to the user-level call that made it: the
# innermost cuplyr function on the stack whose caller is not cuplyr code
# (gpu_entry_frame()). A verb called from the console, from a user function,
# from a data mask (a filter() predicate, a join's `y` argument), or from
# another package is a user-level call. A cuplyr function that cuplyr code
# calls, including through a dplyr generic (UseMethod gives the method its
# generic's caller), works for the user-level call around it. Base closures
# that call back (lapply, vapply, Map, Reduce) are looked through, so
# `lapply(dots, compute)` inside bind_rows() is still bind_rows() working.
#
# The first step of a user-level call opens that call's scope and registers
# its close on the call's frame (gpu_call_scope_ensure()). Each step's tables
# that the step returns go to the call's scope. When the call returns, every
# one of them that is not reachable from its return value is released: the
# step tables of a multi-step verb (a window filter's stages, summarise(.by)'s
# row ids, the keys cur_group_id() joins on, a lazy input that slice_sample()
# or bind_rows() computes) are freed whichever verb runs them, with no code in
# the verb. What a user-level call returns is the user's: no enclosing call's
# scope ever sees it, so a table user code builds (and maybe caches) while a
# verb evaluates its arguments or predicates stays valid. When in doubt the
# rule errs towards the user: a caller cuplyr can't identify (code evaluated
# in a data mask or `eval()` environment) counts as user code, and the table
# is then left to R's GC.
#
# Consequences for code in this package:
# - A step must return the tables it makes, in lists, attributes or a
#   vec_proxy() row-handle vector: a table held only in an environment or a
#   closure is released when the call returns. So is a table parked in a cache
#   that outlives the call (none does).
# - A call that hands its value back by a non-local return (rlang::return_from())
#   settles its scope with that value first (gpu_call_scope_settle()); an
#   unwind would otherwise close it with nothing kept.
# - A GPU op outside wrap_gpu_call() is not tracked (left to R's GC) and gets
#   no OOM retry, so every op that creates a table goes through one.
#
# OOM retry: an out-of-memory failure in the OUTERMOST active wrapper frees the
# attempt's step scope, runs gc() + gpu_gc(), and evaluates `expr` once more, in
# the caller's frame. So `expr` must be safe to evaluate twice: it reads only
# its inputs and assigns nothing it reads before assigning (tables are
# immutable, F57), and it draws no random numbers and has no other effect a
# second run would repeat (slice_sample() draws its rows before the wrapped
# gather). A second OOM raises `cuplyr_error_gpu_oom`. A nested wrapper rethrows
# an OOM unchanged (the outermost one decides), so a step is retried at most
# once. A warning or message the failed attempt signalled has already reached
# the user, so the retry's identical ones are muffled (one recording handler on
# the happy path).
#
# Only what the GPU layer raised is reworded: a C++ exception (cudf, RMM,
# Rcpp::stop) or a plain stop(). A condition R code classed on purpose passes
# through unchanged -- `dplyr_error_join` (J5, `check_join_cardinality()`, whose
# class/message every snapshot against dplyr's own condition depends on),
# `vctrs_error_subscript`, `cuplyr_error_sum_overflow` / `cuplyr_summarise_rerun`
# (F42, caught by class to rerun on the CPU), `cuplyr_error_rowwise_retype`,
# rlang::abort() errors, dplyr's own errors. This holds for every wrap, however
# much R logic runs inside it. `reword = FALSE` passes every error but a
# surviving OOM through as raised (compute(): a lazy plan's errors read as they
# did before F58).
gpu_wrap_state <- new.env(parent = emptyenv())
gpu_wrap_state$depth <- 0L

# What runs in frame `p`: 1L cuplyr code, 2L a base closure (looked through),
# 0L anything else (user code, another package, an eval() context). `ns` is
# cuplyr's namespace.
gpu_frame_kind <- function(p, ns, frames = sys.frames(), env = environment(sys.function(p))) {
  repeat {
    if (is.null(env)) {
      return(0L)
    }
    if (identical(env, ns)) {
      return(1L)
    }
    if (identical(env, .BaseNamespaceEnv)) {
      return(2L)
    }
    # Top-level code: the global environment is no running call's frame (an
    # eval() context that evaluates in it is a builtin's, kind 0 below), so the
    # search below can only end in 0.
    if (identical(env, .GlobalEnv)) {
      return(0L)
    }
    # A closure made inside a function (`lapply(x, function(t) ...)`) is the
    # code of the function that made it, if that call is still running. Under
    # load_all() a test's environment also descends from the namespace, so
    # the namespace is only reached through running frames. (`frames` is a
    # promise: the stack is read only when a frame needs the search, once per
    # gpu_entry_frame() walk.)
    q <- 0L
    for (j in rev(seq_len(p - 1L))) {
      if (identical(frames[[j]], env)) {
        q <- j
        break
      }
    }
    if (q == 0L) {
      return(0L)
    }
    p <- q
    env <- environment(sys.function(p))
  }
}

# Frame number of the user-level call that frame `n` (cuplyr code) works for.
gpu_entry_frame <- function(n) {
  ns <- environment(gpu_entry_frame)
  parents <- sys.parents()
  delayedAssign("frames", sys.frames())
  entry <- n
  i <- n
  repeat {
    p <- parents[[i]]
    # 0: called from the global env; i itself: from an environment that is no
    # running frame (a data mask)
    if (p <= 0L || p >= i) {
      return(entry)
    }
    # (the two common answers, cuplyr code and a base closure, without the call)
    env <- environment(sys.function(p))
    kind <- if (identical(env, ns)) {
      1L
    } else if (identical(env, .BaseNamespaceEnv)) {
      2L
    } else {
      gpu_frame_kind(p, ns, frames, env)
    }
    if (kind == 0L) {
      return(entry)
    }
    if (kind == 1L) {
      entry <- p
    }
    i <- p
  }
}

# Make sure the user-level call that frame `n` works for has an open call
# scope. Its close is registered on that call's frame before it opens, so an
# interrupt can't leave it open (closing a frame without one is a no-op).
gpu_call_scope_ensure <- function(n) {
  frame <- sys.frame(gpu_entry_frame(n))
  if (!gpu_call_scope_active(frame)) {
    close <- as.call(list(gpu_call_scope_exit, frame, quote(base::returnValue(NULL))))
    do.call(base::on.exit, list(close, add = TRUE), envir = frame)
    gpu_call_scope_open(frame)
  }
  invisible(NULL)
}

# Close, keeping `value`, the call scope of the user-level call that frame `n`
# works for: for a call that is about to return `value` by a non-local return.
gpu_call_scope_settle <- function(n, value) {
  gpu_call_scope_exit(sys.frame(gpu_entry_frame(n)), value)
}

# A condition from the GPU layer, or a bare stop(): the ones this wrapper owns.
gpu_rewordable <- function(e) {
  inherits(e, "C++Error") || identical(class(e), c("simpleError", "error", "condition"))
}

# Only memory errors get the device-memory hint and the retry; anything else (a
# cudf logic_error, a type mismatch) would be misdiagnosed by it. The message
# test only applies to errors this wrapper owns: an R condition that merely
# mentions "out of memory" is not retried.
is_gpu_oom_error <- function(e) {
  inherits(e, c("std::bad_alloc", "rmm::bad_alloc", "rmm::out_of_memory")) ||
    (gpu_rewordable(e) &&
       grepl("out[_ ]of[_ ]memory|bad_alloc|cudaErrorMemoryAllocation", conditionMessage(e)))
}

gpu_oom_hint <- " This is often caused by insufficient device memory. Try filtering inputs, calling gpu_gc(), or reducing the workload."

wrap_gpu_call <- function(op_name, expr, reword = TRUE) {
  state <- gpu_wrap_state
  depth <- state$depth
  outermost <- depth == 0L
  # The handler is registered before any state changes, and restores the saved
  # depth instead of decrementing it, so an interrupt anywhere in here can't
  # leave the counter (and with it "outermost") wrong for the session.
  scope <- NULL
  on.exit({
    if (!is.null(scope)) gpu_scope_close(scope, NULL)
    state$depth <- depth
  }, add = TRUE)
  gpu_call_scope_ensure(sys.nframe())
  state$depth <- depth + 1L
  scope <- gpu_scope_open()

  # Warnings/messages of the first attempt (outermost wrapper only), so the
  # retry can muffle the repeats.
  seen <- character()
  err <- tryCatch({
    value <- if (outermost) {
      withCallingHandlers(
        expr,
        warning = function(w) seen[[length(seen) + 1L]] <<- paste(class(w)[[1L]], conditionMessage(w)),
        message = function(m) seen[[length(seen) + 1L]] <<- paste(class(m)[[1L]], conditionMessage(m))
      )
    } else {
      expr
    }
    gpu_scope_close(scope, value)
    scope <- NULL
    return(value)
  }, error = function(e) e)
  gpu_scope_close(scope, NULL)
  scope <- NULL

  if (is_gpu_oom_error(err)) {
    if (!outermost) stop(err)
    # (only the retry needs the expression and the caller's frame)
    ex <- substitute(expr)
    env <- parent.frame()
    # Free what the failed attempt left to R's collector, then try once more.
    invisible(gc(full = TRUE))
    try(gpu_gc(aggressive = FALSE), silent = TRUE)
    scope <- gpu_scope_open()
    muffle <- function(c) {
      key <- paste(class(c)[[1L]], conditionMessage(c))
      hit <- match(key, seen)
      if (!is.na(hit)) {
        seen <<- seen[-hit]
        if (inherits(c, "message")) invokeRestart("muffleMessage") else invokeRestart("muffleWarning")
      }
    }
    err <- tryCatch({
      value <- withCallingHandlers(eval(ex, env), warning = muffle, message = muffle)
      gpu_scope_close(scope, value)
      scope <- NULL
      return(value)
    }, error = function(e) e)
    gpu_scope_close(scope, NULL)
    scope <- NULL
    if (is_gpu_oom_error(err)) {
      rlang::abort(
        paste0(
          "GPU operation '", op_name, "' ran out of device memory. A retry after garbage collection ",
          "(gc() and gpu_gc()) was attempted and failed too.", gpu_oom_hint, "\n",
          "Original error: ", conditionMessage(err)
        ),
        class = "cuplyr_error_gpu_oom",
        call = NULL
      )
    }
  }
  if (!reword || !gpu_rewordable(err)) stop(err)
  hint <- if (is_gpu_oom_error(err)) gpu_oom_hint else ""
  # Classed so an enclosing wrapper (a multi-step verb around push_op() calls)
  # passes it through instead of wrapping the message twice.
  stop(structure(
    class = c("cuplyr_error_gpu", "simpleError", "error", "condition"),
    list(
      message = paste0("GPU operation '", op_name, "' failed.", hint, "\n",
                       "Original error: ", conditionMessage(err)),
      call = NULL
    )
  ))
}

# Internal: integer-backed Date/POSIXct (seq.Date(), structure(1L, class =
# "Date")) -> double storage with class and tzone kept; anything else as is.
# The upload dispatches on TYPEOF, so this keeps them timestamps on the GPU.
timestamp_as_double <- function(x) {
  if (is.integer(x) && (inherits(x, "Date") || inherits(x, "POSIXct"))) {
    storage.mode(x) <- "double"
  }
  x
}

# union()/setdiff()/intersect() of plain vectors without base's per-call class
# probing (base 4.5+ runs `.set_ops_need_as_vector()`, two tryCatch(x[0L]), on
# every call, and dplyr's generics add a dispatch on top): measurable on the
# eager per-call path, where verbs call these dozens of times on short name
# vectors. Same result as base for NULL or a plain atomic vector without
# dim/class; anything else (a factor, a data frame) takes the original call.
# @keywords internal
is_plain_vector <- function(x) {
  is.null(x) || (is.atomic(x) && is.null(dim(x)) && !is.object(x))
}

set_union <- function(x, y) {
  if (is_plain_vector(x) && is_plain_vector(y)) {
    x <- unique(x)
    names(x) <- NULL
    y <- unique(y)
    names(y) <- NULL
    return(c(x, y[match(y, x, 0L) == 0L]))
  }
  union(x, y)
}

set_diff <- function(x, y) {
  if (is_plain_vector(x) && is_plain_vector(y)) {
    if (is.null(x)) return(NULL)
    if (length(x) == 0L) {
      names(x) <- NULL
      return(x)
    }
    x <- unique(x)
    names(x) <- NULL
    y <- unique(y)
    names(y) <- NULL
    return(x[match(x, y, 0L) == 0L])
  }
  setdiff(x, y)
}

set_intersect <- function(x, y) {
  if (is_plain_vector(x) && is_plain_vector(y)) {
    if (is.null(x) || is.null(y)) return(NULL)
    if (length(x) == 0L && typeof(x) == typeof(y)) {
      names(x) <- NULL
      return(x)
    }
    x <- unique(x)
    names(x) <- NULL
    y <- unique(y)
    names(y) <- NULL
    return(c(x[match(x, y, 0L) > 0L], y[0L]))
  }
  intersect(x, y)
}

# The quosure of an argument nobody gave (mutate()'s `.by`, `.before`, `.after`
# default to NULL), so a call doesn't pay an enquo() per absent argument.
null_quo <- rlang::new_quosure(NULL, rlang::empty_env())
