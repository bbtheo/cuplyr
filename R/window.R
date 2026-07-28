# Window-expression decomposition (Phase 5, task W1)
#
# See scratchpad/phase5_window_design.md sections 0-1 for the full design
# authority this file implements: the window registry entries (R/ir.R),
# the decomposition algorithm (section 1.2), and the stage-planning /
# chunking rule this file's plan_window_stages() drives.
#
# This file is dormant in W1: nothing else in the package calls
# ir_extract_windows()/plan_window_stages() yet (no verb is wired to
# window functions). W3/W5/W6/W7 cut mutate()/filter()/slice() over to
# this machinery in later Phase 5 tasks.
#
# -----------------------------------------------------------------------------
# window_spec() -- the per-window-call record a "stage" carries (section 2)
# -----------------------------------------------------------------------------

#' Create a window-spec structure
#'
#' One record per window-function call extracted by [ir_extract_windows()].
#' This is the exact shape section 2 of `scratchpad/phase5_window_design.md`
#' documents for `ast_window()`'s specs (W2), produced here so W2 can
#' consume a finished plan's specs directly with no reshaping.
#'
#' @param output_col Character scalar, the synthetic temp column name this
#'   spec's result will be materialized under (e.g. `"..win1.."`)
#' @param fn Character scalar, the canonical window op name (e.g. `"lag"`,
#'   `"cumsum"`, `"mean"`)
#' @param value_col Character scalar, the (already-resolved-to-a-plain-
#'   column) name of the window function's "value" argument, or
#'   `NA_character_` for a call with no value argument (`n()`, bare
#'   `row_number()`)
#' @param order_cols Character vector of order-by column names for this
#'   spec's frame (empty for W1: only rank-family calls with a value
#'   argument set this, to `value_col` itself, since ranking a column
#'   inherently orders by that same column -- real `order_by=` support is
#'   W4)
#' @param order_desc Logical vector, parallel to `order_cols`, `TRUE` for
#'   descending
#' @param output_type GPU type string for this spec's result column
#' @param n Integer scalar, an op-specific count argument (`lag`/`lead`'s
#'   shift amount, `nth`'s element position), or `NA_integer_` if not
#'   applicable
#' @param default An IR literal node (the op's constant fill/default
#'   value), or `NULL` if not applicable/not supplied
#' @param na_rm Logical scalar, `FALSE` unless a future wave adds an
#'   `na_rm=`-aware window op
#' @return A list with exactly these named fields
#' @keywords internal
window_spec <- function(output_col, fn, value_col = NA_character_,
                         order_cols = character(), order_desc = logical(),
                         output_type, n = NA_integer_, default = NULL,
                         na_rm = FALSE) {
  list(
    output_col = output_col,
    fn = fn,
    value_col = value_col,
    order_cols = order_cols,
    order_desc = order_desc,
    output_type = output_type,
    n = n,
    default = default,
    na_rm = na_rm
  )
}

# -----------------------------------------------------------------------------
# Temp name generation (generalized from R/slice.R's
# unique_slice_order_name(), which now just delegates here -- see that
# function's own docs)
# -----------------------------------------------------------------------------

#' Generate a column name guaranteed not to collide with `existing`
#'
#' Appends an incrementing integer suffix to `base` until the candidate
#' name is not already present in `existing`. The very first candidate
#' tried is the bare `base` itself (no suffix); collisions start the
#' suffix at `1`. Shared by every part of the package that needs a
#' synthetic temp column name: `slice_min()`/`slice_max()`'s computed
#' `order_by` column (`R/slice.R`, via `unique_slice_order_name()`), and
#' the window decomposition's `..win*..`/`..winarg*..` temp columns below.
#'
#' @param base Character scalar, the desired base name (e.g. `"..win.."`)
#' @param existing Character vector of names already in use
#' @return A character scalar not in `existing`
#' @keywords internal
unique_temp_name <- function(base, existing) {
  name <- base
  i <- 0L
  while (name %in% existing) {
    i <- i + 1L
    name <- paste0(base, i)
  }
  name
}

# Allocate the next `..win<k>..`/`..winarg<k>..`-style temp name for a
# window decomposition pass: a per-`state` monotonic counter (stored under
# `counter_field`, e.g. `"win_n"`/`"winarg_n"`) produces an already-numbered
# candidate base (`..win1..`, `..win2..`, ...), which still goes through
# unique_temp_name() as a defensive collision check against
# `state$existing_names` (in the astronomically unlikely event a real
# column already has that exact name) -- this is what makes the FIRST
# extracted spec in any given pass "..win1.." rather than the bare
# "..win.." unique_temp_name() would produce on its own (that bare-first
# scheme is what slice_min()/slice_max()'s single synthetic column still
# wants, via unique_slice_order_name(), so unique_temp_name() itself keeps
# that generic collision-retry behavior; only the *candidate* base differs
# here).
#
# @param state The mutable decomposition state environment
# @param counter_field Character scalar, `"win_n"` or `"winarg_n"`
# @param prefix Character scalar, `"..win"` or `"..winarg"`
# @return A character scalar not in `state$existing_names`
# @keywords internal
next_window_temp_name <- function(state, counter_field, prefix) {
  n <- if (is.null(state[[counter_field]])) 1L else state[[counter_field]] + 1L
  assign(counter_field, n, envir = state)
  unique_temp_name(paste0(prefix, n, ".."), state$existing_names)
}

# -----------------------------------------------------------------------------
# ir_extract_windows() -- one bottom-up pass, innermost window calls only
# (section 1.2)
# -----------------------------------------------------------------------------

# Collect the column names referenced directly inside the ARGS of every
# window-call node within `ir` (not columns referenced anywhere else in the
# tree). Used only by plan_window_stages()'s multi-dot chunking rule below:
# an ordinary (non-window) reference to an earlier dot's output is just
# handled by normal schema sequencing (each dot already sees the previous
# dot's output column once its own ast_mutate node has run) and needs no
# special staging; only a window call whose OWN argument reaches for that
# not-yet-materialized column forces a stage-group boundary, since window
# specs are resolved against a frozen schema snapshot before the group's
# own dots are themselves materialized.
#
# @param ir An IR node (or `NULL`)
# @return Character vector (may contain duplicates; callers only use
#   `%in%`)
# @keywords internal
window_arg_cols <- function(ir) {
  if (is.null(ir) || !identical(ir$kind, "call")) {
    return(character())
  }

  entry <- ir_call_registry[[ir$op]]
  own <- if (!is.null(entry) && !is.null(entry$window)) {
    unique(unlist(lapply(ir$args, ir_cols), use.names = FALSE))
  } else {
    character()
  }

  nested <- unique(unlist(lapply(ir$args, window_arg_cols), use.names = FALSE))
  unique(c(own, nested))
}

# Build a window_spec() for a just-extracted, confirmed-innermost window
# call node, mutating `state` (adds a pre-mutate step to `state$pre` when
# the value argument isn't already a bare column, per section 1.2's
# "..winarg{k}.." rule) and returning the spec's output_col name.
#
# `node` is the ORIGINAL call node (its own, unwalked args) -- safe here
# because "innermost" means those args contain no window call anywhere, so
# there is nothing left to recurse into; every column reference inside
# them already resolves against `schema` (the pass-start schema passed to
# ir_extract_windows(), never a same-pass synthetic temp).
#
# @param node A `cuplyr_ir_call` node, confirmed innermost
# @param schema The pass-start schema (list(names=, types=))
# @param state A mutable environment: `pre` (named list of make_mutate_expr()
#   steps, insertion order), `specs` (named list of window_spec()s,
#   insertion order), `seen` (CSE cache: hash key -> output_col name),
#   `existing_names` (character vector, grows as names are allocated)
# @return Character scalar, the new spec's `output_col`
# @keywords internal
extract_one_window_call <- function(node, schema, state) {
  entry <- ir_call_registry[[node$op]]

  value_col <- NA_character_
  if (length(node$args) > 0) {
    value_arg <- node$args[[1]]
    if (identical(value_arg$kind, "col")) {
      value_col <- value_arg$name
    } else {
      winarg_name <- next_window_temp_name(state, "winarg_n", "..winarg")
      state$existing_names <- c(state$existing_names, winarg_name)
      state$pre[[winarg_name]] <- make_mutate_expr(winarg_name, value_arg, schema)
      value_col <- winarg_name
    }
  }

  order_cols <- character()
  order_desc <- logical()
  if (identical(entry$window$kind, "rank") && !is.na(value_col)) {
    order_cols <- value_col
    order_desc <- FALSE
  }

  out_name <- next_window_temp_name(state, "win_n", "..win")
  state$existing_names <- c(state$existing_names, out_name)

  spec <- window_spec(
    output_col = out_name,
    fn = node$op,
    value_col = value_col,
    order_cols = order_cols,
    order_desc = order_desc,
    output_type = ir_infer_type(node, schema),
    n = if (is.null(node$meta$n)) NA_integer_ else node$meta$n,
    default = node$meta$default,
    na_rm = isTRUE(node$meta$na_rm)
  )
  state$specs[[out_name]] <- spec

  out_name
}

#' Extract innermost window calls from an IR tree (one decomposition pass)
#'
#' Implements section 1.2's decomposition algorithm: a single bottom-up
#' walk that replaces every window-class call node whose OWN arguments
#' contain no (nested) window call anywhere -- i.e. every *innermost*
#' window call -- with an `ir_col()` reference to a freshly materialized
#' (or, on a cache hit, previously materialized) temp column, leaving any
#' window call that is NOT yet innermost (its arguments still contain a
#' deeper window call) untouched under its own op name for a later pass
#' (see [plan_window_stages()]'s driving loop) once that deeper call has
#' been resolved.
#'
#' Critically, the innermost-ness test for a given call node uses that
#' node's ORIGINAL (pre-walk) arguments, not the post-walk/post-substitution
#' ones -- this is what makes `lag(cumsum(x))` take two passes rather than
#' collapsing into one: on pass 1, `lag`'s own argument (`cumsum(x)`) does
#' contain a window call, so `lag` is left as-is (its argument is still
#' recursed into, extracting `cumsum(x)` -- which IS innermost on pass 1 --
#' into a temp column reference); only on pass 2, once `lag`'s argument has
#' become a plain `ir_col()`, does `lag` itself qualify as innermost.
#'
#' CSE (common subexpression elimination): two structurally `identical()`
#' window calls (same op, same args, same meta) anywhere in the tree(s)
#' processed against the same `state` resolve to the SAME temp column,
#' checked via a hash key (`rlang::hash()`, already an indirect dependency
#' via rlang) rather than pairwise `identical()` comparisons, for a cheap
#' single-pass lookup. `state$seen` is the cache, keyed by that hash.
#'
#' Any window call argument that isn't already a bare column reference
#' (e.g. `mean(x + y)`) is hoisted into its own pre-mutate step first (an
#' `..winarg{k}..` temp column, added to `state$pre`) -- see
#' [extract_one_window_call()].
#'
#' @param ir An IR node (typically an entire mutate/filter expression)
#' @param schema The schema this pass starts from (list(names=, types=)) --
#'   used to resolve column references inside window-call arguments and
#'   pre-mutate steps; NOT updated mid-pass (a single call's own arguments
#'   can only ever reference genuinely pre-existing columns, never a
#'   sibling temp created earlier in the SAME pass, since window-bearing
#'   expressions within one pass are independent of each other by
#'   construction -- callers advance to a new schema between passes, see
#'   [plan_window_stages()])
#' @param state A mutable environment as described in
#'   [extract_one_window_call()]'s docs; the caller creates a fresh one per
#'   pass and inspects `state$pre`/`state$specs` afterward
#' @return The IR tree with every innermost window call (this pass)
#'   replaced by an `ir_col()` reference; identical (by structure) to `ir`
#'   if it contained no window calls at all
#' @keywords internal
ir_extract_windows <- function(ir, schema, state) {
  walk <- function(node) {
    if (is.null(node) || !identical(node$kind, "call")) {
      return(node)
    }

    entry <- ir_call_registry[[node$op]]
    is_window <- !is.null(entry) && !is.null(entry$window)

    if (is_window) {
      if (ir_has_window(node$args)) {
        # Not innermost this pass: recurse into args (any deeper innermost
        # window call still gets extracted now), but this node stays a
        # window call, deferred to a later pass.
        new_args <- lapply(node$args, walk)
        return(ir_call(node$op, new_args, node$meta))
      }

      key <- rlang::hash(list(op = node$op, args = node$args, meta = node$meta))
      cached <- state$seen[[key]]
      if (!is.null(cached)) {
        return(ir_col(cached))
      }

      out_name <- extract_one_window_call(node, schema, state)
      state$seen[[key]] <- out_name
      return(ir_col(out_name))
    }

    # Ordinary (non-window) call: recurse bottom-up, rebuild.
    new_args <- lapply(node$args, walk)
    ir_call(node$op, new_args, node$meta)
  }

  walk(ir)
}

# -----------------------------------------------------------------------------
# plan_window_stages() -- the multi-pass driver + multi-dot chunking
# (section 1.2)
# -----------------------------------------------------------------------------

# Run ir_extract_windows() repeatedly over a *jointly*-decomposed group of
# expressions (sharing one CSE/temp-name state per pass, so e.g. `mean(x)`
# appearing in two different dots of the same group still gets a single
# spec) until none of them contain any window call. Each pass becomes one
# "stage" (list(pre=, specs=)); the running schema is advanced after each
# pass to include that pass's pre-mutate and spec output columns, so a
# later pass can see an earlier pass's temp columns (this is what lets
# `lag(cumsum(x))` resolve across its two passes/stages).
#
# @param exprs A list of IR nodes (all belonging to the same stage group --
#   see plan_window_stages()'s chunking loop, which decides group
#   membership before calling this)
# @param schema The schema this group starts from
# @param win_n,winarg_n Integer scalars, the `..win<k>..`/`..winarg<k>..`
#   name counters carried in from the caller (0 at the very start of a
#   plan) -- threaded through (and returned, updated) so temp names stay
#   sequential across passes *and* stage groups within one
#   plan_window_stages() call, rather than restarting at 1 every pass
#   (which would still be correct, since unique_temp_name() always falls
#   back to a fresh suffix on collision, but would produce needlessly ugly
#   double-suffixed names like `..win1..1` once an earlier pass's `..win1..`
#   is already in scope)
# @return `list(post = <rewritten exprs, same length/order as input>,
#   stages = <list of list(pre=, specs=)>, schema = <schema after every
#   stage's outputs are appended>, win_n = , winarg_n = )`
# @keywords internal
decompose_window_group <- function(exprs, schema, win_n = 0L, winarg_n = 0L) {
  pending <- exprs
  stages <- list()
  current_schema <- schema

  repeat {
    if (!any(vapply(pending, ir_has_window, logical(1)))) {
      break
    }

    state <- new.env(parent = emptyenv())
    state$seen <- list()
    state$specs <- list()
    state$pre <- list()
    state$existing_names <- current_schema$names
    state$win_n <- win_n
    state$winarg_n <- winarg_n

    pending <- lapply(pending, function(e) ir_extract_windows(e, current_schema, state))

    win_n <- state$win_n
    winarg_n <- state$winarg_n

    stages <- c(stages, list(list(pre = unname(state$pre), specs = unname(state$specs))))

    pre_names <- names(state$pre)
    pre_types <- if (length(state$pre) > 0) {
      vapply(state$pre, `[[`, character(1), "output_type")
    } else {
      character()
    }
    spec_names <- if (length(state$specs) > 0) {
      vapply(state$specs, `[[`, character(1), "output_col")
    } else {
      character()
    }
    spec_types <- if (length(state$specs) > 0) {
      vapply(state$specs, `[[`, character(1), "output_type")
    } else {
      character()
    }

    current_schema <- list(
      names = c(current_schema$names, pre_names, spec_names),
      types = c(current_schema$types, pre_types, spec_types)
    )
  }

  list(post = pending, stages = stages, schema = current_schema, win_n = win_n, winarg_n = winarg_n)
}

#' Plan the window-decomposition stages for a set of expressions
#'
#' Drives [ir_extract_windows()] to full fixed point (every window call
#' resolved) for each verb-level "dot" in `exprs`, applying the multi-dot
#' chunking rule from `scratchpad/phase5_window_design.md` section 1.2:
#' dots are processed in order and batched together (decomposed jointly,
#' sharing CSE and stage passes) UNLESS a later dot's window-call
#' arguments reference an earlier dot's OWN output column name (from the
#' *current* batch) -- in which case that later dot starts a new batch,
#' after the finished batch's own dot outputs are folded into the running
#' schema as real columns (so the new batch can reference them).
#'
#' This is a pure planning function: it builds the stage/spec/pre-mutate
#' structures a verb will later turn into actual `ast_mutate`/`ast_window`
#' nodes (section 1.3; not implemented until W2/W3) -- no AST node is
#' built here, and nothing is pushed onto any `tbl_gpu`.
#'
#' @param exprs A *named* list of IR nodes, one per dot, in verb-call
#'   order; `names(exprs)` are the dots' intended output column names
#'   (needed for the chunking rule above)
#' @param schema The schema `exprs` were parsed against (list(names=,
#'   types=))
#' @param group_cols Character vector of group-by column names (passed
#'   through unchanged for a future `ast_window()` to consume -- W1's own
#'   decomposition logic doesn't need to inspect these at all)
#' @return `list(stages, post, schema, group_cols)`:
#'   \itemize{
#'     \item `stages`: list of `list(pre = <list of make_mutate_expr()
#'       steps>, specs = <list of window_spec()s>)`, in the order they
#'       must execute
#'     \item `post`: named list (same names/order as `exprs`), the fully
#'       rewritten (window-free) IR for each dot -- identical to the
#'       corresponding input node, unchanged, for any dot that had no
#'       window call at all
#'     \item `schema`: the schema after every stage's outputs, AND every
#'       dot's own output column, have been appended (in dot order)
#'     \item `group_cols`: passed through unchanged
#'   }
#' @keywords internal
plan_window_stages <- function(exprs, schema, group_cols = character()) {
  if (length(exprs) == 0) {
    return(list(stages = list(), post = exprs, schema = schema, group_cols = group_cols))
  }
  out_names <- names(exprs)
  if (is.null(out_names) || any(!nzchar(out_names))) {
    stop("plan_window_stages(): `exprs` must be a fully named list.", call. = FALSE)
  }

  n <- length(exprs)
  stages <- list()
  post <- vector("list", n)
  names(post) <- out_names
  running_schema <- schema
  win_n <- 0L
  winarg_n <- 0L

  i <- 1L
  while (i <= n) {
    # Grow the current batch: start with dot i alone, then keep including
    # subsequent dots as long as none of THEIR window-call arguments
    # reference an output name produced by a dot already in this batch.
    group_end <- i
    group_names <- out_names[i]
    while (group_end < n) {
      next_window_cols <- window_arg_cols(exprs[[group_end + 1L]])
      if (any(next_window_cols %in% group_names)) {
        break
      }
      group_end <- group_end + 1L
      group_names <- c(group_names, out_names[group_end])
    }

    group_exprs <- exprs[i:group_end]
    decomposed <- decompose_window_group(group_exprs, running_schema, win_n, winarg_n)
    win_n <- decomposed$win_n
    winarg_n <- decomposed$winarg_n

    stages <- c(stages, decomposed$stages)
    for (k in seq_along(group_exprs)) {
      post[[i + k - 1L]] <- decomposed$post[[k]]
    }

    running_schema <- decomposed$schema
    for (k in seq_along(group_exprs)) {
      nm <- out_names[i + k - 1L]
      out_type <- ir_infer_type(decomposed$post[[k]], running_schema)
      # update_schema_for_expr() (R/mutate.R) replaces-in-place when `nm`
      # already exists (a dot redefining an existing column) instead of
      # appending a duplicate -- shared with mutate()'s own per-dot schema
      # bookkeeping so a later batch sees exactly the schema mutate() itself
      # would produce.
      running_schema <- update_schema_for_expr(running_schema, list(output_col = nm, output_type = out_type))
    }

    i <- group_end + 1L
  }

  list(stages = stages, post = post, schema = running_schema, group_cols = group_cols)
}
