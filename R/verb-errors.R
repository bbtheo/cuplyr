# The error call and the missing-column errors every verb shares (#13).
#
# dplyr reports an error against the call the user made (`filter()`, `count()`,
# `add_tally()`), found by dplyr_error_call() (R/conditions.R): start at the verb's
# frame and follow the `.__dplyr_error_call__.` binding of the frame that called it,
# as often as there is one. verb_error_call() is that walk, started at the
# user-level call (gpu_entry_frame()), for every error cuplyr raises itself;
# fallback_error_call() makes dplyr's own errors on a CPU route report the same.
# Nothing here runs unless an error is raised or a call falls back to the CPU.

# The user-level call whose frame `frame` belongs to, as dplyr reports errors
# of it: the frame of that call, or the call a `.__dplyr_error_call__.` binding
# up the stack points at (dplyr_error_call()), which is a frame again. NULL when
# `frame` is not a running call (nothing to report).
# @param frame A frame environment (default: the caller's)
# @keywords internal
verb_error_call <- function(frame = parent.frame()) {
  frames <- sys.frames()
  n <- 0L
  for (k in rev(seq_along(frames))) {
    if (identical(frames[[k]], frame)) {
      n <- k
      break
    }
  }
  if (n == 0L) {
    return(NULL)
  }
  call <- sys.frame(gpu_entry_frame(n))
  while (is.environment(call)) {
    caller <- rlang::eval_bare(quote(base::parent.frame()), call)
    caller_call <- caller[[".__dplyr_error_call__."]]
    if (is.null(caller_call)) {
      break
    }
    call <- caller_call
  }
  call
}

# `expr_fn`, the function a CPU fallback runs dplyr's verb in, with the frame of
# the user's call bound as the `.__dplyr_error_call__.` its frame hands dplyr
# (dplyr_local_error_call()): dplyr then builds and signals its own errors and
# warnings with the user's call (`slice_max()`, `count()`, `add_tally()`), where it
# would report its own internal one (`dplyr::summarise(.data = tbl, ...)`).
# @param frame The frame gpu_fallback() was called from
# @keywords internal
fallback_error_call <- function(expr_fn, frame) {
  n <- 0L
  frames <- sys.frames()
  for (k in rev(seq_along(frames))) {
    if (identical(frames[[k]], frame)) {
      n <- k
      break
    }
  }
  if (n == 0L || !is.function(expr_fn) || is.primitive(expr_fn)) {
    return(expr_fn)
  }
  bind <- call("<-", quote(.__dplyr_error_call__.), sys.frame(gpu_entry_frame(n)))
  body(expr_fn) <- call("{", bind, body(expr_fn))
  expr_fn
}

# The error rlang raises for `.data$xx` / `.data[["xx"]]` of a column the data has
# not (class `rlang_error_data_pronoun_not_found`, message "Column `xx` not found
# in `.data`.", call the pronoun as written), raised by rlang itself: `expr` is
# evaluated against the pronoun of a mask holding `names`. Falls back to `stop()`
# with the message if it evaluates anyway.
# @param expr The `.data$xx` call; `env` where it was written; `names` the columns
# @keywords internal
stop_data_pronoun_missing <- function(expr, env, names, name = NULL) {
  mask <- stats::setNames(vector("list", length(names)), names)
  cnd <- tryCatch(
    {
      rlang::eval_tidy(rlang::new_quosure(expr, env), data = mask)
      NULL
    },
    error = function(e) e
  )
  if (inherits(cnd, "rlang_error_data_pronoun_not_found")) {
    rlang::cnd_signal(cnd)
  }
  stop("Column `", name %||% as.character(expr[[3L]]), "` not found in `.data`.", call. = FALSE)
}

# The first `.data$xx` / `.data[["xx"]]` of `expr` (in evaluation order) whose
# column is in none of `names`, as the pronoun call, or NULL. Nothing is
# evaluated; a subscript that is no string or symbol is not looked at, and neither
# are function literals and formulas (their bodies run elsewhere).
# @keywords internal
expr_first_missing_pronoun <- function(expr, env, names) {
  walk <- function(e, env) {
    if (rlang::is_quosure(e)) {
      return(walk(rlang::quo_get_expr(e), rlang::quo_get_env(e)))
    }
    if (!is.call(e)) {
      return(NULL)
    }
    if (is_data_pronoun_call(e)) {
      nm <- pronoun_call_name(e, env, list(names = names))
      return(if (!is.null(nm) && !nm %in% names) list(call = e, env = env, name = nm))
    }
    head <- e[[1L]]
    if (is.symbol(head) && as.character(head) %in% c("function", "~", "quote", "bquote", "expression", "substitute")) {
      return(NULL)
    }
    for (a in as.list(e)) {
      if (!rlang::is_missing(a)) {
        hit <- walk(a, env)
        if (!is.null(hit)) {
          return(hit)
        }
      }
    }
    NULL
  }
  walk(expr, env)
}

# R's own "object not found" error, the parent of dplyr's "In argument: ..." of
# a dot that reads a name that is no column and bound nowhere.
# @keywords internal
object_not_found_error <- function(name) {
  simpleError(paste0("object '", name, "' not found"))
}

# Raise dplyr's "In argument: `<dot>`." error for a dot of `verb` that reads a
# column the data lacks, before anything of the call ran: `parent` is the error
# the dot raises, `run` the frame of the verb's implementation (it holds the
# user's dots, groups and table, see mutate_dot_error_args()), `i` the dot's
# position among the user's. Like dplyr's own, classed `dplyr:::mutate_error`
# for mutate()/transmute() and plain for the others.
# @keywords internal
abort_missing_in_dot <- function(parent, i, verb, run, user_dots) {
  args <- mutate_dot_error_args(parent, i, "expand", NULL, NULL, verb, run, user_dots)
  rlang::abort(
    args$bullets, parent = parent,
    class = if (verb %in% c("mutate", "transmute")) "dplyr:::mutate_error",
    call = verb_error_call(run)
  )
}

# Names whose presence in a summarise() dot makes the missing-column check give
# no verdict: they bind names the walk can't see (`<-`, `with()`, a magrittr
# `.`), evaluate code in another scope, or splice columns of unknown names.
summarise_check_skip <- c(
  "<-", "<<-", "=", "->", "->>", ":=", "assign", "delayedAssign", "local", "with", "within",
  "subset", "transform", "eval", "evalq", "parse", "str2lang", "attach", "%>%", "%<>%",
  "%T>%", "%$%", ".", "across", "if_any", "if_all", "pick", "c_across", "cur_group",
  "cur_data", "cur_data_all", "cur_column"
)

# Reducers that give one value, whose unnamed dot creates no columns of its own
# names for the dots after it.
summarise_scalar_heads <- c(
  "mean", "sum", "n", "min", "max", "median", "sd", "var", "first", "last", "nth",
  "n_distinct", "any", "all", "prod", "quantile", "IQR", "mad"
)

# dplyr's error for a summarise() dot that reads a column the data lacks, raised
# before any CPU fallback is decided (a missing column is the user's typo, not a
# reason to run on the CPU, and under `cuplyr.fallback = "error"` it would be
# reported as one): `object 'xx' not found` or rlang's `.data$xx` error as the
# parent of "In argument: `<dot>`.". A dot reads a name that is no column, bound
# nowhere from its environment up, and made by no earlier dot (dplyr evaluates
# them in order, each seeing the columns the ones before made). It gives no
# verdict where the walk could be wrong: a name bound inside the dot, tidyselect
# calls (their errors are the expansion's), a dot that may splice columns of
# unknown names. The first dot in order that reads a missing name raises.
# @param dots The user's quosures; `schema` the input's; `run` summarise_impl()'s
#   frame (`.data`, `by_cols`, `by_given`)
# @keywords internal
summarise_missing_column_check <- function(dots, schema, run = parent.frame()) {
  scope <- schema$names
  nms <- names(dots)
  for (i in seq_along(dots)) {
    quo <- dots[[i]]
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)
    # (a dot whose every name is a column, and with no `.data` pronoun, has nothing
    # to look up: the common case, settled by one C walk)
    vars <- all.vars(expr)
    if (!all(vars %in% scope) || ".data" %in% vars) {
      if (identical(env, emptyenv()) || any(all.names(expr) %in% summarise_check_skip)) {
        return(invisible(NULL))
      }
      miss <- expr_first_unbound_symbol(expr, env, list(names = scope))
      if (!is.null(miss)) {
        abort_missing_in_dot(object_not_found_error(miss), i, "summarise", run, dots)
      }
      pron <- expr_first_missing_pronoun(expr, env, scope)
      if (!is.null(pron)) {
        cnd <- tryCatch(stop_data_pronoun_missing(pron$call, pron$env, scope, pron$name), error = function(e) e)
        abort_missing_in_dot(cnd, i, "summarise", run, dots)
      }
    }
    if (i == length(dots)) {
      break
    }
    named <- !is.null(nms) && !is.na(nms[[i]]) && nzchar(nms[[i]])
    if (named) {
      scope <- c(scope, nms[[i]])
    } else if (!(is.call(expr) && isTRUE(call_head_name(expr) %in% summarise_scalar_heads))) {
      return(invisible(NULL))
    }
  }
  invisible(NULL)
}

# dplyr's eval_relocate() (R/relocate.R): the positions of a relocate()
# selection, a named integer vector of every column. dplyr resolves the
# selection, `.before` and `.after` with tidyselect::eval_select(), whose errors say
# "Can't select columns that don't exist."; tidyselect's own eval_relocate() says
# "Can't relocate columns...", which dplyr's relocate() never does.
# @keywords internal
eval_relocate_dplyr <- function(expr, data, before = NULL, after = NULL,
                                before_arg = "before", after_arg = "after",
                                env = rlang::caller_env(), error_call = rlang::caller_env()) {
  sel <- tidyselect::eval_select(expr = expr, data = data, env = env, error_call = error_call)
  # a column renamed several times keeps only its last instance
  sel <- vctrs::vec_slice(sel, which(!duplicated(sel, fromLast = TRUE)))
  n <- length(data)

  before <- rlang::as_quosure(before, env = env)
  after <- rlang::as_quosure(after, env = env)
  has_before <- !rlang::quo_is_null(before)
  has_after <- !rlang::quo_is_null(after)

  if (has_before && has_after) {
    rlang::abort(paste0("Can't supply both `", before_arg, "` and `", after_arg, "`."), call = error_call)
  }
  if (has_before) {
    where <- unname(tidyselect::eval_select(before, data, env = env, error_call = error_call))
    # (an empty `.before` selection puts `sel` first)
    where <- if (length(where) == 0L) 1L else min(where)
  } else if (has_after) {
    where <- unname(tidyselect::eval_select(after, data, env = env, error_call = error_call))
    # (an empty `.after` selection puts `sel` last)
    where <- (if (length(where) == 0L) n else max(where)) + 1L
  } else {
    where <- 1L
  }

  lhs <- rlang::seq2(1L, where - 1L)
  rhs <- rlang::seq2(where, n)
  lhs <- setdiff(lhs, sel)
  rhs <- setdiff(rhs, sel)
  names <- names(data)
  names(lhs) <- names[lhs]
  names(rhs) <- names[rhs]
  vctrs::vec_c(lhs, sel, rhs)
}

# The parent of a dot's "In argument: ..." when the dot raised R's "object 'xx' not
# found": the native parser reads a call's arguments, where R looks the function
# up first, so a function that can't be found from the dot's environment (dplyr
# not attached, say) is R's "could not find function" error, with the call as its
# call, when that lookup comes before the missing object in evaluation order. Any
# other error is returned as it is.
# @keywords internal
object_not_found_parent <- function(cnd, expr, env) {
  if (!identical(class(cnd), c("simpleError", "error", "condition"))) {
    return(cnd)
  }
  m <- regmatches(cnd$message, regexec("^object '(.*)' not found$", cnd$message))[[1L]]
  if (length(m) != 2L) {
    return(cnd)
  }
  unevaluated <- c("function", "~", "quote", "bquote", "expression", "substitute", "::", ":::", "$", "@")
  walk <- function(e) {
    if (rlang::is_quosure(e)) {
      return(NULL)
    }
    if (is.symbol(e)) {
      return(if (identical(as.character(e), m[[2L]])) "object")
    }
    if (!is.call(e)) {
      return(NULL)
    }
    head <- e[[1L]]
    if (is.symbol(head)) {
      h <- as.character(head)
      if (h %in% unevaluated) {
        return(NULL)
      }
      if (!exists(h, envir = env, mode = "function", inherits = TRUE)) {
        return(simpleError(paste0("could not find function \"", h, "\""), call = e))
      }
    } else {
      hit <- walk(head)
      if (!is.null(hit)) {
        return(hit)
      }
    }
    for (a in as.list(e)[-1L]) {
      if (!rlang::is_missing(a)) {
        hit <- walk(a)
        if (!is.null(hit)) {
          return(hit)
        }
      }
    }
    NULL
  }
  hit <- walk(expr)
  if (inherits(hit, "error")) hit else cnd
}

# The error handler of parse_filter_ir(), called from filter_run()'s loop over the
# dots (`handler` is the handler's own environment, whose enclosure is the frame
# of parse_filter_ir()): wraps what the parser raises for the dot the loop is on
# as dplyr's "In argument: `<dot>`." (mutate_dot_error()), reading the loop's `i`,
# the dots and the user's dots from filter_run()'s frame. Returns (the error goes
# on) unless it raised. It is the parser's own tryCatch() that catches the error,
# so the success path pays nothing for it.
# @keywords internal
filter_dot_error <- function(cnd, handler) {
  run <- rlang::eval_bare(quote(base::parent.frame()), parent.env(handler))
  dots <- get0("dots", run, inherits = FALSE)
  i <- get0("i", run, inherits = FALSE)
  if (is.null(dots) || is.null(i) || is.null(get0("user_dots", run, inherits = FALSE))) {
    return(invisible(NULL))
  }
  mutate_dot_error(cnd, i, "eval", attr(dots, "provenance"), names(dots), "filter", run = run)
}
