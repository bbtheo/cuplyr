# The error call and the missing-column errors every verb shares (#13).
#
# dplyr reports an error against the call the user made (`filter()`, `count()`,
# `add_tally()`), found by dplyr_error_call() (R/conditions.R): start at the verb's
# frame and follow the `.__dplyr_error_call__.` binding of the frame that called it,
# as often as there is one. verb_error_call() is that walk, started at the
# user-level call (gpu_entry_frame()), for every error cuplyr raises itself;
# fallback_error_call() makes dplyr's own errors on a CPU route report the same.
# None of it runs on the success path of a verb, except the summarise() dot check
# (a walk of the dots up to the last one that reads a name no column has, or
# `.data`), arrange()'s walk of a key it can't sort by natively (on its way to a
# CPU fallback), relocate()'s selection, and a withCallingHandlers() around
# count()/distinct()'s key parses and group_by()'s mutate() of computed dots.

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
# evaluated against the pronoun of a mask holding `names`. An internal error if it
# evaluates anyway.
# @param expr The `.data$xx` call; `env` where it was written; `names` the columns
# @keywords internal
stop_data_pronoun_missing <- function(expr, env, names, name = NULL) {
  names <- unique(names)
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
  rlang::abort(
    paste0("Column `", name %||% as.character(expr[[3L]]), "` is missing from the data but evaluating `", rlang::expr_text(expr), "` did not fail."),
    class = "cuplyr_error_internal"
  )
}

# R's own "object not found" error, the parent of dplyr's "In argument: ..." of
# a dot that reads a name that is no column and bound nowhere.
# @keywords internal
object_not_found_error <- function(name) {
  simpleError(paste0("object '", name, "' not found"))
}

# The error a dot raises for what expr_first_certain_missing() found: R's "object
# not found", or rlang's `.data$xx` error.
# @param miss Its result (not NULL); `scope` the names it was looked up in
# @keywords internal
missing_name_error <- function(miss, scope) {
  if (is.null(miss$pronoun)) {
    object_not_found_error(miss$name)
  } else {
    tryCatch(stop_data_pronoun_missing(miss$pronoun, miss$env, scope, miss$name), error = function(e) e)
  }
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

# The calls the missing-column walk (certain_eval_walk()) goes through, by how R
# evaluates their arguments. A head counts only when it is the function named
# here, as found from the dot's environment (a user's own `mean()` is not); any
# other call ends the walk with no verdict, as it may never read an argument
# (`if`, `switch()`, `&&`, `tryCatch()`, the `...` of an S3 generic, a user
# function) or fail with an error of its own first.
#
# Builtins, and the specials log(), round() and signif(): every argument is
# evaluated, in the order written, before the function runs (and before its
# arity is checked: `round(stop("a"), 1, 2)` raises "a", `round(digits =
# stop("b"), stop("a"))` raises "b"). The walk reads them all.
certain_written_heads <- c(
  "+", "-", "*", "/", "^", "%%", "%/%", "==", "!=", "<", ">", "<=", ">=", "&", "|",
  "!", "(", "is.na", "abs", "sqrt", "exp", "log", "round", "signif",
  "sum", "min", "max", "prod", "any", "all"
)

certain_head_rule <- c(
  stats::setNames(rep("written", length(certain_written_heads)), certain_written_heads),
  # Closures that force the argument matched to `x` before anything else, and may
  # never read the others: mean(), median() and quantile() are `UseMethod()` on
  # `x` (mean.default() returns NA before it reads `trim` or `na.rm` when `x` is
  # not numeric; the `...` of a generic may never be forced); IQR() is
  # `diff(quantile(as.numeric(x), ...))`, so quantile()'s dispatch forces `x`;
  # dplyr's first() and last() call nth(), whose first line is `vec_size(x)`.
  # The walk reads `x` only.
  mean = "x_first", median = "x_first", quantile = "x_first", IQR = "x_first",
  first = "x_first", last = "x_first", nth = "x_first",
  # Closures that read `na.rm` (`if (na.rm)`, var()'s `use`) before `x`: `x` is
  # the first thing that can fail only when every other argument is a literal
  # that can't (a number or TRUE/FALSE). The walk reads `x` only, and only then.
  var = "x_after_literals", sd = "x_after_literals", mad = "x_after_literals",
  # dplyr's n() takes no argument and can't fail in a summarise().
  n = "none",
  # dplyr's n_distinct(): `missing(...)` and check_dots_unnamed() look at the
  # dots without forcing them, then df_list(...) forces them in order; `na.rm`
  # is read after. The walk reads the unnamed arguments when every named one is
  # `na.rm`.
  n_distinct = "dots"
)

# Where the functions above come from (base for the rest).
certain_head_pkg <- c(
  median = "stats", quantile = "stats", IQR = "stats", var = "stats", sd = "stats",
  mad = "stats", first = "dplyr", last = "dplyr", nth = "dplyr", n = "dplyr",
  n_distinct = "dplyr"
)

# Of those, the calls certain to evaluate without an error, to an atomic numeric
# or logical vector, when every argument the walk read is one (and the call has the
# shape below): the builtin operators and math functions (with the arity they
# take and no argument names; any arguments for the Summary group), mean(),
# median(), var(), sd() and mad() of `x` alone, and n(). quantile() fails on an
# NA without `na.rm`, nth() checks `n`, and so on: those never complete.
certain_summary_heads <- c("sum", "min", "max", "prod", "any", "all")
certain_unary_heads <- c("!", "(", "is.na", "abs", "sqrt", "exp")
certain_complete_x_heads <- c("mean", "median", "var", "sd", "mad")

# GPU types a column is an atomic number or logical of, on dplyr's side too.
certain_num_types <- c("BOOL8", "INT8", "INT16", "INT32", "UINT8", "UINT16", "FLOAT32", "FLOAT64")

# The columns of `schema`, split into those that are atomic numbers or logicals
# (`num`: not a factor, a date, a string or an integer64) and the rest (`any`).
# @keywords internal
certain_schema_kinds <- function(schema) {
  nms <- schema$names
  is_num <- schema$types %in% certain_num_types & !nms %in% names(schema$factor_levels)
  list(num = unique(nms[is_num]), any = unique(nms[!is_num]))
}

# The function a head of certain_head_rule stands for.
# @keywords internal
certain_head_fun <- function(h) {
  pkg <- certain_head_pkg[h]
  if (is.na(pkg)) get(h, envir = baseenv()) else getExportedValue(pkg, h)
}

# How the lookup of `nm` from `env` evaluates: "unbound" (R's "object not found"),
# "num" for a number or logical with no class, "any" for any other value, NULL
# for a binding whose lookup runs code (a promise not yet forced, an active
# binding), which may fail first.
# @keywords internal
certain_binding_kind <- function(nm, env) {
  while (!identical(env, emptyenv())) {
    if (exists(nm, envir = env, inherits = FALSE)) {
      if (bindingIsActive(nm, env) || isTRUE(rlang::env_binding_are_lazy(env, nm))) {
        return(NULL)
      }
      v <- get(nm, envir = env, inherits = FALSE)
      return(if ((is.numeric(v) || is.logical(v)) && !is.object(v)) "num" else "any")
    }
    env <- parent.env(env)
  }
  "unbound"
}

# Walk `expr` in R's evaluation order for a name that is certain to be looked up
# before anything else could fail or stop the walk, and that is no column of
# the data (`num`, `any`: the columns as certain_schema_kinds() splits them) and
# bound nowhere from `env` up. Nothing is evaluated.
#
# Returns list(hit =, kind =). `hit` is the verdict: list(name =) for a bare
# name, list(pronoun =, env =, name =) for a `.data$xx` of a column that is not
# there, or NULL. `kind` is how `expr` itself evaluates, when that is certain:
# "num" (without an error, to an atomic numeric or logical vector), "any" (without an
# error, to a value of unknown kind: a lookup), or NULL (not certain: it may fail,
# or the walk stopped).
# @keywords internal
certain_eval_walk <- function(expr, env, num, any) {
  hit <- NULL
  walk <- function(e, env) {
    if (rlang::is_quosure(e)) {
      return(walk(rlang::quo_get_expr(e), rlang::quo_get_env(e)))
    }
    if (rlang::is_missing(e)) {
      return(NULL)
    }
    if (is.symbol(e)) {
      nm <- as.character(e)
      if (nm %in% num) {
        return("num")
      }
      if (nm %in% any) {
        return("any")
      }
      # (`...` and `..1` are read through the dots, which may hold anything)
      if (nm %in% c(".data", ".env", "...") || grepl("^\\.\\.[0-9]+$", nm)) {
        return(NULL)
      }
      kind <- certain_binding_kind(nm, env)
      if (identical(kind, "unbound")) {
        hit <<- list(name = nm)
        return(NULL)
      }
      return(kind)
    }
    if (!is.call(e)) {
      return(if ((is.numeric(e) || is.logical(e)) && length(e) == 1L) "num" else "any")
    }
    if (is_data_pronoun_call(e)) {
      nm <- pronoun_call_name(e, env, list(names = c(num, any)))
      if (is.null(nm)) {
        return(NULL)
      }
      if (nm %in% num) {
        return("num")
      }
      if (nm %in% any) {
        return("any")
      }
      hit <<- list(pronoun = e, env = env, name = nm)
      return(NULL)
    }
    # (`base::sum()` as count(wt =) writes it: the package must be the function's)
    head <- e[[1L]]
    qualified <- rlang::is_call(head, "::", n = 2L) && is.symbol(head[[2L]]) && is.symbol(head[[3L]])
    if (qualified) {
      h <- as.character(head[[3L]])
    } else if (is.symbol(head)) {
      h <- as.character(head)
    } else {
      return(NULL)
    }
    rule <- unname(certain_head_rule[h])
    if (is.na(rule)) {
      return(NULL)
    }
    if (qualified) {
      pkg <- unname(certain_head_pkg[h])
      if (!identical(as.character(head[[2L]]), if (is.na(pkg)) "base" else pkg)) {
        return(NULL)
      }
    } else if (!identical(get0(h, envir = env, mode = "function"), certain_head_fun(h))) {
      return(NULL)
    }
    args <- as.list(e)[-1L]
    arg_names <- names(args) %||% rep("", length(args))
    if (identical(rule, "written")) {
      kinds <- character(length(args))
      for (k in seq_along(args)) {
        kind <- walk(args[[k]], env)
        if (is.null(kind)) {
          return(NULL)
        }
        kinds[k] <- kind
      }
      n <- length(args)
      shape <- if (h %in% certain_summary_heads) {
        TRUE
      } else if (any(nzchar(arg_names))) {
        FALSE
      } else if (h %in% c("+", "-", "log", "round", "signif")) {
        n %in% 1:2
      } else if (h %in% certain_unary_heads) {
        n == 1L
      } else {
        n == 2L
      }
      return(if (shape && all(kinds == "num")) "num")
    }
    if (identical(rule, "none")) {
      return(if (length(args) == 0L) "num")
    }
    if (identical(rule, "dots")) {
      if (!any(arg_names == "") || !all(arg_names %in% c("", "na.rm"))) {
        return(NULL)
      }
      for (a in args[arg_names == ""]) {
        if (is.null(walk(a, env))) {
          return(NULL)
        }
      }
      return(NULL)
    }
    # "x_first", "x_after_literals": the argument R matches to `x` (an empty or
    # `...` argument, or a call R can't match, gives no verdict)
    if (any(vapply(args, function(a) rlang::is_missing(a) || identical(a, quote(...)), logical(1)))) {
      return(NULL)
    }
    matched <- tryCatch(as.list(match.call(certain_head_fun(h), e))[-1L], error = function(err) NULL)
    if (is.null(matched) || !"x" %in% names(matched)) {
      return(NULL)
    }
    others <- matched[names(matched) != "x"]
    if (identical(rule, "x_after_literals") &&
        !all(vapply(others, function(a) {
          (is.numeric(a) || is.logical(a)) && length(a) == 1L && !is.na(a)
        }, logical(1)))) {
      return(NULL)
    }
    kind <- walk(matched[["x"]], env)
    if (identical(kind, "num") && length(args) == 1L && h %in% certain_complete_x_heads) "num"
  }
  kind <- walk(expr, env)
  list(hit = hit, kind = kind)
}

# The first name `expr` is certain to read that is no column of `schema` and bound
# nowhere from `env` up (certain_eval_walk()'s verdict), or NULL. arrange()'s
# check of a key it can't sort by natively.
# @keywords internal
expr_first_certain_missing <- function(expr, env, schema) {
  cols <- certain_schema_kinds(schema)
  certain_eval_walk(expr, env, cols$num, cols$any)$hit
}

# dplyr's error for a summarise() dot that reads a column the data lacks, raised
# before any CPU fallback is decided (a missing column is the user's typo, not a
# reason to run on the CPU, and under `cuplyr.fallback = "error"` it would be
# reported as one): `object 'xx' not found` or rlang's `.data$xx` error as the
# parent of "In argument: `<dot>`.". dplyr evaluates the dots in order, each
# seeing the columns and the names the ones before it make, and reports the
# first that fails. So a dot gives a verdict only for a name it is certain to
# read first (certain_eval_walk()), and only when every dot before it is certain
# to evaluate without an error and to make the columns the check then knows of:
# its name, or, unnamed, its label when its value is certainly one atomic vector
# (any other value may be a data frame, which dplyr unpacks into columns of its
# own). The check ends at the first dot that is not. Anything uncertain leaves
# the call to the route it takes anyway.
# @param dots The user's quosures; `schema` the input's; `run` summarise_impl()'s
#   frame (`.data`, `by_cols`, `by_given`)
# @keywords internal
summarise_missing_column_check <- function(dots, schema, run = parent.frame()) {
  # (a dot whose every name is a column, and with no `.data` pronoun, has nothing
  # to report: the common case, settled by one C walk per dot)
  in_names <- schema$names
  reads_other <- vapply(dots, function(quo) {
    vars <- all.vars(rlang::quo_get_expr(quo))
    !all(vars %in% in_names) || ".data" %in% vars
  }, logical(1), USE.NAMES = FALSE)
  if (!any(reads_other)) {
    return(invisible(NULL))
  }
  last <- max(which(reads_other))
  cols <- certain_schema_kinds(schema)
  num <- cols$num
  any <- cols$any
  nms <- names(dots) %||% rep("", length(dots))
  for (i in seq_len(last)) {
    quo <- dots[[i]]
    res <- certain_eval_walk(rlang::quo_get_expr(quo), rlang::quo_get_env(quo), num, any)
    if (!is.null(res$hit)) {
      abort_missing_in_dot(missing_name_error(res$hit, c(num, any)), i, "summarise", run, dots)
    }
    if (i == last || is.null(res$kind)) {
      break
    }
    new <- if (!is.na(nms[[i]]) && nzchar(nms[[i]])) {
      nms[[i]]
    } else if (identical(res$kind, "num")) {
      dot_label(quo)
    }
    if (is.null(new)) {
      break
    }
    num <- setdiff(num, new)
    any <- setdiff(any, new)
    if (identical(res$kind, "num")) num <- c(num, new) else any <- c(any, new)
  }
  invisible(NULL)
}

# dplyr 1.2.1's eval_relocate() (R/relocate.R): the positions of a relocate()
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

# The error handler of the key parse in count()/add_count()/distinct() (and their
# tally()/add_tally() forms): dplyr evaluates computed keys in a mutate() of the
# ungrouped data (add_computed_columns()), so what the parse raises for a key is
# dplyr's `dplyr:::mutate_error` "In argument: `<key>`." (the key as the user
# named it, no group bullet), with the error as parent and the user's call.
# cuplyr's own runtime errors, and a `&&` of a column (the caller's to send to
# dplyr), go on as they are.
# @param cnd The error; `quo` the key; `name` the user's name for it ("" or NA when
#   unnamed); `frame` the verb's frame
# @keywords internal
computed_key_error <- function(cnd, quo, name, frame) {
  if (inherits(cnd, c(mutate_error_passthrough, "cuplyr_error_short_circuit"))) {
    return(invisible(NULL))
  }
  expr <- rlang::quo_get_expr(quo)
  rlang::abort(
    c(i = paste0("In argument: `", mutate_error_label(name, expr), "`.")),
    class = "dplyr:::mutate_error",
    parent = object_not_found_parent(cnd, expr, rlang::quo_get_env(quo)),
    call = verb_error_call(frame)
  )
}

# dplyr's error for a grouping dot of group_by() that is a variable reference
# (`xx`, `.data$xx`) to a column no earlier dot made, when dplyr runs the dots
# through mutate() (add_computed_columns(): some dot is named or computed): the
# mutate's "In argument: `<dot>`." for it, no group bullet (the data is
# ungrouped there), parent R's "object not found" or rlang's `.data$xx` error.
# @param quo The dot; `name` the column it reads; `cols` the columns at that point;
#   `frame` group_by()'s frame
# @keywords internal
group_by_abort_unbound <- function(quo, name, cols, frame) {
  expr <- rlang::quo_get_expr(quo)
  parent <- if (is.symbol(expr)) {
    object_not_found_error(name)
  } else {
    tryCatch(stop_data_pronoun_missing(expr, rlang::quo_get_env(quo), cols, name), error = function(e) e)
  }
  rlang::abort(
    c(i = paste0("In argument: `", mutate_error_label("", expr), "`.")),
    class = "dplyr:::mutate_error", parent = parent, call = verb_error_call(frame)
  )
}

# The `dplyr:::mutate_error` handler of group_by()'s one mutate() of its computed
# dots, which runs each unnamed one named by its label (so its column has the
# name dplyr gives it): the error then reads "In argument: `x + 1 = x + 1`.",
# where dplyr's mutate() had the dot unnamed and says "In argument: `x + 1`.".
# Re-raised with the dot as the user wrote it; returns for any other error.
# @param cnd The error; `comp_dots` the named dots the mutate() ran; `unnamed`
#   which of them the user left unnamed
# @keywords internal
group_by_mutate_error <- function(cnd, comp_dots, unnamed) {
  for (j in which(unnamed)) {
    expr <- rlang::quo_get_expr(comp_dots[[j]])
    named <- paste0("In argument: `", mutate_error_label(names(comp_dots)[[j]], expr), "`.")
    if (identical(unname(cnd$message), named)) {
      rlang::abort(
        c(i = paste0("In argument: `", mutate_error_label("", expr), "`."), if (is.character(cnd$body)) cnd$body),
        class = "dplyr:::mutate_error", parent = cnd$parent, call = cnd$call
      )
    }
  }
  invisible(NULL)
}
