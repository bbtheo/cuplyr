# Arithmetic around aggregates in summarise() (#22).
#
# A dot such as `mean(x) + 1`, `sum(x) / n()` or `s / n()` (`s` an earlier
# result) is not one reducer call. dplyr evaluates it once per group, where each
# reducer gives one value and the rest is elementwise on those values. The same
# split runs on the GPU: every maximal reducer call inside a dot is a *leaf*,
# computed by the ordinary aggregation routes as a temp aggregate; the rest of the
# dot (its *post* expression) is a mutate on the one-row-per-group result, and a
# final select keeps the keys and the user's dots in the user's order.
#
#   summarise_post_split()   syntactic plan, before anything is judged
#   summarise_post_screen()  judges the post expressions against the types the
#                            aggregates will have, before any GPU work
#   summarise_post_apply()   the mutate + select on the real result
#
# Whatever the plan can't classify declines (NULL) and the call takes the route
# it always had, so dplyr gives the value or the error.

# The function a call names (`pkg::fn` too), or NULL.
# @keywords internal
summarise_post_head <- function(expr) {
  if (!is.call(expr)) {
    return(NULL)
  }
  head <- .subset2(expr, 1L)
  if (is.call(head) && length(head) == 3L && identical(.subset2(head, 1L), quote(`::`)) &&
      is.symbol(.subset2(head, 3L))) {
    head <- .subset2(head, 3L)
  }
  if (is.symbol(head)) as.character(head) else NULL
}

# Every name `expr` reads: its variables and what `.data$x` / `.data[["x"]]`
# name. NA when a `.data[[k]]` subscript can't be resolved (it may name any).
# @keywords internal
summarise_post_reads <- function(expr) {
  pn <- expr_data_pronoun_names(expr)
  if (anyNA(pn)) {
    return(NA_character_)
  }
  c(all.vars(expr), pn)
}

# Split the dots of a summarise() call into aggregate leaves and post expressions.
#
# @param dots Named quosures (expanded, silently named)
# @param expanded The expanded dots, for their provenance (the user dot of each)
# @param in_schema The input's schema
# @param groups The grouping columns
# @return NULL (the call has no composite dot, or one the plan can't classify), or
#   `list(agg_dots, post, out_names, n_keys)`: `agg_dots` the quosures to
#   aggregate (the plain reducer dots under their own names, the leaves as
#   `.post_agg<k>`), carrying the `provenance` that maps each to its user dot
#   (`origin`), the name of the dot it is part of (`label`) and, for a leaf, that
#   dot's quosure (`dot`, NULL for a plain reducer dot); `post` the post dots
#   `list(name, quo, origin)` in dot order; `out_names` the result's dot names in
#   order; `why` an environment where summarise_post_apply() leaves the reason a
#   rerun on the CPU has.
# @keywords internal
summarise_post_split <- function(dots, expanded, in_schema, groups) {
  nms <- names(dots)
  n <- length(dots)
  if (n == 0L || anyNA(nms) || anyDuplicated(nms) > 0L || any(nms %in% groups)) {
    return(NULL)
  }
  # The heads alone say whether any dot is composite. A call whose dots are all
  # reducer calls (`m = mean(z), n = n()`) is not ours, and pays only this.
  composite <- FALSE
  for (i in seq_len(n)) {
    head <- summarise_post_head(.subset2(.subset2(dots, i), 2L))
    if (is.null(head) || match(head, agg_functions, 0L) == 0L) {
      composite <- TRUE
      break
    }
  }
  if (!composite) {
    return(NULL)
  }

  col_names <- in_schema$names
  origin_of <- attr(expanded, "provenance")$origin
  if (length(origin_of) != n) {
    origin_of <- seq_len(n)
  }
  # (a temp name must not hide a variable a dot reads: `mean(x) + .post_agg2`)
  taken <- c(nms, col_names, groups,
             unlist(lapply(dots, function(q) all.vars(rlang::quo_get_expr(q))), use.names = FALSE))

  agg_quos <- list()
  agg_names <- character()
  agg_origin <- integer()
  agg_label <- character()
  agg_src <- list()
  # shared leaves, by the reducer and its first argument when that is a name, else
  # by the call's hash (no deparse: a large constant inside a call costs little;
  # embedded quosures squashed, so no environment is serialized). A bucket holds
  # the leaves its key can't tell apart: list(expr, env, name)
  leaves <- new.env(parent = emptyenv())
  post <- list()
  bail <- FALSE

  add_agg <- function(name, quo, origin, label, src = NULL) {
    agg_quos[[length(agg_quos) + 1L]] <<- quo
    agg_names[[length(agg_names) + 1L]] <<- name
    agg_origin[[length(agg_origin) + 1L]] <<- origin
    agg_label[[length(agg_label) + 1L]] <<- label
    agg_src[length(agg_src) + 1L] <<- list(src)
  }

  for (i in seq_len(n)) {
    quo <- .subset2(dots, i)
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)
    earlier <- nms[seq_len(i - 1L)]
    later <- nms[-seq_len(i)]
    reads_earlier <- function(e) {
      r <- summarise_post_reads(e)
      anyNA(r) || any(r %in% earlier)
    }

    head <- summarise_post_head(expr)
    if (!is.null(head) && head %in% agg_functions) {
      # a plain reducer dot. A dot reading an earlier result is sequential: dplyr's.
      if (identical(head, "count") || reads_earlier(expr)) {
        return(NULL)
      }
      add_agg(nms[[i]], quo, origin_of[[i]], nms[[i]])
      next
    }

    # replace each maximal reducer call by a temp symbol. `env` is the
    # environment the code under `e` is written in: an embedded quosure
    # (`{{ arg }}`, `!!quo`) brings its own.
    walk <- function(e, env) {
      if (!is.call(e)) {
        return(e)
      }
      if (rlang::is_quosure(e)) {
        inner <- rlang::quo_get_expr(e)
        new_inner <- walk(inner, rlang::quo_get_env(e))
        if (identical(new_inner, inner)) {
          return(e)
        }
        # (a quosure that was one reducer call is its leaf)
        return(if (is.symbol(new_inner)) new_inner else rlang::quo_set_expr(e, new_inner))
      }
      h <- summarise_post_head(e)
      if (!is.null(h) && h %in% agg_functions) {
        if (identical(h, "count") || reads_earlier(e)) {
          bail <<- TRUE
          return(e)
        }
        # (min()/max() warn per call, as dplyr evaluates each: never shared)
        key <- NULL
        if (!h %in% c("min", "max")) {
          a <- if (length(e) >= 2L) .subset2(e, 2L)
          key <- if (is.symbol(a)) {
            paste0(h, "(", as.character(a))
          } else {
            paste0(h, "#", rlang::hash(rlang::quo_squash(e)))
          }
        }
        share <- NULL
        for (l in if (!is.null(key)) leaves[[key]]) {
          if (identical(l$expr, e, num.eq = FALSE) && identical(l$env, env)) {
            share <- l
            break
          }
        }
        if (!is.null(share)) {
          return(as.name(share$name))
        }
        name <- unique_temp_name(paste0(".post_agg", length(agg_quos) + 1L), taken)
        taken <<- c(taken, name)
        if (!is.null(key)) {
          leaves[[key]] <- c(leaves[[key]], list(list(expr = e, env = env, name = name)))
        }
        add_agg(name, rlang::new_quosure(e, env), origin_of[[i]], nms[[i]], quo)
        return(as.name(name))
      }
      # (only calls change; `e[k] <- list()` keeps a NULL argument, `round(m, NULL)`)
      for (k in seq_along(e)[-1L]) {
        if (is.call(e[[k]])) {
          e[k] <- list(walk(e[[k]], env))
        }
      }
      e
    }
    new_expr <- tryCatch(walk(expr, env), error = function(cnd) {
      bail <<- TRUE
      expr
    })
    if (bail) {
      return(NULL)
    }
    # Outside a reducer a column of the input is the group's chunk (dplyr's "must be
    # size 1"); only results of earlier dots are one value. The name of a later dot
    # is not one yet: the table the post step reads holds every plain reducer dot,
    # later ones included, which would shadow the variable or column dplyr reads.
    reads <- summarise_post_reads(new_expr)
    if (anyNA(reads) || any(reads %in% later) || any(reads %in% col_names & !reads %in% earlier)) {
      return(NULL)
    }
    post[[length(post) + 1L]] <- list(name = nms[[i]], quo = rlang::new_quosure(new_expr, env),
                                      origin = origin_of[[i]])
  }

  if (length(post) == 0L || length(agg_quos) == 0L) {
    return(NULL)
  }
  agg_dots <- rlang::new_quosures(stats::setNames(agg_quos, agg_names))
  attr(agg_dots, "provenance") <- list(origin = agg_origin, kind = rep("dot", length(agg_quos)),
                                       label = agg_label, dot = agg_src)
  list(agg_dots = agg_dots, post = post, out_names = nms, n_keys = length(groups),
       why = new.env(parent = emptyenv()))
}

# Why the GPU engine would give a value R does not, for a value an aggregate can
# take (NA, or NaN: `mean()` of an empty group; -Inf: `max()` of one; -0), in post
# expression `ir`; NULL when it gives R's. A reducer's result is one of these far
# more often than a column value is, so these, which mutate() shares, are kept off
# the post step:
# - `^`: R's `NA^0` and `1^NA` are 1, the engine's pow is NA when either side is
#   (and R's `(-1)^Inf` is NaN): the exponent must be a constant other than 0, NA
#   and an infinity. R_pow() also special-cases a `-Inf` or `-0` base, where C's
#   pow() differs: `(-Inf)^0.5` is NaN in R, Inf in pow; `(-0)^-1` Inf, -Inf;
#   `(-0)^3` 0, -0; `(-Inf)^-3` 0, -0. Both agree for an even integer exponent,
#   and for any base that can be neither: `nonneg`, the aggregates that are n(),
#   n_distinct(), sd() or var() (a base computed from them may be -0: `-sd(x)`).
#   (Finite values also differ in the last bit where R uses powl(): `9^0.5` is
#   3 in R, 2.9999999999999996 on the GPU, as in mutate().)
# - `&`, `|`, `!` of anything but a logical or an integer: of a double R reads
#   NaN as NA, the engine as TRUE; of a string R's error, the engine's failure.
# - `if_else()`/`case_when()` with a condition that is not a logical expression:
#   dplyr's error, where the engine fails at execution. An NA constant condition
#   is dplyr's NA: kept on dplyr's route too (case_when() leaves a bare NA
#   untyped, which the engine can't take as a condition).
# @param nonneg The names in `schema` that are never -Inf or -0
# @keywords internal
summarise_post_na_rule <- function(ir, schema, nonneg = character()) {
  if (!inherits(ir, "cuplyr_ir") || !identical(.subset2(ir, "kind"), "call")) {
    return(NULL)
  }
  op <- .subset2(ir, "op")
  args <- .subset2(ir, "args")
  # (a bare NA literal adopts its sibling's type: never a double here)
  type_of <- function(a) {
    if (identical(.subset2(a, "kind"), "lit") && isTRUE(.subset2(a, "na"))) {
      return("BOOL8")
    }
    tryCatch(unname(ir_infer_type(a, schema)), error = function(e) NA_character_)
  }
  if (identical(op, "^")) {
    # (the exponent's value: a literal, under a unary minus or parentheses)
    ex <- .subset2(args, 2L)
    sign <- 1
    while (identical(.subset2(ex, "kind"), "call") && length(.subset2(ex, "args")) == 1L &&
           match(.subset2(ex, "op"), c("-", "("), 0L) > 0L) {
      if (identical(.subset2(ex, "op"), "-")) sign <- -sign
      ex <- .subset2(.subset2(ex, "args"), 1L)
    }
    v <- if (identical(.subset2(ex, "kind"), "lit") && !isTRUE(.subset2(ex, "na"))) .subset2(ex, "value")
    if (is.numeric(v) || is.logical(v)) v <- sign * v
    if (!((is.numeric(v) || is.logical(v)) && length(v) == 1L && is.finite(v) && v != 0)) {
      return(paste0(
        "`^` around a reducer runs on the GPU only with a constant exponent other than 0, NA or an ",
        "infinity: R's `NA^0` and `1^NA` are 1, where the GPU engine gives NA."
      ))
    }
    base <- .subset2(args, 1L)
    if (v %% 2 != 0 &&
        !(identical(.subset2(base, "kind"), "col") && match(.subset2(base, "name"), nonneg, 0L) > 0L)) {
      return(paste0(
        "`^` around a reducer runs on the GPU only with an even exponent, or over `n()`, ",
        "`n_distinct()`, `sd()` or `var()`: R's power of a `-Inf` or `-0` (`max()` of an empty ",
        "group, `-mean(x)` of a zero mean) differs from the GPU engine's, such as `(-Inf)^0.5`, ",
        "NaN in R and Inf on the GPU."
      ))
    }
  } else if (match(op, c("&", "|", "!"), 0L) > 0L) {
    for (a in args) {
      t <- type_of(a)
      if (identical(t, "FLOAT64")) {
        return(paste0("`&`, `|` and `!` of a double around a reducer read NaN as TRUE on the GPU, ",
                      "where R gives NA; dplyr evaluates it."))
      }
      if (is.na(t) || match(t, c("BOOL8", "INT32"), 0L) == 0L) {
        return(paste0("`&`, `|` and `!` around a reducer run on the GPU only on logical or integer ",
                      "values; dplyr evaluates it."))
      }
    }
  } else if (identical(op, "if_else") || identical(op, "case_when")) {
    conds <- if (identical(op, "if_else")) {
      args[1L]
    } else {
      args[seq(1L, by = 2L, length.out = .subset2(.subset2(ir, "meta"), "n_when"))]
    }
    # (an NA constant condition too, though if_else()'s parse types it logical)
    for (a in conds) {
      if ((identical(.subset2(a, "kind"), "lit") && isTRUE(.subset2(a, "na"))) ||
          !identical(tryCatch(unname(ir_infer_type(a, schema)), error = function(e) NA_character_), "BOOL8")) {
        return(sprintf("a condition of `%s()` that is not a logical expression has no GPU form; dplyr evaluates it.",
                       op))
      }
    }
  }
  for (a in args) {
    why <- summarise_post_na_rule(a, schema, nonneg)
    if (!is.null(why)) {
      return(why)
    }
  }
  NULL
}

# Judge the post expressions of a split before any GPU work: each must parse, with
# no window, against the schema the aggregated table will have (the keys are
# unreadable, so it holds the aggregates and the earlier post results only). A
# dot it declines is claimed for the F88 notice, by the user's dot it comes from.
#
# @param post The result of summarise_post_split()
# @param simple_aggs The aggregations of the agg dots when they are all simple
# @return `post`, or NULL to decline
# @keywords internal
summarise_post_screen <- function(post, in_schema, ir_cache, simple_aggs = NULL) {
  dots <- post$agg_dots
  origin <- attr(dots, "provenance")$origin
  no <- function(dot, reason = fallback_cause_take()) {
    fallback_dot_failed(dot, reason)
    NULL
  }
  types <- character(length(dots))
  for (j in seq_along(dots)) {
    if (!is.null(simple_aggs)) {
      types[[j]] <- unname(.subset2(simple_aggs[[j]], "output_type"))
      next
    }
    fallback_dot_begin()
    parts <- tryCatch(agg_dot_parts(ir_cache, j, dots[[j]], in_schema), error = function(e) NULL)
    if (is.null(parts)) {
      return(no(origin[[j]]))
    }
    expr <- rlang::quo_get_expr(parts$quo)
    decomposed <- decompose_agg_call(expr)
    if (is.null(decomposed) || !decomposed$fn_name %in% names(agg_fn_map)) {
      return(no(origin[[j]]))
    }
    fn <- decomposed$fn_name
    arg <- decomposed$arg
    arg_type <- NULL
    if (!is.null(arg)) {
      if (is.symbol(arg) && as.character(arg) %in% in_schema$names) {
        arg_type <- unname(in_schema$types[[match(as.character(arg), in_schema$names)]])
      } else {
        arg_type <- tryCatch({
          ir <- agg_arg_ir(ir_cache, j, rlang::new_quosure(arg, rlang::quo_get_env(parts$quo)), in_schema)
          if (is.null(ir)) NA_character_ else ir_infer_type(ir, in_schema)
        }, error = function(e) NA_character_)
        if (is.na(arg_type)) {
          return(no(origin[[j]]))
        }
        # (a logical argument is an integer for every reducer that doesn't keep it)
        if (identical(arg_type, "BOOL8") && !agg_keeps_bool(fn)) {
          arg_type <- "INT32"
        }
      }
    }
    types[[j]] <- make_aggregation("o", "i", unname(agg_fn_map[fn]), arg_type)$output_type
  }

  schema <- list(names = names(dots), types = types)
  # (the aggregates that are never -Inf or -0, for a power's base)
  nonneg <- names(dots)[vapply(dots, function(q) {
    h <- summarise_post_head(rlang::quo_get_expr(q))
    !is.null(h) && match(h, c("n", "n_distinct", "sd", "var"), 0L) > 0L
  }, logical(1))]
  for (p in post$post) {
    fallback_dot_begin()
    ir <- tryCatch(
      ir_with_post_agg_cols(schema$names, ir_parse_quo(p$quo, schema)),
      cuplyr_vector_literal = function(e) {
        fallback_cause("a vector constant inside a larger expression is recycled by dplyr.")
        NULL
      },
      error = function(e) NULL
    )
    if (is.null(ir)) {
      cause <- fallback_cause_take()
      # (the expression engine's type text names mutate(); the dot is summarise()'s)
      if (is.character(cause)) {
        cause <- sub("mutate() does not support", "the GPU engine does not support", cause, fixed = TRUE)
      }
      return(no(p$origin, cause))
    }
    if (ir_has_window(ir)) {
      return(no(p$origin, paste0(
        "a function of the group's rows around a reducer (such as `lag()` or `cumsum()`) ",
        "is evaluated per group by dplyr."
      )))
    }
    why <- summarise_post_na_rule(ir, schema, nonneg)
    if (!is.null(why)) {
      return(no(p$origin, why))
    }
    type <- tryCatch(ir_infer_type(ir, schema), error = function(e) NA_character_)
    if (is.na(type) || type %in% c("DICTIONARY32", "LIST", "STRUCT")) {
      return(no(p$origin, "the result of the arithmetic around the reducers has a type the GPU engine does not take here."))
    }
    schema$names <- c(schema$names, p$name)
    schema$types <- c(schema$types, type)
  }
  post
}

# Run the post step on the aggregated table `res` (eager or lazy): one mutate of
# every post expression in dot order (a later one reads an earlier one, as the
# expressions of one mutate node do), then a select of the keys and the user's dot
# names, which drops the temp aggregates. The expressions are parsed against the
# real result (an INT32 sum that left the integer range is a double there). A
# parse that fails now reruns the call on the CPU, as an integer overflow does,
# with its own reason (left in `post$why` for the caller's CPU route).
# @keywords internal
summarise_post_apply <- function(res, post) {
  schema <- current_schema(res)
  rerun <- function() {
    if (is.environment(post$why)) {
      post$why$reason <- paste0("the arithmetic around the reducers has no GPU form for the types ",
                                "the aggregates came out with; dplyr evaluates it.")
    }
    rlang::abort("The arithmetic around the aggregates has no GPU form here.",
                 class = "cuplyr_summarise_rerun")
  }
  exprs <- vector("list", length(post$post))
  for (k in seq_along(post$post)) {
    p <- post$post[[k]]
    expr <- tryCatch({
      ir <- ir_with_post_agg_cols(schema$names[seq_along(schema$names) > post$n_keys], ir_parse_quo(p$quo, schema))
      if (is.null(ir) || ir_has_window(ir)) NULL else make_mutate_expr(p$name, ir, schema)
    }, error = function(e) NULL)
    if (is.null(expr)) {
      rerun()
    }
    exprs[[k]] <- expr
    schema$names <- c(schema$names, p$name)
    schema$types <- c(schema$types, expr$output_type)
  }
  cols <- c(current_schema(res)$names[seq_len(post$n_keys)], post$out_names)
  push_chain(res, function(x) {
    y <- push_op(x, ast_mutate(input_node(x), exprs), carry = FALSE)
    push_op(y, ast_select(input_node(y), cols), input_schema = .subset2(y, "schema"), carry = FALSE)
  })
}
