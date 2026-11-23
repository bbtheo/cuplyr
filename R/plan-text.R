# Plan text (#10): the readable plan explain() and show_query() print
#
# A plan prints in pipeline order, source first, one step per line, each step
# the dplyr call it stands for, with expressions as R code. Nothing here runs a
# plan: the renderer only reads the AST (R/ast.R) and its expression IR
# (R/ir.R). The format is the contract of tests/testthat/_snaps/parity-10-explain.md.
#
# A rendered plan is a "block": parallel vectors `text` (the lines), `level`
# (how many CPU-fallback blocks the line sits inside, 0 = none) and `role`
# ("input" marks the `input (computed at ...)` line of a CPU block, "" any
# other). The levels let the CPU history be capped at the newest few steps
# (plan_apply_cap()) and are plain integers, so a block is plain data.

# Longest one-line step before it is split one argument per line
plan_wrap_width <- 80L

# How many nested CPU steps are shown (the newest ones)
plan_cpu_cap <- 3L

# R's operator precedence, tightest first: `^`, unary minus, `:`, `%any%`,
# `* /`, `+ -`, comparisons, `!`, `&`, `|`. A call or an atom binds tightest.
plan_prec_atom <- 100
plan_prec_unary_minus <- 13
plan_prec_not <- 7
plan_prec_special <- 11  # %in%, %%, %/%, ...
plan_prec_compare <- 8
plan_infix_prec <- c(
  "^" = 14, ":" = 12,
  "*" = 10, "/" = 10,
  "+" = 9, "-" = 9,
  "==" = 8, "!=" = 8, "<" = 8, "<=" = 8, ">" = 8, ">=" = 8,
  "&" = 6, "|" = 5
)

# Internal ops printed under the R function they stand for
plan_op_names <- c(
  ceil = "ceiling", cast_float64 = "as.double", cast_int32 = "as.integer",
  factor_codes = "as.integer", factor_labels = "as.character"
)

# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------

plan_block <- function(text, level = 0L, role = "") {
  list(text = text, level = rep_len(as.integer(level), length(text)),
       role = rep_len(role, length(text)))
}

plan_block_c <- function(blocks) {
  list(
    text = as.character(unlist(lapply(blocks, .subset2, "text"), use.names = FALSE)),
    level = as.integer(unlist(lapply(blocks, .subset2, "level"), use.names = FALSE)),
    role = as.character(unlist(lapply(blocks, .subset2, "role"), use.names = FALSE))
  )
}

plan_indent <- function(block, n) {
  if (length(block$text) > 0L) {
    block$text <- paste0(strrep(" ", n), block$text)
  }
  block
}

# One line of text on one line
plan_one_line <- function(x) {
  gsub("[[:space:]]*\n[[:space:]]*", " ", x)
}

# -----------------------------------------------------------------------------
# Names, numbers, literals
# -----------------------------------------------------------------------------

# Column names as R code: backticks around a name that isn't syntactic
plan_names <- function(x) {
  x <- as.character(x)
  bad <- is.na(x) | make.names(x) != x
  x[bad] <- paste0("`", gsub("`", "\\\\`", x[bad]), "`")
  x
}

# `a, b` / `c(a, b)`: a `.by` / `by` value
plan_by <- function(cols) {
  if (length(cols) == 1L) plan_names(cols) else paste0("c(", paste(plan_names(cols), collapse = ", "), ")")
}

# A number as R code, without the `L` of an integer
plan_num <- function(x) {
  sub("L$", "", paste(deparse(x, width.cutoff = 500L), collapse = ""))
}

# `columns read: all 5` / `columns read: 2 of 5 (a, b)`; more than 8 names show the first 8
plan_columns_text <- function(read, n_total) {
  k <- length(read)
  if (is.null(read) || k >= n_total) {
    return(sprintf("columns read: all %d", n_total))
  }
  shown <- if (k > 8L) {
    paste0(paste(read[seq_len(8L)], collapse = ", "), ", and ", k - 8L, " more")
  } else {
    paste(read, collapse = ", ")
  }
  sprintf("columns read: %d of %d (%s)", k, n_total, shown)
}

# A literal vector as R code; more than `cap` values print as `<n values>`
plan_value <- function(value, type, cap) {
  n <- length(value)
  if (n > cap) {
    return(sprintf("<%d values>", n))
  }
  type <- unname(type %||% "")
  tryCatch({
    if (identical(type, "TIMESTAMP_DAYS") && is.numeric(unclass(value))) {
      days <- as.numeric(unclass(value))
      text <- format(as.Date(days, origin = "1970-01-01"))
      text[is.na(days)] <- NA_character_
      return(paste0("as.Date(", paste(deparse(text, width.cutoff = 500L), collapse = ""), ")"))
    }
    if (identical(type, "TIMESTAMP_MICROSECONDS") && is.numeric(unclass(value))) {
      secs <- as.numeric(unclass(value))
      tz <- attr(value, "tzone")
      tz <- if (is.null(tz) || !nzchar(tz[[1L]])) "UTC" else tz[[1L]]
      fmt <- if (all(is.na(secs) | secs == floor(secs))) "%Y-%m-%d %H:%M:%S" else "%Y-%m-%d %H:%M:%OS6"
      text <- format(as.POSIXct(secs, origin = "1970-01-01", tz = tz), fmt)
      text[is.na(secs)] <- NA_character_
      return(paste0("as.POSIXct(", paste(deparse(text, width.cutoff = 500L), collapse = ""),
                    ", tz = ", deparse(tz), ")"))
    }
    if (is.factor(value)) {
      value <- as.character(value)
    }
    if (inherits(value, "integer64")) {
      text <- as.character(value)
      return(if (n == 1L) text else paste0("c(", paste(text, collapse = ", "), ")"))
    }
    if (is.object(value)) {
      return(sprintf("<%s>", class(value)[[1L]]))
    }
    paste(deparse(unname(value), width.cutoff = 500L), collapse = "")
  }, error = function(e) "<value>")
}

# -----------------------------------------------------------------------------
# Expressions: IR -> R code, with explicit parentheses
# -----------------------------------------------------------------------------

plan_is_na_lit <- function(ir) {
  if (!identical(.subset2(ir, "kind"), "lit")) {
    return(FALSE)
  }
  value <- .subset2(ir, "value")
  isTRUE(.subset2(ir, "na")) ||
    (length(value) == 1L && is.atomic(value) && !is.object(value) && is.na(value) && !is.nan(value))
}

# @return `list(text, prec)`: the code and how tightly its top operator binds
plan_ir <- function(ir) {
  atom <- function(text, prec = plan_prec_atom) list(text = text, prec = prec)
  switch(as.character(.subset2(ir, "kind") %||% ""),
    col = atom(plan_names(.subset2(ir, "name"))),
    lit = {
      value <- .subset2(ir, "value")
      if (plan_is_na_lit(ir)) {
        atom("NA")
      } else if (is.null(value)) {
        atom("NULL")
      } else {
        negative <- length(value) == 1L && is.numeric(value) && !is.object(value) && !is.na(value) &&
          value < 0
        atom(plan_value(value, .subset2(ir, "type"), cap = 10L),
             if (negative) plan_prec_unary_minus else plan_prec_atom)
      }
    },
    vec = atom(plan_value(.subset2(ir, "value"), .subset2(ir, "type"), cap = 6L)),
    call = plan_call(ir),
    atom("<expression>")
  )
}

plan_ir_text <- function(ir) {
  .subset2(plan_ir(ir), "text")
}

plan_op_prec <- function(op) {
  p <- plan_infix_prec[op]
  if (is.na(p)) plan_prec_special else unname(p)
}

# `left op right`, parenthesised where a child binds looser than `op` (and, for
# equal precedence, on the side R's associativity doesn't group)
plan_infix <- function(op, left, right) {
  p <- plan_op_prec(op)
  right_assoc <- identical(op, "^")
  non_assoc <- p == plan_prec_compare
  paren_left <- if (non_assoc || right_assoc) left$prec <= p else left$prec < p
  paren_right <- if (right_assoc) right$prec < p else right$prec <= p
  lt <- if (paren_left) paste0("(", left$text, ")") else left$text
  rt <- if (paren_right) paste0("(", right$text, ")") else right$text
  sep <- if (op %in% c("^", ":")) op else paste0(" ", op, " ")
  list(text = paste0(lt, sep, rt), prec = p)
}

# `-x` / `!x`
plan_prefix <- function(op, arg) {
  p <- if (identical(op, "!")) plan_prec_not else plan_prec_unary_minus
  text <- if (arg$prec <= p) paste0("(", arg$text, ")") else arg$text
  list(text = paste0(op, text), prec = p)
}

plan_call <- function(ir) {
  op <- as.character(.subset2(ir, "op"))
  args <- .subset2(ir, "args")
  meta <- .subset2(ir, "meta")
  n <- length(args)
  parts <- lapply(args, plan_ir)

  if (n == 2L && (op %in% names(plan_infix_prec) || grepl("^%.*%$", op))) {
    return(plan_infix(op, parts[[1L]], parts[[2L]]))
  }
  if (n == 1L && op %in% c("-", "+", "!")) {
    return(plan_prefix(op, parts[[1L]]))
  }
  # a date shifted by whole days, a POSIXct by seconds (stored in microseconds)
  if (n == 2L && op %in% c("date_add_days", "date_sub_days")) {
    return(plan_infix(if (identical(op, "date_add_days")) "+" else "-", parts[[1L]], parts[[2L]]))
  }
  if (n == 2L && op %in% c("ts_add_us", "ts_sub_us")) {
    shift <- .subset2(args[[2L]], "value")
    if (is.numeric(shift) && length(shift) == 1L) {
      parts[[2L]] <- list(text = plan_num(shift / 1e6), prec = if (shift < 0) plan_prec_unary_minus else plan_prec_atom)
    }
    return(plan_infix(if (identical(op, "ts_add_us")) "+" else "-", parts[[1L]], parts[[2L]]))
  }

  texts <- vapply(parts, .subset2, character(1), "text")
  user_call <- plan_user_call(op, args, texts, meta)
  if (!is.null(user_call)) {
    return(list(text = user_call, prec = plan_prec_atom))
  }
  fn <- if (op %in% names(plan_op_names)) unname(plan_op_names[op]) else op

  if (identical(op, "factor_labels") && n >= 1L) {
    texts <- texts[1L]  # the levels are an implementation detail
  }
  if (identical(op, "if_else") && n == 4L) {
    texts[4L] <- paste0("missing = ", texts[4L])
  }
  nw <- meta[["n_when"]]
  if (identical(op, "case_when") && is.numeric(nw) && length(nw) == 1L && n == 2L * nw + 1L) {
    idx <- seq_len(nw)
    pairs <- paste0(texts[2L * idx - 1L], " ~ ", texts[2L * idx])
    texts <- c(pairs, if (!plan_is_na_lit(args[[n]])) paste0(".default = ", texts[n]))
  }
  if (isTRUE(meta[["rank_desc"]]) && n == 1L) {
    texts[1L] <- paste0("desc(", texts[1L], ")")
  }
  k <- meta[["n"]]
  if (is.numeric(k) && length(k) == 1L && !is.na(k) && !(op %in% c("lag", "lead") && k == 1)) {
    texts <- c(texts, plan_num(k))
  }
  default <- meta[["default"]]
  if (inherits(default, "cuplyr_ir") && !plan_is_na_lit(default)) {
    texts <- c(texts, paste0("default = ", plan_ir_text(default)))
  }
  if (isTRUE(meta[["na_rm"]])) {
    texts <- c(texts, paste0(if (op %in% c("first", "last", "nth")) "na_rm" else "na.rm", " = TRUE"))
  }
  list(text = paste0(fn, "(", paste(texts, collapse = ", "), ")"), prec = plan_prec_atom)
}

# The IR encodes some calls in an order or a form of its own (R/ir.R): the
# string functions take the column first, a paste carries its separator last,
# substr() is 0-based, a recode is one lookup. This is the call the user wrote,
# or an equivalent one, for those ops; NULL for any other op.
# @param texts The printed arguments
plan_user_call <- function(op, args, texts, meta) {
  n <- length(args)
  call_text <- function(fn, ...) paste0(fn, "(", paste(c(...), collapse = ", "), ")")
  flag <- function(arg, name) {
    if (isTRUE(.subset2(arg, "value"))) paste0(name, " = TRUE")
  }
  switch(op,
    # (x, pattern, fixed): the pattern comes first, `fixed` only when set
    grepl = if (n == 3L) call_text("grepl", texts[2L], texts[1L], flag(args[[3L]], "fixed")),
    # (x, pattern, replacement, fixed)
    sub = , gsub = if (n == 4L) {
      call_text(op, texts[2L], texts[3L], texts[1L], flag(args[[4L]], "fixed"))
    },
    starts_with = if (n == 2L) call_text("startsWith", texts),
    ends_with = if (n == 2L) call_text("endsWith", texts),
    # (x, start - 1, stop): the start is 0-based
    substr = if (n == 3L) {
      start <- .subset2(args[[2L]], "value")
      if (identical(.subset2(args[[2L]], "kind"), "lit") && is.numeric(start) && length(start) == 1L &&
          !is.na(start)) {
        call_text("substr", texts[1L], plan_value(as.integer(start) + 1L, NULL, 10L), texts[3L])
      }
    },
    # (..., sep)
    paste_cols = if (n >= 2L) {
      sep <- .subset2(args[[n]], "value")
      if (identical(sep, "")) {
        call_text("paste0", texts[-n])
      } else if (identical(sep, " ")) {
        call_text("paste", texts[-n])
      } else {
        call_text("paste", texts[-n], paste0("sep = ", texts[n]))
      }
    },
    # (x, which): `which` only when it isn't the default
    trimws = if (n == 2L) {
      call_text("trimws", texts[1L], if (!identical(.subset2(args[[2L]], "value"), "both")) texts[2L])
    },
    # a cast to the UTC day; a POSIXct keeps its zone and a Date becomes UTC
    # midnight, which as.POSIXct() does with no `tz`
    as_date = if (n == 1L) call_text("as.Date", texts, 'tz = "UTC"'),
    as_posixct = if (n == 1L) call_text("as.POSIXct", texts),
    # horizontal aggregates of a rowwise table, its operands as written
    # (meta$form): `max(a, b)`, `sum(c(a, b))`, `sum(c_across(c(a, b)))`; the
    # step that holds one is marked rowwise (plan_step())
    rw_agg_i32 = , rw_agg_f64 = {
      fn <- as.character(meta[["fn"]] %||% "")
      if (nzchar(fn) && n >= 1L) {
        form <- meta[["form"]]
        vec <- if (n == 1L) texts else call_text("c", texts)
        operands <- if (identical(form, "args")) {
          texts
        } else if (identical(form, "c_across")) {
          call_text("c_across", vec)
        } else if (identical(form, "c")) {
          call_text("c", texts)
        } else {
          vec
        }
        call_text(fn, operands, if (isTRUE(meta[["na_rm"]])) "na.rm = TRUE")
      }
    },
    # (x, keys, values, default, missing): recode()
    recode_lookup = plan_recode(args, texts),
    # (x, cond1, value1, ...): replace_when()
    replace_when = {
      nw <- meta[["n_when"]]
      if (is.numeric(nw) && length(nw) == 1L && n == 2L * nw + 1L) {
        idx <- seq_len(nw)
        call_text("replace_when", texts[1L], paste0(texts[2L * idx], " ~ ", texts[2L * idx + 1L]))
      }
    },
    {
      round_op <- regmatches(op, regexec("^(floor_date|ceiling_date|round_date)_(second|minute|hour|day|week)$", op))[[1L]]
      if (length(round_op) == 3L && n == 1L) {
        call_text(round_op[[2L]], texts, deparse(round_op[[3L]]))
      }
    }
  )
}

# Most replacements a printed recode() lists; the rest are counted
plan_recode_cap <- 20L

# `recode(x, a = "A", `1` = "b", .default = d, .missing = m)` from the lookup node
# the parser builds for it; past `plan_recode_cap` pairs the first ones are
# listed and the others counted (`<5 more>`). A node of another shape (none is
# built) still prints as recode(), never as the lookup op.
plan_recode <- function(args, texts) {
  keys <- if (length(args) == 5L) .subset2(args[[2L]], "value")
  vals <- if (length(args) == 5L) .subset2(args[[3L]], "value")
  if (length(keys) == 0L || length(keys) != length(vals)) {
    return(paste0("recode(", paste(c(texts[1L], "<replacements>"), collapse = ", "), ")"))
  }
  shown <- seq_len(min(length(keys), plan_recode_cap))
  pairs <- paste0(plan_names(as.character(keys[shown])), " = ",
                  vapply(shown, function(i) plan_value(vals[[i]], NULL, 1L), character(1)))
  if (length(keys) > plan_recode_cap) {
    pairs <- c(pairs, sprintf("<%d more>", length(keys) - plan_recode_cap))
  }
  default <- if (!identical(args[[4L]], args[[1L]])) paste0(".default = ", texts[4L])
  missing <- if (!plan_is_na_lit(args[[5L]])) paste0(".missing = ", texts[5L])
  paste0("recode(", paste(c(texts[1L], pairs, default, missing), collapse = ", "), ")")
}

# -----------------------------------------------------------------------------
# Steps
# -----------------------------------------------------------------------------

# `verb(a, b)`, or `verb(` + one argument per line + `)` when the one-line form
# is longer than the wrap width (the optional `comment` is not measured)
plan_step_lines <- function(verb, args, comment = NULL, wrap = TRUE) {
  one <- paste0(verb, "(", paste(args, collapse = ", "), ")")
  tail <- if (is.null(comment)) "" else paste0("  # ", comment)
  if (!wrap || length(args) == 0L || nchar(one) <= plan_wrap_width) {
    return(paste0(one, tail))
  }
  c(paste0(verb, "("),
    paste0("  ", args, c(rep(",", length(args) - 1L), "")),
    paste0(")", tail))
}

plan_aggregation <- function(agg) {
  fn <- .subset2(agg, "fn")
  input <- .subset2(agg, "input_col")
  args <- if (is.null(input) || is.na(input)) character() else plan_names(input)
  extra <- .subset2(agg, "extra")
  if (length(extra) == 1L && !is.na(extra)) {
    args <- c(args, plan_num(extra))
  }
  if (isTRUE(.subset2(agg, "na_rm"))) {
    args <- c(args, paste0(if (fn %in% c("first", "last", "nth")) "na_rm" else "na.rm", " = TRUE"))
  }
  paste0(fn, "(", paste(args, collapse = ", "), ")")
}

plan_summarise <- function(node) {
  aggs <- .subset2(node, "aggregations")
  groups <- .subset2(node, "groups")
  args <- vapply(aggs, function(a) {
    paste0(plan_names(.subset2(a, "output_col")), " = ", plan_aggregation(a))
  }, character(1), USE.NAMES = FALSE)
  by_first <- identical(.subset2(node, "group_order"), "first")
  if (length(groups) > 0L && by_first) {
    return(plan_block(plan_step_lines("summarise", c(args, paste0(".by = ", plan_by(groups))))))
  }
  group_line <- if (length(groups) > 0L) paste0("group_by(", paste(plan_names(groups), collapse = ", "), ")")
  plan_block(c(group_line, plan_step_lines("summarise", args)))
}

plan_slice <- function(node) {
  mode <- as.character(.subset2(node, "mode") %||% "")
  amount <- .subset2(node, "amount")
  size <- paste0(if (isTRUE(.subset2(node, "is_prop"))) "prop = " else "n = ",
                 if (is.null(amount)) "?" else plan_num(amount))
  lines <- switch(mode,
    head = plan_step_lines("slice_head", size),
    tail = plan_step_lines("slice_tail", size),
    index = plan_step_lines("slice", plan_value(.subset2(node, "raw_indices"), NULL, cap = 10L)),
    bracket = {
      idx <- .subset2(node, "raw_indices")
      plan_step_lines("slice", if (is.atomic(idx)) plan_value(idx, NULL, cap = 10L) else "...",
                      comment = "x[i, ]")
    },
    rank = , rank_grouped = {
      args <- c(plan_names(.subset2(node, "order_col")), size)
      if (identical(.subset2(node, "with_ties"), FALSE)) args <- c(args, "with_ties = FALSE")
      if (isTRUE(.subset2(node, "na_rm"))) args <- c(args, "na_rm = TRUE")
      if (identical(mode, "rank_grouped")) {
        args <- c(args, paste0("by = ", plan_by(.subset2(node, "group_cols"))))
      }
      plan_step_lines(if (isTRUE(.subset2(node, "descending"))) "slice_max" else "slice_min", args,
                      comment = if (isTRUE(.subset2(node, "top_k"))) "top-k: arrange() |> head()")
    },
    plan_step_lines("slice", character())
  )
  plan_block(lines)
}

plan_window_spec <- function(spec) {
  fn <- .subset2(spec, "fn")
  value <- .subset2(spec, "value_col")
  order_cols <- .subset2(spec, "order_cols")
  order_desc <- .subset2(spec, "order_desc")
  has_value <- length(value) == 1L && !is.na(value)
  args <- character()
  if (has_value) {
    arg <- plan_names(value)
    # a rank call orders by its own argument: descending reads desc(x)
    if (identical(order_cols, value) && length(order_desc) >= 1L && isTRUE(order_desc[[1L]])) {
      arg <- paste0("desc(", arg, ")")
    }
    args <- arg
  }
  k <- .subset2(spec, "n")
  if (is.numeric(k) && length(k) == 1L && !is.na(k) && !(fn %in% c("lag", "lead") && k == 1)) {
    args <- c(args, plan_num(k))
  }
  default <- .subset2(spec, "default")
  if (inherits(default, "cuplyr_ir") && !plan_is_na_lit(default)) {
    args <- c(args, paste0("default = ", plan_ir_text(default)))
  }
  if (isTRUE(.subset2(spec, "na_rm"))) {
    args <- c(args, paste0(if (fn %in% c("first", "last", "nth")) "na_rm" else "na.rm", " = TRUE"))
  }
  if (length(order_cols) > 0L && !(has_value && identical(order_cols, value))) {
    ord <- plan_names(order_cols)
    if (length(order_desc) == length(ord)) ord[order_desc] <- paste0("desc(", ord[order_desc], ")")
    args <- c(args, paste0("order_by = ", if (length(ord) == 1L) ord else paste0("c(", paste(ord, collapse = ", "), ")")))
  }
  paste0(fn, "(", paste(args, collapse = ", "), ")")
}

plan_window <- function(node) {
  specs <- .subset2(node, "specs")
  args <- vapply(specs, function(s) {
    paste0(plan_names(.subset2(s, "output_col")), " = ", plan_window_spec(s))
  }, character(1), USE.NAMES = FALSE)
  groups <- .subset2(node, "group_cols")
  if (length(groups) > 0L) args <- c(args, paste0(".by = ", plan_by(groups)))
  plan_block(plan_step_lines("mutate", args))
}

# `join_by(k, a >= b, closest(t >= ty))`
plan_join_by <- function(by) {
  left <- as.character(.subset2(by, "left"))
  right <- as.character(.subset2(by, "right"))
  op <- as.character(.subset2(by, "op"))
  cond <- ifelse(op == "==" & left == right, plan_names(left),
                 paste(plan_names(left), op, plan_names(right)))
  filter <- .subset2(by, "filter")
  rolling <- length(filter) == length(cond) & filter != "none"
  cond[rolling] <- paste0("closest(", cond[rolling], ")")
  paste0("join_by(", paste(cond, collapse = ", "), ")")
}

# The join step, then `y:` and the right input's plan under it
plan_join <- function(node, optimized) {
  type <- .subset2(node, "join_type")
  by <- .subset2(node, "by")
  cross <- identical(type, "cross") || isTRUE(.subset2(node, "cross_by"))
  filtering <- type %in% c("semi", "anti")
  args <- "y"
  if (!cross) {
    args <- c(args, paste0("by = ", plan_join_by(by)))
  }
  if (!cross && !filtering && isTRUE(.subset2(node, "keep"))) {
    args <- c(args, "keep = TRUE")
  }
  # (only the arguments the dplyr call has: no suffix on a semi/anti join, no
  # na_matches on a cross join)
  suffix <- .subset2(node, "suffix")
  if (!filtering && !is.null(suffix) && !identical(unname(suffix), c(".x", ".y"))) {
    args <- c(args, paste0("suffix = ", paste(deparse(unname(suffix), width.cutoff = 500L), collapse = "")))
  }
  na_matches <- .subset2(node, "na_matches")
  if (!cross && !is.null(na_matches) && !identical(na_matches, "na")) {
    args <- c(args, paste0("na_matches = ", deparse(na_matches)))
  }
  if (!cross && !filtering) {
    multiple <- .subset2(node, "multiple")
    if (!is.null(multiple) && !identical(multiple, "all")) {
      args <- c(args, paste0("multiple = ", deparse(multiple)))
    }
    unmatched <- .subset2(node, "unmatched")
    if (!is.null(unmatched) && !all(unmatched == "drop")) {
      args <- c(args, paste0("unmatched = ", paste(deparse(unname(unmatched), width.cutoff = 500L), collapse = "")))
    }
    # dplyr only checks an all-`==` join: for any other the node carries
    # "many-to-many" as the default, which the user didn't write
    relationship <- .subset2(node, "relationship")
    equi <- length(.subset2(by, "op")) > 0L && all(.subset2(by, "op") == "==")
    if (!is.null(relationship) && !(identical(relationship, "many-to-many") && !equi)) {
      args <- c(args, paste0("relationship = ", deparse(relationship)))
    }
  }
  right <- plan_indent(plan_node_block(.subset2(node, "right"), optimized), 4L)
  plan_block_c(list(
    plan_block(plan_step_lines(if (cross) "cross_join" else paste0(type, "_join"), args, wrap = FALSE)),
    plan_block("  y:"),
    right
  ))
}

# TRUE iff the IR `ir` holds a rowwise horizontal aggregate (rw_agg_*), which
# only a rowwise table builds (R/rowwise.R): `max(x, i)` there is per row
plan_has_rw_agg <- function(ir) {
  if (!inherits(ir, "cuplyr_ir") || !identical(.subset2(ir, "kind"), "call")) {
    return(FALSE)
  }
  .subset2(ir, "op") %in% c("rw_agg_i32", "rw_agg_f64") ||
    any(vapply(.subset2(ir, "args"), plan_has_rw_agg, logical(1)))
}

# One node as the dplyr call(s) it stands for (not a source, not a join). A
# filter or mutate on a rowwise table is marked `# rowwise` (its aggregates are
# per row), a filter on a grouped one `# grouped`.
plan_step <- function(node) {
  type <- .subset2(node, "type")
  switch(type,
    filter = {
      irs <- lapply(.subset2(node, "predicates"), .subset2, "ir")
      rowwise <- isTRUE(.subset2(node, "rowwise")) || any(vapply(irs, plan_has_rw_agg, logical(1)))
      plan_block(plan_step_lines(
        "filter", vapply(irs, plan_ir_text, character(1), USE.NAMES = FALSE),
        comment = if (rowwise) "rowwise" else if (isTRUE(.subset2(node, "grouped"))) "grouped"
      ))
    },
    mutate = {
      exprs <- .subset2(node, "expressions")
      plan_block(plan_step_lines(
        "mutate",
        vapply(exprs, function(e) {
          paste0(plan_names(.subset2(e, "output_col")), " = ", plan_ir_text(.subset2(e, "ir")))
        }, character(1), USE.NAMES = FALSE),
        comment = if (any(vapply(exprs, function(e) plan_has_rw_agg(.subset2(e, "ir")), logical(1)))) "rowwise"
      ))
    },
    select = plan_block(plan_step_lines("select", plan_names(.subset2(node, "columns")))),
    rename = plan_block(plan_step_lines(
      "rename", paste0(plan_names(.subset2(node, "new_names")), " = ", plan_names(.subset2(node, "old_names")))
    )),
    distinct = plan_block(plan_step_lines(
      "distinct", c(plan_names(.subset2(node, "key_cols")),
                    if (isTRUE(.subset2(node, "keep_all"))) ".keep_all = TRUE")
    )),
    arrange = {
      specs <- .subset2(node, "sort_specs")
      keys <- vapply(specs, function(s) {
        col <- plan_names(.subset2(s, "col_name"))
        if (isTRUE(.subset2(s, "descending"))) paste0("desc(", col, ")") else col
      }, character(1), USE.NAMES = FALSE)
      by_group <- length(.subset2(node, "groups")) > 0L
      plan_block(plan_step_lines("arrange", c(keys, if (by_group) ".by_group = TRUE")))
    },
    summarise = plan_summarise(node),
    slice = plan_slice(node),
    window = plan_window(node),
    barrier = plan_block("collapse()"),
    reframe = {
      names <- vapply(.subset2(node, "outputs"), function(o) as.character(.subset2(o, "name")), character(1),
                      USE.NAMES = FALSE)
      plan_block(plan_step_lines("reframe", plan_names(names),
                                 comment = paste0(.subset2(node, "k"), " rows per group")))
    },
    plan_block(paste0(type, "()"))
  )
}

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

# A source: the table on the GPU, a parquet scan, or the result of a CPU step.
# `columns` is what a select the optimizer added on top of it keeps (folded into
# the source's `columns read:` line).
plan_source_block <- function(node, optimized, columns = NULL) {
  n_total <- length(.subset2(.subset2(node, "schema"), "names"))
  origin <- .subset2(node, "origin")
  scan <- .subset2(node, "scan")

  if (!is.null(origin)) {
    verb <- as.character(.subset2(origin, "verb"))
    expr <- .subset2(origin, "expr")
    head <- if (is.null(expr) || is.na(expr)) {
      sprintf("CPU fallback: %s()", verb)
    } else {
      sprintf("CPU fallback: %s() for %s", verb, plan_one_line(expr))
    }
    input <- .subset2(.subset2(origin, "input"), if (optimized) "optimized" else "as_written")
    if (is.null(input)) {
      input <- plan_block("plan not recorded")
    }
    input <- plan_indent(input, 4L)
    return(list(
      text = c(head,
               paste0("  reason: ", plan_one_line(.subset2(origin, "reason"))),
               paste0("  ", plan_columns_text(columns %||% .subset2(.subset2(node, "schema"), "names"), n_total)),
               sprintf("  input (computed at the %s() call):", verb),
               input$text),
      level = c(rep(1L, 4L), input$level + 1L),
      role = c("", "", "", "input", input$role)
    ))
  }

  if (!is.null(scan)) {
    files <- basename(.subset2(scan, "paths"))
    head <- if (length(files) == 1L) {
      paste0("parquet scan: ", files)
    } else {
      sprintf("parquet scan: %d files (%s%s)", length(files), paste(utils::head(files, 3L), collapse = ", "),
              if (length(files) > 3L) ", ..." else "")
    }
    file_names <- .subset2(scan, "file_names")
    read <- .subset2(scan, "columns") %||% file_names
    preds <- .subset2(scan, "predicates")
    filter_line <- if (length(preds) > 0L) {
      # R code (`code`), the conjunct as the plan would write it; `text` is the reader's note
      texts <- vapply(preds, function(p) {
        as.character(.subset2(p, "code") %||% .subset2(p, "text") %||% "?")
      }, character(1))
      paste0("  reader filter: ", paste(texts, collapse = ", "))
    }
    return(plan_block(c(head, paste0("  ", plan_columns_text(read, length(file_names))), filter_line)))
  }

  plan_block(c("table on the GPU",
               paste0("  ", plan_columns_text(columns %||% .subset2(.subset2(node, "schema"), "names"), n_total))))
}

# -----------------------------------------------------------------------------
# A whole plan
# -----------------------------------------------------------------------------

# The plan under `node`, source first. A select the optimizer inserted
# (`pruned = TRUE`) directly on an in-memory or CPU-result source is folded into
# the source's `columns read:` line; on a parquet scan it stays a step, as the reader
# then reads every column (pushdown off).
plan_node_block <- function(node, optimized) {
  steps <- list()
  while (!is.null(node) && !identical(.subset2(node, "type"), "source")) {
    steps[[length(steps) + 1L]] <- node
    node <- if (identical(.subset2(node, "type"), "join")) .subset2(node, "left") else .subset2(node, "input")
  }
  if (is.null(node)) {
    base <- plan_block(character())
  } else {
    fold <- NULL
    n_steps <- length(steps)
    if (n_steps > 0L) {
      last <- steps[[n_steps]]
      if (identical(.subset2(last, "type"), "select") && isTRUE(.subset2(last, "pruned")) &&
          is.null(.subset2(node, "scan"))) {
        fold <- .subset2(last, "columns")
        steps <- steps[-n_steps]
      }
    }
    base <- plan_source_block(node, optimized, columns = fold)
  }
  blocks <- vector("list", length(steps) + 1L)
  blocks[[1L]] <- base
  for (i in seq_along(steps)) {
    step <- steps[[length(steps) - i + 1L]]
    blocks[[i + 1L]] <- if (identical(.subset2(step, "type"), "join")) {
      plan_join(step, optimized)
    } else {
      plan_step(step)
    }
  }
  plan_block_c(blocks)
}

# The block with the CPU history cut to the newest `cap` steps: lines of a
# deeper CPU block go, and the input of the oldest block shown, when it holds
# one, becomes a single line.
plan_apply_cap <- function(block, cap = plan_cpu_cap) {
  n <- length(block$text)
  if (n == 0L || all(block$level <= cap)) {
    return(block)
  }
  indent <- nchar(block$text) - nchar(sub("^ +", "", block$text))
  keep <- rep(TRUE, n)
  for (i in seq_len(n)) {
    if (!keep[[i]]) next
    if (block$level[[i]] > cap) {
      keep[[i]] <- FALSE
    } else if (identical(block$role[[i]], "input") && block$level[[i]] == cap) {
      j <- i
      while (j < n && indent[[j + 1L]] > indent[[i]]) j <- j + 1L
      if (j > i && any(block$level[(i + 1L):j] > cap)) {
        block$text[[i]] <- paste0(strrep(" ", indent[[i]]), "input: not shown (older CPU steps)")
        keep[(i + 1L):j] <- FALSE
      }
    }
  }
  list(text = block$text[keep], level = block$level[keep], role = block$role[keep])
}

# The root a table's plan is read from: its pending plan, or its source (with the
# CPU step that produced it)
plan_root_node <- function(x) {
  if (has_pending_ops(x)) {
    return(.subset2(x, "lazy_ops"))
  }
  ast_source(.subset2(x, "schema"), origin = table_origin(x))
}

# The plan of `node` as the lines `show_query()` prints, optimized or as written
plan_node_text <- function(node, optimized) {
  if (optimized && !identical(.subset2(node, "type"), "source")) {
    node <- optimize_ast(node)
  }
  plan_node_block(node, optimized)
}

# What `show_query()` and `explain()` print for `x`; nothing runs
plan_show <- function(x, optimized) {
  if (!identical(.subset2(x, "exec_mode"), "lazy")) {
    cat("<cuplyr plan>\n",
        "No pending operations: an eager table runs each verb when it is called.\n", sep = "")
    return(invisible(NULL))
  }
  block <- plan_apply_cap(plan_node_text(plan_root_node(x), optimized))
  cat(paste0(c(paste0("<cuplyr plan: ", if (optimized) "optimized" else "as written", ">"), block$text), "\n"),
      sep = "")
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# The CPU history
# -----------------------------------------------------------------------------

# The origin of a table a CPU step produced from the lazy table `.data`: what
# the step was and why it ran there, and the plan it computed, rendered now in
# both modes. Plain character and integer data only: nothing that holds an R
# environment or a GPU table (a stored AST would keep the sides of a join
# alive). The history is cut at 3 nested CPU steps, the most a plan shows.
# Rendering can never break a fallback: a failure is stored as one line.
# @param .data The lazy `tbl_gpu` the verb was called on
# @param note What `cuplyr_fallback_notify()` reported (verb, expr, reason)
# @param result The lazy `tbl_gpu` the step produced: its columns and its GPU
#   table (by address) tell a stale origin from a true one
# @keywords internal
plan_origin <- function(.data, note, result) {
  input_of <- function(optimized) {
    block <- plan_node_text(plan_root_node(.data), optimized)
    keep <- block$level <= plan_cpu_cap
    list(text = block$text[keep], level = block$level[keep], role = block$role[keep])
  }
  input <- tryCatch(
    suppressWarnings(list(optimized = input_of(TRUE), as_written = input_of(FALSE))),
    error = function(e) {
      msg <- list(text = plan_one_line(paste0("plan could not be shown: ", conditionMessage(e))),
                  level = 0L, role = "")
      list(optimized = msg, as_written = msg)
    }
  )
  list(
    verb = as.character(note$verb %||% "?"),
    expr = as.character(note$expr %||% NA_character_),
    reason = as.character(note$reason %||% ""),
    names = unname(.subset2(.subset2(result, "schema"), "names")),
    ptr = rlang::obj_address(.subset2(result, "ptr")),
    input = input
  )
}

# The origin of `x`'s data if it still describes it (the table holds the GPU
# table the CPU step produced, with its columns), else NULL
table_origin <- function(x) {
  origin <- .subset2(x, "origin")
  if (is.null(origin) ||
      !identical(.subset2(origin, "ptr"), rlang::obj_address(.subset2(x, "ptr"))) ||
      !identical(.subset2(origin, "names"), unname(.subset2(.subset2(x, "schema"), "names")))) {
    return(NULL)
  }
  origin
}

# `out` with the CPU history of `from` (a table that keeps `from`'s GPU table and
# schema, like group_by() and ungroup() do)
carry_origin <- function(out, from) {
  origin <- .subset2(from, "origin")
  if (is.null(origin)) out else set_tbl_gpu_field(out, "origin", origin)
}
