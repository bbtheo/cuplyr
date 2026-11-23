# Common subexpressions (#17)
#
# `ir_mark_repeats()` finds call subtrees that occur more than once in the
# BOUND IR of one mutate()/filter() lowering and tags them with an integer
# `cse` id. The C++ evaluator (src/expr_eval.hpp) keeps a memo from id to the
# slot of the column that holds the value: when it is about to materialize a
# tagged node a second time it reuses the slot. It never hoists a subtree that
# the AST inlines (that measured slower), so a tag costs nothing where the
# evaluator would not have materialized the subtree anyway.
#
# Keys are structural and bitwise: a bound `col` is its index (an overwritten
# column binds to a new index, so the same text before and after the overwrite
# differs), a literal and the meta of a call go through rlang::hash(), which
# tells 0 from -0 and NA from NaN.

#' Mark repeated call subtrees of bound IR with a `cse` id
#'
#' Marks the maximal repeats: a call subtree that occurs at least twice and, in
#' at least one occurrence, whose parent occurs fewer times than it does (a
#' top-level occurrence has parent count 0). A repeated chain is tagged at its
#' root, not at every inner node.
#'
#' @param irs A list of bound IR trees evaluated by one C++ call (the dots of a
#'   mutate(), or the one predicate of a filter()).
#' @return `irs`, with `cse` set on the tagged nodes; `irs` itself (no copy)
#'   when nothing repeats.
#' @keywords internal
ir_mark_repeats <- function(irs) {
  # A scalar literal is keyed by its bits (`%a` tells -0 from 0 and NA from
  # NaN); anything else is hashed, which is bitwise too but slower.
  lit_key <- function(node) {
    v <- .subset2(node, "value")
    if (length(node) == 4L && length(v) == 1L && is.null(attributes(v))) {
      tag <- if (is.double(v)) {
        sprintf("d%a", v)
      } else if (is.integer(v) || is.logical(v)) {
        paste0(typeof(v), v)
      }
      if (!is.null(tag)) {
        return(paste0(tag, "|", .subset2(node, "type"), "|", .subset2(node, "na")))
      }
    }
    rlang::hash(node)
  }

  keys <- character(8L)
  parents <- integer(8L)
  n <- 0L

  # Returns the structural key of `node`; records every call occurrence in
  # pre-order with its key and its parent's occurrence number.
  leaf_key <- function(node) {
    kind <- .subset2(node, "kind")
    if (identical(kind, "col")) {
      idx <- .subset2(node, "index")
      if (!is.null(idx)) return(paste0("c", idx))
    } else if (identical(kind, "lit")) {
      return(lit_key(node))
    }
    rlang::hash(node)
  }

  walk <- function(node, parent) {
    if (!identical(.subset2(node, "kind"), "call")) return(leaf_key(node))
    n <<- n + 1L
    me <- n
    args <- .subset2(node, "args")
    ck <- character(length(args))
    for (j in seq_along(args)) {
      a <- .subset2(args, j)
      ck[[j]] <- if (identical(.subset2(a, "kind"), "call")) walk(a, me) else leaf_key(a)
    }
    meta <- .subset2(node, "meta")
    key <- paste0(
      .subset2(node, "op"), "(", paste(ck, collapse = ","), ")",
      if (!is.null(names(args))) paste(names(args), collapse = ","),
      if (length(meta) > 0L) rlang::hash(meta)
    )
    if (nchar(key) > 96L) key <- rlang::hash(key)
    if (me > length(keys)) {
      length(keys) <<- 2L * length(keys)
      length(parents) <<- length(keys)
    }
    keys[[me]] <<- key
    parents[[me]] <<- parent
    key
  }

  for (ir in irs) walk(ir, 0L)
  if (n < 2L) return(irs)
  keys <- keys[seq_len(n)]
  if (anyDuplicated(keys) == 0L) return(irs)
  parents <- parents[seq_len(n)]

  u <- unique(keys)
  m <- match(keys, u)
  own <- tabulate(m, length(u))[m]
  parent_count <- c(0L, own)[parents + 1L]
  marked <- unique(m[own >= 2L & parent_count < own])
  ids <- integer(length(u))
  ids[marked] <- seq_along(marked)
  occ_id <- ids[m]

  i <- 0L
  tag <- function(node) {
    if (!identical(.subset2(node, "kind"), "call")) return(node)
    i <<- i + 1L
    me <- i
    args <- .subset2(node, "args")
    for (j in seq_along(args)) args[[j]] <- tag(.subset2(args, j))
    node[["args"]] <- args
    if (occ_id[[me]] > 0L) node[["cse"]] <- occ_id[[me]]
    node
  }
  lapply(irs, tag)
}
