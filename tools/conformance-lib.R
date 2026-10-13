# Library for the dplyr conformance harness (tools/run-dplyr-conformance.R).
#
# Functions only: sourcing this file has no side effects (no load_all, no
# argument parsing, no GPU check). It assumes cuplyr is already loaded
# (pkgload::load_all() or library(cuplyr)); the CLI does that. Keeping the
# logic here lets tests/testthat/test-parity-F01-harness.R drive it end to end
# on fake vendored-style files.
#
# Public API
#   conformance_run_file(path, exclusions = NULL)   run one test file
#   conformance_load_exclusions(path)               read + validate the manifest
#   conformance_write_ids(ids, path)                per-test TSV (merge gate input)
#   conformance_read_ids(path)                      read such a TSV back
#   conformance_compare(old_ids, new_ids)           per-file counts, lost passes
#   conformance_load_known_failures(path)           read + validate known failures
#   conformance_gate_feature_name(out_dir)          merging feature from the gate dir
#   conformance_gate_check(...)                     merge gate decision (see there)
#   conformance_read_gate_ids(path)                 gate input TSV, columns checked
#   conformance_gate_added(...)                     NOTE lines for rows a candidate adds
#   conformance_classify_reason(msg)                heuristic failure category
#   conformance_report(...)                         markdown report lines
#
# conformance_run_file(upload = FALSE) is a control run: nothing is uploaded, so
# the file runs on plain dplyr/tibble (used to check that an excluded test or a
# suspected harness artifact fails without cuplyr too).
#
# Status vocabulary: pass, fail, error, skip, not_exercised, excluded.
#   raw_status is what the test did (pass/fail/error/skip); status is what the
#   ledger counts. status differs from raw_status only for
#     excluded       the test is in tools/conformance-exclusions.tsv
#     not_exercised  raw pass, but the test built a table with a shimmed
#                    constructor and no tbl_gpu was ever created or uploaded
#                    (the fixture silently fell back to a plain tibble, so a
#                    "pass" says nothing about cuplyr)
#
# ---------------------------------------------------------------------------
# HOW THE SHIM WORKS (and why it changed in F01)
#
# dplyr's own tests build fixtures with bare tibble()/data.frame()/tribble()
# and never mention a backend. To run them against tbl_gpu the harness evals
# each test file in a clone of dplyr's test environment in which those three
# constructors are masked (so dplyr's internals, which call tibble::tibble()
# from their own namespace, are untouched). The old shim uploaded EVERY value
# those masks returned, so tables built inside helper closures, nested column
# constructors (tibble(z = data.frame(...))), verb dot expressions and the
# expected values of expectations all became tbl_gpu. That tested the shim
# rather than cuplyr (C1), and round-tripped expected values through the GPU,
# which hides upload bugs.
#
# Now the masks return plain objects and only the CONSTRUCTION SITES that feed
# cuplyr are wrapped in an upload helper, by rewriting the test file's AST
# before testthat sees it:
#   U1  right-hand side of an assignment to a name at statement level (the
#       top level of the file, or any statement of a `{` block, which includes
#       test_that() bodies and nested blocks);
#   U2  the table argument of a verb at any depth outside function literals:
#       first argument / pipe head of every verb, `y` of two-table verbs
#       (joins, set operations, rows_*), every data argument of bind_rows()/
#       bind_cols();
#   everything else stays plain: function bodies and lambdas, nested column
#   constructors, verb dot expressions, arguments of non-verb helpers,
#   expected values inside expectations, quote()d code and formulas.
# Why an AST rewrite instead of a call-frame check: a masked tibble() cannot
# tell `x <- tibble()` from `list(tibble())` or `helper(tibble())` by looking
# at its frames (arguments are promises evaluated in the caller's frame), and
# it cannot see file-level code at all. The rewritten file is deparsed into a
# temporary copy of the tests directory (helpers and _snaps are copied
# alongside), so nothing is ever written into the vendored tree and testthat
# still runs the file unchanged otherwise. Wrapping is the ONLY change; the
# identity of everything else is checked by test-parity-F01-rewrite.R
# (parse(deparse(x)) round-trips for every vendored file).
#
# Comparisons: expect_equal()/expect_identical()/expect_s3_class() collect a
# tbl_gpu with its grouping restored (so grouped vs ungrouped now differs, as in
# dplyr), recurse into plain lists and list-columns (C7), and, only when
# exactly one side came from a tbl_gpu and the other is a bare data.frame,
# compare the collected side as a bare data.frame (the harness turned that
# data.frame fixture into a tbl_gpu; the tibble class is its artifact). Data,
# names, types, row count and grouping are always compared.
# expect_s3_class(<tbl_gpu result>, "data.frame", exact = TRUE) follows the same
# rule (review N6): it is a HARNESS RELAXATION, since cuplyr always collects a
# tibble, and it turns e.g. arrange's "preserves input class" into a pass.
#
# class() in the shim env reports a tbl_gpu's collected class (tibble, plus
# grouped_df when grouped), or the class of the fixture for the uploaded object
# itself, so `expect_s3_class(out, class(df), exact = TRUE)` keeps its meaning.
# A bare data.frame fixture still reports "data.frame" and cuplyr hands back a
# tibble, so such tests fail honestly instead of being waved through.
#
# Snapshots (C4): the run is pinned to CRAN-mode snapshot semantics
# (NOT_CRAN=false) and testthat edition 3, whatever the caller's environment
# (restored afterwards). expect_snapshot() code still runs and the output is
# never compared, as on CRAN. An `error = TRUE` snapshot is credited (a
# success, "relaxed") only when, statement by statement, cuplyr errors exactly
# where dplyr's reference in _snaps errors (a statement the reference shows
# succeeding must not error, one it shows erroring must), and no credited error
# is an infrastructure or missing-feature error. A credited message is that
# only when cf_infra_error_pattern matches it (libcudf failures, "must be a
# vector, not a <tbl_gpu>", missing methods/functions, R-level programming
# errors, cuplyr's own unsupported-expression errors, rlang machinery errors
# cuplyr triggers by not validating as dplyr does) AND the matched text is
# absent from the reference error lines of the SAME statement: "unused argument
# (1)" is dplyr's own error in join_by(closest(a >= b, 1)) and is credited
# there. Messages are taken with their parent chain (rlang::cnd_message()).
# A reference block that cannot be found or parsed credits nothing: the
# snapshot records a skip expectation without unwinding the test (later
# assertions still run; the test is `skip` only if nothing else in it passed
# or failed) and is counted per test in snapshots_uncredited. No error at all,
# a disagreement on a statement or an infrastructure error is a failure. Every
# credited snapshot is recorded per test (snapshots_relaxed) with ALL of
# cuplyr's messages next to dplyr's reference error lines, plus a flag when
# the final "!" line (innermost error) differs from the reference's, so the
# report can be audited. Wording is never compared.

# Instrumentation (C5): per test the harness counts successful uploads, failed
# uploads, shim constructor calls and tbl_gpu objects created (cuplyr:::
# new_tbl_gpu is wrapped for the duration of a file and restored on exit).
# exercised = a tbl_gpu was created or uploaded during the test.
#
# Exclusions (C2/C3/C6/C8...): tools/conformance-exclusions.tsv lists tests
# that cannot measure cuplyr, each with a C-item and a one-line reason.
# Excluded tests still run; their status is `excluded` and raw_status keeps the
# real outcome. A genuine cuplyr gap is never excluded.

cf_or <- function(x, y) if (is.null(x)) y else x
`%||%` <- cf_or

# ---- Exclusion manifest ------------------------------------------------------

conformance_items <- paste0("C", 1:8)

conformance_load_exclusions <- function(path) {
  if (!file.exists(path)) {
    stop("exclusion manifest does not exist: ", path, call. = FALSE)
  }
  m <- utils::read.delim(path, quote = "", comment.char = "", colClasses = "character",
    check.names = FALSE, stringsAsFactors = FALSE, na.strings = character(0))
  required <- c("file", "test", "item", "reason")
  missing_cols <- setdiff(required, names(m))
  if (length(missing_cols) > 0) {
    stop("exclusion manifest ", path, " lacks column(s): ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }
  m <- m[, required, drop = FALSE]
  if (any(!nzchar(trimws(m$reason)))) {
    stop("exclusion manifest ", path, ": every row needs a non-empty reason (row(s) ",
      paste(which(!nzchar(trimws(m$reason))), collapse = ", "), ")", call. = FALSE)
  }
  if (any(!nzchar(m$file)) || any(!nzchar(m$test))) {
    stop("exclusion manifest ", path, ": file and test must be non-empty", call. = FALSE)
  }
  bad_item <- !(m$item %in% conformance_items)
  if (any(bad_item)) {
    stop("exclusion manifest ", path, ": item must be one of C1..C8, got ",
      paste(unique(m$item[bad_item]), collapse = ", "), call. = FALSE)
  }
  key <- paste(m$file, m$test, sep = "\t")
  if (anyDuplicated(key)) {
    stop("exclusion manifest ", path, ": duplicate (file, test): ",
      paste(m$file[anyDuplicated(key)], m$test[anyDuplicated(key)], sep = " :: "), call. = FALSE)
  }
  rownames(m) <- NULL
  m
}

# ---- Source rewriting (C1 construction sites) --------------------------------

cf_ctor_names <- c("tibble", "data.frame", "tribble")

# Verbs whose first argument is the table.
cf_verbs_one <- c(
  "filter", "mutate", "transmute", "select", "rename", "rename_with", "relocate",
  "arrange", "distinct", "summarise", "summarize", "reframe", "group_by", "ungroup",
  "rowwise", "count", "tally", "add_count", "add_tally", "slice", "slice_head",
  "slice_tail", "slice_min", "slice_max", "slice_sample", "pull", "collect", "compute",
  "nest_by", "group_split", "group_nest", "group_map", "group_modify", "group_walk",
  "group_trim", "group_keys", "group_data", "group_vars", "group_indices", "group_rows",
  "groups", "n_groups", "group_size", "mutate_all", "mutate_if", "mutate_at",
  "transmute_all", "transmute_if", "transmute_at", "summarise_all", "summarise_if",
  "summarise_at", "summarize_all", "summarize_if", "summarize_at", "filter_all",
  "filter_if", "filter_at", "arrange_all", "arrange_if", "arrange_at", "select_all",
  "select_if", "select_at", "rename_all", "rename_if", "rename_at", "group_by_all",
  "group_by_if", "group_by_at", "distinct_all", "distinct_if", "distinct_at", "top_n",
  "top_frac", "sample_n", "sample_frac", "do", "glimpse", "with_groups", "by_slice",
  "filter_out", "dplyr_row_slice", "dplyr_col_modify", "dplyr_reconstruct"
)
# Verbs whose first two arguments are tables (x, y).
cf_verbs_two <- c(
  "inner_join", "left_join", "right_join", "full_join", "semi_join", "anti_join",
  "nest_join", "cross_join", "union", "union_all", "intersect", "setdiff", "symdiff",
  "setequal", "rows_insert", "rows_update", "rows_patch", "rows_upsert", "rows_delete",
  "rows_append"
)
cf_verbs_bind <- c("bind_rows", "bind_cols")
cf_bind_options <- c(".id", ".name_repair", ".ptype", ".size")

cf_call_name <- function(e) {
  h <- e[[1]]
  if (is.name(h)) {
    return(as.character(h))
  }
  if (is.call(h) && length(h) == 3L && is.name(h[[3]]) &&
    (identical(h[[1]], as.name("::")) || identical(h[[1]], as.name(":::")))) {
    pkg <- as.character(h[[2]])
    if (pkg %in% c("dplyr", "cuplyr")) {
      return(as.character(h[[3]]))
    }
  }
  NA_character_
}

cf_missing_at <- function(e, i) identical(e[[i]], quote(expr = ))

# A bare (unqualified) shimmed constructor call.
cf_is_ctor <- function(x) {
  is.call(x) && is.name(x[[1]]) && as.character(x[[1]]) %in% cf_ctor_names
}

cf_is_verb_call <- function(x) {
  if (!is.call(x)) {
    return(FALSE)
  }
  nm <- cf_call_name(x)
  !is.na(nm) && nm %in% c(cf_verbs_one, cf_verbs_two, cf_verbs_bind)
}

# Indices (into the call) of the arguments of a verb call that are tables.
cf_table_args <- function(e, nm) {
  n <- length(e)
  if (n < 2L) {
    return(integer())
  }
  args <- 2:n
  an <- names(e)
  an <- if (is.null(an)) rep("", n) else an
  an <- an[args]
  if (nm %in% cf_verbs_bind) {
    return(args[!(an %in% cf_bind_options)])
  }
  # A named argument is a table only when its name is the verb's own formal
  # (`.data`, `.tbl` or `x` for the first, `y` for the second): a dot argument
  # that happens to be called `x` (summarise(df, x = x + 1)) is not the table.
  f1 <- cf_verb_formal(nm, 1L)
  f2 <- cf_verb_formal(nm, 2L)
  unnamed <- args[an == ""]
  named_x <- if (is.na(f1)) integer() else args[an == f1]
  t1 <- if (length(named_x)) named_x[[1]] else if (length(unnamed)) unnamed[[1]] else NA_integer_
  if (nm %in% cf_verbs_one) {
    return(if (is.na(t1)) integer() else t1)
  }
  rest <- setdiff(unnamed, t1)
  named_y <- if (is.na(f2)) integer() else args[an == f2]
  t2 <- if (length(named_y)) named_y[[1]] else if (length(rest)) rest[[1]] else NA_integer_
  c(t1, t2)[!is.na(c(t1, t2))]
}

# The k-th formal of dplyr's verb (NA when dplyr has no such export).
cf_verb_formal <- function(nm, k) {
  f <- tryCatch(names(formals(getExportedValue("dplyr", nm))), error = function(e) character())
  if (length(f) >= k) f[[k]] else NA_character_
}

# The helper is reached through an option, not a name in the shim env, because
# some tests evaluate fixtures in the global environment (eval_bare(globalenv())).
cf_upload_option <- "cuplyr.conformance.upload"

cf_wrap_upload <- function(x) {
  getter <- call("::", as.name("base"), as.name("getOption"))
  as.call(list(as.call(list(getter, cf_upload_option)), x))
}

cf_is_assignment <- function(x) {
  is.call(x) && is.name(x[[1]]) && as.character(x[[1]]) %in% c("<-", "=", "<<-")
}

# Rewrite one expression. `stmt` is TRUE when `e` sits at statement level
# (top of the file, or a statement of a `{` block).
cf_rewrite <- function(e, stmt = FALSE) {
  if (!is.call(e)) {
    return(e)
  }
  hn <- if (is.name(e[[1]])) as.character(e[[1]]) else ""
  # Function literals, quoted code and formulas are never rewritten.
  if (hn %in% c("function", "quote", "bquote", "expression", "~")) {
    return(e)
  }
  n <- length(e)
  nm <- cf_call_name(e)
  stmt_child <- rep(FALSE, n)
  wrap <- integer()
  skip <- integer()

  if (hn == "{") {
    stmt_child[-1] <- TRUE
  } else if (hn == "if") {
    stmt_child[-(1:2)] <- stmt
  } else if (hn %in% c("for", "while")) {
    stmt_child[n] <- stmt
  } else if (hn == "repeat" || hn == "(") {
    stmt_child[n] <- stmt
  } else if (hn == "test_that") {
    stmt_child[n] <- TRUE
  } else if (cf_is_assignment(e) && stmt && n == 3L) {
    lhs <- e[[2]]
    if ((is.name(lhs) || is.character(lhs)) && !cf_missing_at(e, 3L)) {
      if (cf_is_ctor(e[[3]])) {
        wrap <- 3L
      } else if (cf_is_assignment(e[[3]])) {
        stmt_child[3] <- TRUE
      }
    }
  } else if (hn %in% c("%>%", "%T>%") && n == 3L) {
    # lhs %>% verb(...): the pipe head is the verb's table. The rhs call gets a
    # placeholder first argument so the verb rules below pick its other table
    # arguments (y of a join, bind_rows() arguments) and skip its dots.
    if (!cf_missing_at(e, 2L) && !cf_missing_at(e, 3L) && cf_is_verb_call(e[[3]])) {
      if (cf_is_ctor(e[[2]])) wrap <- 2L
      rhs <- e[[3]]
      fake <- as.call(c(list(rhs[[1]], quote(.)), as.list(rhs)[-1]))
      e[[3]] <- cf_rewrite(fake, FALSE)[-2]
      skip <- 3L
    }
  } else if (!is.na(nm) && nm %in% c(cf_verbs_one, cf_verbs_two, cf_verbs_bind)) {
    tbl <- cf_table_args(e, nm)
    # Everything but the table arguments is a dot expression: left untouched.
    skip <- setdiff(2:n, tbl)
    for (i in tbl) {
      if (!cf_missing_at(e, i) && cf_is_ctor(e[[i]])) wrap <- c(wrap, i)
    }
  }

  for (i in seq_len(n)) {
    if (i %in% skip || cf_missing_at(e, i)) next
    child <- e[[i]]
    if (!is.call(child)) next
    new <- cf_rewrite(child, stmt_child[i])
    if (i %in% wrap) new <- cf_wrap_upload(new)
    e[[i]] <- new
  }
  e
}

# deparse() prints a call whose head is an operator symbol but whose arity
# does not match (join_by(`>`(x))) as unparseable text. For such expressions
# only, every operator head is aliased to a plain name and restored after.
cf_alias_ops <- function(e, table) {
  if (!is.call(e)) {
    return(e)
  }
  if (is.name(e[[1]])) {
    h <- as.character(e[[1]])
    if (grepl("^([-+*/^<>=!&|:$@~?]+|%.*%)$", h)) {
      alias <- paste0(".cf_op_", length(table$ops) + 1L)
      table$ops[[alias]] <- h
      e[[1]] <- as.name(alias)
    }
  }
  if (identical(e[[1]], as.name("function"))) {
    return(e)
  }
  for (i in seq_along(e)[-1]) {
    if (!cf_missing_at(e, i) && is.call(e[[i]])) e[[i]] <- cf_alias_ops(e[[i]], table)
  }
  e
}

cf_deparse_one <- function(x) {
  ctl <- c("keepNA", "keepInteger", "niceNames", "showAttributes", "digits17")
  txt <- paste(deparse(x, width.cutoff = 500L, control = ctl), collapse = "\n")
  if (!inherits(try(parse(text = txt, keep.source = FALSE), silent = TRUE), "try-error")) {
    return(txt)
  }
  table <- new.env()
  table$ops <- list()
  txt <- paste(deparse(cf_alias_ops(x, table), width.cutoff = 500L, control = ctl), collapse = "\n")
  for (alias in rev(names(table$ops))) {
    txt <- gsub(alias, paste0("`", table$ops[[alias]], "`"), txt, fixed = TRUE)
  }
  txt
}

cf_deparse_exprs <- function(exprs) vapply(exprs, cf_deparse_one, character(1))

# Rewrite the source lines of a test file. With wrap = FALSE the code is only
# round-tripped (used to check that deparse changes nothing).
conformance_rewrite_source <- function(lines, wrap = TRUE) {
  exprs <- parse(text = lines, keep.source = FALSE, encoding = "UTF-8")
  out <- lapply(as.list(exprs), function(e) if (wrap) cf_rewrite(e, TRUE) else e)
  paste(cf_deparse_exprs(out), collapse = "\n\n")
}

# Copy the test file's directory (helpers, _snaps, data files) to a temp dir
# and write the rewritten test file into it. The vendored tree is never written.
cf_stage_file <- function(path) {
  if (!file.exists(path)) {
    stop("test file does not exist: ", path, call. = FALSE)
  }
  src_dir <- dirname(path)
  fname <- basename(path)
  dir <- tempfile("conformance-")
  dir.create(dir)
  entries <- list.files(src_dir, all.files = TRUE, no.. = TRUE)
  entries <- entries[!(grepl("^test-.*\\.[Rr]$", entries) & entries != fname)]
  entries <- setdiff(entries, fname)
  if (length(entries) > 0) {
    file.copy(file.path(src_dir, entries), dir, recursive = TRUE)
  }
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  rewritten <- conformance_rewrite_source(lines)
  new_path <- file.path(dir, fname)
  con <- file(new_path, open = "wb")
  on.exit(close(con), add = TRUE)
  writeLines(enc2utf8(rewritten), con, useBytes = TRUE)
  list(dir = dir, path = new_path)
}

# ---- Normalization of comparison operands ------------------------------------

# Collect a tbl_gpu with its grouping restored. collect() returns an ungrouped
# tibble by design, so grouping is re-applied from the tbl_gpu's metadata.
cf_collect <- function(x) {
  out <- tryCatch(tibble::as_tibble(dplyr::collect(x)), error = function(e) NULL)
  if (is.null(out)) {
    return(x) # collect() itself failed: let the real comparison surface it
  }
  groups <- .subset2(x, "groups")
  if (length(groups) > 0) {
    drop <- .subset2(x, "group_drop") %||% TRUE
    grouped <- tryCatch(dplyr::group_by(out, !!!rlang::syms(groups), .drop = drop), error = function(e) NULL)
    if (!is.null(grouped)) out <- grouped
  }
  out
}

cf_contains_gpu <- function(x, depth = 0L) {
  if (inherits(x, "tbl_gpu")) {
    return(TRUE)
  }
  if (depth > 8L || !is.list(x)) {
    return(FALSE)
  }
  if (is.data.frame(x)) {
    for (col in unclass(x)) {
      if (is.list(col) && !is.data.frame(col) && cf_contains_gpu(col, depth + 1L)) return(TRUE)
    }
    return(FALSE)
  }
  for (el in unclass(x)) {
    if (cf_contains_gpu(el, depth + 1L)) return(TRUE)
  }
  FALSE
}

# tbl_gpu -> collected (grouped) tibble; recurse into plain lists and into
# list-columns of data frames (C7). Everything else is returned untouched.
cf_normalize <- function(x) {
  if (inherits(x, "tbl_gpu")) {
    return(cf_collect(x))
  }
  if (!cf_contains_gpu(x)) {
    return(x)
  }
  if (is.data.frame(x)) {
    for (nm in names(x)) {
      col <- .subset2(x, nm)
      if (is.list(col) && !is.data.frame(col) && cf_contains_gpu(col)) {
        x[[nm]] <- cf_normalize(col)
      }
    }
    return(x)
  }
  cls <- attr(x, "class", exact = TRUE)
  out <- unclass(x)
  for (i in seq_along(out)) {
    if (cf_contains_gpu(out[[i]])) out[[i]] <- cf_normalize(out[[i]])
  }
  attr(out, "class") <- cls
  out
}

cf_is_bare_df <- function(x) identical(class(x), "data.frame")

# The collected tibble of a data.frame fixture, as the bare data.frame it came from.
cf_as_bare <- function(x) {
  if (identical(class(x), c("tbl_df", "tbl", "data.frame"))) as.data.frame(x) else x
}

# Normalize both operands of a comparison. The only relaxation: when exactly
# one side is a tbl_gpu and the other is a bare data.frame, the collected side
# is compared as a bare data.frame (the harness uploaded a data.frame fixture;
# the tibble class of the collected result is the harness's doing).
cf_normalize_pair <- function(object, expected) {
  object_gpu <- inherits(object, "tbl_gpu")
  expected_gpu <- inherits(expected, "tbl_gpu")
  o <- cf_normalize(object)
  e <- cf_normalize(expected)
  if (object_gpu && !expected_gpu && cf_is_bare_df(e)) o <- cf_as_bare(o)
  if (expected_gpu && !object_gpu && cf_is_bare_df(o)) e <- cf_as_bare(e)
  list(object = o, expected = e)
}

# ---- Per-file state, instrumentation -----------------------------------------

cf_reset_counters <- function(state) {
  state$uploads <- 0L
  state$upload_failures <- 0L
  state$constructor_calls <- 0L
  state$tbl_gpu_created <- 0L
  state$relaxed <- list()
  state$uncredited <- 0L # error snapshots with no usable reference block
  state$snap_index <- 0L # expect_snapshot() calls so far in this test
  invisible(state)
}

cf_new_state <- function() {
  state <- cf_reset_counters(new.env(parent = emptyenv()))
  state$test_name <- NA_character_
  state$snap_file <- NA_character_
  # Uploaded fixtures of this file, list(fields, class): see .cf_upload() in
  # cf_make_shim_env(). Holding the fields keeps each fixture's GPU pointer
  # alive, so its address cannot be reused by a later table.
  state$fixtures <- list()
  state
}

# A tbl_gpu's fields without any attributes (other than their names): two
# objects with identical fields are the same GPU table with the same metadata.
cf_fixture_fields <- function(x) {
  f <- unclass(x)
  attributes(f) <- list(names = names(f))
  f
}

# The class of the fixture `x` was uploaded from, or NULL when `x` is not (a
# copy of) an uploaded fixture.
cf_fixture_class <- function(state, x) {
  fields <- cf_fixture_fields(x)
  for (entry in rev(state$fixtures)) {
    if (identical(entry$fields, fields)) return(entry$class)
  }
  NULL
}

cf_read_counters <- function(state) {
  list(
    uploads = state$uploads, upload_failures = state$upload_failures,
    constructor_calls = state$constructor_calls, tbl_gpu_created = state$tbl_gpu_created,
    relaxed = state$relaxed, uncredited = state$uncredited
  )
}

# Wrap cuplyr:::new_tbl_gpu to count tbl_gpu creations. Returns a function that
# restores the original binding; the caller must run it on exit.
cf_instrument_new_tbl_gpu <- function(state) {
  ns <- asNamespace("cuplyr")
  orig <- get("new_tbl_gpu", envir = ns)
  counting <- function(...) {
    state$tbl_gpu_created <- state$tbl_gpu_created + 1L
    orig(...)
  }
  set_binding <- function(value) {
    locked <- bindingIsLocked("new_tbl_gpu", ns)
    if (locked) unlockBinding("new_tbl_gpu", ns)
    assign("new_tbl_gpu", value, envir = ns)
    if (locked) lockBinding("new_tbl_gpu", ns)
  }
  set_binding(counting)
  function() set_binding(orig)
}

# A ListReporter that also snapshots the per-test counters.
cf_make_reporter <- function(state) {
  gen <- R6::R6Class("ConformanceReporter",
    inherit = testthat::ListReporter,
    public = list(
      per_test = NULL,
      start_test = function(context, test) {
        cf_reset_counters(state)
        state$test_name <- as.character(test)
        super$start_test(context, test)
      },
      end_test = function(context, test) {
        super$end_test(context, test)
        self$per_test[[length(self$per_test) + 1L]] <- cf_read_counters(state)
      }
    )
  )
  gen$new()
}

# ---- Shim environment --------------------------------------------------------

cf_make_shim_env <- function(state, upload = TRUE) {
  # Same base as dplyr's own test_check("dplyr"): a clone of dplyr's namespace,
  # so unqualified dplyr-internal helpers (join_cols(), count_regroups(), ...)
  # and the imported rlang/vctrs functions resolve exactly as in a real run.
  env <- testthat::test_env("dplyr")

  # Re-shadow every cuplyr-exported name over the cloned dplyr binding (cuplyr
  # masks dplyr's non-generic bind_rows()/bind_cols() on the search path in
  # real use; the clone would otherwise bypass that).
  for (nm in getNamespaceExports("cuplyr")) {
    env[[nm]] <- get(nm, envir = asNamespace("cuplyr"))
  }

  # Constructors stay plain; construction sites chosen by conformance_rewrite_
  # source() are uploaded through .cf_upload().
  count_ctor <- function() state$constructor_calls <- state$constructor_calls + 1L
  env$data.frame <- function(...) {
    count_ctor()
    base::data.frame(...)
  }
  env$tibble <- function(...) {
    count_ctor()
    tibble::tibble(...)
  }
  env$tribble <- function(...) {
    count_ctor()
    tibble::tribble(...)
  }
  do_upload <- function(x) {
    if (!isTRUE(upload) || !is.data.frame(x)) {
      return(x)
    }
    res <- tryCatch(cuplyr::tbl_gpu(x), error = function(e) NULL, warning = function(w) NULL)
    if (is.null(res)) {
      state$upload_failures <- state$upload_failures + 1L
      return(x)
    }
    state$uploads <- state$uploads + 1L
    # class(df) in a test means "the class of this fixture", see `class` below.
    # Recorded in the state, not as attributes of `res`: a tbl_gpu's extra R
    # attributes are its data-frame attributes (F31), which verbs carry and
    # collect() returns, so harness bookkeeping stored there would show up in
    # results and attributes() of derived tables.
    state$fixtures[[length(state$fixtures) + 1L]] <- list(fields = cf_fixture_fields(res), class = class(x))
    res
  }

  # class() of a tbl_gpu reports the class its collected form has (derived from
  # the collected object), or the class of the fixture it was uploaded from (tests write expect_s3_class(out,
  # class(df), exact = TRUE) and class(df) would otherwise be c("tbl_gpu",
  # "tbl")). A data.frame fixture therefore still reports "data.frame", and
  # since cuplyr collects a tibble the test fails, as it should.
  env$class <- function(x) {
    if (inherits(x, "tbl_gpu")) {
      fixture <- cf_fixture_class(state, x)
      is_fixture <- !is.null(fixture) &&
        length(.subset2(x, "groups")) == 0L && is.null(.subset2(x, "lazy_ops"))
      if (is_fixture) {
        return(fixture)
      }
      return(base::class(cf_collect(x)))
    }
    base::class(x)
  }

  # Harness-only column accessors (dplyr tests extract columns with `$`/`[[`
  # in plain test-body code). NOT part of the cuplyr package.
  env$`$.tbl_gpu` <- function(x, name) {
    tibble::as_tibble(dplyr::collect(x))[[name]]
  }
  env$`[[.tbl_gpu` <- function(x, i, ...) {
    tibble::as_tibble(dplyr::collect(x))[[i, ...]]
  }

  env$expect_s3_class <- function(object, class, ..., exact = FALSE) {
    # The one thing checked on the raw object: that it is a tbl_gpu.
    o <- if (inherits(object, "tbl_gpu") && all(class == "tbl_gpu")) object else cf_normalize(object)
    # Same rule as expect_equal(): a result compared against exactly a bare
    # data.frame class is compared as the bare data.frame (the tibble class of
    # a collected result is a harness artifact, the fixture was a data.frame).
    if (inherits(object, "tbl_gpu") && isTRUE(exact) && identical(class, "data.frame")) o <- cf_as_bare(o)
    testthat::expect_s3_class(o, class, ..., exact = exact)
  }
  cf_label <- function(e) paste(deparse(e, width.cutoff = 60L)[1], collapse = "")
  env$expect_equal <- function(object, expected, ..., label = NULL, expected.label = NULL) {
    label <- label %||% cf_label(substitute(object))
    expected.label <- expected.label %||% cf_label(substitute(expected))
    p <- cf_normalize_pair(object, expected)
    testthat::expect_equal(p$object, p$expected, ..., label = label, expected.label = expected.label)
  }
  env$expect_identical <- function(object, expected, ..., label = NULL, expected.label = NULL) {
    label <- label %||% cf_label(substitute(object))
    expected.label <- expected.label %||% cf_label(substitute(expected))
    p <- cf_normalize_pair(object, expected)
    testthat::expect_identical(p$object, p$expected, ..., label = label, expected.label = expected.label)
  }

  env$expect_snapshot <- function(x, cran = FALSE, error = FALSE, transform = NULL,
                                  variant = NULL, cnd_class = FALSE) {
    expr <- substitute(x)
    caller <- parent.frame()
    state$snap_index <- idx <- state$snap_index + 1L
    if (!isTRUE(error)) {
      # CRAN-mode behavior: the code runs, an unexpected error is an error,
      # output is not compared (a skip if nothing else is asserted).
      call <- as.call(list(quote(testthat::expect_snapshot), expr, cran = cran, error = error,
        transform = transform, variant = variant, cnd_class = cnd_class))
      return(eval(call, caller))
    }
    stmts <- if (is.call(expr) && identical(expr[[1]], as.name("{"))) as.list(expr)[-1] else list(expr)
    got <- rep(FALSE, length(stmts))
    msgs <- rep(NA_character_, length(stmts))
    for (i in seq_along(stmts)) {
      tryCatch(
        withCallingHandlers(
          utils::capture.output(eval(stmts[[i]], caller)),
          warning = function(w) invokeRestart("muffleWarning"),
          message = function(m) invokeRestart("muffleMessage")
        ),
        error = function(e) {
          got[[i]] <<- TRUE
          msgs[[i]] <<- cf_cnd_text(e)
        }
      )
    }
    if (!any(got)) {
      return(testthat::fail("`code` did not throw an error (snapshot with `error = TRUE`)."))
    }
    # dplyr's own reference decides which statements must error. Without a
    # usable reference nothing is credited (testthat's CRAN mode skips too).
    # The snapshot records a skip expectation and the test goes on: skip()
    # would unwind it and drop every later assertion. The test ends up `skip`
    # only when nothing else in it passed or failed (cf_test_outcome()).
    ref <- cf_snapshot_ref_block(state$snap_file, state$test_name, idx)
    if (is.null(ref) || length(ref$errors) != length(stmts)) {
      state$uncredited <- state$uncredited + 1L
      return(testthat::exp_signal(testthat::expectation("skip", paste0("`error = TRUE` snapshot not credited: ",
        if (is.null(ref)) "no parsable reference block in _snaps for this test"
        else sprintf("reference has %d statement(s), the snapshot %d", length(ref$errors), length(stmts))))))
    }
    wrong <- which(got != ref$errors)
    if (length(wrong) > 0L) {
      w <- wrong[[1]]
      what <- if (got[[w]]) {
        paste0("raised an error where dplyr's reference shows none: ", cf_sanitize(msgs[[w]], 160L))
      } else {
        "did not raise the error dplyr's reference shows"
      }
      return(testthat::fail(sprintf("statement %d of %d (`%s`) %s (%d statement(s) disagree with the reference).",
        w, length(stmts), cf_label(stmts[[w]]), what, length(wrong))))
    }
    # An infrastructure-looking error is still dplyr's own when the reference
    # error of the same statement contains the matched text.
    for (i in which(got)) {
      hits <- cf_infra_hits(msgs[[i]], ref$stmt_lines[[i]])
      if (length(hits) > 0L) {
        return(testthat::fail(paste0("`error = TRUE` snapshot errored with an infrastructure or missing-feature error, not an error cuplyr raises on purpose: ",
          cf_sanitize(msgs[[i]], 200L))))
      }
    }
    differs <- any(vapply(which(got), function(i) {
      !identical(cf_final_bang(msgs[[i]], cuplyr = TRUE), cf_final_bang(ref$stmt_lines[[i]]))
    }, NA))
    state$relaxed[[length(state$relaxed) + 1L]] <- list(cuplyr = msgs[got], reference = ref$error_lines,
      final_differs = differs)
    testthat::succeed()
  }

  list(env = env, upload = do_upload)
}

# ---- Result handling ---------------------------------------------------------

cf_sanitize <- function(x, width = 240L) {
  x <- paste(x, collapse = " ")
  x <- iconv(x, "UTF-8", "UTF-8", sub = "?")
  x <- cli::ansi_strip(x)
  x <- gsub("[\t\r\n]+", " ", x)
  x <- trimws(gsub(" {2,}", " ", x))
  if (nchar(x) > width) x <- paste0(substr(x, 1L, width - 3L), "...")
  x
}

cf_test_outcome <- function(t) {
  cls <- vapply(t$results, function(x) class(x)[1], character(1))
  bad <- function(type) Filter(function(x) inherits(x, type), t$results)
  if (any(cls == "expectation_error")) {
    list(status = "error", message = conditionMessage(bad("expectation_error")[[1]]))
  } else if (any(cls == "expectation_failure")) {
    list(status = "fail", message = conditionMessage(bad("expectation_failure")[[1]]))
  } else if (length(cls) > 0 && all(cls == "expectation_skip")) {
    list(status = "skip", message = NA_character_)
  } else {
    list(status = "pass", message = NA_character_)
  }
}

# Errors that mean "cuplyr could not do this" rather than "cuplyr raised the
# error dplyr raises". A relaxed `error = TRUE` snapshot whose credited errors
# match is a failure, however closely the reference agrees on where it errors,
# unless the matched text also appears in the reference error of the same
# statement (then dplyr raises that error too: cf_infra_hits()).
# One place, one regex (perl, case-sensitive unless marked):
#   - libcudf / GPU runtime failures surfaced by the Rcpp layer
#   - a tbl_gpu reaching a vctrs/dplyr generic cuplyr has no method for
#   - a missing function or S3 method
#   - cuplyr's own unsupported-feature / unsupported-expression / own-wording
#     validation errors (R/*.R), which are not the error dplyr raises
#   - R-level programming errors (unused argument, non-function call, ...)
cf_infra_error_pattern <- paste(c(
  "GPU operation '[^']*' failed",
  "CUDF failure|CUDA error|std::bad_alloc|cudaError",
  "must be a vector, not a <tbl_gpu",
  "no applicable method for .* applied to an object of class \"c\\('tbl_gpu'",
  "could not find function",
  "arrange\\(\\) expressions must be column names",
  "filter\\(\\) only supports comparisons",
  "is not supported in filter\\(\\)",
  "(?i:unsupported|not (yet )?supported|does not support|only supports|not implemented)",
  "Unknown (IR op|AST node|slice mode|join type)",
  "Invalid (aggregation expression|lazy_ops|column index)",
  "^internal:",
  "Undefined columns selected",
  "across\\(\\)'s `\\.fns` must be",
  "unused argument|attempt to apply non-function|subscript out of bounds|non-numeric argument to",
  # an error raised by rlang/dplyr machinery because cuplyr did not get as far
  # as dplyr does (pick() evaluated outside a mask, an unvalidated `name`)
  "Must only be used inside data-masking verbs",
  "The LHS of `:=` must be"
), collapse = "|")

# The infrastructure pattern's matches in `msg` that do not occur in
# `ref_lines` (dplyr's reference error lines for the same statement). Line
# wrapping and runs of whitespace are ignored on both sides.
cf_infra_hits <- function(msg, ref_lines) {
  hits <- unique(cf_squash(regmatches(msg, gregexpr(cf_infra_error_pattern, msg, perl = TRUE))[[1]]))
  ref <- cf_squash(paste(ref_lines, collapse = " "))
  hits[!vapply(hits, function(h) grepl(h, ref, fixed = TRUE), NA)]
}

cf_squash <- function(x) trimws(gsub("\\s+", " ", cli::ansi_strip(x)))

# Full text of a condition, parents included ("Caused by error: ! ..."), so
# an infrastructure error wrapped by a verb's error is still seen.
cf_cnd_text <- function(e) {
  tryCatch(cli::ansi_strip(rlang::cnd_message(e)), error = function(err) conditionMessage(e))
}

# The final "! " line of an error (the innermost cause), wrapped continuation
# lines joined. For a cuplyr message whose top-level header carries no "! "
# prefix, the first line stands in when there is no "! " line. Used only to
# flag credited snapshots for audit (the report's "final line differs").
cf_final_bang <- function(lines, cuplyr = FALSE) {
  lines <- unlist(strsplit(cli::ansi_strip(paste(lines, collapse = "\n")), "\n", fixed = TRUE))
  bang <- which(startsWith(lines, "! "))
  if (length(bang) == 0L && !(cuplyr && length(lines))) {
    return(NA_character_)
  }
  from <- if (length(bang)) bang[[length(bang)]] else 1L
  to <- from
  # A new bullet (ASCII as in _snaps, or cli's Unicode bullets) ends the line.
  bullet <- "^([!ix*>v\u2139\u2716\u2714\u2022\u2192] |Caused by|Error\\b|Warning\\b)"
  while (to < length(lines) && !grepl(bullet, lines[[to + 1L]])) {
    to <- to + 1L
  }
  cf_squash(sub("^! ", "", paste(lines[from:to], collapse = " ")))
}

cf_snap_path <- function(path) {
  file.path(dirname(path), "_snaps", paste0(sub("^test-", "", sub("\\.[Rr]$", "", basename(path))), ".md"))
}

# dplyr's reference blocks for one test, from the vendored _snaps file: a list
# with one element per `---`-separated block (one per expect_snapshot() call),
# each list(errors = logical per statement, error_lines = character (all of
# the block's), stmt_lines = list of character, the error lines of each
# statement). A
# statement errors in the reference when the Condition section that follows
# its Code section has an `Error...:` header; consecutive statements without
# output share one Code section, so only the last of a group can be the one
# that errored. NULL when the file or test is missing; a block that does not
# parse is NULL as well (the caller then credits nothing).
cf_snapshot_ref_blocks <- function(snap_file, test) {
  if (is.na(snap_file) || is.na(test) || !file.exists(snap_file)) {
    return(NULL)
  }
  lines <- readLines(snap_file, warn = FALSE, encoding = "UTF-8")
  start <- which(lines == paste0("# ", test))
  if (length(start) == 0L) {
    return(NULL)
  }
  rest <- lines[-seq_len(start[[1]])]
  nxt <- which(grepl("^# ", rest))
  if (length(nxt)) rest <- rest[seq_len(nxt[[1]] - 1L)]
  sep <- c(0L, which(rest == "---"), length(rest) + 1L)
  lapply(seq_len(length(sep) - 1L), function(i) {
    from <- sep[[i]] + 1L
    to <- sep[[i + 1L]] - 1L
    cf_parse_ref_block(if (to >= from) rest[from:to] else character())
  })
}

cf_snapshot_ref_block <- function(snap_file, test, idx) {
  blocks <- cf_snapshot_ref_blocks(snap_file, test)
  if (is.null(blocks) || idx > length(blocks)) NULL else blocks[[idx]]
}

cf_parse_ref_block <- function(lines) {
  groups <- list()
  kind <- NA_character_
  in_err <- FALSE
  for (ln in lines) {
    if (grepl("^    [A-Za-z]+$", ln)) {
      new_kind <- trimws(ln)
      if (new_kind == "Code" && !identical(kind, "Code")) {
        groups[[length(groups) + 1L]] <- list(code = character(), err = FALSE, err_lines = character())
      }
      kind <- new_kind
      in_err <- FALSE
    } else if (startsWith(ln, "      ") && length(groups) > 0L) {
      txt <- substring(ln, 7L)
      g <- length(groups)
      if (identical(kind, "Code")) {
        groups[[g]]$code <- c(groups[[g]]$code, txt)
      } else if (identical(kind, "Condition")) {
        if (grepl("^Error\\b.*:$", txt)) {
          groups[[g]]$err <- TRUE
          in_err <- TRUE
        } else if (grepl("^(Warning|Message)\\b.*:$", txt)) {
          in_err <- FALSE
        } else if (in_err) {
          groups[[g]]$err_lines <- c(groups[[g]]$err_lines, txt)
        }
      }
    }
  }
  if (length(groups) == 0L) {
    return(NULL)
  }
  errors <- logical()
  err_lines <- character()
  stmt_lines <- list()
  for (g in groups) {
    n <- tryCatch(length(parse(text = paste(g$code, collapse = "\n"), keep.source = FALSE)),
      error = function(e) NA_integer_)
    if (is.na(n) || n == 0L) {
      return(NULL)
    }
    errors <- c(errors, rep(FALSE, n - 1L), g$err)
    err_lines <- c(err_lines, g$err_lines)
    stmt_lines <- c(stmt_lines, rep(list(character()), n - 1L), list(g$err_lines))
  }
  list(errors = errors, error_lines = err_lines, stmt_lines = stmt_lines)
}

cf_empty_ids <- function() {
  data.frame(
    file = character(), test = character(), status = character(), raw_status = character(),
    exercised = logical(), uploads = integer(), item = character(),
    upload_failures = integer(), constructor_calls = integer(), tbl_gpu_created = integer(),
    snapshots_relaxed = integer(), relaxed_only = logical(), snapshots_uncredited = integer(),
    message = character(),
    stringsAsFactors = FALSE
  )
}

cf_file_result <- function(fname, ids, file_error = NA_character_, unmatched = NULL,
                           relaxed = NULL) {
  status <- ids$status
  list(
    file = fname, n_tests = nrow(ids),
    passed = sum(status == "pass"), failed = sum(status == "fail"),
    errored = sum(status == "error"), skipped = sum(status == "skip"),
    not_exercised = sum(status == "not_exercised"), excluded = sum(status == "excluded"),
    file_error = file_error,
    reasons = ids$message[ids$status %in% c("fail", "error") & !is.na(ids$message)],
    ids = ids,
    unmatched_exclusions = cf_or(unmatched, data.frame(file = character(), test = character(),
      item = character(), reason = character(), stringsAsFactors = FALSE)),
    relaxed_snapshots = cf_or(relaxed, data.frame(file = character(), test = character(),
      cuplyr_message = character(), reference = character(), final_line_differs = logical(),
      stringsAsFactors = FALSE))
  )
}

cf_unmatched <- function(exclusions, fname, ids_tests) {
  if (is.null(exclusions)) {
    return(NULL)
  }
  mine <- exclusions[exclusions$file == fname, , drop = FALSE]
  mine[!(mine$test %in% ids_tests), , drop = FALSE]
}

conformance_run_file <- function(path, exclusions = NULL, upload = TRUE) {
  fname <- basename(path)
  stage <- tryCatch(cf_stage_file(path), error = function(e) e)
  if (inherits(stage, "error")) {
    return(cf_file_result(fname, cf_empty_ids(), file_error = conditionMessage(stage),
      unmatched = cf_unmatched(exclusions, fname, character())))
  }
  on.exit(unlink(stage$dir, recursive = TRUE), add = TRUE)

  state <- cf_new_state()
  state$snap_file <- normalizePath(cf_snap_path(path), mustWork = FALSE) # testthat changes the working directory
  shim <- cf_make_shim_env(state, upload = upload)
  env <- shim$env
  reporter <- cf_make_reporter(state)

  # CRAN-mode snapshot semantics and edition 3 whatever the caller has set.
  withr::local_envvar(NOT_CRAN = "false", TESTTHAT_EDITION = "3")
  withr::local_options(structure(list(shim$upload), names = cf_upload_option))
  restore <- cf_instrument_new_tbl_gpu(state)
  on.exit(restore(), add = TRUE)

  res <- tryCatch(
    # The snapshot reporter announces cleaning up the staged _snaps copy.
    withCallingHandlers(
      testthat::test_file(stage$path, reporter = reporter, env = env),
      message = function(m) {
        if (grepl("unused snapshot", conditionMessage(m))) invokeRestart("muffleMessage")
      }
    ),
    error = function(e) structure(list(), file_level_error = conditionMessage(e))
  )
  file_error <- attr(res, "file_level_error")
  if (!is.null(file_error)) {
    return(cf_file_result(fname, cf_empty_ids(), file_error = file_error,
      unmatched = cf_unmatched(exclusions, fname, character())))
  }

  n <- length(res)
  if (n == 0L) {
    return(cf_file_result(fname, cf_empty_ids(),
      unmatched = cf_unmatched(exclusions, fname, character())))
  }
  counters <- reporter$per_test
  if (length(counters) != n) {
    counters <- rep(list(list(uploads = 0L, upload_failures = 0L, constructor_calls = 0L,
      tbl_gpu_created = 0L, relaxed = list(), uncredited = 0L)), n)
  }

  # Test names can repeat within a file; suffix the occurrence so every row
  # has a stable id for the merge gate's per-test regression check.
  test_names <- vapply(res, function(t) {
    nm <- t$test
    if (is.null(nm) || is.na(nm)) "" else as.character(nm)
  }, character(1))
  occurrence <- stats::ave(seq_along(test_names), test_names, FUN = seq_along)
  test_ids <- ifelse(occurrence > 1, paste0(test_names, " #", occurrence), test_names)
  test_ids <- gsub("[\t\n\r]+", " ", test_ids)

  outcomes <- lapply(res, cf_test_outcome)
  raw <- vapply(outcomes, `[[`, character(1), "status")
  msg <- vapply(outcomes, function(o) {
    if (is.na(o$message)) NA_character_ else cf_sanitize(o$message)
  }, character(1))
  uploads <- vapply(counters, function(c) as.integer(c$uploads), integer(1))
  upload_failures <- vapply(counters, function(c) as.integer(c$upload_failures), integer(1))
  ctor_calls <- vapply(counters, function(c) as.integer(c$constructor_calls), integer(1))
  created <- vapply(counters, function(c) as.integer(c$tbl_gpu_created), integer(1))
  n_relaxed <- vapply(counters, function(c) length(c$relaxed), integer(1))
  n_uncredited <- vapply(counters, function(c) as.integer(c$uncredited %||% 0L), integer(1))
  exercised <- (created > 0L) | (uploads > 0L)
  # Passing evidence that is nothing but relaxed `error = TRUE` snapshots.
  n_success <- vapply(res, function(t) {
    sum(vapply(t$results, function(x) inherits(x, "expectation_success"), NA))
  }, integer(1))
  relaxed_only <- n_relaxed > 0L & n_success <= n_relaxed

  item <- rep(NA_character_, n)
  if (!is.null(exclusions)) {
    mine <- exclusions[exclusions$file == fname, , drop = FALSE]
    hit <- match(test_ids, mine$test)
    item[!is.na(hit)] <- mine$item[hit[!is.na(hit)]]
  }
  status <- raw
  status[raw == "pass" & !exercised & ctor_calls > 0L] <- "not_exercised"
  status[!is.na(item)] <- "excluded"

  ids <- data.frame(
    file = rep(fname, n), test = test_ids, status = status, raw_status = raw,
    exercised = exercised, uploads = uploads, item = item,
    upload_failures = upload_failures, constructor_calls = ctor_calls,
    tbl_gpu_created = created, snapshots_relaxed = n_relaxed, relaxed_only = relaxed_only,
    snapshots_uncredited = n_uncredited, message = msg,
    stringsAsFactors = FALSE
  )

  relaxed <- NULL
  if (any(n_relaxed > 0L)) {
    # One row per credited snapshot: every cuplyr error message (not just the
    # first) next to the reference's error lines, for audit.
    rows <- lapply(which(n_relaxed > 0L), function(i) {
      do.call(rbind, lapply(counters[[i]]$relaxed, function(r) {
        data.frame(
          file = fname, test = test_ids[[i]],
          cuplyr_message = paste(vapply(r$cuplyr, cf_sanitize, character(1)), collapse = " ; "),
          reference = paste(vapply(r$reference, cf_sanitize, character(1)), collapse = " | "),
          final_line_differs = isTRUE(r$final_differs),
          stringsAsFactors = FALSE
        )
      }))
    })
    relaxed <- do.call(rbind, rows)
  }

  cf_file_result(fname, ids,
    unmatched = cf_unmatched(exclusions, fname, test_ids), relaxed = relaxed)
}

# ---- TSV in / out, comparison ------------------------------------------------

conformance_write_ids <- function(ids, path) {
  first <- c("file", "test", "status")
  ids <- ids[, c(first, setdiff(names(ids), first)), drop = FALSE]
  for (nm in names(ids)) {
    if (is.character(ids[[nm]])) ids[[nm]] <- gsub("[\t\n\r]+", " ", ids[[nm]])
  }
  utils::write.table(ids, path, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
  invisible(path)
}

conformance_read_ids <- function(path) {
  utils::read.delim(path, quote = "", comment.char = "", colClasses = "character",
    stringsAsFactors = FALSE, check.names = FALSE, na.strings = character(0))
}

conformance_compare <- function(old_ids, new_ids) {
  key <- function(d) paste(d$file, d$test, sep = "\t")
  ko <- key(old_ids)
  kn <- key(new_ids)
  files <- sort(unique(c(old_ids$file, new_ids$file)))
  count <- function(d, f, st) sum(d$file == f & d$status %in% st)
  per_file <- do.call(rbind, lapply(files, function(f) {
    data.frame(
      file = f,
      tests_before = sum(old_ids$file == f), tests_after = sum(new_ids$file == f),
      pass_before = count(old_ids, f, "pass"), pass_after = count(new_ids, f, "pass"),
      fail_after = count(new_ids, f, "fail"), error_after = count(new_ids, f, "error"),
      skip_after = count(new_ids, f, "skip"),
      not_exercised_after = count(new_ids, f, "not_exercised"),
      excluded_after = count(new_ids, f, "excluded"),
      stringsAsFactors = FALSE
    )
  }))
  if (is.null(per_file)) {
    per_file <- data.frame(file = character(), pass_before = integer(), pass_after = integer())
  }

  old_pass <- old_ids[old_ids$status == "pass", , drop = FALSE]
  idx <- match(key(old_pass), kn)
  new_status <- new_ids$status[idx]
  lost <- is.na(idx) | new_status != "pass"
  lost_passes <- data.frame(
    file = old_pass$file[lost], test = old_pass$test[lost], new_status = new_status[lost],
    message = if ("message" %in% names(new_ids)) new_ids$message[idx[lost]] else rep(NA_character_, sum(lost)),
    stringsAsFactors = FALSE
  )

  new_pass <- new_ids[new_ids$status == "pass", , drop = FALSE]
  idx2 <- match(key(new_pass), ko)
  old_status <- old_ids$status[idx2]
  gained <- is.na(idx2) | old_status != "pass"
  new_passes <- data.frame(
    file = new_pass$file[gained], test = new_pass$test[gained], old_status = old_status[gained],
    stringsAsFactors = FALSE
  )
  list(per_file = per_file, lost_passes = lost_passes, new_passes = new_passes)
}

# ---- Merge gate decision (tools/conformance-gate.R) --------------------------
#
# The merge gate compares the candidate's per-test TSV with the ratcheted
# baseline. A baseline `pass` that is no longer `pass` is a regression unless:
#   reclassified  now `excluded` (always: it needs a manifest row, which the
#                 diff shows), or now `not_exercised` AND the candidate changes
#                 the harness relative to dev (harness_changed). On a
#                 package-only merge the harness is identical, so a fixture
#                 that stops uploading means cuplyr broke an upload: that stays
#                 a regression.
#   accepted      now `fail` or `error` and listed in
#                 tools/conformance-known-failures.tsv (the candidate's or
#                 dev's copy). skip and missing are never accepted.
# Known-failure rows are owned by a feature. When the owner merges, every row
# it owns (in either copy) must pass, and its branch must have deleted them
# from the candidate's copy; otherwise the gate fails. A listed id that passes
# on a non-owner's merge only warns.

conformance_known_failures_cols <- c("file", "test", "owner", "reason")

# Reads and validates tools/conformance-known-failures.tsv. NULL or a missing
# path is an empty list (dev predates the file).
conformance_load_known_failures <- function(path) {
  empty <- as.data.frame(stats::setNames(rep(list(character()), 4), conformance_known_failures_cols),
    stringsAsFactors = FALSE)
  if (is.null(path) || !file.exists(path)) {
    return(empty)
  }
  m <- utils::read.delim(path, quote = "", comment.char = "", colClasses = "character",
    check.names = FALSE, stringsAsFactors = FALSE, na.strings = character(0))
  missing_cols <- setdiff(conformance_known_failures_cols, names(m))
  if (length(missing_cols) > 0) {
    stop("known-failures file ", path, " lacks column(s): ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }
  m <- m[, conformance_known_failures_cols, drop = FALSE]
  for (nm in conformance_known_failures_cols) {
    blank <- which(!nzchar(trimws(m[[nm]])))
    if (length(blank) > 0) {
      stop("known-failures file ", path, ": `", nm, "` must be non-empty (row(s) ",
        paste(blank, collapse = ", "), ")", call. = FALSE)
    }
  }
  bad_owner <- !grepl("^F[0-9]{2}(-a[0-9]+)?$", m$owner)
  if (any(bad_owner)) {
    stop("known-failures file ", path, ": `owner` must look like F07 or F07-a2, got ",
      paste(unique(m$owner[bad_owner]), collapse = ", "), call. = FALSE)
  }
  key <- paste(m$file, m$test, sep = "\t")
  if (anyDuplicated(key)) {
    d <- key[duplicated(key)]
    stop("known-failures file ", path, ": duplicate (file, test): ",
      paste(sub("\t", " :: ", unique(d), fixed = TRUE), collapse = "; "), call. = FALSE)
  }
  rownames(m) <- NULL
  m
}

# Reads a per-test TSV for the gate; one without file/test/status columns
# (truncated, wrong file) is an error, not an empty run.
conformance_read_gate_ids <- function(path) {
  m <- conformance_read_ids(path)
  missing_cols <- setdiff(c("file", "test", "status"), names(m))
  if (length(missing_cols) > 0) {
    stop("per-test TSV ", path, " lacks column(s): ", paste(missing_cols, collapse = ", "),
      call. = FALSE)
  }
  m
}

# Informational NOTE lines for known-failure rows and exclusions the candidate
# adds relative to dev. Arguments are data frames with file and test columns
# (known failures also owner, exclusions also item).
conformance_gate_added <- function(known_candidate, known_base, excl_candidate, excl_base) {
  key <- function(d) paste(d$file, d$test, sep = "\t")
  if (is.null(excl_candidate)) excl_candidate <- data.frame(file = character(), test = character(), item = character())
  if (is.null(excl_base)) excl_base <- excl_candidate[0, , drop = FALSE]
  k <- known_candidate[!(key(known_candidate) %in% key(known_base)), , drop = FALSE]
  e <- excl_candidate[!(key(excl_candidate) %in% key(excl_base)), , drop = FALSE]
  c(
    if (nrow(k) > 0) sprintf("conformance: NOTE: candidate adds known-failure row %s :: %s (owner %s)",
      k$file, k$test, k$owner),
    if (nrow(e) > 0) sprintf("conformance: NOTE: candidate adds exclusion %s :: %s (%s)",
      e$file, e$test, e$item)
  )
}

# Feature name from the gate's output dir. tools/merge-queue creates it as
# gates/<name>-$(date +%Y%m%dT%H%M%S), where <name> is the branch suffix
# (feat/<name>), e.g. F07 or F07-a2 for a second attempt. The timestamp has a
# fixed shape, so stripping exactly one trailing -YYYYmmddTHHMMSS recovers
# <name> even when it contains dashes. Only the candidate's copy of the gate
# runs (the orchestrator's merge-queue passes nothing else), so the name has
# to come from here; anything that doesn't fit the pattern is an error rather
# than a guess, because a wrong name would silently skip the owner checks.
conformance_gate_feature_name <- function(out_dir) {
  b <- basename(sub("/+$", "", out_dir))
  pat <- "^([A-Za-z0-9][A-Za-z0-9._-]*)-[0-9]{8}T[0-9]{6}$"
  if (!grepl(pat, b)) {
    stop("gate output dir '", out_dir, "' does not end in <name>-YYYYmmddTHHMMSS; ",
      "cannot tell which feature is merging", call. = FALSE)
  }
  sub(pat, "\\1", b)
}

# F07-a2 (attempt 2 of F07) owns F07's rows.
cf_feature_base <- function(x) sub("-a[0-9]+$", "", x)

conformance_gate_check <- function(baseline, new, known_candidate, known_base, feature,
                                   harness_changed) {
  key <- function(d) paste(d$file, d$test, sep = "\t")
  kn <- key(new)
  new_status <- function(k) new$status[match(k, kn)]
  known <- rbind(known_candidate, known_base)
  known_keys <- unique(key(known))

  bp <- baseline[baseline$status == "pass", , drop = FALSE]
  bk <- key(bp)
  st <- new_status(bk)
  reclass <- !is.na(st) & (st == "excluded" | (st == "not_exercised" & isTRUE(harness_changed)))
  accepted <- !is.na(st) & st %in% c("fail", "error") & bk %in% known_keys
  regressed <- (is.na(st) | st != "pass") & !reclass & !accepted
  trans <- function(sel) data.frame(file = bp$file[sel], test = bp$test[sel],
    new_status = ifelse(is.na(st[sel]), "missing", st[sel]), stringsAsFactors = FALSE)

  np <- new[new$status == "pass", , drop = FALSE]
  new_passes <- np[!(key(np) %in% bk), c("file", "test"), drop = FALSE]

  me <- cf_feature_base(feature)
  owned <- known[cf_feature_base(known$owner) == me, , drop = FALSE]
  owned <- owned[!duplicated(key(owned)), , drop = FALSE]
  owned_st <- new_status(key(owned))
  owned_st[is.na(owned_st)] <- "missing"
  owner_failures <- character()
  bad <- owned_st != "pass"
  if (any(bad)) {
    owner_failures <- c(owner_failures, sprintf(
      "%s owns known failure %s :: %s, which is %s (must pass when its owner merges)",
      feature, owned$file[bad], owned$test[bad], owned_st[bad]))
  }
  still <- known_candidate[cf_feature_base(known_candidate$owner) == me, , drop = FALSE]
  if (nrow(still) > 0) {
    owner_failures <- c(owner_failures, sprintf(
      "%s must delete its row %s :: %s from tools/conformance-known-failures.tsv",
      feature, still$file, still$test))
  }

  others <- known[cf_feature_base(known$owner) != me, , drop = FALSE]
  others <- others[!duplicated(key(others)), , drop = FALSE]
  passing <- which(new_status(key(others)) %in% "pass")
  warnings <- sprintf("known failure %s :: %s (owner %s) passes now; %s should delete its row",
    others$file[passing], others$test[passing], others$owner[passing], others$owner[passing])

  regressions <- trans(regressed)
  list(
    regressions = regressions, reclassified = trans(reclass), accepted = trans(accepted),
    owner_failures = owner_failures, warnings = warnings, new_passes = new_passes,
    n_pass = sum(new$status == "pass"),
    ok = nrow(regressions) == 0 && length(owner_failures) == 0
  )
}

# The gate's summary lines (the first starts "conformance: PASS|FAIL").
conformance_gate_summary <- function(res) {
  counts <- sprintf("%d passing, +%d new, %d regressions, %d reclassified, %d accepted known failures",
    res$n_pass, nrow(res$new_passes), nrow(res$regressions), nrow(res$reclassified),
    nrow(res$accepted))
  why <- c(
    if (nrow(res$regressions) > 0) "previously-passing tests regressed; see conformance_regressions.txt",
    if (length(res$owner_failures) > 0) "known-failure owner check failed; see below"
  )
  head <- if (res$ok) {
    sprintf("conformance: PASS (%s)", counts)
  } else {
    sprintf("conformance: FAIL (%s; %s)", counts, paste(why, collapse = "; "))
  }
  # paste0() of a zero-length vector is "prefix", not character(0)
  tag <- function(prefix, x) if (length(x) > 0) paste0(prefix, x) else character()
  c(head, tag("conformance: OWNER FAIL: ", res$owner_failures),
    tag("conformance: WARNING: ", res$warnings))
}

# ---- Reason classification (heuristic, for the report's honesty mandate) -----

conformance_classify_reason <- function(msg) {
  if (is.na(msg) || !nzchar(msg)) {
    return("unknown")
  }
  if (grepl("not an exported object|dplyr:::|:::.*dplyr|non-exported|could not find function", msg, ignore.case = TRUE)) {
    return("dplyr-internal usage")
  }
  if (grepl("only failed with a missing-feature error", msg, fixed = TRUE)) {
    return("genuine cuplyr gap: verb/function missing for tbl_gpu (snapshot error is a missing-method error)")
  }
  if (grepl("to have class \"data.frame\"|to have class \"tbl_df\"", msg)) {
    return("genuine cuplyr design gap: bare data.frame input comes back as a tibble (collect() always returns tibbles)")
  }
  if (grepl("did not throw an error|did not throw the expected error|Expected .* to throw", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: expected error not raised")
  }
  # Checked BEFORE the generic tbl_gpu bucket: both messages mention "tbl_gpu"
  # but are genuine, actionable cuplyr-side gaps, not harness plumbing.
  if (grepl("must be a vector, not a", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: tbl_gpu has no vctrs vec_size()/vec_slice() contract (breaks coalesce()/if_else()/between()/consecutive_id()/n_distinct()/lead()/lag()/rank()/nth_value()/with_order() when passed a whole table)")
  }
  if (grepl("2-d .\\[. indexing", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: 2-d `[` (row+col) indexing not implemented for tbl_gpu (R/generics.R only implements the 1-d column-select form; dplyr's own tests routinely do `result[1:2, ]` to check a row subset)")
  }
  if (grepl("inherit from|Actual class|no applicable method|not a data\\.frame|no method for", msg, ignore.case = TRUE)) {
    return("shim artifact: class/representation mismatch")
  }
  if (grepl("tbl_gpu|gpu_normalize|GPU|has_gpu|cuplyr", msg, ignore.case = TRUE)) {
    return("shim artifact: tbl_gpu representation")
  }
  if (grepl("not supported|not implemented|unsupported|no ir_call_registry|Unrecognized|not recognized|Unknown function", msg, ignore.case = TRUE)) {
    return("genuine cuplyr gap: unsupported expression/verb shape")
  }
  "genuine cuplyr gap (needs triage)"
}

# ---- Report ------------------------------------------------------------------

cf_md <- function(x) gsub("\\n", " ", gsub("\\|", "\\\\|", x))

cf_pct <- function(num, den) if (den > 0) sprintf("%.1f%%", 100 * num / den) else "n/a"

# `results`: named list of conformance_run_file() values (in run order).
# `exclusions`: the manifest (or NULL). `cmp`: conformance_compare() output
# (or NULL). Returns markdown lines.
conformance_report <- function(results, exclusions = NULL, cmp = NULL, header = character()) {
  ids <- do.call(rbind, lapply(results, function(r) r$ids))
  if (is.null(ids)) ids <- cf_empty_ids()
  n_of <- function(st) sum(ids$status == st)
  total <- nrow(ids)
  passed <- n_of("pass")
  denom <- passed + n_of("fail") + n_of("error")
  file_errors <- vapply(results, function(r) !is.na(r$file_error), logical(1))
  green <- vapply(results, function(r) {
    is.na(r$file_error) && r$n_tests > 0 && r$failed == 0 && r$errored == 0
  }, logical(1))

  lines <- c(header, "", "## Headline numbers", "",
    sprintf("- **%d of %d runnable files fully green** (0 failed, 0 errored, ran to completion)", sum(green), length(results)),
    sprintf("- **%d individual tests** executed across those files", total),
    sprintf("- **old headline** (pass / all tests): %d / %d = **%s**", passed, total, cf_pct(passed, total)),
    sprintf("- **new headline** (pass / (pass + fail + error)): %d / %d = **%s**", passed, denom, cf_pct(passed, denom)),
    sprintf("- **pass rate over exercised tests** (a tbl_gpu was created or uploaded; pass / (pass + fail + error)): %d / %d = **%s**",
      sum(ids$status == "pass" & ids$exercised), sum(ids$status %in% c("pass", "fail", "error") & ids$exercised),
      cf_pct(sum(ids$status == "pass" & ids$exercised), sum(ids$status %in% c("pass", "fail", "error") & ids$exercised))),
    sprintf("- status counts: %d pass, %d fail, %d error, %d skip, %d not_exercised, %d excluded",
      passed, n_of("fail"), n_of("error"), n_of("skip"), n_of("not_exercised"), n_of("excluded")),
    sprintf("- passing tests that never created or uploaded a tbl_gpu (vector-only / helper-only, exercised = FALSE): %d",
      sum(ids$status == "pass" & !ids$exercised)),
    sprintf("- passing tests whose only passing evidence is relaxed `error = TRUE` snapshots (cuplyr errored on exactly the statements dplyr's reference errors on, with no infrastructure error; wording not compared): %d",
      sum(ids$status == "pass" & ids$relaxed_only)),
    sprintf("- `error = TRUE` snapshots not credited for lack of a usable reference block in _snaps (recorded as a skip expectation; the test went on): %d, in %d test(s)",
      sum(ids$snapshots_uncredited %||% 0L), sum((ids$snapshots_uncredited %||% 0L) > 0L)),
    sprintf("- %d file(s) errored before a single `test_that()` block completed (parse/load-time failure)", sum(file_errors)),
    "")

  # Failure categories.
  bad <- ids[ids$status %in% c("fail", "error"), , drop = FALSE]
  lines <- c(lines, "## Top failure-reason categories (heuristic classification of the first failure message)", "")
  if (nrow(bad) > 0) {
    cats <- sort(table(vapply(bad$message, conformance_classify_reason, character(1))), decreasing = TRUE)
    lines <- c(lines, "| Category | Count |", "|---|---|",
      sprintf("| %s | %d |", cf_md(names(cats)), as.integer(cats)))
  } else {
    lines <- c(lines, "(no failures/errors recorded)")
  }

  # Per-file ledger.
  lines <- c(lines, "", "## Per-file ledger", "",
    "| File | Tests | Pass | Fail | Error | Skip | Not exercised | Excluded | Status | Top reason (first failure/error) |",
    "|---|---|---|---|---|---|---|---|---|---|")
  for (r in results) {
    status <- if (!is.na(r$file_error)) "FILE ERROR" else if (r$n_tests == 0) "NO TESTS" else
      if (r$failed == 0 && r$errored == 0) "GREEN" else "RED"
    top <- if (!is.na(r$file_error)) cf_md(substr(r$file_error, 1, 160)) else
      if (length(r$reasons) > 0) cf_md(substr(r$reasons[[1]], 1, 160)) else ""
    lines <- c(lines, sprintf("| %s | %d | %d | %d | %d | %d | %d | %d | %s | %s |",
      r$file, r$n_tests, r$passed, r$failed, r$errored, r$skipped, r$not_exercised, r$excluded, status, top))
  }

  # Exclusions.
  excl <- ids[ids$status == "excluded", , drop = FALSE]
  lines <- c(lines, "", "## Exclusions (tests removed from the pass/fail denominator)", "")
  if (nrow(excl) > 0) {
    by_item <- table(excl$item)
    lines <- c(lines, "| C-item | Tests | Raw pass | Raw fail/error | Raw skip |", "|---|---|---|---|---|")
    for (it in sort(names(by_item))) {
      e <- excl[excl$item == it, ]
      lines <- c(lines, sprintf("| %s | %d | %d | %d | %d |", it, nrow(e), sum(e$raw_status == "pass"),
        sum(e$raw_status %in% c("fail", "error")), sum(e$raw_status == "skip")))
    }
    lines <- c(lines, "", "Full list (every row has a one-line reason in tools/conformance-exclusions.tsv):", "",
      "| Item | File | Test | Raw status | Reason |", "|---|---|---|---|---|")
    reason_of <- if (is.null(exclusions)) NULL else
      stats::setNames(exclusions$reason, paste(exclusions$file, exclusions$test, sep = "\t"))
    for (i in seq_len(nrow(excl))) {
      rs <- if (is.null(reason_of)) "" else cf_or(unname(reason_of[paste(excl$file[i], excl$test[i], sep = "\t")]), "")
      lines <- c(lines, sprintf("| %s | %s | %s | %s | %s |", excl$item[i], excl$file[i],
        cf_md(excl$test[i]), excl$raw_status[i], cf_md(rs)))
    }
  } else {
    lines <- c(lines, "(none)")
  }
  stale <- do.call(rbind, lapply(results, function(r) r$unmatched_exclusions))
  lines <- c(lines, "", "### Stale manifest entries (match no test in a file that was run)", "")
  if (!is.null(stale) && nrow(stale) > 0) {
    lines <- c(lines, "| Item | File | Test |", "|---|---|---|",
      sprintf("| %s | %s | %s |", stale$item, stale$file, cf_md(stale$test)))
  } else {
    lines <- c(lines, "(none)")
  }

  # not_exercised.
  ne <- ids[ids$status == "not_exercised", , drop = FALSE]
  lines <- c(lines, "", "## Not exercised (passed, but no tbl_gpu was ever created or uploaded)", "")
  if (nrow(ne) > 0) {
    lines <- c(lines, "| File | Test | Constructor calls | Upload failures |", "|---|---|---|---|",
      sprintf("| %s | %s | %d | %d |", ne$file, cf_md(ne$test), ne$constructor_calls, ne$upload_failures))
  } else {
    lines <- c(lines, "(none)")
  }

  # Relaxed snapshots.
  relaxed <- do.call(rbind, lapply(results, function(r) r$relaxed_snapshots))
  lines <- c(lines, "", "## Relaxed snapshot checks (`error = TRUE` snapshots credited; wording not compared)", "")
  if (!is.null(relaxed) && nrow(relaxed) > 0) {
    key <- paste(relaxed$file, relaxed$test, sep = "\t")
    uniq <- unique(key)
    trunc <- function(x, n = 300L) if (nchar(x) > n) paste0(substr(x, 1L, n - 3L), "...") else x
    differs <- if (is.null(relaxed$final_line_differs)) rep(FALSE, nrow(relaxed)) else relaxed$final_line_differs
    lines <- c(lines, sprintf("%d credited snapshot check(s) in %d test(s). Each errored on exactly the statements dplyr's reference errors on, with no infrastructure/missing-feature error; wording is not compared (all cuplyr messages shown for audit).",
      nrow(relaxed), length(uniq)),
      sprintf("%d test(s) have a check whose final `!` line (innermost error) differs from the reference's: marked AUDIT, review those first.",
        length(unique(key[differs]))), "",
      "| File | Test | Checks | Final `!` line | cuplyr message(s) | dplyr reference |", "|---|---|---|---|---|---|")
    for (k in uniq) {
      r <- relaxed[key == k, , drop = FALSE]
      ref <- unique(stats::na.omit(r$reference))
      lines <- c(lines, sprintf("| %s | %s | %d | %s | %s | %s |", r$file[[1]], cf_md(r$test[[1]]), nrow(r),
        if (any(differs[key == k])) "AUDIT (differs)" else "same",
        cf_md(trunc(paste(unique(r$cuplyr_message), collapse = " ; "))),
        cf_md(trunc(paste(ref, collapse = " ; ")))))
    }
  } else {
    lines <- c(lines, "(none)")
  }

  # Before/after.
  if (!is.null(cmp)) {
    pf <- cmp$per_file
    lines <- c(lines, "", "## Before / after per file (against the --compare ids)", "",
      "| File | Tests before | Tests after | Pass before | Pass after | Fail | Error | Skip | Not exercised | Excluded |",
      "|---|---|---|---|---|---|---|---|---|---|",
      sprintf("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |", pf$file, pf$tests_before,
        pf$tests_after, pf$pass_before, pf$pass_after, pf$fail_after, pf$error_after, pf$skip_after,
        pf$not_exercised_after, pf$excluded_after),
      sprintf("| **all** | %d | %d | %d | %d | %d | %d | %d | %d | %d |", sum(pf$tests_before),
        sum(pf$tests_after), sum(pf$pass_before), sum(pf$pass_after), sum(pf$fail_after),
        sum(pf$error_after), sum(pf$skip_after), sum(pf$not_exercised_after), sum(pf$excluded_after)))
    lp <- cmp$lost_passes
    lines <- c(lines, "", sprintf("### Lost passes: %d test(s) that passed before and do not pass now", nrow(lp)), "")
    if (nrow(lp) > 0) {
      lines <- c(lines, "| File | Test | New status | Message |", "|---|---|---|---|",
        sprintf("| %s | %s | %s | %s |", lp$file, cf_md(lp$test),
          ifelse(is.na(lp$new_status), "(missing)", lp$new_status),
          cf_md(ifelse(is.na(lp$message), "", lp$message))))
    }
    lines <- c(lines, "", sprintf("### New passes: %d", nrow(cmp$new_passes)))
  }
  lines
}
