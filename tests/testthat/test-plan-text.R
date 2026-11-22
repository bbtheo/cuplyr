# The plan renderer (R/plan-text.R) on hand-built IR and plans: no GPU needed.
# The printed format of whole pipelines is pinned by test-parity-10-explain.R.

pt_col <- function(x) ir_col(x)
pt_num <- function(v) ir_lit(v, "FLOAT64")
pt_call <- function(op, ...) ir_call(op, list(...))
pt_text <- function(ir) plan_ir_text(ir)

pt_schema <- list(names = c("a", "b", "c"), types = rep("FLOAT64", 3L))

pt_pred <- function(ir) make_predicate(ir, pt_schema)

test_that("parentheses are inserted where a child binds looser than its parent", {
  expect_identical(pt_text(pt_call("*", pt_call("+", pt_col("x"), pt_num(1)), pt_num(2))), "(x + 1) * 2")
  expect_identical(pt_text(pt_call("+", pt_call("*", pt_col("x"), pt_num(2)), pt_num(1))), "x * 2 + 1")
  expect_identical(pt_text(pt_call("-", pt_call("+", pt_col("x"), pt_num(1)))), "-(x + 1)")
  expect_identical(pt_text(pt_call("-", pt_col("i"))), "-i")
  # left-associative: the right child of an equal-precedence operator is grouped
  expect_identical(pt_text(pt_call("-", pt_col("a"), pt_call("-", pt_col("b"), pt_col("c")))), "a - (b - c)")
  expect_identical(pt_text(pt_call("-", pt_call("-", pt_col("a"), pt_col("b")), pt_col("c"))), "a - b - c")
  # `^` is right-associative and binds tighter than unary minus
  expect_identical(pt_text(pt_call("^", pt_col("x"), pt_num(2))), "x^2")
  expect_identical(pt_text(pt_call("^", pt_call("-", pt_col("x")), pt_num(2))), "(-x)^2")
  expect_identical(pt_text(pt_call("-", pt_call("^", pt_col("x"), pt_num(2)))), "-x^2")
  expect_identical(pt_text(pt_call("^", pt_call("^", pt_col("a"), pt_col("b")), pt_col("c"))), "(a^b)^c")
  # `!` binds looser than a comparison and tighter than `&`
  gt <- pt_call(">", pt_col("x"), pt_num(1))
  lt <- pt_call("<", pt_col("i"), ir_lit(3L, "INT32"))
  expect_identical(pt_text(pt_call("!", pt_call("&", gt, lt))), "!(x > 1 & i < 3L)")
  expect_identical(pt_text(pt_call("!", pt_call("is.na", pt_col("x")))), "!is.na(x)")
  expect_identical(pt_text(pt_call("&", gt, pt_call("|", lt, gt))), "x > 1 & (i < 3L | x > 1)")
  expect_identical(pt_text(pt_call("|", pt_call("&", gt, lt), gt)), "x > 1 & i < 3L | x > 1")
  # a comparison of comparisons is grouped
  expect_identical(pt_text(pt_call("==", gt, lt)), "(x > 1) == (i < 3L)")
  expect_identical(pt_text(pt_call("%in%", pt_col("x"), pt_num(c(1, 3)))), "x %in% c(1, 3)")
  expect_identical(pt_text(pt_call("*", pt_call("%%", pt_col("a"), pt_col("b")), pt_col("c"))), "a %% b * c")
})

test_that("literals print as R code", {
  expect_identical(pt_text(ir_lit(1000L, "INT32")), "1000L")
  expect_identical(pt_text(pt_num(0.5)), "0.5")
  expect_identical(pt_text(ir_lit("q", "STRING")), "\"q\"")
  expect_identical(pt_text(ir_lit(TRUE, "BOOL8")), "TRUE")
  expect_identical(pt_text(ir_lit(NULL, NULL, na = TRUE)), "NA")
  expect_identical(pt_text(ir_lit(NA_real_, "FLOAT64")), "NA")
  expect_identical(pt_text(ir_lit(NaN, "FLOAT64")), "NaN")
  expect_identical(pt_text(ir_lit(as.Date("2024-02-01"), "TIMESTAMP_DAYS")), "as.Date(\"2024-02-01\")")
  expect_identical(
    pt_text(ir_lit(as.POSIXct("2024-02-01 03:04:05", tz = "UTC"), "TIMESTAMP_MICROSECONDS")),
    "as.POSIXct(\"2024-02-01 03:04:05\", tz = \"UTC\")"
  )
  expect_identical(pt_text(pt_num(c(1, 2, 3))), "c(1, 2, 3)")
  expect_identical(pt_text(pt_num(as.numeric(1:11))), "<11 values>")
  # a negative literal groups like a unary minus
  expect_identical(pt_text(pt_call("^", pt_num(-2), pt_col("x"))), "(-2)^x")
  expect_identical(plan_ir_text(ir_vec(1:5, "INT32", NULL, "v")), "1:5")
  expect_identical(plan_ir_text(ir_vec(as.numeric(1:7), "FLOAT64", NULL, "v")), "<7 values>")
})

test_that("internal ops print under the R function they stand for", {
  expect_identical(pt_text(pt_call("ceil", pt_col("x"))), "ceiling(x)")
  expect_identical(pt_text(pt_call("cast_float64", pt_col("l"))), "as.double(l)")
  expect_identical(pt_text(pt_call("cast_int32", pt_col("l"))), "as.integer(l)")
  expect_identical(pt_text(pt_call("factor_codes", pt_col("f"))), "as.integer(f)")
  expect_identical(pt_text(pt_call("factor_labels", pt_col("f"), ir_lit(c("a", "b"), "STRING"))), "as.character(f)")
  expect_identical(pt_text(pt_call("date_add_days", pt_col("d"), ir_lit(3L, "INT32"))), "d + 3L")
  expect_identical(pt_text(pt_call("date_sub_days", pt_col("d"), ir_lit(3L, "INT32"))), "d - 3L")
  expect_identical(pt_text(pt_call("ts_add_us", pt_col("t"), ir_lit(3600e6, "INT64"))), "t + 3600")
  expect_identical(pt_text(pt_call("if_else", pt_call(">", pt_col("x"), pt_num(1)), pt_num(10), pt_num(20))),
                   "if_else(x > 1, 10, 20)")
  expect_identical(pt_text(pt_call("some_op", pt_col("x"))), "some_op(x)")
  expect_identical(pt_text(ir_call("n", list())), "n()")
  # a window call: its lag, rank direction and NA flag
  expect_identical(pt_text(ir_call("lag", list(pt_col("x")), meta = list(n = 2L, default = NULL))), "lag(x, 2)")
  expect_identical(pt_text(ir_call("lag", list(pt_col("x")), meta = list(n = 1L, default = NULL))), "lag(x)")
  expect_identical(pt_text(ir_call("min_rank", list(pt_col("x")), meta = list(rank_desc = TRUE))), "min_rank(desc(x))")
  expect_identical(pt_text(ir_call("mean", list(pt_col("x")), meta = list(na_rm = TRUE))), "mean(x, na.rm = TRUE)")
  expect_identical(
    pt_text(ir_call("case_when", list(pt_call(">", pt_col("x"), pt_num(1)), ir_lit("hi", "STRING"),
                                      ir_lit("lo", "STRING")), meta = list(n_when = 1L))),
    "case_when(x > 1 ~ \"hi\", .default = \"lo\")"
  )
})

test_that("column names that are not syntactic get backticks", {
  expect_identical(pt_text(pt_call("+", pt_col("my col"), pt_num(1))), "`my col` + 1")
  expect_identical(plan_names(c("a", "1a", "if", "..win1..")), c("a", "`1a`", "`if`", "..win1.."))
})

test_that("a step longer than 80 characters is split one argument per line", {
  short <- plan_step_lines("select", c("a", "b"))
  expect_identical(short, "select(a, b)")
  args <- paste0("col", 1:12, " = x", 1:12, " + 1")
  expect_gt(nchar(paste0("mutate(", paste(args, collapse = ", "), ")")), 80L)
  long <- plan_step_lines("mutate", args, comment = "note")
  expect_identical(long[[1L]], "mutate(")
  expect_identical(long[[2L]], "  col1 = x1 + 1,")
  expect_identical(long[[length(long) - 1L]], "  col12 = x12 + 1")
  expect_identical(long[[length(long)]], ")  # note")
  # the comment does not count towards the width; a join is never split
  expect_identical(plan_step_lines("x", strrep("a", 60L), comment = strrep("c", 40L)),
                   paste0("x(", strrep("a", 60L), ")  # ", strrep("c", 40L)))
  expect_length(plan_step_lines("left_join", args, wrap = FALSE), 1L)
})

test_that("a plan prints source first, with a pruning select folded into the source", {
  src <- ast_source(pt_schema)
  sel <- ast_select(src, c("a", "b"), pruned = TRUE)
  flt <- ast_filter(sel, list(pt_pred(pt_call(">", pt_col("a"), pt_num(1)))))
  mut <- ast_mutate(flt, list(make_mutate_expr("d", pt_call("+", pt_col("a"), pt_col("b")), pt_schema)))
  expect_identical(
    plan_node_block(mut, TRUE)$text,
    c("table on the GPU", "  columns read: 2 of 3 (a, b)", "filter(a > 1)", "mutate(d = a + b)")
  )
  # not folded when it is the user's select
  user <- ast_select(src, c("a", "b"))
  expect_identical(plan_node_block(user, FALSE)$text,
                   c("table on the GPU", "  columns read: all 3", "select(a, b)"))
  # a grouped filter says so
  grp <- ast_filter(src, list(pt_pred(pt_call(">", pt_col("a"), pt_num(1)))), grouped = TRUE)
  expect_identical(plan_node_block(grp, FALSE)$text[[3L]], "filter(a > 1)  # grouped")
})

test_that("more than 8 columns are listed as the first 8 and a count", {
  expect_identical(plan_columns_text(letters[1:5], 5L), "columns read: all 5")
  expect_identical(plan_columns_text(letters[1:3], 9L), "columns read: 3 of 9 (a, b, c)")
  expect_identical(plan_columns_text(letters[1:10], 20L),
                   "columns read: 10 of 20 (a, b, c, d, e, f, g, h, and 2 more)")
})

test_that("a join prints its right side under `y:`", {
  x <- ast_source(list(names = c("k", "a"), types = c("INT32", "FLOAT64")))
  y <- ast_source(list(names = c("k", "b"), types = c("INT32", "FLOAT64")))
  by <- list(left = "k", right = "k", op = "==", filter = "none")
  expect_identical(
    plan_node_block(ast_join("left", x, y, by), FALSE)$text,
    c("table on the GPU", "  columns read: all 2", "left_join(y, by = join_by(k))",
      "  y:", "    table on the GPU", "      columns read: all 2")
  )
  uneq <- list(left = c("k", "a"), right = c("k", "b"), op = c("==", ">="), filter = c("none", "closest"))
  one <- plan_node_block(ast_join("inner", x, y, uneq, relationship = "many-to-many", multiple = "first"), FALSE)$text[[3L]]
  # "many-to-many" on a join dplyr doesn't check is the default, not the user's
  expect_identical(one, "inner_join(y, by = join_by(k, closest(a >= b)), multiple = \"first\")")
  expect_identical(
    plan_node_block(ast_join("inner", x, y, by, relationship = "many-to-many", keep = TRUE), FALSE)$text[[3L]],
    "inner_join(y, by = join_by(k), keep = TRUE, relationship = \"many-to-many\")"
  )
  expect_identical(plan_node_block(ast_join("cross", x, y, list(left = character(), right = character(),
                                                            op = character(), filter = character())),
                                   FALSE)$text[[3L]], "cross_join(y)")
})

test_that("arguments that change a semi, anti or cross join are printed", {
  x <- ast_source(list(names = c("k", "a"), types = c("INT32", "FLOAT64")))
  y <- ast_source(list(names = c("k", "b"), types = c("INT32", "FLOAT64")))
  by <- list(left = "k", right = "k", op = "==", filter = "none")
  no_by <- list(left = character(), right = character(), op = character(), filter = character())
  line <- function(node) plan_node_block(node, FALSE)$text[[3L]]
  expect_identical(line(ast_join("semi", x, y, by, na_matches = "never")),
                   "semi_join(y, by = join_by(k), na_matches = \"never\")")
  expect_identical(line(ast_join("anti", x, y, by, na_matches = "never")),
                   "anti_join(y, by = join_by(k), na_matches = \"never\")")
  expect_identical(line(ast_join("semi", x, y, by)), "semi_join(y, by = join_by(k))")
  expect_identical(line(ast_join("cross", x, y, no_by, suffix = c("_l", "_r"))),
                   "cross_join(y, suffix = c(\"_l\", \"_r\"))")
})

test_that("string, paste and substr calls print as the R call the user wrote", {
  s <- pt_col("s")
  lit <- function(v) ir_lit(v, if (is.logical(v)) "BOOL8" else "STRING")
  expect_identical(pt_text(pt_call("grepl", s, lit("^A"), lit(FALSE))), "grepl(\"^A\", s)")
  expect_identical(pt_text(pt_call("grepl", s, lit("a"), lit(TRUE))), "grepl(\"a\", s, fixed = TRUE)")
  expect_identical(pt_text(pt_call("sub", s, lit("a"), lit("o"), lit(FALSE))), "sub(\"a\", \"o\", s)")
  expect_identical(pt_text(pt_call("gsub", s, lit("a"), lit("o"), lit(TRUE))), "gsub(\"a\", \"o\", s, fixed = TRUE)")
  expect_identical(pt_text(pt_call("starts_with", s, lit("x"))), "startsWith(s, \"x\")")
  expect_identical(pt_text(pt_call("ends_with", s, lit("x"))), "endsWith(s, \"x\")")
  # substr(): the IR's start is 0-based
  expect_identical(pt_text(pt_call("substr", s, ir_lit(1L, "INT32"), ir_lit(4L, "INT32"))), "substr(s, 2L, 4L)")
  expect_identical(pt_text(pt_call("paste_cols", s, pt_col("t"), lit(""))), "paste0(s, t)")
  expect_identical(pt_text(pt_call("paste_cols", s, lit("_"), pt_col("t"), lit(""))), "paste0(s, \"_\", t)")
  expect_identical(pt_text(pt_call("paste_cols", s, pt_col("t"), lit(" "))), "paste(s, t)")
  expect_identical(pt_text(pt_call("paste_cols", s, pt_col("t"), lit("-"))), "paste(s, t, sep = \"-\")")
  # a call inside another keeps its parentheses rules
  expect_identical(pt_text(pt_call("==", pt_call("grepl", s, lit("a"), lit(FALSE)), ir_lit(TRUE, "BOOL8"))),
                   "grepl(\"a\", s) == TRUE")
})

test_that("date, rowwise, recode and replace_when calls print as R calls", {
  expect_identical(pt_text(pt_call("as_date", pt_col("t"))), "as.Date(t, tz = \"UTC\")")
  expect_identical(pt_text(pt_call("as_posixct", pt_col("d"))), "as.POSIXct(d, tz = \"UTC\")")
  expect_identical(pt_text(pt_call("floor_date_hour", pt_col("t"))), "floor_date(t, \"hour\")")
  expect_identical(pt_text(pt_call("ceiling_date_day", pt_col("t"))), "ceiling_date(t, \"day\")")
  expect_identical(pt_text(pt_call("round_date_minute", pt_col("t"))), "round_date(t, \"minute\")")
  rw <- function(op, fn, na_rm, ...) ir_call(op, lapply(c(...), ir_col), meta = list(fn = fn, na_rm = na_rm))
  expect_identical(pt_text(rw("rw_agg_f64", "sum", FALSE, "x", "i")), "sum(c(x, i))")
  expect_identical(pt_text(rw("rw_agg_i32", "max", TRUE, "x", "i")), "max(c(x, i), na.rm = TRUE)")
  expect_identical(pt_text(rw("rw_agg_f64", "mean", FALSE, "x")), "mean(x)")
  rec <- ir_call("recode_lookup", list(
    pt_col("s"), ir_lit(c("ab", "cd"), "STRING"), ir_lit(c("A", "C"), "STRING"),
    ir_lit("z", "STRING"), ir_lit(NULL, "STRING", na = TRUE)
  ), meta = list(na_step = TRUE))
  expect_identical(pt_text(rec), "recode(s, ab = \"A\", cd = \"C\", .default = \"z\")")
  num <- ir_call("recode_lookup", list(
    pt_col("i"), ir_lit(1:2, "INT32"), ir_lit(c(10L, 20L), "INT32"), pt_col("i"), ir_lit(0L, "INT32")
  ), meta = list(na_step = TRUE))
  expect_identical(pt_text(num), "recode(i, `1` = 10L, `2` = 20L, .missing = 0L)")
  rep <- ir_call("replace_when", list(
    pt_col("x"), ir_call(">", list(pt_col("x"), pt_num(2))), pt_num(0),
    ir_call("<", list(pt_col("x"), pt_num(1))), pt_num(9)
  ), meta = list(n_when = 2L))
  expect_identical(pt_text(rep), "replace_when(x, x > 2 ~ 0, x < 1 ~ 9)")
  # no internal op name reaches the text
  expect_false(grepl("rw_agg|recode_lookup|as_date|floor_date_|paste_cols", pt_text(rec)))
})

test_that("a pushed reader filter prints as R code, falling back to the reader's note", {
  schema <- list(names = c("d", "i"), types = c("TIMESTAMP_DAYS", "INT32"))
  scan <- list(paths = "/tmp/dates.parquet", file_names = c("d", "i"), file_types = schema$types,
               columns = NULL, predicates = list(
                 list(text = "d > 2024-02-01", code = "d > as.Date(\"2024-02-01\")", atoms = list()),
                 list(text = "i < 5", atoms = list())
               ))
  text <- plan_node_block(ast_source(schema, scan = scan), TRUE)$text
  expect_identical(text[[length(text)]], "  reader filter: d > as.Date(\"2024-02-01\"), i < 5")
})

# --- the CPU history ----------------------------------------------------------

pt_origin <- function(verb, expr, input_block) {
  list(verb = verb, expr = expr, reason = "no GPU version.", names = "a",
       input = list(optimized = input_block, as_written = input_block))
}

test_that("a CPU step prints its verb, reason and input, and the newest three are kept", {
  schema <- list(names = "a", types = "FLOAT64")
  # blocks[[i]] is the plan of a table that i CPU steps made, each one's input the one before
  blocks <- list()
  block <- plan_block(c("table on the GPU", "  columns read: all 1"))
  for (i in 1:4) {
    origin <- pt_origin("mutate", paste0("f", i, "(a)"), block)
    block <- plan_node_block(ast_source(schema, origin = origin), TRUE)
    blocks[[i]] <- block
  }
  one <- blocks[[1L]]
  expect_identical(one$text, c("CPU fallback: mutate() for f1(a)", "  reason: no GPU version.",
                               "  columns read: all 1", "  input (computed at the mutate() call):",
                               "    table on the GPU", "      columns read: all 1"))
  expect_identical(one$level, rep(1L, 6L))
  expect_identical(one$role, c("", "", "", "input", "", ""))
  nothing <- plan_node_block(
    ast_source(schema, origin = pt_origin("mutate", NA_character_, plan_block("table on the GPU"))), TRUE)
  expect_identical(nothing$text[[1L]], "CPU fallback: mutate()")

  # three steps: nothing is cut
  expect_identical(plan_apply_cap(blocks[[3L]]), blocks[[3L]])
  # four: the oldest goes, and the input of the oldest shown becomes one line
  out <- plan_apply_cap(blocks[[4L]])$text
  heads <- trimws(out[startsWith(trimws(out), "CPU fallback: ")])
  expect_identical(heads, c("CPU fallback: mutate() for f4(a)", "CPU fallback: mutate() for f3(a)",
                            "CPU fallback: mutate() for f2(a)"))
  expect_true("input: not shown (older CPU steps)" %in% trimws(out))
  expect_false(any(grepl("table on the GPU", out, fixed = TRUE)))
})

pt_holds_ref <- function(v) {
  if (typeof(v) %in% c("externalptr", "environment", "closure", "language", "symbol", "promise")) {
    return(TRUE)
  }
  is.list(v) && any(vapply(v, pt_holds_ref, logical(1)))
}

test_that("plan_origin() records plain text only, and can never fail", {
  schema <- list(names = c("a", "b", "c"), types = rep("FLOAT64", 3L))
  plan <- ast_filter(ast_source(schema), list(pt_pred(pt_call(">", pt_col("a"), pt_num(1)))))
  fake <- structure(list(lazy_ops = plan, schema = schema, exec_mode = "lazy"), class = c("tbl_gpu", "tbl"))
  result <- structure(list(schema = list(names = c("a", "b", "c", "z"), types = rep("FLOAT64", 4L)),
                           ptr = new.env()), class = c("tbl_gpu", "tbl"))
  note <- list(verb = "mutate", expr = "pnorm(a)", reason = "no GPU version.")
  origin <- plan_origin(fake, note, result)
  expect_false(pt_holds_ref(origin))
  expect_identical(origin$verb, "mutate")
  expect_identical(origin$input$as_written$text, c("table on the GPU", "  columns read: all 3", "filter(a > 1)"))
  expect_identical(origin$input$optimized$level, c(0L, 0L, 0L))

  local_mocked_bindings(plan_node_text = function(...) stop("boom"))
  broken <- plan_origin(fake, note, result)
  expect_identical(broken$input$optimized$text, "plan could not be shown: boom")
  expect_identical(broken$input$as_written$text, "plan could not be shown: boom")
})

test_that("an origin only describes the table, and the columns, the CPU step produced", {
  schema <- list(names = c("a", "b"), types = c("FLOAT64", "FLOAT64"))
  ptr <- new.env()
  origin <- pt_origin("mutate", "f(a)", plan_block("table on the GPU"))
  origin$names <- c("a", "b")
  origin$ptr <- rlang::obj_address(ptr)
  tbl <- structure(list(schema = schema, ptr = ptr, origin = origin), class = c("tbl_gpu", "tbl"))
  expect_identical(table_origin(tbl), origin)
  other <- structure(list(schema = list(names = c("a", "c"), types = schema$types), ptr = ptr, origin = origin),
                     class = c("tbl_gpu", "tbl"))
  expect_null(table_origin(other))
  # the same columns on another GPU table (the origin was copied along by mistake)
  elsewhere <- structure(list(schema = schema, ptr = new.env(), origin = origin), class = c("tbl_gpu", "tbl"))
  expect_null(table_origin(elsewhere))
  # a table that keeps the GPU table and schema (group_by(), ungroup()) hands the origin on
  plain <- structure(list(schema = schema, ptr = ptr), class = c("tbl_gpu", "tbl"))
  expect_identical(carry_origin(plain, tbl)$origin, origin)
  expect_identical(carry_origin(plain, plain), plain)
})
