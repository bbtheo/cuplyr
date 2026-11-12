# F118: a randomized differential test of cuplyr against dplyr.
#
# A case is built from an integer seed: a random table, an optional second
# table, a few scalar variables (what `.env` reads) and a list of dplyr calls
# (the data argument left out). The same case runs on a tibble (the oracle) and
# on a tbl_gpu, eager and lazy, and the two results are compared.
#
# Function definitions only, and no testthat dependency: tools/run-fuzz.R
# sources this file outside testthat. Nothing is computed at source time.
#
# The comparator is deliberately not helper-oracle.R's expect_equal(): testthat
# 3e compares with waldo's tolerance, which reports no difference for
# compare(1L, 1), compare(NA_real_, NaN) and compare(<integer64 2^53+1>,
# <integer64 2^53>). Here equality is per column and canonical: type signature,
# then NA mask, NaN mask, then values (integer64 through character, plain
# doubles with one documented tolerance, everything else exact).
#
# Layout: 0 the session's normal form, 1 random helpers, 2 table generator, 3 expression generator, 4 verb
# generator, 5 case + code, 6 runner, 7 comparator, 8 known differences,
# 9 check / shrink / campaign, 10 quick subset.


# =============================================================================
# 0. The session's normal form
# =============================================================================
# The session state found to change a verdict is set here, so a seed gives the
# same pipeline and the same verdict whatever ran before it (not pinned yet:
# OutDec and scipen, which review round 3 measured). Called on entry to
# fuzz_case() (the generator evaluates each step and draws the next one from the
# result, so this has to hold while drawing too) and to fz_run(); the local_*
# calls undo themselves when that caller exits.
#  - lubridate loaded: loading it removes base R's tzone warning from comparing
#    POSIXct columns, and dplyr folds a verb's warnings into one condition, so
#    no text mask can undo it (see known difference `lubridate_posixct_warning`).
#  - collation "C": base min()/max()/`<` on strings sort by it.
#  - UTC and English messages: the NaN-math mask matches English text.
#  - the options that make dplyr, rlang and lifecycle warn or stay quiet.
#    dplyr.legacy_locale is unset, not FALSE: dplyr 1.2 gives a deprecation
#    warning for any non-NULL value, and that would count as the oracle warning.
fz_normal_form <- function(env = parent.frame()) {
  if (!requireNamespace("lubridate", quietly = TRUE)) {
    stop("the F118 fuzz harness needs the lubridate package: verdicts depend on whether it is loaded")
  }
  withr::local_collate("C", .local_envir = env)
  withr::local_timezone("UTC", .local_envir = env)
  withr::local_language("en", .local_envir = env)
  withr::local_options(list(
    cuplyr.fallback = "silent", cuplyr.plan_break = "silent",
    lifecycle_verbosity = "warning", warn = 0,
    dplyr.legacy_locale = NULL, rlib_warning_verbosity = "default"
  ), .local_envir = env)
  invisible()
}


# =============================================================================
# 1. Random helpers (all run inside withr::with_seed in fuzz_case())
# =============================================================================

fz_pick <- function(x, prob = NULL) x[[sample.int(length(x), 1L, prob = prob)]]
fz_chance <- function(p) stats::runif(1L) < p
fz_between <- function(lo, hi) lo + sample.int(hi - lo + 1L, 1L) - 1L

fz_family <- function(x) {
  if (inherits(x, "integer64")) "i64"
  else if (is.factor(x)) "fct"
  else if (inherits(x, "POSIXct")) "dttm"
  else if (inherits(x, "Date")) "date"
  else if (is.integer(x)) "int"
  else if (is.double(x)) "dbl"
  else if (is.logical(x)) "lgl"
  else if (is.character(x)) "chr"
  else "other"
}


# =============================================================================
# 2. Table generator
# =============================================================================

# Column `k` is a key with few distinct values (duplicates, NA); the others are
# one of each type the GPU holds. NA and NaN at chosen rates.
fz_gen_table <- function() {
  n <- fz_pick(c(0L, 1L, 2L, 3L, 5L, 8L, 12L, 20L, 40L),
               prob = c(.05, .07, .08, .08, .14, .16, .18, .14, .10))
  cols <- list(k = fz_gen_key(fz_pick(c("int", "chr", "fct"), prob = c(.5, .35, .15)),
                              n, fz_between(1L, 5L)))
  types <- c("int", "dbl", "lgl", "chr", "fct", "date", "dttm", "i64")
  prob <- c(.22, .26, .10, .14, .07, .08, .06, .07)
  for (nm in c("a", "b", "c", "d", "e")[seq_len(fz_between(1L, 5L))]) {
    cols[[nm]] <- fz_gen_column(fz_pick(types, prob = prob), n)
  }
  tibble::as_tibble(cols)
}

fz_gen_key <- function(type, n, ndist) {
  na <- if (fz_chance(.2)) .15 else 0
  v <- if (type == "int") sample.int(ndist, n, TRUE) else letters[sample.int(ndist, n, TRUE)]
  v[stats::runif(n) < na] <- NA
  if (type == "fct") factor(v, levels = letters[seq_len(ndist + fz_between(0L, 1L))]) else v
}

fz_gen_column <- function(type, n) {
  na <- if (fz_chance(.4)) fz_pick(c(.1, .25, .5)) else 0
  hole <- stats::runif(n) < na
  switch(type,
    int = {
      pool <- if (fz_chance(.1)) c(-1L, 0L, 1L, 2147483647L) else -3L:9L
      v <- sample(pool, n, TRUE)
      v[hole] <- NA
      v
    },
    dbl = {
      v <- switch(fz_pick(c("half", "round", "wide")),
        half = sample(-8:16, n, TRUE) / 4,
        round = round(stats::runif(n, -5, 10), 2),
        wide = sample(c(0, 1, -1, 0.5, 1e6, 1e-3, 123456.789), n, TRUE)
      )
      if (fz_chance(.15)) {
        idx <- which(stats::runif(n) < .2)
        v[idx] <- sample(c(NaN, NaN, Inf, -Inf), length(idx), TRUE)
      }
      v[hole] <- NA
      v
    },
    lgl = {
      v <- sample(c(TRUE, FALSE), n, TRUE)
      v[hole] <- NA
      v
    },
    chr = {
      pool <- fz_pick(list(letters[1:4], c("a", "B", "c", "D"),
                           c("", "a", "bb", "abc"), c("x y", "y", "z")))
      v <- sample(pool, n, TRUE)
      v[hole] <- NA
      v
    },
    fct = {
      lv <- fz_pick(list(letters[1:3], c("lo", "mid", "hi"), letters[1:5]))
      v <- sample(lv[seq_len(length(lv) - fz_between(0L, 1L))], n, TRUE)
      v[hole] <- NA
      factor(v, levels = lv, ordered = fz_chance(.2))
    },
    date = {
      v <- 18262 + sample(0:60, n, TRUE)
      v[hole] <- NA
      structure(as.numeric(v), class = "Date")
    },
    dttm = {
      v <- 1.5778e9 + sample.int(86400L * 5L, n, TRUE)
      if (fz_chance(.15)) v <- v + 0.25
      v[hole] <- NA
      .POSIXct(as.numeric(v), tz = fz_pick(c("UTC", "Asia/Tokyo"), prob = c(.8, .2)))
    },
    i64 = {
      v <- sample(c("0", "1", "2", "3", "9007199254740993", "-9007199254740993",
                    "4611686018427387905"), n, TRUE)
      v[hole] <- NA
      bit64::as.integer64(v)
    }
  )
}

# The second table of a join / bind_rows step, built from the table the step
# will see (`cur`), so keys can match.
fz_gen_y_join <- function(cur, key) {
  pool <- cur[[key]]
  pool <- pool[!duplicated(pool)]
  n <- if (length(pool) == 0L) 0L else fz_between(0L, 8L)
  kv <- if (n == 0L) pool[0] else pool[sample.int(length(pool), n, TRUE)]
  if (n > 0L && fz_chance(.25) && (is.numeric(kv) || is.character(kv)) && !inherits(kv, "integer64")) {
    idx <- which(stats::runif(n) < .3)
    kv[idx] <- if (is.numeric(kv)) kv[idx] + 100 else "zz"
  }
  y <- list()
  y[[key]] <- kv
  extra <- c("j1", "j2")[seq_len(fz_between(1L, 2L))]
  if (fz_chance(.15) && ncol(cur) > 1L) extra[1L] <- fz_pick(setdiff(names(cur), key))
  for (nm in extra) {
    y[[nm]] <- fz_gen_column(fz_pick(c("int", "dbl", "chr", "lgl", "date"),
                                     prob = c(.3, .3, .2, .1, .1)), n)
  }
  tibble::as_tibble(y)
}

fz_gen_y_bind <- function(cur) {
  n <- fz_between(0L, 6L)
  rows <- if (nrow(cur) == 0L) integer() else sample.int(nrow(cur), n, TRUE)
  y <- cur[rows, , drop = FALSE]
  y <- dplyr::ungroup(y)
  nms <- names(y)
  if (length(nms) > 1L && fz_chance(.3)) y <- y[, sample(nms), drop = FALSE]
  if (length(nms) > 1L && fz_chance(.2)) y <- y[, -fz_between(1L, length(nms)), drop = FALSE]
  if (fz_chance(.2)) y$extra <- sample(c(1L, 2L), nrow(y), TRUE)
  if (fz_chance(.1) && ncol(y) > 0L) y[[1L]] <- as.character(y[[1L]])
  y
}


# =============================================================================
# 3. Expression generator
# =============================================================================
# `st` is an environment: cols (named chr: column -> family), groups, vars
# (what `.env` reads), nvar, window_ok / agg_ok (which calls make sense).

fz_cols_of <- function(st, fams) names(st$cols)[st$cols %in% fams]

fz_var <- function(st, value) {
  st$nvar <- st$nvar + 1L
  nm <- paste0("v", st$nvar)
  st$vars[[nm]] <- value
  nm
}

# A column reference, in the forms real bugs lived in: bare, .data$x,
# .data[["x"]], and .data[[s]] with s a variable holding the name.
fz_ref <- function(st, name, bare = .78) {
  r <- stats::runif(1L)
  if (r < bare) as.name(name)
  else if (r < bare + .08) call("$", quote(.data), as.name(name))
  else if (r < bare + .15) call("[[", quote(.data), name)
  else call("[[", quote(.data), as.name(fz_var(st, name)))
}

fz_lit <- function(fam) {
  switch(fam,
    int = fz_between(-2L, 6L),
    dbl = fz_pick(c(0, .5, 1, 2.5, -1, 10, 1e6)),
    lgl = fz_chance(.5),
    chr = fz_pick(c("a", "b", "c", "x", "")),
    date = rlang::call2("as.Date", fz_pick(c("2020-01-05", "2020-01-20", "2020-02-10"))),
    dttm = rlang::call2("as.POSIXct", "2020-01-03 12:00:00", tz = "UTC"),
    i64 = rlang::call2(quote(bit64::as.integer64), fz_pick(c(0L, 1L, 2L, 3L)))
  )
}

# A scalar of a family: a literal, or a variable read through `.env$v` /
# `.env[[nm]]` (nm a variable holding the name).
fz_scalar <- function(st, fam) {
  lit <- fz_lit(fam)
  if (!fz_chance(.22)) return(lit)
  nm <- fz_var(st, eval(lit))
  if (fz_chance(.6)) call("$", quote(.env), as.name(nm))
  else call("[[", quote(.env), as.name(fz_var(st, nm)))
}

fz_weighted <- function(w) {
  if (sum(w) == 0) return(names(w)[1L])
  names(w)[sample.int(length(w), 1L, prob = w)]
}

gx_num <- function(st, depth = 0L) {
  cols <- fz_cols_of(st, c("int", "dbl"))
  w <- c(col = if (length(cols)) .36 else 0, lit = .06, arith = .18, unary = .07,
         ifelse = .06, coalesce = .05, casewhen = .03,
         window = if (st$window_ok) .07 else 0, agg = if (st$agg_ok) .08 else 0,
         misc = .04)
  if (depth >= 2L) w[c("arith", "unary", "ifelse", "coalesce", "casewhen", "window", "agg", "misc")] <- 0
  if (depth == 1L) w[c("window", "agg")] <- w[c("window", "agg")] / 2
  if (sum(w) == 0) w[["lit"]] <- 1
  switch(fz_weighted(w),
    col = fz_ref(st, fz_pick(cols)),
    lit = fz_scalar(st, fz_pick(c("int", "dbl"), prob = c(.6, .4))),
    arith = {
      op <- fz_pick(c("+", "-", "*", "/", "%%", "%/%"), prob = c(.3, .2, .25, .12, .07, .06))
      call(op, gx_num(st, depth + 1L), gx_num(st, depth + 1L))
    },
    unary = {
      x <- gx_num(st, depth + 1L)
      switch(fz_pick(c("abs", "neg", "round", "sqrtabs", "sqrt"), prob = c(.3, .25, .25, .12, .08)),
        abs = call("abs", x), neg = call("-", x), round = call("round", x, 1L),
        sqrtabs = call("sqrt", call("abs", x)), sqrt = call("sqrt", x))
    },
    ifelse = rlang::call2("if_else", gx_lgl(st, depth + 1L), gx_num(st, depth + 1L), gx_num(st, depth + 1L)),
    coalesce = rlang::call2("coalesce", gx_num(st, depth + 1L), gx_num(st, 2L)),
    casewhen = rlang::call2("case_when",
                            call("~", gx_lgl(st, depth + 1L), gx_num(st, 2L)),
                            call("~", gx_lgl(st, depth + 1L), gx_num(st, 2L)),
                            .default = gx_num(st, 2L)),
    window = {
      x <- gx_num(st, depth + 1L)
      switch(fz_pick(c("lag", "lead", "lag2", "cumsum", "cummax", "cummin", "row_number", "min_rank", "dense_rank", "n", "ntile")),
        lag = rlang::call2("lag", x), lead = rlang::call2("lead", x),
        lag2 = rlang::call2("lag", x, 2L, default = 0L),
        cumsum = rlang::call2("cumsum", x), cummax = rlang::call2("cummax", x),
        cummin = rlang::call2("cummin", x), dense_rank = rlang::call2("dense_rank", x),
        row_number = rlang::call2("row_number"), min_rank = rlang::call2("min_rank", x),
        n = rlang::call2("n"), ntile = rlang::call2("ntile", x, 2L))
    },
    agg = {
      x <- gx_num(st, depth + 1L)
      fn <- fz_pick(c("sum", "mean", "min", "max", "median"))
      if (fz_chance(.5)) rlang::call2(fn, x, na.rm = TRUE) else rlang::call2(fn, x)
    },
    misc = {
      chrs <- fz_cols_of(st, "chr")
      switch(fz_pick(c("pmax", "pmin", "na_if", "as_int", "nchar")),
        pmax = rlang::call2("pmax", gx_num(st, 1L), gx_num(st, 2L)),
        pmin = rlang::call2("pmin", gx_num(st, 1L), gx_num(st, 2L), na.rm = TRUE),
        na_if = rlang::call2("na_if", gx_num(st, 1L), fz_between(0L, 2L)),
        as_int = rlang::call2("as.integer", gx_num(st, 1L)),
        nchar = if (length(chrs)) rlang::call2("nchar", fz_ref(st, fz_pick(chrs))) else gx_num(st, 2L))
    }
  )
}

gx_lgl <- function(st, depth = 0L) {
  nums <- fz_cols_of(st, c("int", "dbl")); chrs <- fz_cols_of(st, "chr")
  fcts <- fz_cols_of(st, "fct"); dates <- fz_cols_of(st, "date")
  dttms <- fz_cols_of(st, "dttm"); lgls <- fz_cols_of(st, "lgl")
  i64s <- fz_cols_of(st, "i64"); all <- names(st$cols)
  w <- c(cmp_num = if (length(nums)) .30 else 0, cmp_chr = if (length(chrs)) .08 else 0,
         cmp_fct = if (length(fcts)) .05 else 0, cmp_date = if (length(dates)) .05 else 0,
         cmp_dttm = if (length(dttms)) .03 else 0, cmp_i64 = if (length(i64s)) .03 else 0,
         is_na = if (length(all)) .12 else 0,
         in_set = if (length(nums) + length(chrs)) .09 else 0, logic = .16,
         col = if (length(lgls)) .06 else 0, between = if (length(nums)) .03 else 0,
         if_any = if (length(all)) .04 else 0, text = if (length(chrs)) .04 else 0)
  if (depth >= 2L) w[c("logic", "if_any", "between")] <- 0
  if (sum(w) == 0) w[["logic"]] <- 1
  cmp <- function() fz_pick(c("==", "!=", "<", "<=", ">", ">="))
  switch(fz_weighted(w),
    cmp_num = call(cmp(), gx_num(st, depth + 1L), gx_num(st, 2L)),
    cmp_chr = call(fz_pick(c("==", "!=", "==", "<")), fz_ref(st, fz_pick(chrs)), fz_scalar(st, "chr")),
    cmp_fct = if (fz_chance(.6)) {
      call(fz_pick(c("==", "!=")), fz_ref(st, fz_pick(fcts)), fz_lit("chr"))
    } else {
      call("%in%", fz_ref(st, fz_pick(fcts)), rlang::call2("c", fz_lit("chr"), fz_lit("chr")))
    },
    cmp_date = call(cmp(), fz_ref(st, fz_pick(dates)), fz_scalar(st, "date")),
    cmp_dttm = call(cmp(), fz_ref(st, fz_pick(dttms)), fz_lit("dttm")),
    cmp_i64 = call(fz_pick(c("==", "!=", "<", ">")), fz_ref(st, fz_pick(i64s)), fz_lit("i64")),
    is_na = call("is.na", fz_ref(st, fz_pick(all))),
    in_set = {
      col <- fz_pick(c(nums, chrs))
      vals <- if (col %in% nums) {
        c(sample(-1:5, fz_between(1L, 3L)), if (fz_chance(.2)) NA)
      } else {
        c(sample(c("a", "b", "c", "x"), fz_between(1L, 3L)), if (fz_chance(.2)) NA)
      }
      set <- if (fz_chance(.25)) call("$", quote(.env), as.name(fz_var(st, vals)))
             else as.call(c(list(as.name("c")), as.list(vals)))
      call("%in%", fz_ref(st, col), set)
    },
    logic = switch(fz_pick(c("and", "or", "not"), prob = c(.4, .35, .25)),
      and = call("&", gx_lgl(st, depth + 1L), gx_lgl(st, depth + 1L)),
      or = call("|", gx_lgl(st, depth + 1L), gx_lgl(st, depth + 1L)),
      not = call("!", gx_lgl(st, depth + 1L))),
    col = fz_ref(st, fz_pick(lgls)),
    text = if (fz_chance(.5)) rlang::call2("grepl", fz_pick(c("a", "b", "x")), fz_ref(st, fz_pick(chrs)))
           else rlang::call2("startsWith", fz_ref(st, fz_pick(chrs)), fz_pick(c("a", "b", ""))),
    between = rlang::call2("between", fz_ref(st, fz_pick(nums)), -1L, 4L),
    if_any = {
      sel <- fz_selection(st, all)
      switch(fz_pick(c("na", "num")),
        na = rlang::call2(fz_pick(c("if_any", "if_all")), sel, quote(is.na)),
        num = rlang::call2(fz_pick(c("if_any", "if_all")), sel, quote(~ !is.na(.x))))
    }
  )
}

gx_chr <- function(st, depth = 0L) {
  chrs <- fz_cols_of(st, "chr"); fcts <- fz_cols_of(st, "fct"); nums <- fz_cols_of(st, "int")
  w <- c(col = if (length(chrs)) .40 else 0, lit = .12, coalesce = if (length(chrs)) .10 else 0,
         ifelse = .10, casewhen = .06, upper = if (length(chrs)) .06 else 0,
         paste = if (length(chrs)) .06 else 0, asint = if (length(nums)) .05 else 0,
         asfct = if (length(fcts)) .05 else 0)
  if (depth >= 1L) w[c("coalesce", "ifelse", "casewhen", "upper", "paste")] <- 0
  switch(fz_weighted(w),
    col = fz_ref(st, fz_pick(chrs)),
    lit = fz_scalar(st, "chr"),
    coalesce = rlang::call2("coalesce", fz_ref(st, fz_pick(chrs)), fz_lit("chr")),
    ifelse = rlang::call2("if_else", gx_lgl(st, 1L), gx_chr(st, 1L), gx_chr(st, 1L)),
    casewhen = rlang::call2("case_when", call("~", gx_lgl(st, 1L), gx_chr(st, 1L)), .default = gx_chr(st, 1L)),
    upper = rlang::call2("toupper", fz_ref(st, fz_pick(chrs))),
    paste = rlang::call2("paste0", fz_ref(st, fz_pick(chrs)), fz_lit("chr")),
    asint = rlang::call2("as.character", fz_ref(st, fz_pick(nums))),
    asfct = rlang::call2("as.character", fz_ref(st, fz_pick(fcts)))
  )
}

# Date and POSIXct share one generator: `fam` picks the columns and literals.
gx_time <- function(st, fam, depth = 0L) {
  cols <- fz_cols_of(st, fam)
  w <- c(col = if (length(cols)) .45 else 0, lit = .10, shift = if (fam == "date" && length(cols)) .15 else 0,
         ifelse = .12, coalesce = if (length(cols)) .10 else 0,
         lag = if (length(cols) && st$window_ok) .08 else 0)
  if (depth >= 1L) w[c("shift", "ifelse", "coalesce", "lag")] <- 0
  if (sum(w) == 0) w[["lit"]] <- 1
  switch(fz_weighted(w),
    col = fz_ref(st, fz_pick(cols)),
    lit = fz_lit(fam),
    shift = call(fz_pick(c("+", "-")), fz_ref(st, fz_pick(cols)), fz_between(1L, 5L)),
    ifelse = rlang::call2("if_else", gx_lgl(st, 1L), gx_time(st, fam, 1L), gx_time(st, fam, 1L)),
    coalesce = rlang::call2("coalesce", fz_ref(st, fz_pick(cols)), fz_lit(fam)),
    lag = rlang::call2(fz_pick(c("lag", "lead")), fz_ref(st, fz_pick(cols)))
  )
}

gx_any <- function(st, fam) {
  switch(fam,
    num = gx_num(st), lgl = gx_lgl(st), chr = gx_chr(st),
    date = gx_time(st, "date"), dttm = gx_time(st, "dttm"),
    fct = {
      cols <- fz_cols_of(st, "fct")
      if (length(cols) && fz_chance(.5)) fz_ref(st, fz_pick(cols))
      else rlang::call2("factor", gx_chr(st, 1L))
    },
    i64 = {
      cols <- fz_cols_of(st, "i64")
      if (length(cols) == 0L) fz_lit("i64")
      else if (fz_chance(.5)) fz_ref(st, fz_pick(cols))
      else rlang::call2("coalesce", fz_ref(st, fz_pick(cols)), fz_lit("i64"))
    }
  )
}

# A set of columns written the way users write them.
fz_selection <- function(st, pool) {
  switch(fz_pick(c("where", "c", "all_of", "everything"), prob = c(.35, .4, .1, .15)),
    where = rlang::call2("where", fz_pick(c(quote(is.numeric), quote(is.character),
                                            quote(is.logical), quote(is.integer)))),
    c = as.call(c(list(as.name("c")),
                  lapply(sample(pool, min(length(pool), fz_between(1L, 3L))), as.name))),
    all_of = rlang::call2("all_of", sample(pool, min(length(pool), fz_between(1L, 2L)))),
    everything = rlang::call2("everything")
  )
}

# One aggregate for summarise().
gx_agg <- function(st) {
  nums <- fz_cols_of(st, c("int", "dbl")); lgls <- fz_cols_of(st, "lgl")
  chrs <- fz_cols_of(st, "chr"); dates <- fz_cols_of(st, "date")
  dttms <- fz_cols_of(st, "dttm"); fcts <- fz_cols_of(st, "fct")
  i64s <- fz_cols_of(st, "i64"); all <- names(st$cols)
  w <- c(num = if (length(nums)) .55 else 0, n = .12, lgl = if (length(lgls)) .06 else 0,
         chr = if (length(chrs)) .06 else 0, date = if (length(dates)) .06 else 0,
         dttm = if (length(dttms)) .03 else 0, fct = if (length(fcts)) .03 else 0,
         i64 = if (length(i64s)) .03 else 0, cond = if (length(nums)) .06 else 0)
  narm <- function(fn, x) if (fz_chance(.55)) rlang::call2(fn, x, na.rm = TRUE) else rlang::call2(fn, x)
  # no aggregate or window call inside an aggregate. `st` is an environment shared
  # with the nested gx_num() calls (they write st$vars / st$nvar), so the flags are
  # switched off in place and put back when this step's aggregate is built.
  was <- c(agg = st$agg_ok, window = st$window_ok)
  st$agg_ok <- FALSE; st$window_ok <- FALSE
  on.exit({ st$agg_ok <- was[["agg"]]; st$window_ok <- was[["window"]] }, add = TRUE)
  switch(fz_weighted(w),
    num = {
      x <- if (fz_chance(.75)) fz_ref(st, fz_pick(nums)) else gx_num(st, 1L)
      fn <- fz_pick(c("sum", "mean", "min", "max", "median", "sd", "var", "n_distinct", "first", "last"),
                    prob = c(.22, .2, .12, .12, .07, .06, .03, .06, .06, .06))
      if (fn %in% c("first", "last")) rlang::call2(fn, x) else narm(fn, x)
    },
    n = if (fz_chance(.8)) rlang::call2("n") else rlang::call2("n_distinct", fz_ref(st, fz_pick(all))),
    lgl = narm(fz_pick(c("any", "all", "sum", "mean")), fz_ref(st, fz_pick(lgls))),
    chr = {
      fn <- fz_pick(c("min", "max", "first", "last", "n_distinct"))
      if (fn %in% c("first", "last")) rlang::call2(fn, fz_ref(st, fz_pick(chrs))) else narm(fn, fz_ref(st, fz_pick(chrs)))
    },
    date = {
      fn <- fz_pick(c("min", "max", "first", "last", "mean"), prob = c(.3, .3, .15, .15, .1))
      if (fn %in% c("first", "last")) rlang::call2(fn, fz_ref(st, fz_pick(dates))) else narm(fn, fz_ref(st, fz_pick(dates)))
    },
    dttm = narm(fz_pick(c("min", "max")), fz_ref(st, fz_pick(dttms))),
    fct = rlang::call2(fz_pick(c("first", "last")), fz_ref(st, fz_pick(fcts))),
    i64 = rlang::call2(fz_pick(c("first", "last", "n_distinct")), fz_ref(st, fz_pick(i64s))),
    cond = narm("sum", gx_lgl(st, 1L))
  )
}


# =============================================================================
# 4. Verb generator
# =============================================================================
# Each fz_step_*() returns list(step =, y =): the call with the data argument
# omitted, and the second table if the step needs one.

fz_newname <- function(st, prefix = "m") {
  st$nname <- st$nname + 1L
  paste0(prefix, st$nname)
}

fz_call <- function(verb, args) rlang::call2(verb, !!!args)

# group_by/count/distinct keys: a column, or a small expression
fz_group_args <- function(st, n) {
  keys <- names(st$cols)
  if (length(keys) == 0L) return(list())
  prob <- ifelse(st$cols %in% c("int", "chr", "fct", "lgl", "date"), 3, 1)
  pick <- unique(keys[sample.int(length(keys), min(n, length(keys)), prob = prob)])
  args <- lapply(pick, function(nm) fz_ref(st, nm, bare = .88))
  names(args) <- rep("", length(args))
  if (fz_chance(.1) && length(fz_cols_of(st, c("int", "dbl")))) {
    args <- c(args[0], stats::setNames(list(call(">", gx_num(st, 1L), 1L)), fz_newname(st, "g")))
  }
  args
}

fz_by_arg <- function(st, args) {
  if (length(st$groups) == 0L && length(st$cols) && fz_chance(.12)) {
    args[[".by"]] <- as.name(fz_pick(names(st$cols)))
  }
  args
}

fz_step_filter <- function(st) {
  args <- lapply(seq_len(fz_pick(1:2, prob = c(.75, .25))), function(i) gx_lgl(st))
  if (fz_chance(.07)) {
    args <- list(call(fz_pick(c("<=", "==", ">")), rlang::call2("row_number"), fz_between(1L, 3L)))
  }
  list(step = fz_call("filter", fz_by_arg(st, args)))
}

fz_step_mutate <- function(st, verb = "mutate") {
  args <- list()
  if (fz_chance(.12) && length(st$cols)) {
    # across()
    grp <- fz_pick(c("num", "chr", "lgl", "date"), prob = c(.6, .15, .1, .15))
    pool <- fz_cols_of(st, switch(grp, num = c("int", "dbl"), chr = "chr", lgl = "lgl", date = "date"))
    if (length(pool) == 0L) pool <- names(st$cols)
    fn <- switch(grp,
      num = fz_pick(list(quote(~ .x + 1L), quote(~ .x * 2), quote(~ coalesce(.x, 0)), quote(abs),
                         quote(~ .x - mean(.x, na.rm = TRUE)), quote(~ if_else(is.na(.x), 0, .x)))),
      chr = fz_pick(list(quote(toupper), quote(~ coalesce(.x, "z")))),
      lgl = quote(~ !.x),
      date = quote(~ .x + 1L))
    sel <- if (fz_chance(.5) && grp == "num") quote(where(is.numeric)) else fz_selection(st, pool)
    acr <- rlang::call2("across", sel, fn)
    if (fz_chance(.3)) acr$.names <- "{.col}_n"
    args <- list(acr)
  } else {
    for (i in seq_len(fz_pick(1:3, prob = c(.6, .3, .1)))) {
      fam <- fz_pick(c("num", "lgl", "chr", "date", "dttm", "fct", "i64"),
                     prob = c(.5, .16, .12, .07, .04, .04, .03))
      nm <- if (length(st$cols) && fz_chance(.25)) fz_pick(names(st$cols)) else fz_newname(st)
      args[[nm]] <- if (fz_chance(.06)) rlang::call2("row_number") else gx_any(st, fam)
    }
  }
  if (verb == "mutate") {
    if (fz_chance(.08)) args[[".keep"]] <- fz_pick(c("none", "used", "unused"))
    if (fz_chance(.08) && length(st$cols)) {
      args[[fz_pick(c(".before", ".after"))]] <- as.name(fz_pick(names(st$cols)))
    }
  }
  list(step = fz_call(verb, fz_by_arg(st, args)))
}

fz_step_select <- function(st) {
  all <- names(st$cols)
  args <- switch(fz_pick(c("names", "neg", "where", "starts", "everything", "rename", "any_of", "range")),
    names = lapply(sample(all, fz_between(1L, length(all))), as.name),
    neg = list(call("-", as.name(fz_pick(all)))),
    where = list(rlang::call2("where", fz_pick(c(quote(is.numeric), quote(is.character))))),
    starts = list(rlang::call2("starts_with", fz_pick(c("m", "a", "j")))),
    everything = list(rlang::call2("everything")),
    rename = {
      args <- lapply(sample(all, fz_between(1L, length(all))), as.name)
      names(args) <- c(fz_newname(st, "r"), rep("", length(args) - 1L))
      args
    },
    any_of = list(rlang::call2("any_of", c(sample(all, 1L), "zz_absent"))),
    range = if (length(all) > 1L) list(call(":", as.name(all[1L]), as.name(all[length(all)]))) else list()
  )
  list(step = fz_call("select", args))
}

fz_step_rename <- function(st) {
  all <- names(st$cols)
  if (fz_chance(.25)) {
    return(list(step = if (fz_chance(.6)) rlang::call2("rename_with", quote(toupper))
                       else rlang::call2("rename_with", quote(~ paste0(.x, "_r")),
                                         rlang::call2("where", quote(is.numeric)))))
  }
  from <- sample(all, min(length(all), fz_between(1L, 2L)))
  args <- lapply(from, as.name)
  names(args) <- vapply(from, function(f) fz_newname(st, "r"), "")
  list(step = fz_call("rename", args))
}

fz_step_relocate <- function(st) {
  all <- names(st$cols)
  what <- if (fz_chance(.25)) rlang::call2("where", quote(is.numeric))
          else if (fz_chance(.3) && length(all) > 1L) call("c", as.name(all[1L]), as.name(all[length(all)]))
          else as.name(fz_pick(all))
  args <- list(what)
  r <- fz_pick(c("none", "before", "after", "last"), prob = c(.35, .25, .2, .2))
  if (r == "before") args$.before <- as.name(fz_pick(all))
  if (r == "after") args$.after <- as.name(fz_pick(all))
  if (r == "last") args$.after <- rlang::call2("last_col")
  list(step = fz_call("relocate", args))
}

fz_step_arrange <- function(st) {
  all <- names(st$cols)
  args <- lapply(seq_len(fz_pick(1:2, prob = c(.7, .3))), function(i) {
    key <- if (fz_chance(.85)) fz_ref(st, fz_pick(all)) else gx_num(st, 1L)
    if (fz_chance(.35)) rlang::call2("desc", key) else key
  })
  if (length(st$groups) && fz_chance(.25)) args$.by_group <- TRUE
  list(step = fz_call("arrange", args))
}

fz_step_distinct <- function(st) {
  args <- if (fz_chance(.25)) list() else fz_group_args(st, fz_between(1L, 2L))
  if (length(args) && fz_chance(.3)) args$.keep_all <- TRUE
  list(step = fz_call("distinct", args))
}

fz_step_group_by <- function(st) {
  args <- fz_group_args(st, fz_between(1L, 2L))
  if (length(st$groups) && fz_chance(.2)) args$.add <- TRUE
  has_fct <- any(st$cols %in% "fct")
  if (fz_chance(if (has_fct) .4 else .15)) args$.drop <- FALSE
  list(step = fz_call("group_by", args))
}

fz_step_ungroup <- function(st) {
  list(step = if (length(st$groups) && fz_chance(.3)) rlang::call2("ungroup", as.name(fz_pick(st$groups)))
              else rlang::call2("ungroup"))
}

fz_step_summarise <- function(st) {
  args <- list()
  if (fz_chance(.1) && length(fz_cols_of(st, c("int", "dbl")))) {
    fn <- fz_pick(list(quote(~ mean(.x, na.rm = TRUE)), quote(sum), quote(max),
                       quote(list(mn = min, mx = max))))
    sel <- if (fz_chance(.5)) quote(where(is.numeric)) else fz_selection(st, fz_cols_of(st, c("int", "dbl")))
    args <- list(rlang::call2("across", sel, fn))
  } else {
    for (i in seq_len(fz_pick(1:3, prob = c(.5, .35, .15)))) args[[fz_newname(st, "s")]] <- gx_agg(st)
  }
  if (length(st$groups) && fz_chance(.25)) args$.groups <- fz_pick(c("drop", "keep", "drop_last"))
  list(step = fz_call("summarise", fz_by_arg(st, args)))
}

fz_step_count <- function(st) {
  args <- if (fz_chance(.15)) list() else fz_group_args(st, fz_between(1L, 2L))
  if (fz_chance(.2)) args$sort <- TRUE
  nums <- fz_cols_of(st, c("int", "dbl"))
  if (fz_chance(.2) && length(nums)) args$wt <- as.name(fz_pick(nums))
  if (fz_chance(.1)) args$name <- "N"
  list(step = fz_call("count", args))
}

fz_step_slice <- function(st) {
  all <- names(st$cols)
  by <- if (length(st$groups) == 0L && fz_chance(.12)) as.name(fz_pick(all)) else NULL
  step <- switch(fz_pick(c("slice", "head", "tail", "min", "max")),
    slice = {
      idx <- fz_pick(list(1L, 1:2, c(1L, 3L), -1L, quote(n()), 0L, 2:1, quote(1:2 + 1L)))
      args <- list(idx)
      if (!is.null(by)) args$.by <- by
      fz_call("slice", args)
    },
    head = , tail = {
      fn <- paste0("slice_", if (fz_chance(.5)) "head" else "tail")
      args <- if (fz_chance(.7)) list(n = fz_between(0L, 3L)) else list(prop = fz_pick(c(0.25, 0.5, 1)))
      if (!is.null(by)) args$by <- by
      fz_call(fn, args)
    },
    min = , max = {
      fn <- paste0("slice_", if (fz_chance(.5)) "min" else "max")
      ord <- fz_cols_of(st, c("int", "dbl", "date", "chr"))
      key <- if (length(ord)) fz_ref(st, fz_pick(ord)) else as.name(fz_pick(all))
      args <- list(key)
      if (fz_chance(.7)) args$n <- fz_between(1L, 3L) else args$prop <- 0.5
      if (fz_chance(.4)) args$with_ties <- FALSE
      if (fz_chance(.15)) args$na_rm <- TRUE
      if (!is.null(by)) args$by <- by
      fz_call(fn, args)
    }
  )
  list(step = step)
}

# `cur` is the table the step will see: keys of y are drawn from it.
fz_step_join <- function(st, cur) {
  keys <- names(st$cols)[st$cols %in% c("int", "chr", "fct", "date", "lgl", "dbl")]
  if (length(keys) == 0L) keys <- names(st$cols)
  key <- if ("k" %in% keys && fz_chance(.7)) "k" else fz_pick(keys)
  y <- fz_gen_y_join(cur, key)
  verb <- fz_pick(c("left_join", "inner_join", "right_join", "full_join", "semi_join", "anti_join"),
                  prob = c(.3, .25, .1, .1, .12, .13))
  by <- switch(fz_pick(c("chr", "join_by", "named", "eq", "noneq"), prob = c(.35, .25, .15, .2, .05)),
    chr = key,
    join_by = rlang::call2("join_by", as.name(key)),
    named = stats::setNames(key, key),
    eq = rlang::call2("join_by", call("==", as.name(key), as.name(key))),
    noneq = rlang::call2("join_by", call(">=", as.name(key), as.name(key)))
  )
  args <- list(as.name("y"), by = by)
  if (!(verb %in% c("semi_join", "anti_join"))) {
    if (fz_chance(.1)) args$suffix <- c(".l", ".r")
    if (fz_chance(.08)) args$keep <- TRUE
    if (fz_chance(.06)) args$relationship <- fz_pick(c("many-to-one", "one-to-many", "many-to-many"))
    if (fz_chance(.06)) args$na_matches <- "never"
  }
  list(step = fz_call(verb, args), y = y)
}

fz_step_bind <- function(st, cur) {
  args <- list(as.name("y"))
  if (fz_chance(.2)) args$.id <- "src"
  list(step = fz_call("bind_rows", args), y = fz_gen_y_bind(cur))
}

# Replace the first column reference in the call by one that does not exist,
# or give up (the step then stays valid).
fz_make_invalid <- function(step, cols) {
  done <- FALSE
  walk <- function(e) {
    if (done) return(e)
    if (is.symbol(e) && as.character(e) %in% cols) {
      done <<- TRUE
      return(as.name("no_such_col"))
    }
    if (is.call(e)) {
      for (i in seq_along(e)[-1L]) {
        v <- walk(e[[i]])
        if (!is.null(v)) e[[i]] <- v
      }
    }
    e
  }
  out <- walk(step)
  if (done) out else NULL
}


# =============================================================================
# 5. Cases and code
# =============================================================================

fuzz_make_case <- function(x, steps, y = NULL, vars = list(), seed = NA_integer_) {
  if (is.data.frame(x) && !tibble::is_tibble(x)) x <- tibble::as_tibble(x)
  if (is.data.frame(y) && !tibble::is_tibble(y)) y <- tibble::as_tibble(y)
  structure(list(seed = seed, x = x, y = y, steps = steps, vars = vars), class = "fuzz_case")
}

# What a generated call resolves its functions in. Its parent is the global env
# so the calls count as user code: dplyr only gives its many-to-many join warning
# to a caller whose topenv is global (dplyr:::is_direct(), ported as
# join_is_direct()), and a dplyr namespace parent silenced that on both sides.
# It holds base, stats, then dplyr's exports (so `lag` is dplyr's) because
# anything left to the search path would see whatever the session attached:
# bit64 masks `%in%`. Built once per sourcing of this file.
fz_lookup_env <- local({
  shim <- NULL
  function() {
    if (is.null(shim)) {
      shim <<- new.env(parent = globalenv())
      list2env(mget(ls(baseenv(), all.names = TRUE), envir = baseenv()), envir = shim)
      for (pkg in c("stats", "dplyr")) {
        for (nm in getNamespaceExports(pkg)) assign(nm, getExportedValue(pkg, nm), envir = shim)
      }
    }
    shim
  }
})

# The environment a case's calls are evaluated in: the lookup env above, the
# case's variables and `y`, and cuplyr's bind_rows (which routes non-GPU input to
# dplyr unchanged).
fz_env <- function(case, y = case$y) {
  env <- new.env(parent = fz_lookup_env())
  for (nm in names(case$vars)) assign(nm, case$vars[[nm]], envir = env)
  env$y <- y
  env$bind_rows <- getExportedValue("cuplyr", "bind_rows")
  env
}

fz_apply_step <- function(tbl, step, env) {
  assign(".tbl", tbl, envir = env)
  eval(as.call(c(list(step[[1L]], quote(.tbl)), as.list(step)[-1L])), env)
}

# Only the variables the steps read: by name, or by the name another variable
# holds (`.env[[nm]]`).
fz_used_vars <- function(steps, vars) {
  used <- unique(all.names(as.call(c(list(quote(list)), steps))))
  names_held <- unlist(lapply(vars[names(vars) %in% used], function(v) if (is.character(v)) v))
  vars[names(vars) %in% c(used, names_held)]
}

fuzz_case <- function(seed) {
  fz_normal_form()
  seed <- as.integer(seed)
  withr::with_seed(seed, fz_build_case(seed),
                   .rng_kind = "Mersenne-Twister", .rng_normal_kind = "Inversion",
                   .rng_sample_kind = "Rejection")
}

fz_build_case <- function(seed) {
  x <- fz_gen_table()
  cur <- x
  st <- new.env()
  st$vars <- list(); st$nvar <- 0L; st$nname <- 0L; st$window_ok <- TRUE; st$agg_ok <- TRUE
  steps <- list()
  y <- NULL
  verbs <- c("filter", "mutate", "select", "rename", "relocate", "arrange", "distinct",
             "group_by", "ungroup", "summarise", "count", "slice", "join", "bind_rows", "transmute")
  vprob <- c(.16, .20, .06, .04, .03, .08, .04, .08, .03, .09, .05, .07, .08, .04, .02)
  for (i in seq_len(fz_pick(1:6, prob = c(.12, .2, .24, .2, .14, .1)))) {
    stop_here <- FALSE
    for (attempt in 1:3) {
      st$cols <- vapply(cur, fz_family, "")
      st$groups <- dplyr::group_vars(cur)
      if (length(st$cols) == 0L) {
        stop_here <- TRUE
        break
      }
      verb <- fz_pick(verbs, prob = vprob)
      if (verb == "bind_rows" && !is.null(y)) verb <- "filter"
      if (verb == "join" && !is.null(y)) verb <- "arrange"
      cand <- switch(verb,
        filter = fz_step_filter(st), mutate = fz_step_mutate(st), transmute = fz_step_mutate(st, "transmute"),
        select = fz_step_select(st), rename = fz_step_rename(st), relocate = fz_step_relocate(st),
        arrange = fz_step_arrange(st), distinct = fz_step_distinct(st), group_by = fz_step_group_by(st),
        ungroup = fz_step_ungroup(st), summarise = fz_step_summarise(st), count = fz_step_count(st),
        slice = fz_step_slice(st), join = fz_step_join(st, cur), bind_rows = fz_step_bind(st, cur))
      invalid <- fz_chance(.04)
      if (invalid) {
        bad <- fz_make_invalid(cand$step, names(st$cols))
        if (is.null(bad)) invalid <- FALSE else cand$step <- bad
      }
      env <- fz_env(list(vars = st$vars), y = if (is.null(cand$y)) y else cand$y)
      res <- tryCatch(suppressMessages(suppressWarnings(fz_apply_step(cur, cand$step, env))),
                      error = function(e) e)
      failed <- inherits(res, "error")
      if (!failed || invalid || (attempt == 3L && fz_chance(.3))) {
        steps[[length(steps) + 1L]] <- cand$step
        if (!is.null(cand$y)) y <- cand$y
        if (failed) stop_here <- TRUE else cur <- res
        break
      }
    }
    if (stop_here) break
  }
  fuzz_make_case(x, steps, y = y, vars = fz_used_vars(steps, st$vars), seed = seed)
}

# ---- code

# 15 digits when they read back exactly, 17 otherwise
fz_deparse <- function(v) {
  short <- paste(deparse(v, width.cutoff = 500L, control = c("keepNA", "keepInteger")), collapse = "")
  if (!is.double(v) || identical(eval(parse(text = short)), v)) return(short)
  paste(deparse(v, width.cutoff = 500L, control = c("keepNA", "keepInteger", "digits17")), collapse = "")
}

fz_value_code <- function(v) {
  if (inherits(v, "integer64")) {
    sprintf("bit64::as.integer64(%s)", fz_deparse(as.character(v)))
  } else if (is.factor(v)) {
    sprintf("factor(%s, levels = %s%s)", fz_deparse(as.character(v)), fz_deparse(levels(v)),
            if (is.ordered(v)) ", ordered = TRUE" else "")
  } else if (inherits(v, "POSIXct")) {
    tz <- attr(v, "tzone")
    sprintf(".POSIXct(%s%s)", fz_deparse(as.numeric(v)),
            if (is.null(tz)) "" else paste0(", tz = ", fz_deparse(tz)))
  } else if (inherits(v, "Date")) {
    n <- as.numeric(v)
    if (all(is.na(n) | n == floor(n))) sprintf("as.Date(%s)", fz_deparse(format(v)))
    else sprintf("structure(%s, class = \"Date\")", fz_deparse(n))
  } else {
    fz_deparse(v)
  }
}

fz_table_code <- function(df, name) {
  nm <- names(df)
  lhs <- ifelse(make.names(nm) == nm, nm, sprintf("`%s`", nm))
  cols <- sprintf("  %s = %s", lhs, vapply(df, fz_value_code, ""))
  c(sprintf("%s <- tibble::tibble(", name), paste0(cols, c(rep(",", length(cols) - 1L), "")), ")")
}

fuzz_code <- function(case) {
  lines <- c(sprintf("# fuzz case, seed %s", case$seed), "library(dplyr)", "",
             fz_table_code(case$x, "x"))
  if (!is.null(case$y)) lines <- c(lines, "", fz_table_code(case$y, "y"))
  for (nm in names(case$vars)) {
    lines <- c(lines, sprintf("%s <- %s", nm, fz_value_code(case$vars[[nm]])))
  }
  calls <- vapply(case$steps, function(s) {
    if (identical(s[[1L]], quote(bind_rows))) s[[1L]] <- quote(cuplyr::bind_rows)
    paste(trimws(deparse(s, width.cutoff = 500L)), collapse = " ")
  }, "")
  body <- if (length(calls)) paste0("    ", calls, c(rep(" |>", length(calls) - 1L), "")) else character()
  c(lines, "",
    "# gpu <- cuplyr::tbl_gpu(x); y <- cuplyr::tbl_gpu(y); then pipeline(gpu) |> collect()",
    "pipeline <- function(.tbl) {",
    if (length(body)) c("  .tbl |>", body) else "  .tbl",
    "}") |> paste(collapse = "\n")
}

print.fuzz_case <- function(x, ...) {
  cat(fuzz_code(x), "\n", sep = "")
  invisible(x)
}


# =============================================================================
# 6. Runner
# =============================================================================

fuzz_outcome <- function(value = NULL, error = NULL, warnings = list(), groups = character(),
                         fallbacks = 0L, plan_breaks = 0L) {
  structure(list(value = value, error = error, warnings = warnings, groups = groups,
                 fallbacks = fallbacks, plan_breaks = plan_breaks), class = "fuzz_outcome")
}

# `mode` NULL: the oracle (dplyr on the tibble). "eager" / "lazy": a tbl_gpu.
# `inject` (tests only) is applied to the collected GPU tibble.
fz_run <- function(case, mode = NULL, inject = NULL) {
  fz_normal_form()
  warnings <- list(); fallbacks <- 0L; breaks <- 0L
  value <- NULL; error <- NULL; groups <- character()
  up <- function(df) {
    if (is.null(mode) || is.null(df)) return(df)
    # an upload warning (integer64, out-of-range timestamps) is not the pipeline's
    suppressMessages(suppressWarnings(cuplyr::tbl_gpu(df, lazy = identical(mode, "lazy"))))
  }
  tryCatch({
    x <- up(case$x)
    y <- up(case$y)
    env <- fz_env(case, y = y)
    withCallingHandlers({
      tbl <- x
      for (s in case$steps) tbl <- fz_apply_step(tbl, s, env)
      groups <- dplyr::group_vars(tbl)
      out <- dplyr::collect(tbl)
      # .drop travels with the grouping: it decides whether later verbs see empty groups
      if (length(groups)) attr(groups, "drop") <- dplyr::group_by_drop_default(out)
      value <- tibble::as_tibble(dplyr::ungroup(out))
    },
    warning = function(w) {
      warnings[[length(warnings) + 1L]] <<- w
      invokeRestart("muffleWarning")
    },
    message = function(m) invokeRestart("muffleMessage"),
    cuplyr_cpu_fallback = function(c) fallbacks <<- fallbacks + 1L,
    cuplyr_plan_break = function(c) breaks <<- breaks + 1L)
    if (!is.null(inject) && !is.null(mode)) value <- inject(value)
  }, error = function(e) {
    error <<- e
    value <<- NULL
  })
  fuzz_outcome(value, error, warnings, groups, fallbacks, breaks)
}

fuzz_oracle <- function(case) fz_run(case, NULL)

fuzz_gpu <- function(case, mode = c("eager", "lazy"), inject = NULL) {
  fz_run(case, match.arg(mode), inject)
}


# =============================================================================
# 7. Comparator
# =============================================================================

fz_generic_errors <- c("rlang_error", "simpleError", "error", "condition")

# cuplyr_error_gpu is what wrap_gpu_call() rewords a C++ failure into; lazy mode
# lets cuplyr's own stop("internal: ...") assertions through as a plain error.
fz_is_raw_error <- function(e) {
  inherits(e, c("Rcpp::exception", "C++Error", "cuplyr_error_gpu")) ||
    any(grepl("^std::", class(e))) ||
    grepl("^internal( error)?:", fz_cond_text(e))
}

# The message with the parts that vary between seeds taken out, for signatures.
# All the ways of quoting a name become the same token. A single quote only opens
# a name after a non-word character, so the apostrophe of can't doesn't.
fz_norm_msg <- function(msg) {
  msg <- gsub("`[^`]*`", "`_`", msg)
  msg <- gsub("\"[^\"]*\"", "`_`", msg)
  msg <- gsub("\u2018[^\u2019]*\u2019", "`_`", msg)
  msg <- gsub("(?<![[:alnum:]])'[^']*'", "`_`", msg, perl = TRUE)
  msg <- gsub("[0-9]+(\\.[0-9]+)?", "#", msg)
  msg <- gsub("\\s+", " ", msg)
  # eager wraps a GPU failure in this; lazy shows the same failure bare
  msg <- sub("^GPU operation `_` failed\\. Original error: ", "", msg)
  substr(trimws(msg), 1L, 90L)
}

fz_cond_text <- function(cnd) {
  txt <- tryCatch(conditionMessage(cnd), error = function(e) "")
  par <- cnd$parent
  if (!is.null(par) && inherits(par, "condition")) txt <- paste(txt, fz_cond_text(par))
  txt
}

fz_mismatch <- function(kind, category, signature, detail = "") {
  structure(list(kind = kind, category = category, signature = signature,
                 detail = detail), class = "fuzz_mismatch")
}

fz_type_diff <- function(a, b) {
  if (!identical(typeof(a), typeof(b))) return(sprintf("typeof:%s->%s", typeof(a), typeof(b)))
  if (!identical(class(a), class(b))) {
    return(sprintf("class:%s->%s", paste(class(a), collapse = "/"), paste(class(b), collapse = "/")))
  }
  if (!identical(levels(a), levels(b))) return("levels")
  if (!identical(attr(a, "tzone"), attr(b, "tzone"))) {
    return(sprintf("tzone:%s->%s", paste(attr(a, "tzone"), collapse = "|"), paste(attr(b, "tzone"), collapse = "|")))
  }
  ""
}

fz_is_na <- function(x) {
  if (inherits(x, "integer64")) is.na(as.character(x)) else is.na(x)
}

# Relative tolerance of plain doubles: GPU reductions add in another order than
# R, so a sum can differ in the last bits (known difference `float_tolerance`).
# Tight enough that 1e12 vs 1e12 + 10 and 1 vs 1 + 1e-6 are differences.
fz_float_tol <- 1e-12

fz_plain_double <- function(x) {
  is.double(x) && !inherits(x, c("Date", "POSIXct", "integer64", "difftime"))
}

# "" when the columns are equal, otherwise the shape of the first difference.
fz_col_diff <- function(a, b, ctx) {
  n <- length(a)
  if (n == 0L) return("")
  keep <- rep(TRUE, n)
  if (inherits(a, c("Date", "POSIXct"))) {
    ua <- as.numeric(a); ub <- as.numeric(b)
    na_only_b <- is.na(ub) & !is.nan(ub)
    m <- is.infinite(ua) & na_only_b
    if (any(m)) {
      keep[m] <- FALSE
      ctx$masked <- c(ctx$masked, "infinite_timestamp")
    }
    m <- is.nan(ua) & na_only_b
    if (any(m)) {
      keep[m] <- FALSE
      ctx$masked <- c(ctx$masked, "nan_timestamp")
    }
    if (inherits(a, "Date")) {
      m <- is.finite(ua) & ua != floor(ua) & !is.na(ub) & ub == floor(ua)
      if (any(m)) {
        keep[m] <- FALSE
        ctx$masked <- c(ctx$masked, "fractional_date")
      }
    }
  }
  na_a <- fz_is_na(a)[keep]; na_b <- fz_is_na(b)[keep]
  if (any(na_a != na_b)) return("na_mask")
  if ((fz_plain_double(a) || inherits(a, c("Date", "POSIXct"))) &&
      any(is.nan(as.numeric(a)[keep]) != is.nan(as.numeric(b)[keep]))) {
    return("nan_mask")
  }
  ok <- !na_a
  if (!any(ok)) return("")
  sel <- which(keep)[ok]
  same <- if (inherits(a, "integer64")) {
    as.character(a)[sel] == as.character(b)[sel]
  } else if (fz_plain_double(a)) {
    p <- a[sel]; q <- b[sel]
    eq <- p == q
    fin <- is.finite(p) & is.finite(q) & !eq
    eq[fin] <- abs(p[fin] - q[fin]) <= fz_float_tol * pmax(1, abs(p[fin]), abs(q[fin]))
    eq
  } else if (is.factor(a)) {
    as.integer(a)[sel] == as.integer(b)[sel]
  } else if (is.atomic(a)) {
    as.numeric_or_self <- function(v) if (inherits(v, c("Date", "POSIXct", "difftime"))) as.numeric(v) else v
    as.numeric_or_self(a)[sel] == as.numeric_or_self(b)[sel]
  } else {
    return(if (identical(a, b)) "" else "values")
  }
  if (all(same)) "" else "values"
}

# One string per element, same for equal elements, for the multiset test that
# tells a row-order difference from a wrong value.
fz_canon <- function(col) {
  na <- fz_is_na(col)
  s <- if (inherits(col, "integer64")) as.character(col)
       else if (is.factor(col)) paste0("f:", as.character(col))
       else if (fz_plain_double(col)) sprintf("%.12g", col)
       else if (inherits(col, c("Date", "POSIXct"))) sprintf("%.6f", as.numeric(col))
       else paste0("v:", as.character(col))
  s[na] <- "<NA>"
  if (is.double(col) && !inherits(col, "integer64")) s[is.nan(as.numeric(col))] <- "<NaN>"
  s
}

fz_row_keys <- function(tbl) {
  if (ncol(tbl) == 0L) return(rep("", nrow(tbl)))
  do.call(paste, c(lapply(tbl, fz_canon), sep = "\r"))
}

fz_short_type <- function(x) {
  paste(c(class(x)[1L], if (is.double(x) || is.integer(x)) typeof(x)), collapse = "/")
}

fz_compare_tables <- function(a, b, ctx) {
  if (!identical(names(a), names(b))) {
    return(fz_mismatch("names", "silent_wrong", "names",
                       sprintf("oracle: %s; gpu: %s", paste(names(a), collapse = ","), paste(names(b), collapse = ","))))
  }
  for (i in seq_along(a)) {
    d <- fz_type_diff(a[[i]], b[[i]])
    if (nzchar(d)) {
      return(fz_mismatch("type", "silent_wrong", paste0("type:", sub(":.*", "", d), ":", fz_short_type(a[[i]])),
                         sprintf("column `%s`: %s", names(a)[i], d)))
    }
  }
  if (nrow(a) != nrow(b)) {
    return(fz_mismatch("value", "silent_wrong", "value:nrow",
                       sprintf("rows: oracle %d, gpu %d", nrow(a), nrow(b))))
  }
  shapes <- vapply(seq_along(a), function(i) fz_col_diff(a[[i]], b[[i]], ctx), "")
  bad <- which(nzchar(shapes))
  if (length(bad) == 0L) return(NULL)
  if (identical(sort(fz_row_keys(a), method = "radix"), sort(fz_row_keys(b), method = "radix"))) {
    return(fz_mismatch("order", "silent_wrong", "order", "the same rows in another order"))
  }
  i <- bad[1L]
  fz_mismatch("value", "silent_wrong", sprintf("value:%s:%s", fz_short_type(a[[i]]), shapes[i]),
              sprintf("column `%s` (%s): %s", names(a)[i], fz_short_type(a[[i]]), shapes[i]))
}

fz_warn_sig <- function(w) {
  paste(class(w)[1L], fz_norm_msg(fz_cond_text(w)))
}

# Warnings after the known-difference masks (ids appended to ctx$masked).
fz_masked_warnings <- function(oracle, gpu, ctx) {
  wa <- oracle$warnings; wb <- gpu$warnings
  if (length(wa) == 0L && length(wb) == 0L) return(list(oracle = wa, gpu = wb))
  val <- oracle$value
  # nan_math_warning: only when every dplyr warning is "NaNs produced" and the GPU has none.
  # dplyr folds a verb's warnings into one condition that quotes only the first
  # ("There were 2 warnings in `mutate()`..."), so that text can hide a different
  # warning: a condition that says "There were N warnings" is never masked.
  if (length(wa) && length(wb) == 0L) {
    nan_w <- vapply(wa, function(w) {
      txt <- fz_cond_text(w)
      grepl("NaNs produced", txt, fixed = TRUE) && !grepl("There were [0-9]+ warnings", txt)
    }, NA)
    if (all(nan_w)) {
      ctx$masked <- c(ctx$masked, "nan_math_warning")
      wa <- list()
    }
  }
  # infinite_timestamp / fractional_date: the GPU upload warns, dplyr has no upload
  has_inf_ts <- !is.null(val) && any(vapply(val, function(col) inherits(col, c("Date", "POSIXct")) && any(is.infinite(as.numeric(col))), NA))
  has_frac <- !is.null(val) && any(vapply(val, function(col) inherits(col, "Date") && any(is.finite(as.numeric(col)) & as.numeric(col) != floor(as.numeric(col)), na.rm = TRUE), NA))
  drop_cls <- c(if (has_inf_ts) "cuplyr_warning_infinite_timestamp", if (has_frac) "cuplyr_warning_fractional_date")
  if (length(drop_cls) && length(wb)) {
    hit <- vapply(wb, function(w) inherits(w, drop_cls), NA)
    if (any(hit)) {
      ctx$masked <- c(ctx$masked, if (has_inf_ts) "infinite_timestamp", if (has_frac) "fractional_date")
      wb <- wb[!hit]
    }
  }
  list(oracle = wa, gpu = wb)
}

# The mismatch (or NULL) plus the ids of the known differences that applied.
fz_compare_ex <- function(oracle, gpu, case = NULL) {
  ctx <- new.env()
  ctx$masked <- character()
  res <- fz_compare_core(oracle, gpu, ctx)
  list(mismatch = res, masked = unique(ctx$masked))
}

fz_compare_core <- function(oracle, gpu, ctx) {
  eo <- oracle$error; eg <- gpu$error
  if (!is.null(eo) || !is.null(eg)) {
    if (is.null(eo)) {
      return(fz_mismatch("error_presence", "crash_or_raw",
                         paste0("error_gpu_only:", fz_norm_msg(fz_cond_text(eg))),
                         sprintf("gpu errored (%s): %s", class(eg)[1L], substr(fz_cond_text(eg), 1L, 200L))))
    }
    if (is.null(eg)) {
      return(fz_mismatch("error_presence", "silent_wrong",
                         paste0("error_oracle_only:", class(eo)[1L]),
                         sprintf("dplyr errored (%s): %s; gpu returned a result", class(eo)[1L], substr(fz_cond_text(eo), 1L, 200L))))
    }
    if (fz_is_raw_error(eg)) {
      return(fz_mismatch("error_class", "crash_or_raw",
                         paste0("error_raw:", fz_norm_msg(fz_cond_text(eg))),
                         sprintf("raw %s: %s", class(eg)[1L], substr(fz_cond_text(eg), 1L, 200L))))
    }
    oc <- class(eo)[1L]
    if (!(oc %in% fz_generic_errors) && !inherits(eg, oc)) {
      return(fz_mismatch("error_class", "error_class",
                         sprintf("error_class:%s->%s", oc, class(eg)[1L]),
                         sprintf("dplyr: %s; gpu: %s (%s)", oc, paste(class(eg), collapse = "/"),
                                 substr(fz_cond_text(eg), 1L, 150L))))
    }
    return(NULL)
  }
  m <- fz_compare_tables(oracle$value, gpu$value, ctx)
  if (!is.null(m)) return(m)
  if (!identical(oracle$groups, gpu$groups)) {
    return(fz_mismatch("groups", "silent_wrong", "groups",
                       sprintf("oracle: [%s]; gpu: [%s]", paste(oracle$groups, collapse = ","), paste(gpu$groups, collapse = ","))))
  }
  w <- fz_masked_warnings(oracle, gpu, ctx)
  if ((length(w$oracle) > 0L) != (length(w$gpu) > 0L)) {
    one <- if (length(w$oracle)) w$oracle[[1L]] else w$gpu[[1L]]
    side <- if (length(w$oracle)) "oracle_only" else "gpu_only"
    return(fz_mismatch("warning", "warning_only", paste0("warning_", side, ":", fz_warn_sig(one)),
                       sprintf("%s warns: %s", side, substr(fz_cond_text(one), 1L, 200L))))
  }
  NULL
}

fuzz_compare <- function(oracle, gpu, case = NULL) fz_compare_ex(oracle, gpu, case)$mismatch


# =============================================================================
# 8. Known differences
# =============================================================================
# The single list the comparator skips. A new entry needs a narrow condition, a
# reason and the ledger line it comes from; it is not a way to get a green run.
# `where` says the exact condition under which the comparator applies it.

fuzz_known_differences <- function() {
  list(
    list(
      id = "row_names",
      reason = "Row names are not data: both sides are normalised with tibble::as_tibble() before comparing.",
      ledger = "F118 brief, known differences: row names",
      where = "always"
    ),
    list(
      id = "float_tolerance",
      reason = "GPU reductions add in another order than R, so a double sum or mean can differ in the last bits where dplyr's own result is not exact; plain doubles compare with a relative tolerance of 1e-12 (against a floor of 1).",
      ledger = "F118 brief, known differences: float rounding within a tolerance",
      where = "finite plain doubles only; Date, POSIXct and integer64 compare exactly"
    ),
    list(
      id = "nan_math_warning",
      reason = "dplyr warns 'NaNs produced' for sqrt()/log() of a negative; the GPU computes NaN without a warning. Undecided whether cuplyr should warn.",
      ledger = "F118 brief, known differences: NaN-math warnings (undecided candidate)",
      where = "only when every dplyr warning is 'NaNs produced' and the GPU side has no warning at all; never a combined 'There were N warnings' condition, which quotes only the first"
    ),
    list(
      id = "nan_timestamp",
      reason = "A Date/POSIXct of NaN (mean() of an empty group in dplyr) has no GPU representation: the GPU gives NA where dplyr gives a NaN Date.",
      ledger = "F81 (documents empty-group mean(): NA vs NaN-Date)",
      where = "only elements where dplyr has NaN and the GPU has plain NA, in a Date or POSIXct column; the reverse (NA vs NaN) is a finding"
    ),
    list(
      id = "lubridate_posixct_warning",
      reason = "Base R's 'tzone' attributes are inconsistent warning from comparing POSIXct columns depends on whether lubridate is loaded (its namespace supplies an `==` for POSIXt that does not warn), so the harness always runs with lubridate loaded and dplyr gives no such warning. A documented consequence of fz_normal_form(), not a mask: the comparator has no branch for it, and dplyr's fold of a verb's warnings means none could remove it.",
      ledger = "F118 round 2 review, B2",
      where = "no comparator branch: every run loads lubridate in fz_normal_form()"
    ),
    list(
      id = "infinite_timestamp",
      reason = "A Date/POSIXct of +-Inf (only reachable by computation, e.g. min() of an empty group) cannot be held on the GPU: it becomes NA and the upload warns.",
      ledger = "F81 (unrepresentable timestamps at upload; architecture reference)",
      where = "only elements where dplyr has +-Inf and the GPU has NA, plus the GPU's cuplyr_warning_infinite_timestamp when dplyr's result holds one"
    ),
    list(
      id = "fractional_date",
      reason = "A TIMESTAMP_DAYS column holds whole days, so a fractional Date (mean(<Date>), d + 0.5) is floored at upload, with a warning dplyr does not give.",
      ledger = "architecture reference, 'Fractional Dates at upload' (cuplyr_warning_fractional_date)",
      where = "only elements where the GPU value is floor() of dplyr's non-whole Date, plus that warning class"
    )
  )
}


# =============================================================================
# 9. Check, shrink, campaign
# =============================================================================

fz_empty_findings <- function() {
  data.frame(mode = character(), kind = character(), category = character(),
             signature = character(), detail = character(), stringsAsFactors = FALSE)
}

fz_check_full <- function(case, inject = NULL, modes = c("eager", "lazy")) {
  oracle <- fuzz_oracle(case)
  rows <- list(); gpus <- list(); masked <- character()
  for (m in modes) {
    g <- fuzz_gpu(case, m, inject)
    gpus[[m]] <- g
    cmp <- fz_compare_ex(oracle, g, case)
    masked <- c(masked, cmp$masked)
    if (!is.null(cmp$mismatch)) {
      mm <- cmp$mismatch
      rows[[length(rows) + 1L]] <- data.frame(mode = m, kind = mm$kind, category = mm$category,
                                              signature = mm$signature, detail = mm$detail,
                                              stringsAsFactors = FALSE)
    }
  }
  findings <- if (length(rows)) do.call(rbind, rows) else fz_empty_findings()
  list(findings = findings, oracle = oracle, gpu = gpus, masked = masked)
}

fuzz_check <- function(case, inject = NULL, modes = c("eager", "lazy")) {
  fz_check_full(case, inject, modes)$findings
}

# ---- shrinker

# Smaller variants of one expression: an operand in place of its call, anywhere.
fz_expr_variants <- function(e) {
  out <- list()
  if (is.call(e)) {
    kids <- as.list(e)[-1L]
    for (k in kids) {
      if (!missing(k) && !is.null(k) && (is.language(k) || is.atomic(k))) out <- c(out, list(k))
    }
    for (i in seq_along(kids)) {
      k <- kids[[i]]
      if (is.null(k) || !is.call(k)) next
      for (v in fz_expr_variants(k)) {
        e2 <- e
        e2[[i + 1L]] <- v
        out <- c(out, list(e2))
      }
    }
  }
  out
}

fz_step_variants <- function(step) {
  args <- as.list(step)[-1L]
  out <- list()
  for (j in seq_along(args)) {
    out <- c(out, list(as.call(c(list(step[[1L]]), args[-j]))))
  }
  for (j in seq_along(args)) {
    a <- args[[j]]
    if (!is.call(a)) next
    for (v in fz_expr_variants(a)) {
      new <- args
      new[[j]] <- v
      out <- c(out, list(as.call(c(list(step[[1L]]), new))))
    }
  }
  out
}

# Passes run in this order; each may use attempts up to its cumulative share of
# `max_attempts`, so a pass that finds nothing hands its budget on and a greedy
# early pass can't starve a later one. When a pass hits its cap, its name is
# listed in attr(<result>, "budget_exhausted").
fz_shrink_shares <- c(steps = .25, rows = .55, columns = .65, expressions = .90, columns_again = 1)

fuzz_shrink <- function(case, fails = NULL, inject = NULL, max_attempts = 150L) {
  if (is.null(fails)) {
    before <- fuzz_check(case, inject)
    if (nrow(before) == 0L) return(case)
    target <- before[1L, ]
    modes <- unique(before$mode[before$kind == target$kind & before$signature == target$signature])
    fails <- function(k) {
      r <- fuzz_check(k, inject, modes = modes)
      any(r$kind == target$kind & r$signature == target$signature)
    }
  }
  attempts <- 0L
  cap <- max_attempts
  pass <- ""
  exhausted <- character()
  try_case <- function(cand) {
    if (attempts >= cap) {
      exhausted <<- union(exhausted, pass)
      return(FALSE)
    }
    attempts <<- attempts + 1L
    isTRUE(tryCatch(fails(cand), error = function(e) FALSE))
  }
  start_pass <- function(name) {
    pass <<- name
    cap <<- ceiling(max_attempts * fz_shrink_shares[[name]])
  }
  set_fields <- function(k, ...) {
    f <- list(...)
    for (nm in names(f)) k[nm] <- list(f[[nm]])
    k
  }
  shrink_columns <- function(case) {
    for (nm in rev(names(case$x))) {
      if (ncol(case$x) <= 1L) break
      cand <- set_fields(case, x = case$x[, setdiff(names(case$x), nm), drop = FALSE])
      if (try_case(cand)) case <- cand
    }
    if (!is.null(case$y)) {
      for (nm in rev(names(case$y))) {
        if (ncol(case$y) <= 1L) break
        cand <- set_fields(case, y = case$y[, setdiff(names(case$y), nm), drop = FALSE])
        if (try_case(cand)) case <- cand
      }
    }
    case
  }
  # steps: drop one at a time (last first) until nothing goes
  start_pass("steps")
  repeat {
    changed <- FALSE
    for (i in rev(seq_along(case$steps))) {
      cand <- set_fields(case, steps = case$steps[-i])
      if (try_case(cand)) {
        case <- cand
        changed <- TRUE
      }
    }
    if (!changed) break
  }
  # the second table
  if (!is.null(case$y) && !("y" %in% all.names(as.call(c(list(quote(list)), case$steps))))) {
    case <- set_fields(case, y = NULL)
  }
  shrink_rows <- function(df, rebuild) {
    n <- nrow(df)
    if (n > 0L && try_case(rebuild(df[0L, , drop = FALSE]))) return(df[0L, , drop = FALSE])
    chunk <- max(1L, ceiling(n / 2))
    repeat {
      start <- 1L
      while (start <= nrow(df)) {
        drop <- seq.int(start, min(nrow(df), start + chunk - 1L))
        if (length(drop) < nrow(df) && try_case(rebuild(df[-drop, , drop = FALSE]))) {
          df <- df[-drop, , drop = FALSE]
        } else {
          start <- start + chunk
        }
      }
      if (chunk == 1L) break
      chunk <- max(1L, chunk %/% 2L)
    }
    df
  }
  start_pass("rows")
  case$x <- shrink_rows(case$x, function(d) set_fields(case, x = d))
  if (!is.null(case$y)) case$y <- shrink_rows(case$y, function(d) set_fields(case, y = d))
  start_pass("columns")
  case <- shrink_columns(case)
  # expressions: simpler variants of each step, restarted after every success
  start_pass("expressions")
  for (i in seq_along(case$steps)) {
    repeat {
      progressed <- FALSE
      for (v in fz_step_variants(case$steps[[i]])) {
        steps <- case$steps
        steps[[i]] <- v
        cand <- set_fields(case, steps = steps)
        if (try_case(cand)) {
          case <- cand
          progressed <- TRUE
          break
        }
      }
      if (!progressed || attempts >= cap) break
    }
  }
  # an expression that went may have been the only reader of a column
  start_pass("columns_again")
  case <- shrink_columns(case)
  case$vars <- fz_used_vars(case$steps, case$vars)
  attr(case, "budget_exhausted") <- exhausted
  case
}

# ---- campaign

fz_tsv_text <- function(x) {
  x <- gsub("\r", "", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  gsub("\t", " ", x, fixed = TRUE)
}

# The code column keeps "\n" as two characters; this puts the lines back.
fuzz_decode <- function(code) gsub("\\n", "\n", code, fixed = TRUE)

fz_findings_columns <- c("seed", "mode", "kind", "category", "signature", "detail", "code")

# `progress`: a file that gets the current seed before each case, so a driver
# can tell which one killed the process.
fuzz_campaign <- function(seeds, out = NULL, inject = NULL, progress = NULL) {
  rows <- list(); n_native <- 0L; n_native_ok <- 0L; n_break <- 0L; masked <- character()
  for (s in seeds) {
    if (!is.null(progress)) writeLines(as.character(s), progress)
    case <- fuzz_case(s)
    full <- fz_check_full(case, inject)
    masked <- c(masked, full$masked)
    native <- all(vapply(full$gpu, function(g) g$fallbacks == 0L, NA))
    if (native) n_native <- n_native + 1L
    # native_ok: also no side errored (an error stops the pipeline before it works the GPU)
    errored <- !is.null(full$oracle$error) || any(vapply(full$gpu, function(g) !is.null(g$error), NA))
    if (native && !errored) n_native_ok <- n_native_ok + 1L
    if (any(vapply(full$gpu, function(g) g$plan_breaks > 0L, NA))) n_break <- n_break + 1L
    f <- full$findings
    if (nrow(f)) {
      f$seed <- as.integer(s)
      f$code <- fz_tsv_text(fuzz_code(case))
      f$detail <- fz_tsv_text(f$detail)
      f$signature <- fz_tsv_text(f$signature)
      rows[[length(rows) + 1L]] <- f[fz_findings_columns]
    }
  }
  res <- if (length(rows)) do.call(rbind, rows) else {
    data.frame(seed = integer(), mode = character(), kind = character(), category = character(),
               signature = character(), detail = character(), code = character(),
               stringsAsFactors = FALSE)
  }
  rownames(res) <- NULL
  n <- length(seeds)
  attr(res, "stats") <- list(
    cases = n,
    native_share = if (n) n_native / n else NA_real_,
    native_ok_share = if (n) n_native_ok / n else NA_real_,
    plan_break_share = if (n) n_break / n else NA_real_,
    masked = fz_table_counts(masked)
  )
  if (!is.null(out)) {
    utils::write.table(res, out, sep = "\t", quote = FALSE, row.names = FALSE)
  }
  res
}

fz_table_counts <- function(ids) {
  if (length(ids) == 0L) return(integer())
  t <- table(ids)
  stats::setNames(as.integer(t), names(t))
}


# =============================================================================
# 10. The quick subset (test-fuzz-dplyr.R)
# =============================================================================

# Seeds that pass at merge, plus the seeds listed in fuzz_quick_skips(), one or
# more per distinct finding. The list is explicit: changing the generator
# changes every case, so the seeds are re-chosen with it (a campaign over
# 1:600; from seeds 1:159 the ones that pass, with a non-error, non-empty dplyr
# result, plus 6 where dplyr and the GPU both error and 10 where both return
# zero rows, so the error and empty-result paths stay covered).
fuzz_quick_seeds <- function() {
  c(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 10L, 11L, 12L, 13L, 15L, 16L, 17L, 18L, 19L, 20L,
    21L, 22L, 23L, 24L, 25L, 26L, 27L, 29L, 30L, 31L, 32L, 33L, 34L, 35L, 36L, 37L, 38L,
    39L, 40L, 41L, 43L, 44L, 45L, 46L, 47L, 48L, 49L, 50L, 51L, 54L, 55L, 56L, 57L, 58L,
    61L, 62L, 64L, 65L, 68L, 69L, 70L, 71L, 73L, 74L, 75L, 76L, 77L, 79L, 83L, 85L, 86L,
    87L, 88L, 89L, 90L, 91L, 92L, 93L, 94L, 96L, 97L, 98L, 99L, 100L, 102L, 103L, 107L,
    109L, 110L, 111L, 112L, 114L, 115L, 116L, 117L, 118L, 119L, 120L, 121L, 123L, 126L,
    127L, 128L, 129L, 131L, 133L, 134L, 135L, 136L, 137L, 138L, 139L, 140L, 141L, 142L,
    143L, 144L, 145L, 147L, 148L, 149L, 150L, 151L, 155L, 157L, 159L,
    # F122 turned these on (the .data-pronoun column name)
    9L, 28L, 122L, 146L, 551L,
    as.integer(names(fuzz_quick_skips())))
}

# seed (as a name) -> the finding that makes it fail. Provisional ids: the
# ledger rows for them are the PM's to open (see the F118 report).
fuzz_quick_skips <- function() {
  c(`42` = "F118-unknown-column-error-class",
    `209` = "F118-mutate-after-missing-column",
    `423` = "F118-zero-column-collect",
    `435` = "F118-distinct-window-lowering",
    `451` = "F118-min-empty-integer-type",
    `567` = "F118-min-empty-string-warning")
}
