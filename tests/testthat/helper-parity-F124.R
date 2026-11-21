# F124: min()/max() over empty or all-NA input, compared with dplyr.
#
# f124_check() runs `pipeline` on a tibble (dplyr, the oracle) and on a gibble,
# eager and lazy, and compares:
#   - the collected values and column types (exactly);
#   - the group vars;
#   - the warnings: dplyr folds a verb's warnings into ONE condition
#     ("There was 1 warning in `summarise()`." / "There were N warnings in
#     ..."), so the GPU side must give the same list of conditions with the
#     same (ANSI-stripped) messages, each inheriting dplyr's classes. A GPU
#     Date/POSIXct can't hold +-Inf: where dplyr's result has one, the GPU has
#     NA and gives one extra `cuplyr_warning_infinite_timestamp` (the upload
#     rule, R/tbl-gpu.R), which is checked on its own;
#   - eager: every warning at the verb, none at collect();
#   - lazy: the schema the plan reports before collect() has the collected
#     types (no `<int>` preview that collects a `<dbl>`), and, with
#     `lazy_warns_at = "collect"`, no warning at the verb (F80: a plan warns
#     when it runs) and no plan break;
#   - `native = TRUE`: no CPU fallback in either mode.

f124_capture <- function(expr) {
  ws <- list()
  fb <- character()
  pb <- character()
  val <- withCallingHandlers(
    expr,
    warning = function(w) {
      ws[[length(ws) + 1L]] <<- w
      invokeRestart("muffleWarning")
    },
    cuplyr_cpu_fallback = function(c) fb <<- c(fb, conditionMessage(c)),
    cuplyr_plan_break = function(c) pb <<- c(pb, conditionMessage(c))
  )
  list(value = val, warnings = ws, fallback = fb, plan_break = pb)
}

f124_msgs <- function(ws) {
  vapply(ws, function(w) cli::ansi_strip(conditionMessage(w)), character(1))
}

# R class + typeof of each column, the way collect() gives them.
f124_col_types <- function(frame) {
  frame <- dplyr::ungroup(tibble::as_tibble(frame))
  vapply(frame, function(col) paste(c(class(col), typeof(col)), collapse = "/"), character(1))
}

# dplyr's +-Inf in a Date/POSIXct is NA on the GPU (it can't hold one).
f124_ts_inf_to_na <- function(frame) {
  for (nm in names(frame)) {
    col <- frame[[nm]]
    if (inherits(col, c("Date", "POSIXct"))) {
      bad <- is.infinite(unclass(col))
      if (any(bad)) {
        col[bad] <- NA
        frame[[nm]] <- col
      }
    }
  }
  frame
}

f124_has_ts_inf <- function(frame) {
  any(vapply(frame, function(col) {
    inherits(col, c("Date", "POSIXct")) && any(is.infinite(unclass(col)))
  }, logical(1)))
}

f124_check <- function(df, pipeline, native = TRUE, arrange_by = NULL,
                       lazy_warns_at = c("any", "collect"), modes = c("eager", "lazy")) {
  lazy_warns_at <- match.arg(lazy_warns_at)
  oracle <- f124_capture(pipeline(tibble::as_tibble(df)))
  oracle_groups <- dplyr::group_vars(oracle$value)
  oracle_tbl <- oracle_normalize(oracle$value, arrange_by)
  oracle_msgs <- f124_msgs(oracle$warnings)
  ts_inf <- f124_has_ts_inf(oracle_tbl)

  for (mode in modes) {
    lazy <- identical(mode, "lazy")
    verb <- f124_capture(pipeline(tbl_gpu(df, lazy = lazy)))
    expect_true(is_tbl_gpu(verb$value), info = mode)
    expect_identical(verb$value$groups, oracle_groups, info = paste(mode, "group vars"))
    planned <- if (lazy) {
      f124_col_types(tidyselect_proxy(current_schema(verb$value)))
    }
    col <- f124_capture(collect(verb$value))
    gpu_tbl <- oracle_normalize(col$value, arrange_by)

    # values and types
    expect_identical(f124_col_types(gpu_tbl), f124_col_types(oracle_tbl),
                     info = paste(mode, "collected column types"))
    expect_equal(gpu_tbl, f124_ts_inf_to_na(oracle_tbl), info = paste(mode, "values"))
    if (lazy) {
      expect_identical(planned, f124_col_types(oracle_tbl),
                       info = "lazy schema before collect() vs collected types")
    }

    # warnings
    all_ws <- c(verb$warnings, col$warnings)
    is_inf <- vapply(all_ws, inherits, logical(1), "cuplyr_warning_infinite_timestamp")
    expect_identical(f124_msgs(all_ws[!is_inf]), oracle_msgs, info = paste(mode, "warning messages"))
    for (i in seq_along(oracle$warnings)) {
      if (i <= sum(!is_inf)) {
        expect_true(inherits(all_ws[!is_inf][[i]], class(oracle$warnings[[i]])[[1L]]),
                    info = paste(mode, "warning class"))
      }
    }
    expect_identical(sum(is_inf), if (ts_inf) 1L else 0L,
                     info = paste(mode, "cuplyr_warning_infinite_timestamp count"))
    if (!lazy) {
      expect_length(col$warnings, 0L)
    } else if (identical(lazy_warns_at, "collect")) {
      expect_length(verb$warnings, 0L)
      expect_length(verb$plan_break, 0L)
    }

    if (native) {
      expect_identical(c(verb$fallback, col$fallback), character(), info = paste(mode, "CPU fallback"))
    }
  }
  invisible(oracle)
}
