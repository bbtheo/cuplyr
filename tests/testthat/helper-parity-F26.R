# Shared helpers for the F26 acceptance tests (test-parity-F26-*.R):
# native rows_insert/append/update/patch/upsert/delete.

# A table covering every plain GPU type, with an NA key (row 4) and an NA in
# every other column.
f26_x <- function() {
  tibble::tibble(
    id  = c(1L, 2L, 3L, NA, 5L),
    k2  = c("a", "b", "a", "c", NA),
    lg  = c(TRUE, NA, FALSE, TRUE, NA),
    int = c(10L, NA, 30L, 40L, 50L),
    dbl = c(1.5, NA, 3.5, 4.5, NA),
    chr = c("p", "q", NA, "s", "t"),
    dt  = as.Date("2024-01-01") + c(0, 1, NA, 3, 4),
    ts  = as.POSIXct("2024-01-01 12:00:00", tz = "UTC") + c(0, 60, 120, NA, 240)
  )
}

# Run `thunk()`, recording its value, its error (if any) and every message
# (text and first class), muffled.
f26_run <- function(thunk) {
  msgs <- character()
  classes <- character()
  err <- NULL
  value <- withCallingHandlers(
    tryCatch(thunk(), error = function(e) {
      err <<- e
      NULL
    }),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      classes <<- c(classes, class(m)[[1]])
      invokeRestart("muffleMessage")
    }
  )
  list(value = value, error = err, messages = msgs, message_classes = classes)
}

# Per column, the rows holding NaN (double-backed columns only).
f26_nan_positions <- function(df) {
  lapply(as.list(tibble::as_tibble(dplyr::ungroup(df))), function(col) {
    if (is.double(col)) which(is.nan(unclass(col))) else integer()
  })
}

# x/y execution modes: both eager, both lazy, and the two mixed cases.
f26_modes <- list(c(FALSE, FALSE), c(TRUE, TRUE), c(FALSE, TRUE), c(TRUE, FALSE))

# Frame attributes worth comparing (everything but the structural ones).
f26_user_attrs <- function(df) {
  a <- attributes(df)
  a[setdiff(names(a), c("names", "row.names", "class", "groups"))]
}

# Compare `fn(x, y)` on tbl_gpu inputs against real dplyr on the plain inputs.
#
# `fn` gets two plain data frames for the oracle and two tbl_gpu for the GPU
# side (in every mode of f26_modes). Checks, per mode:
# - the same messages (text and class), in the same order, each exactly once;
# - if dplyr errors: the GPU call itself errors (at call time, also when x is
#   lazy) with the same class vector and the same conditionMessage();
# - otherwise: a tbl_gpu whose exec mode is x's, and, after collect(), the same
#   class, group_vars, .drop, group_data (empty groups included), user
#   attributes and data (names, types, values, row order, NaN vs NA).
# `native = TRUE` runs the GPU call and collect() with
# options(cuplyr.fallback = "error"), so any CPU fallback fails the test.
f26_check <- function(x, y, fn, native = TRUE) {
  cpu <- f26_run(function() fn(x, y))
  mode <- if (native) "error" else "silent"
  for (m in f26_modes) {
    label <- sprintf("x %s, y %s", if (m[[1]]) "lazy" else "eager",
                     if (m[[2]]) "lazy" else "eager")
    gx <- tbl_gpu(x, lazy = m[[1]])
    gy <- tbl_gpu(y, lazy = m[[2]])
    gpu <- withr::with_options(list(cuplyr.fallback = mode), {
      r <- f26_run(function() fn(gx, gy))
      if (is.null(r$error) && is_tbl_gpu(r$value)) {
        r$collected <- collect(r$value)
      }
      r
    })

    expect_identical(gpu$messages, cpu$messages,
                     info = paste0(label, ": messages differ from dplyr"))
    expect_identical(gpu$message_classes, cpu$message_classes,
                     info = paste0(label, ": message classes differ from dplyr"))

    if (!is.null(cpu$error)) {
      expect_false(is.null(gpu$error),
                   info = paste0(label, ": dplyr errors (", conditionMessage(cpu$error),
                                 ") but the GPU call did not"))
      if (!is.null(gpu$error)) {
        expect_identical(class(gpu$error), class(cpu$error),
                         info = paste0(label, ": error class differs; GPU said: ",
                                       conditionMessage(gpu$error)))
        expect_identical(conditionMessage(gpu$error), conditionMessage(cpu$error),
                         info = paste0(label, ": error message differs"))
      }
      next
    }

    expect_null(gpu$error,
                info = paste0(label, ": GPU errored: ",
                              if (!is.null(gpu$error)) conditionMessage(gpu$error) else ""))
    if (!is.null(gpu$error)) next
    expect_true(is_tbl_gpu(gpu$value), info = paste0(label, ": result is not a tbl_gpu"))
    if (!is_tbl_gpu(gpu$value)) next
    expect_identical(is_lazy(gpu$value), m[[1]],
                     info = paste0(label, ": the result's exec mode must be x's"))

    got <- gpu$collected
    want <- cpu$value
    expect_identical(class(got), class(want), info = paste0(label, ": class differs"))
    expect_identical(dplyr::group_vars(got), dplyr::group_vars(want),
                     info = paste0(label, ": group_vars differ"))
    expect_identical(dplyr::group_by_drop_default(got), dplyr::group_by_drop_default(want),
                     info = paste0(label, ": .drop differs"))
    # the full group structure, empty groups included (review round 1, B1)
    expect_equal(dplyr::group_data(got), dplyr::group_data(want),
                 info = paste0(label, ": group_data differs"))
    expect_identical(f26_user_attrs(got), f26_user_attrs(want),
                     info = paste0(label, ": frame attributes differ"))
    expect_equal(tibble::as_tibble(dplyr::ungroup(got)),
                 tibble::as_tibble(dplyr::ungroup(want)),
                 info = paste0(label, ": data differs from dplyr"))
    # expect_equal() doesn't tell NaN from NA; dplyr does (review round 2)
    expect_identical(f26_nan_positions(got), f26_nan_positions(want),
                     info = paste0(label, ": NaN positions differ from dplyr"))
  }
}
