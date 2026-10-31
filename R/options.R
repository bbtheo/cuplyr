# The options registry (F93): every cuplyr option is declared once here, read
# through cuplyr_opt(), warned about the same way when its value is invalid, and
# reported by cuplyr_options(). The registry is built once at namespace load.
#
# Resolution, per option: the R option wins, then the environment variable, then
# the default. A *set* option decides even when invalid (it warns once and the
# default is used; the environment variable is not consulted). What counts as
# "unset" (the environment variable decides) and as "blank" (the option is set
# but empty: the default, silently, the environment variable not consulted)
# differs per option (`unset`, `blank`), as it always did.

# Allowed values that hot paths test inline (resolve_exec_mode(), the rolling
# readers, check_schema_cache_on()); the registry below uses the same vectors.
cuplyr_exec_modes <- c("lazy", "eager")
cuplyr_roll_medians <- c("exact", "frollmedian")
cuplyr_roll_paths <- c("auto", "scans", "cudf")
cuplyr_check_env_on <- c("1", "true", "TRUE", "yes")

# Returned by a `parse` function for a value that is not valid.
cuplyr_invalid <- structure(list(), class = "cuplyr_invalid")

# One warning per (option, source, value) per session, shared by every option
# (the generalised state of F88's cuplyr_notice_mode()).
cuplyr_opt_state <- new.env(parent = emptyenv())

opt_unset_null <- function(x) is.null(x)

# NULL, zero-length, or NA (as the first element); a blank string too when
# `blank`. `guard` skips the test for a non-atomic value (a function) instead
# of failing on it.
opt_unset_na <- function(blank = FALSE, guard = TRUE) {
  force(blank)
  force(guard)
  function(x) {
    if (is.null(x) || length(x) == 0L) {
      return(TRUE)
    }
    if (guard && !is.atomic(x)) {
      return(FALSE)
    }
    is.na(x[[1L]]) || (blank && is.character(x) && !nzchar(trimws(x[[1L]])))
  }
}

# "Blank" options: set, but empty. They mean the default and do not consult the
# environment variable (the option was set, so it decides).
# A blank (or all-space) string.
opt_blank_chr <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && !nzchar(trimws(x))
}

# Exactly "" (the rolling options: a blank of spaces is not "").
opt_blank_empty <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && !nzchar(x)
}

# Zero-length, NA, or a blank string; never a function or other non-atomic.
opt_blank_na <- function(x) {
  length(x) == 0L ||
    (is.atomic(x) && (is.na(x[[1L]]) || (is.character(x) && !nzchar(trimws(x[[1L]])))))
}

# A blank (or all-space) first element (cuplyr.memory_resource, which reads the
# first element of a vector, as it always did).
opt_blank_chr1 <- function(x) {
  is.character(x) && length(x) >= 1L && !is.na(x[[1L]]) && !nzchar(trimws(x[[1L]]))
}

# A character scalar that is one of `allowed` (after mapping `alias`, a named
# vector old -> new); `fold` ignores case and surrounding blanks. `first`
# reads the first element of a vector or list as character, as
# cuplyr.memory_resource always did (c("pool", "async") is "pool").
opt_choice <- function(allowed, alias = NULL, fold = FALSE, first = FALSE) {
  force(allowed)
  force(alias)
  force(fold)
  force(first)
  function(x, env = FALSE) {
    if (first && (is.atomic(x) || is.list(x)) && length(x) >= 1L) {
      x <- as.character(x)[1L]
    }
    if (is.character(x) && length(x) == 1L && !is.na(x)) {
      if (fold) {
        x <- tolower(trimws(x))
      }
      if (!is.null(alias) && x %in% names(alias)) {
        x <- alias[[x]]
      }
      if (x %in% allowed) {
        return(x)
      }
    }
    cuplyr_invalid
  }
}

opt_quote <- function(x) paste0("\"", x, "\"", collapse = ", ")

opt_parse_pool <- function(x, env = FALSE) {
  if (isTRUE(x)) {
    return(TRUE)
  }
  if (isFALSE(x)) {
    return(FALSE)
  }
  flag <- tolower(trimws(as.character(x[[1L]])))
  if (flag %in% c("0", "false", "no", "off")) {
    FALSE
  } else if (flag %in% c("1", "true", "yes", "on")) {
    TRUE
  } else {
    cuplyr_invalid
  }
}

# Bytes: a number, 0 or more. NA is the default ("computed / none").
opt_parse_bytes <- function(x, env = FALSE) {
  num <- if (is.atomic(x)) suppressWarnings(as.numeric(x[[1L]])) else NA_real_
  if (is.na(num) || num < 0) cuplyr_invalid else num
}

opt_parse_release <- function(x, env = FALSE) {
  if (is.numeric(x) || is.character(x)) opt_parse_bytes(x) else cuplyr_invalid
}

opt_parse_check <- function(x, env = FALSE) {
  if (env) x %in% cuplyr_check_env_on else isTRUE(x)
}

# `blank`: NULL, or a test for a set-but-empty option (see above).
# `default_result` is what a lookup of an unset or blank option returns, built
# once so that path allocates nothing.
opt_new <- function(name, default, env, parse, unset, description, expect,
                    internal = FALSE, message = NULL, blank = NULL,
                    default_label = NULL) {
  list(
    name = name, option = paste0("cuplyr.", name), default = default, env = env,
    parse = parse, unset = unset, blank = blank, description = description,
    expect = expect, internal = internal, message = message,
    default_label = default_label %||% opt_quote(default),
    default_result = list(value = default, source = "default", bad = FALSE)
  )
}

cuplyr_option_registry <- list(
  exec_mode = opt_new(
    "exec_mode", "lazy", "CUPLYR_EXEC_MODE",
    opt_choice(cuplyr_exec_modes), opt_unset_na(blank = TRUE),
    "Execution mode of new tables: lazy plans (default) or eager, one verb at a time.",
    paste0("one of ", opt_quote(cuplyr_exec_modes))
  ),
  fallback = opt_new(
    "fallback", "silent", NA_character_,
    opt_choice(c("silent", "warn", "error"), alias = c(warning = "warn")),
    opt_unset_null,
    "What a CPU fallback does: stay silent, warn, or error.",
    paste0("one of ", opt_quote(c("silent", "warn", "error")),
           " (or \"warning\" for \"warn\")")
  ),
  plan_break = opt_new(
    "plan_break", "silent", NA_character_,
    opt_choice(c("silent", "warn"), alias = c(warning = "warn")),
    opt_unset_null,
    "What a plan break does: stay silent or warn.",
    paste0("one of ", opt_quote(c("silent", "warn")),
           " (or \"warning\" for \"warn\")")
  ),
  memory_resource = opt_new(
    "memory_resource", "async", "CUPLYR_MEMORY_RESOURCE",
    opt_choice(c("async", "pool", "cuda", "none"), fold = TRUE, first = TRUE),
    opt_unset_na(guard = FALSE),
    "GPU memory resource: async (default), pool, or cuda (none). Read at load.",
    paste0("one of ", opt_quote(c("async", "pool", "cuda")), " (or \"none\", no resource)"),
    blank = opt_blank_chr1
  ),
  memory_pool = opt_new(
    "memory_pool", TRUE, "CUPLYR_MEMORY_POOL",
    opt_parse_pool, opt_unset_na(guard = FALSE),
    "Set FALSE (or 0, no, off) to use no memory resource. Read at load.",
    paste0("TRUE, FALSE, or one of ",
           opt_quote(c("1", "true", "yes", "on", "0", "false", "no", "off"))),
    blank = opt_blank_chr,
    default_label = "the default (on: the resource is installed)"
  ),
  pool_initial = opt_new(
    "pool_initial", NA_real_, "CUPLYR_POOL_INITIAL",
    opt_parse_bytes, opt_unset_null,
    "Initial pool size in bytes (default: a quarter of free GPU memory, at most 1 GiB). Read at load.",
    "a number of bytes, 0 or more",
    blank = opt_blank_na,
    default_label = "the default size"
  ),
  release_threshold = opt_new(
    "release_threshold", NA_real_, "CUPLYR_RELEASE_THRESHOLD",
    opt_parse_release, opt_unset_na(blank = TRUE),
    "Bytes the async resource keeps before returning memory to the driver (unset: never). Read at load.",
    "a number of bytes, 0 or more",
    message = function(shown, r) {
      sprintf(paste0("Invalid release threshold '%s' in %s (expected a number of ",
                     "bytes, 0 or more); no automatic release."),
              opt_typed(r), opt_source_label(r))
    }
  ),
  roll_median = opt_new(
    "roll_median", "exact", "CUPLYR_ROLL_MEDIAN",
    opt_choice(cuplyr_roll_medians), opt_unset_null,
    "Engine of roll_median(): exact (default) or frollmedian (data.table).",
    paste0("one of ", opt_quote(cuplyr_roll_medians)),
    blank = opt_blank_empty
  ),
  roll_path = opt_new(
    "roll_path", "auto", "CUPLYR_ROLL_PATH",
    opt_choice(cuplyr_roll_paths), opt_unset_null,
    "Internal: force the rolling-window path (auto, scans or cudf).",
    paste0("one of ", opt_quote(cuplyr_roll_paths)),
    internal = TRUE, blank = opt_blank_empty
  ),
  check_schema_cache = opt_new(
    "check_schema_cache", FALSE, "CUPLYR_CHECK_SCHEMA_CACHE",
    opt_parse_check, opt_unset_null,
    "Internal: verify the cached schema of a lazy table on every use (debug).",
    "TRUE or FALSE",
    internal = TRUE
  )
)

# The value as typed, for a message: an atomic value as character, anything else
# (a function) by its class.
opt_typed <- function(r) {
  if (is.atomic(r$raw) && length(r$raw)) {
    as.character(r$raw)[1L]
  } else {
    opt_shown(r$raw)
  }
}

# The value for a message: deparsed (its first line, as F88 showed it), except
# a function or environment, shown by its class (its deparse spans lines or
# says nothing).
opt_shown <- function(raw) {
  if (is.function(raw) || is.environment(raw)) {
    paste0("<", class(raw)[1L], ">")
  } else {
    paste(deparse(raw, nlines = 1L), collapse = "")
  }
}

# The dedupe key's value: the whole deparse, so values that differ past the
# first line (or list values) each warn once. Functions and environments share
# one key per class.
opt_key <- function(raw) {
  if (is.function(raw) || is.environment(raw)) {
    paste0("<", class(raw)[1L], ">")
  } else {
    paste(deparse(raw), collapse = "\n")
  }
}

# The dedupe key of one warning: option, where the value came from, the value.
opt_dedupe_key <- function(option, from, raw) {
  paste(option, from, opt_key(raw))
}

# Forget that the invalid-option warnings among `conds` were given, so the next
# read of each warns again (for a warning whose signal did not return).
opt_forget_warned <- function(conds) {
  for (w in conds) {
    if (inherits(w, "cuplyr_warning_invalid_option")) {
      key <- opt_dedupe_key(w$option, w$source, w$value)
      if (exists(key, envir = cuplyr_opt_state, inherits = FALSE)) {
        rm(list = key, envir = cuplyr_opt_state)
      }
    }
  }
  invisible()
}

# Re-raise the warnings `held` back. When one does not return (options(warn =
# 2)) the invalid-option warnings from it on stay un-given: the next read of
# each reports them.
opt_reraise <- function(held) {
  raised <- 0L
  on.exit(opt_forget_warned(held[seq_along(held) > raised]))
  for (w in held) {
    warning(w)
    raised <- raised + 1L
  }
  invisible()
}

# suppressWarnings(), except that an invalid-option warning passes through: a
# probe that evaluates a function reading a bad option would otherwise record
# the warning as given while nobody saw it.
suppress_probe_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (!inherits(w, "cuplyr_warning_invalid_option")) {
      tryInvokeRestart("muffleWarning")
    }
  })
}

# "cuplyr.x" or "CUPLYR_X", for a value that came from the option or the env var.
opt_source_label <- function(r) {
  if (identical(r$from, "env")) {
    sprintf("%s (option `%s`)", r$env, r$option)
  } else {
    r$option
  }
}

# Resolve `name` (without the "cuplyr." prefix): list(value, source, bad, ...).
# No warning here; `bad` is TRUE for an invalid value, which resolves to the
# default with source "default".
cuplyr_opt_read <- function(name) {
  e <- cuplyr_option_registry[[name]]
  raw <- getOption(e$option)
  from <- "option"
  if (e$unset(raw)) {
    raw <- NULL
    if (!is.na(e$env)) {
      v <- Sys.getenv(e$env, "")
      # (nzchar() first: the common case is an empty variable, and trimws() is
      # two regex calls)
      if (nzchar(v) && nzchar(trimws(v))) {
        raw <- v
        from <- "env"
      }
    }
    if (is.null(raw)) {
      return(e$default_result)
    }
  } else if (!is.null(e$blank) && e$blank(raw)) {
    return(e$default_result)
  }
  value <- e$parse(raw, from == "env")
  if (identical(value, cuplyr_invalid)) {
    return(list(value = e$default, source = "default", bad = TRUE, raw = raw,
                from = from, option = e$option, env = e$env))
  }
  list(value = value, source = from, bad = FALSE)
}

# The resolved value of option `name`; an invalid value warns once (see
# cuplyr_opt_warn()) and gives the default. `message` is an optional
# function(shown, r) building the warning text for the call site.
cuplyr_opt <- function(name, message = NULL) {
  r <- cuplyr_opt_read(name)
  if (r$bad) {
    cuplyr_opt_warn(name, r, message)
  }
  r$value
}

# As cuplyr_opt(), but the whole list (with the source).
cuplyr_opt_resolve <- function(name, message = NULL) {
  r <- cuplyr_opt_read(name)
  if (r$bad) {
    cuplyr_opt_warn(name, r, message)
  }
  r
}

# Warn about an invalid value of `name`, once per session and (option, source,
# value). Class cuplyr_warning_invalid_option; fields `option` (the R option,
# also for an env value), `value` (as given) and `source`.
cuplyr_opt_warn <- function(name, r, message = NULL) {
  e <- cuplyr_option_registry[[name]]
  # (the key first: a repeat read of a value already warned about costs one
  # deparse, and builds no text)
  key <- opt_dedupe_key(e$option, r$from, r$raw)
  if (!is.null(cuplyr_opt_state[[key]])) {
    return(invisible())
  }
  shown <- opt_shown(r$raw)
  text <- (message %||% e$message)
  text <- if (is.function(text)) {
    text(shown, r)
  } else {
    what <- if (identical(r$from, "env")) {
      sprintf("Environment variable %s (option `%s`) is %s", e$env, e$option, shown)
    } else {
      sprintf("`options(%s = )` is %s", e$option, shown)
    }
    sprintf("%s, which is not %s. Using %s.", what, e$expect, e$default_label)
  }
  # (recorded only once the text exists: a failing builder must not silence
  # every later read. Recorded during the signal, so a handler that reads the
  # option again does not warn again; and forgotten when the signal does not
  # return: when warnings are errors (options(warn = 2)) or a tryCatch() exits,
  # the error may be caught by code that never shows it (a verb trying an
  # expression as a constant), so the next read must warn again.)
  cuplyr_opt_state[[key]] <- TRUE
  done <- FALSE
  on.exit(if (!done) rm(list = key, envir = cuplyr_opt_state))
  rlang::warn(text, class = "cuplyr_warning_invalid_option",
              option = e$option, value = r$raw, source = r$from)
  done <- TRUE
  invisible()
}

# cuplyr_opt_resolve(), but a value that can't even be inspected (a function:
# the readers fail on it) is reported as invalid, with the usual one warning.
# Only the read is in the tryCatch: the warning is signalled outside it, so
# when warnings are errors (options(warn = 2), a calling handler) that error
# reaches the caller instead of being taken for an unreadable value. A read can
# fail only on an R option (an environment variable is a string, which every
# reader handles), so the failed value is the option's.
opt_resolve_safe <- function(name) {
  r <- tryCatch(cuplyr_opt_read(name), error = function(err) {
    e <- cuplyr_option_registry[[name]]
    list(value = e$default, source = "default", bad = TRUE,
         raw = getOption(e$option), from = "option", option = e$option,
         env = e$env)
  })
  if (r$bad) {
    cuplyr_opt_warn(name, r)
  }
  r
}

opt_render <- function(x) {
  if (length(x) != 1L || is.na(x)) {
    NA_character_
  } else if (is.numeric(x)) {
    format(x, scientific = FALSE, trim = TRUE, digits = 15)
  } else {
    as.character(x)
  }
}

#' cuplyr options
#'
#' Every cuplyr option is set with `options()` (scope one with
#' `withr::local_options()`) and, for most, also with an environment variable.
#' The option wins over the environment variable, which wins over the default.
#' `cuplyr_options()` lists them all with the value in use and where it came
#' from.
#'
#' @section Options:
#' * `cuplyr.exec_mode` (environment variable `CUPLYR_EXEC_MODE`): `"lazy"`
#'   (the default) or `"eager"`. Whether `tbl_gpu()` makes lazy tables or runs
#'   each verb at once; `tbl_gpu(lazy = )` overrides it. See [tbl_gpu()].
#' * `cuplyr.fallback`: `"silent"` (the default), `"warn"` or `"error"`
#'   (`"warning"` means `"warn"`). What a CPU fallback does. See
#'   [cuplyr-fallback].
#' * `cuplyr.plan_break`: `"silent"` (the default) or `"warn"`. What a plan
#'   break does. See [cuplyr-fallback].
#' * `cuplyr.memory_resource` (`CUPLYR_MEMORY_RESOURCE`): `"async"` (the
#'   default), `"pool"`, or `"cuda"` (`"none"`; no memory resource). See
#'   [gpu_memory_state()].
#' * `cuplyr.memory_pool` (`CUPLYR_MEMORY_POOL`): `TRUE` (the default) or
#'   `FALSE`; `0`, `"no"` and `"off"` also switch the resource off, and
#'   `1`, `"yes"` and `"on"` on.
#' * `cuplyr.pool_initial` (`CUPLYR_POOL_INITIAL`): the initial pool size in
#'   bytes, 0 or more. Unset: a quarter of the free GPU memory, at most 1 GiB.
#' * `cuplyr.release_threshold` (`CUPLYR_RELEASE_THRESHOLD`): bytes the async
#'   resource keeps before returning memory to the driver, 0 or more. Unset
#'   (the default): no automatic release.
#' * `cuplyr.roll_median` (`CUPLYR_ROLL_MEDIAN`): `"exact"` (the default) or
#'   `"frollmedian"`, the engine of [roll_median()].
#'
#' The memory options are read once, when cuplyr is loaded: changing them
#' later has no effect until the next R session.
#'
#' @section Internal switches:
#' Listed by `cuplyr_options(internal = TRUE)`; not for normal use.
#' * `cuplyr.roll_path` (`CUPLYR_ROLL_PATH`): `"auto"` (the default),
#'   `"scans"` or `"cudf"`; forces the path of rolling-window functions.
#' * `cuplyr.check_schema_cache` (`CUPLYR_CHECK_SCHEMA_CACHE`): `TRUE` or
#'   `FALSE` (the default); checks the cached schema of every lazy table on use
#'   (the environment variable is `1`, `true` or `yes`).
#'
#' @section Invalid values:
#' A value that is not allowed (from the option or the environment variable)
#' warns once per session and value, with class
#' `cuplyr_warning_invalid_option` (fields `option`, `value` and `source`).
#' The message names the allowed values, and the default is used. A set option
#' decides even when invalid: the environment variable is then not consulted.
#' An invalid value never makes `cuplyr_options()` or a verb fail; a value that
#' cannot even be inspected (a function, say) is reported as invalid. When
#' warnings are errors (`options(warn = 2)`), the warning is raised as an error
#' from the call that read the value, naming the option or environment
#' variable, and every later read raises it again until the value is fixed,
#' unless something in between caught the warning and raised it again (a dplyr
#' verb run on the CPU does). A warning counts as given only when its signal
#' returned (one caught by `tryCatch(warning = )` is given again on the next
#' read).
#'
#' A blank option is not invalid. `""` means the default, without a warning and
#' without consulting the environment variable, for `cuplyr.memory_resource`,
#' `cuplyr.memory_pool`, `cuplyr.pool_initial`, `cuplyr.roll_median` and
#' `cuplyr.roll_path` (`NA` too for `cuplyr.pool_initial`). For
#' `cuplyr.exec_mode` and `cuplyr.release_threshold`, `""` and `NA` count as
#' unset, so the environment variable decides, as `NA` does for
#' `cuplyr.memory_resource` and `cuplyr.memory_pool`. An environment variable
#' that is empty or only spaces counts as unset for every option: the default
#' is used, without a warning.
#'
#' `cuplyr.memory_resource` reads the first element of a vector, as it always
#' did: `c("pool", "async")` means `"pool"`.
#'
#' @examples
#' cuplyr_options()
#'
#' # Scope an option to a block
#' withr::with_options(list(cuplyr.fallback = "warn"), {
#'   # ... a pipeline that should say when it leaves the GPU
#' })
#' withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"), cuplyr_options()[1, ])
#' @name cuplyr-options
#' @aliases cuplyr-options
#' @aliases cuplyr.exec_mode cuplyr.fallback cuplyr.plan_break
#' @aliases cuplyr.memory_resource cuplyr.memory_pool cuplyr.pool_initial
#' @aliases cuplyr.release_threshold cuplyr.roll_median cuplyr.roll_path
#' @aliases cuplyr.check_schema_cache
NULL

#' @rdname cuplyr-options
#' @param internal Also list the internal switches (`cuplyr.roll_path`,
#'   `cuplyr.check_schema_cache`).
#' @return `cuplyr_options()` returns a data frame of class `cuplyr_options`
#'   with one row per option and the character columns `option`, `value` (the
#'   value cuplyr uses; `NA` for an unset one), `source` (`"option"`, `"env"`
#'   or `"default"`), `env_var` (`NA` when there is none) and `description`.
#'   An invalid value is reported as the default, after its warning. Printing
#'   it shows the same table.
#' @export
cuplyr_options <- function(internal = FALSE) {
  reg <- cuplyr_option_registry
  if (!isTRUE(internal)) {
    reg <- reg[!vapply(reg, function(e) e$internal, NA)]
  }
  res <- lapply(reg, function(e) opt_resolve_safe(e$name))
  out <- data.frame(
    option = vapply(reg, function(e) e$option, ""),
    value = vapply(res, function(r) opt_render(r$value), ""),
    source = vapply(res, function(r) r$source, ""),
    env_var = vapply(reg, function(e) e$env, ""),
    description = vapply(reg, function(e) e$description, ""),
    stringsAsFactors = FALSE, row.names = NULL
  )
  class(out) <- c("cuplyr_options", "data.frame")
  out
}

#' @export
print.cuplyr_options <- function(x, ...) {
  cat("<cuplyr options>\n")
  for (i in seq_len(nrow(x))) {
    value <- if (is.na(x$value[i])) "<unset>" else x$value[i]
    env <- if (is.na(x$env_var[i])) "" else paste0(", ", x$env_var[i])
    cat(sprintf("%s = %s  (%s%s)\n    %s\n", x$option[i], value, x$source[i], env,
                x$description[i]))
  }
  invisible(x)
}
