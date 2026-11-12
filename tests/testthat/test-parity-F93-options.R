# F93: one options registry, validated and documented.
#
# Contract (scratchpad/parity/features/F93.md, Contract + Design):
# - Every cuplyr option is declared once (R/options.R): name, default, env var
#   (or NA), allowed values / validator, description, internal flag.
# - User options: cuplyr.exec_mode, cuplyr.fallback, cuplyr.plan_break,
#   cuplyr.memory_resource, cuplyr.memory_pool, cuplyr.pool_initial,
#   cuplyr.release_threshold, cuplyr.roll_median. Internal switches:
#   cuplyr.roll_path, cuplyr.check_schema_cache. cuplyr.gpu_available is load
#   state, not an option; cuplyr.lazy / cuplyr.verbose are gone.
# - An invalid value (option or env var) warns once per session and value with
#   class `cuplyr_warning_invalid_option` (fields `option` = the R option name,
#   `value` = the value as given), the message names the allowed values (and
#   the env var when the value came from it), and the default is used. A set
#   option decides even when invalid (the env var is not consulted then).
#   Never an error.
# - cuplyr_options() (exported) returns a data frame of class
#   "cuplyr_options": columns option, value (character, the value cuplyr
#   uses), source ("option" / "env" / "default"), env_var, description; one row
#   per user option (internal = TRUE adds the internal switches). An invalid
#   value is reported as the default with source "default" (after warning).
# - One help page ?"cuplyr-options" with an alias for every option name.
# - Valid values behave exactly as before.
#
# Invalid values below are unique to this file ("...-f93...") so the
# once-per-session state of other files can't hide a warning here.

f93_user_opts <- c(
  "cuplyr.exec_mode", "cuplyr.fallback", "cuplyr.plan_break",
  "cuplyr.memory_resource", "cuplyr.memory_pool", "cuplyr.pool_initial",
  "cuplyr.release_threshold", "cuplyr.roll_median"
)
f93_internal_opts <- c("cuplyr.roll_path", "cuplyr.check_schema_cache")

f93_env_vars <- c(
  cuplyr.exec_mode = "CUPLYR_EXEC_MODE",
  cuplyr.fallback = NA,
  cuplyr.plan_break = NA,
  cuplyr.memory_resource = "CUPLYR_MEMORY_RESOURCE",
  cuplyr.memory_pool = "CUPLYR_MEMORY_POOL",
  cuplyr.pool_initial = "CUPLYR_POOL_INITIAL",
  cuplyr.release_threshold = "CUPLYR_RELEASE_THRESHOLD",
  cuplyr.roll_median = "CUPLYR_ROLL_MEDIAN",
  cuplyr.roll_path = "CUPLYR_ROLL_PATH",
  cuplyr.check_schema_cache = "CUPLYR_CHECK_SCHEMA_CACHE"
)

# Clear every switch so the package defaults decide.
f93_local_clean <- function(env = parent.frame()) {
  withr::local_options(
    cuplyr.exec_mode = NULL, cuplyr.fallback = NULL, cuplyr.plan_break = NULL,
    cuplyr.memory_resource = NULL, cuplyr.memory_pool = NULL,
    cuplyr.pool_initial = NULL, cuplyr.release_threshold = NULL,
    cuplyr.roll_median = NULL, cuplyr.roll_path = NULL,
    .local_envir = env
  )
  withr::local_envvar(
    CUPLYR_EXEC_MODE = NA, CUPLYR_MEMORY_RESOURCE = NA, CUPLYR_MEMORY_POOL = NA,
    CUPLYR_POOL_INITIAL = NA, CUPLYR_RELEASE_THRESHOLD = NA,
    CUPLYR_ROLL_MEDIAN = NA, CUPLYR_ROLL_PATH = NA,
    .local_envir = env
  )
}

# Evaluate `expr`, collecting (and muffling) every warning.
f93_capture <- function(expr) {
  warns <- list()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warns[[length(warns) + 1L]] <<- w
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warns = warns)
}

f93_invalid <- function(got) {
  Filter(function(w) inherits(w, "cuplyr_warning_invalid_option"), got$warns)
}

# One invalid-option warning, for `option`, quoting `value` and every allowed
# value in its message.
f93_expect_one_invalid <- function(got, option, value, allowed = character(),
                                   mentions = character()) {
  inv <- f93_invalid(got)
  expect_length(inv, 1L)
  expect_length(got$warns, 1L)
  if (length(inv) != 1L) {
    return(invisible())
  }
  expect_identical(inv[[1L]]$option, option)
  expect_identical(inv[[1L]]$value, value)
  msg <- conditionMessage(inv[[1L]])
  expect_match(msg, option, fixed = TRUE)
  expect_match(msg, as.character(value), fixed = TRUE)
  for (a in allowed) {
    expect_match(msg, paste0("\"", a, "\""), fixed = TRUE, info = a)
  }
  for (m in mentions) {
    expect_match(msg, m, fixed = TRUE, info = m)
  }
}

f93_row <- function(tab, option) {
  tab[tab$option == option, , drop = FALSE]
}

# ---------------------------------------------------------------------------
# (1) cuplyr_options(): the registry, as a table

test_that("F93: cuplyr_options() is exported and lists every user option once", {
  f93_local_clean()
  expect_true("cuplyr_options" %in% getNamespaceExports("cuplyr"))
  tab <- cuplyr_options()
  expect_s3_class(tab, "cuplyr_options")
  expect_s3_class(tab, "data.frame")
  for (col in c("option", "value", "source", "env_var", "description")) {
    expect_true(col %in% names(tab), info = col)
  }
  expect_type(tab$option, "character")
  expect_type(tab$value, "character")
  expect_type(tab$source, "character")
  expect_type(tab$env_var, "character")
  expect_type(tab$description, "character")
  expect_setequal(tab$option, f93_user_opts)
  expect_false(anyDuplicated(tab$option) > 0L)
  for (bad in c("cuplyr.gpu_available", "cuplyr.lazy", "cuplyr.verbose",
                f93_internal_opts)) {
    expect_false(bad %in% tab$option, info = bad)
  }
  expect_true(all(nzchar(tab$description)))
  for (o in f93_user_opts) {
    expect_identical(f93_row(tab, o)$env_var, unname(f93_env_vars[[o]]), info = o)
  }
})

test_that("F93: cuplyr_options(internal = TRUE) adds the internal switches", {
  f93_local_clean()
  tab <- cuplyr_options(internal = TRUE)
  expect_s3_class(tab, "cuplyr_options")
  expect_setequal(tab$option, c(f93_user_opts, f93_internal_opts))
  for (o in f93_internal_opts) {
    expect_identical(f93_row(tab, o)$env_var, unname(f93_env_vars[[o]]), info = o)
  }
})

test_that("F93: with nothing set, every option reports its default", {
  f93_local_clean()
  got <- f93_capture(cuplyr_options())
  expect_length(got$warns, 0L)
  tab <- got$value
  expect_true(all(tab$source == "default"))
  expect_identical(f93_row(tab, "cuplyr.exec_mode")$value, "lazy")
  expect_identical(f93_row(tab, "cuplyr.fallback")$value, "silent")
  expect_identical(f93_row(tab, "cuplyr.plan_break")$value, "silent")
  expect_identical(f93_row(tab, "cuplyr.memory_resource")$value, "async")
  expect_identical(f93_row(tab, "cuplyr.memory_pool")$value, "TRUE")
  expect_identical(f93_row(tab, "cuplyr.roll_median")$value, "exact")
  tab2 <- cuplyr_options(internal = TRUE)
  expect_identical(f93_row(tab2, "cuplyr.roll_path")$value, "auto")
})

test_that("F93: cuplyr_options() reports the source: option, env var, default", {
  f93_local_clean()
  withr::local_options(cuplyr.exec_mode = "eager", cuplyr.fallback = "warn",
                       cuplyr.roll_median = "frollmedian",
                       cuplyr.pool_initial = 268435456)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = "pool", CUPLYR_MEMORY_POOL = "0",
                      CUPLYR_RELEASE_THRESHOLD = "1000000000")
  got <- f93_capture(cuplyr_options())
  expect_length(got$warns, 0L)
  tab <- got$value
  src <- stats::setNames(tab$source, tab$option)
  val <- stats::setNames(tab$value, tab$option)
  expect_identical(src[["cuplyr.exec_mode"]], "option")
  expect_identical(val[["cuplyr.exec_mode"]], "eager")
  expect_identical(src[["cuplyr.fallback"]], "option")
  expect_identical(val[["cuplyr.fallback"]], "warn")
  expect_identical(src[["cuplyr.roll_median"]], "option")
  expect_identical(val[["cuplyr.roll_median"]], "frollmedian")
  expect_identical(src[["cuplyr.pool_initial"]], "option")
  expect_identical(val[["cuplyr.pool_initial"]], "268435456")
  expect_identical(src[["cuplyr.memory_resource"]], "env")
  expect_identical(val[["cuplyr.memory_resource"]], "pool")
  expect_identical(src[["cuplyr.memory_pool"]], "env")
  expect_identical(val[["cuplyr.memory_pool"]], "FALSE")
  expect_identical(src[["cuplyr.release_threshold"]], "env")
  expect_equal(as.numeric(val[["cuplyr.release_threshold"]]), 1e9)
  expect_identical(src[["cuplyr.plan_break"]], "default")

  # the option beats the env var
  withr::local_envvar(CUPLYR_EXEC_MODE = "lazy", CUPLYR_ROLL_MEDIAN = "exact")
  tab <- cuplyr_options()
  expect_identical(f93_row(tab, "cuplyr.exec_mode")$source, "option")
  expect_identical(f93_row(tab, "cuplyr.exec_mode")$value, "eager")
  expect_identical(f93_row(tab, "cuplyr.roll_median")$value, "frollmedian")

  # the env var alone
  withr::local_options(cuplyr.exec_mode = NULL, cuplyr.memory_pool = FALSE)
  tab <- cuplyr_options()
  expect_identical(f93_row(tab, "cuplyr.exec_mode")$source, "env")
  expect_identical(f93_row(tab, "cuplyr.exec_mode")$value, "lazy")
  expect_identical(f93_row(tab, "cuplyr.memory_pool")$source, "option")
  expect_identical(f93_row(tab, "cuplyr.memory_pool")$value, "FALSE")
})

test_that("F93: cuplyr.fallback = 'warning' is reported as 'warn'", {
  f93_local_clean()
  withr::local_options(cuplyr.fallback = "warning")
  got <- f93_capture(cuplyr_options())
  expect_length(got$warns, 0L)
  expect_identical(f93_row(got$value, "cuplyr.fallback")$value, "warn")
  expect_identical(f93_row(got$value, "cuplyr.fallback")$source, "option")
  expect_identical(cuplyr_notice_mode("cuplyr.fallback"), "warn")
})

test_that("F93: printing cuplyr_options() shows each option, its value and description", {
  f93_local_clean()
  withr::local_options(cuplyr.exec_mode = "eager")
  tab <- cuplyr_options()
  out <- paste(utils::capture.output(print(tab)), collapse = "\n")
  for (o in f93_user_opts) {
    expect_match(out, o, fixed = TRUE, info = o)
  }
  expect_match(out, "eager", fixed = TRUE)
  expect_match(out, "CUPLYR_EXEC_MODE", fixed = TRUE)
  # the descriptions are shown (at least their first words)
  d <- f93_row(tab, "cuplyr.exec_mode")$description
  expect_match(out, substr(d, 1L, 12L), fixed = TRUE)
  expect_invisible(print(tab))
})

# ---------------------------------------------------------------------------
# (2) invalid values: warn once, use the default

test_that("F93: cuplyr.exec_mode = 'Lazy' warns once and the default (lazy) is used", {
  f93_local_clean()
  # a set option decides even when invalid: the env var is not consulted
  withr::local_envvar(CUPLYR_EXEC_MODE = "eager")
  withr::local_options(cuplyr.exec_mode = "Lazy")
  got <- f93_capture(resolve_exec_mode(NULL))
  expect_identical(got$value, "lazy")
  f93_expect_one_invalid(got, "cuplyr.exec_mode", "Lazy", c("lazy", "eager"))

  # the same value again: silent
  got <- f93_capture(resolve_exec_mode(NULL))
  expect_identical(got$value, "lazy")
  expect_length(got$warns, 0L)
  got <- f93_capture(cuplyr_options())
  expect_length(got$warns, 0L)
  expect_identical(f93_row(got$value, "cuplyr.exec_mode")$value, "lazy")
  expect_identical(f93_row(got$value, "cuplyr.exec_mode")$source, "default")

  # an explicit argument still wins, without any warning
  got <- f93_capture(resolve_exec_mode(FALSE))
  expect_identical(got$value, "eager")
  expect_length(got$warns, 0L)
})

test_that("F93: an invalid CUPLYR_EXEC_MODE warns once, naming the env var", {
  f93_local_clean()
  withr::local_envvar(CUPLYR_EXEC_MODE = "eagre-f93")
  got <- f93_capture(resolve_exec_mode(NULL))
  expect_identical(got$value, "lazy")
  f93_expect_one_invalid(got, "cuplyr.exec_mode", "eagre-f93", c("lazy", "eager"),
                         mentions = "CUPLYR_EXEC_MODE")
  got <- f93_capture(resolve_exec_mode(NULL))
  expect_length(got$warns, 0L)
})

test_that("F93: an invalid exec mode warns once over many tbl_gpu() calls", {
  skip_if_no_gpu()
  f93_local_clean()
  withr::local_options(cuplyr.exec_mode = "eager-f93-many")
  got <- f93_capture({
    a <- tbl_gpu(data.frame(x = 1:3))
    b <- tbl_gpu(data.frame(x = 4:6)) |> dplyr::mutate(y = x + 1L)
    c <- tbl_gpu(data.frame(x = 7:9)) |> dplyr::filter(x > 7L)
    list(a, b, c)
  })
  expect_length(f93_invalid(got), 1L)
  expect_length(got$warns, 1L)
  for (t in got$value) {
    expect_identical(.subset2(t, "exec_mode"), "lazy")
  }
  expect_identical(dplyr::collect(got$value[[3]])$x, c(8L, 9L))
})

test_that("F93: an invalid cuplyr.fallback / cuplyr.plan_break warns once across cuplyr_options() and the notify path", {
  f93_local_clean()
  withr::local_options(cuplyr.fallback = "loud-f93", cuplyr.plan_break = "loud-f93")
  got <- f93_capture(cuplyr_options())
  inv <- f93_invalid(got)
  expect_length(inv, 2L)
  expect_length(got$warns, 2L)
  expect_setequal(vapply(inv, function(w) w$option, ""),
                  c("cuplyr.fallback", "cuplyr.plan_break"))
  for (w in inv) {
    expect_identical(w$value, "loud-f93")
    expect_match(conditionMessage(w), "\"silent\"", fixed = TRUE)
    expect_match(conditionMessage(w), "\"warn\"", fixed = TRUE)
  }
  expect_identical(f93_row(got$value, "cuplyr.fallback")$value, "silent")
  expect_identical(f93_row(got$value, "cuplyr.fallback")$source, "default")
  # the notify path shares the once-per-session state
  got <- f93_capture(list(cuplyr_notice_mode("cuplyr.fallback"),
                          cuplyr_notice_mode("cuplyr.plan_break")))
  expect_identical(got$value, list("silent", "silent"))
  expect_length(got$warns, 0L)
})

test_that("F93: an invalid cuplyr.roll_median warns once and the exact engine is used", {
  f93_local_clean()
  withr::local_options(cuplyr.roll_median = "fast-f93")
  got <- f93_capture(roll_median_engine())
  expect_identical(got$value, "exact")
  f93_expect_one_invalid(got, "cuplyr.roll_median", "fast-f93", c("exact", "frollmedian"))
  got <- f93_capture(roll_median(c(1, 5, 2, 8), 3))
  expect_length(got$warns, 0L)
  expect_identical(got$value, c(NA, NA, 2, 5))

  withr::local_options(cuplyr.roll_median = NULL)
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "Exact-f93")
  got <- f93_capture(roll_median(c(1, 5, 2, 8), 3))
  expect_identical(got$value, c(NA, NA, 2, 5))
  f93_expect_one_invalid(got, "cuplyr.roll_median", "Exact-f93", c("exact", "frollmedian"),
                         mentions = "CUPLYR_ROLL_MEDIAN")
  got <- f93_capture(roll_median_engine())
  expect_length(got$warns, 0L)
})

test_that("F93: an invalid cuplyr.roll_path warns once and the auto path is used", {
  f93_local_clean()
  withr::local_options(cuplyr.roll_path = "gpu-f93")
  got <- f93_capture(roll_path_option())
  expect_identical(got$value, "auto")
  f93_expect_one_invalid(got, "cuplyr.roll_path", "gpu-f93", c("auto", "scans", "cudf"))
  got <- f93_capture(roll_path_option())
  expect_length(got$warns, 0L)

  withr::local_options(cuplyr.roll_path = NULL)
  withr::local_envvar(CUPLYR_ROLL_PATH = "Scans-f93")
  got <- f93_capture(roll_path_option())
  expect_identical(got$value, "auto")
  f93_expect_one_invalid(got, "cuplyr.roll_path", "Scans-f93", c("auto", "scans", "cudf"),
                         mentions = "CUPLYR_ROLL_PATH")
})

test_that("F93: an invalid roll path never errors a pipeline", {
  skip_if_no_gpu()
  f93_local_clean()
  withr::local_options(cuplyr.roll_path = "fastest-f93")
  df <- data.frame(x = c(1.5, 2.5, 4, 8, 16))
  got <- f93_capture(dplyr::collect(dplyr::mutate(tbl_gpu(df), y = roll_mean(x, 2))))
  expect_length(f93_invalid(got), 1L)
  expect_equal(got$value$y, c(NA, 2, 3.25, 6, 12))
})

test_that("F93: an invalid memory resource kind warns with the class, once, and means async", {
  f93_local_clean()
  withr::local_options(cuplyr.memory_resource = "managed-f93")
  got <- f93_capture(memory_resource_config())
  expect_identical(got$value$kind, "async")
  expect_true(got$value$enabled)
  f93_expect_one_invalid(got, "cuplyr.memory_resource", "managed-f93",
                         c("async", "pool", "cuda"))
  got <- f93_capture(memory_resource_config())
  expect_length(got$warns, 0L)
  expect_identical(got$value$kind, "async")

  withr::local_options(cuplyr.memory_resource = NULL)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = "Managed-F93")
  got <- f93_capture(memory_resource_config())
  expect_identical(got$value$kind, "async")
  f93_expect_one_invalid(got, "cuplyr.memory_resource", "Managed-F93",
                         c("async", "pool", "cuda"), mentions = "CUPLYR_MEMORY_RESOURCE")
})

test_that("F93: cuplyr.memory_pool accepts on/off spellings silently; anything else warns and means on", {
  f93_local_clean()
  for (off in list(FALSE, "0", "false", "no", "off", " OFF ", 0)) {
    withr::with_options(list(cuplyr.memory_pool = off), {
      got <- f93_capture(memory_resource_config())
      expect_length(got$warns, 0L)
      expect_false(got$value$enabled, label = paste(deparse(off), collapse = ""))
    })
  }
  for (on in list(TRUE, "1", "true", "yes", "on", " TRUE ", 1)) {
    withr::with_options(list(cuplyr.memory_pool = on), {
      got <- f93_capture(memory_resource_config())
      expect_length(got$warns, 0L)
      expect_true(got$value$enabled, label = paste(deparse(on), collapse = ""))
    })
  }
  for (on in c("0", "1", "true", "false", "yes", "no", "on", "off")) {
    withr::with_envvar(c(CUPLYR_MEMORY_POOL = on), {
      got <- f93_capture(memory_resource_config())
      expect_length(got$warns, 0L)
    })
  }
  withr::with_options(list(cuplyr.memory_pool = "flase-f93"), {
    got <- f93_capture(memory_resource_config())
    expect_true(got$value$enabled)
    f93_expect_one_invalid(got, "cuplyr.memory_pool", "flase-f93")
  })
  withr::with_envvar(c(CUPLYR_MEMORY_POOL = "maybe-f93"), {
    got <- f93_capture(memory_resource_config())
    expect_true(got$value$enabled)
    f93_expect_one_invalid(got, "cuplyr.memory_pool", "maybe-f93",
                           mentions = "CUPLYR_MEMORY_POOL")
    got <- f93_capture(memory_resource_config())
    expect_length(got$warns, 0L)
  })
})

test_that("F93: an invalid cuplyr.pool_initial warns once and the default size is used", {
  f93_local_clean()
  withr::with_options(list(cuplyr.pool_initial = 2^28), {
    got <- f93_capture(memory_resource_config())
    expect_length(got$warns, 0L)
    expect_identical(got$value$initial, 2^28)
  })
  withr::with_envvar(c(CUPLYR_POOL_INITIAL = "67108864"), {
    got <- f93_capture(memory_resource_config())
    expect_length(got$warns, 0L)
    expect_identical(got$value$initial, 67108864)
  })
  withr::with_options(list(cuplyr.pool_initial = "lots-f93"), {
    got <- f93_capture(memory_resource_config())
    f93_expect_one_invalid(got, "cuplyr.pool_initial", "lots-f93")
    expect_true(is.finite(got$value$initial))
    expect_true(got$value$initial >= 0 && got$value$initial <= 2^30)
    got <- f93_capture(memory_resource_config())
    expect_length(got$warns, 0L)
  })
  withr::with_options(list(cuplyr.pool_initial = -7.93), {
    got <- f93_capture(memory_resource_config())
    expect_length(f93_invalid(got), 1L)
    expect_true(got$value$initial >= 0 && got$value$initial <= 2^30)
  })
  withr::with_envvar(c(CUPLYR_POOL_INITIAL = "1GiB-f93"), {
    got <- f93_capture(memory_resource_config())
    f93_expect_one_invalid(got, "cuplyr.pool_initial", "1GiB-f93",
                           mentions = "CUPLYR_POOL_INITIAL")
    expect_true(got$value$initial >= 0 && got$value$initial <= 2^30)
  })
})

test_that("F93: an invalid cuplyr.release_threshold warns with the class, once", {
  f93_local_clean()
  withr::with_options(list(cuplyr.release_threshold = "2GB-f93"), {
    got <- f93_capture(memory_resource_config())
    expect_true(is.na(got$value$release_threshold))
    f93_expect_one_invalid(got, "cuplyr.release_threshold", "2GB-f93")
    got <- f93_capture(memory_resource_config())
    expect_length(got$warns, 0L)
  })
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "-93"), {
    got <- f93_capture(memory_resource_config())
    expect_true(is.na(got$value$release_threshold))
    f93_expect_one_invalid(got, "cuplyr.release_threshold", "-93",
                           mentions = "CUPLYR_RELEASE_THRESHOLD")
  })
})

test_that("F93: invalid values are reported by cuplyr_options() as the default, after one warning", {
  f93_local_clean()
  withr::local_options(cuplyr.roll_median = "median-f93-tab",
                       cuplyr.memory_resource = "gpu-f93-tab")
  got <- f93_capture(cuplyr_options())
  inv <- f93_invalid(got)
  expect_setequal(vapply(inv, function(w) w$option, ""),
                  c("cuplyr.roll_median", "cuplyr.memory_resource"))
  expect_length(got$warns, 2L)
  tab <- got$value
  expect_identical(f93_row(tab, "cuplyr.roll_median")$value, "exact")
  expect_identical(f93_row(tab, "cuplyr.roll_median")$source, "default")
  expect_identical(f93_row(tab, "cuplyr.memory_resource")$value, "async")
  expect_identical(f93_row(tab, "cuplyr.memory_resource")$source, "default")
  got <- f93_capture(list(cuplyr_options(), roll_median_engine()))
  expect_length(got$warns, 0L)
})

# ---------------------------------------------------------------------------
# (3) valid values behave as before

test_that("F93: exec mode precedence is argument > option > env var > lazy", {
  f93_local_clean()
  got <- f93_capture({
    r <- c(
      resolve_exec_mode(NULL),
      withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"), resolve_exec_mode(NULL)),
      withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"),
        withr::with_options(list(cuplyr.exec_mode = "lazy"), resolve_exec_mode(NULL))),
      withr::with_options(list(cuplyr.exec_mode = "eager"), resolve_exec_mode(NULL)),
      withr::with_options(list(cuplyr.exec_mode = "eager"), resolve_exec_mode(TRUE)),
      withr::with_options(list(cuplyr.exec_mode = "lazy"), resolve_exec_mode("eager")),
      withr::with_envvar(c(CUPLYR_EXEC_MODE = "lazy"), resolve_exec_mode(FALSE))
    )
    r
  })
  expect_identical(got$value, c("lazy", "eager", "lazy", "eager", "lazy", "eager", "eager"))
  expect_length(got$warns, 0L)
  # a bad argument is still an error, as before
  expect_error(resolve_exec_mode("Lazy"), "lazy=")
})

test_that("F93: valid exec modes reach tbl_gpu() unchanged", {
  skip_if_no_gpu()
  f93_local_clean()
  df <- data.frame(x = 1:3)
  withr::with_options(list(cuplyr.exec_mode = "eager"),
                      expect_identical(.subset2(tbl_gpu(df), "exec_mode"), "eager"))
  withr::with_envvar(c(CUPLYR_EXEC_MODE = "eager"),
                     expect_identical(.subset2(tbl_gpu(df), "exec_mode"), "eager"))
  expect_identical(.subset2(tbl_gpu(df), "exec_mode"), "lazy")
  withr::with_options(list(cuplyr.exec_mode = "eager"),
                      expect_identical(.subset2(tbl_gpu(df, lazy = TRUE), "exec_mode"), "lazy"))
})

test_that("F93: valid notice, roll and memory values are read as before, silently", {
  f93_local_clean()
  got <- f93_capture({
    list(
      withr::with_options(list(cuplyr.fallback = "warning"), cuplyr_notice_mode("cuplyr.fallback")),
      withr::with_options(list(cuplyr.fallback = "error"), cuplyr_notice_mode("cuplyr.fallback")),
      withr::with_options(list(cuplyr.plan_break = "warn"), cuplyr_notice_mode("cuplyr.plan_break")),
      cuplyr_notice_mode("cuplyr.fallback"),
      withr::with_options(list(cuplyr.roll_median = "frollmedian"), roll_median_engine()),
      withr::with_envvar(c(CUPLYR_ROLL_MEDIAN = "frollmedian"), roll_median_engine()),
      roll_median_engine(),
      withr::with_envvar(c(CUPLYR_ROLL_PATH = "cudf"), roll_path_option()),
      withr::with_envvar(c(CUPLYR_ROLL_PATH = "cudf"),
        withr::with_options(list(cuplyr.roll_path = "scans"), roll_path_option())),
      withr::with_options(list(cuplyr.memory_resource = " CUDA "), memory_resource_config()$enabled),
      withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "pool"), memory_resource_config()$kind),
      withr::with_envvar(c(CUPLYR_MEMORY_RESOURCE = "async"),
        withr::with_options(list(cuplyr.memory_resource = "pool"), memory_resource_config()$kind)),
      withr::with_options(list(cuplyr.release_threshold = 0), memory_resource_config()$release_threshold),
      withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "268435456"),
                         memory_resource_config()$release_threshold),
      memory_resource_config()$release_threshold
    )
  })
  expect_length(got$warns, 0L)
  expect_identical(got$value, list(
    "warn", "error", "warn", "silent",
    "frollmedian", "frollmedian", "exact",
    "cudf", "scans",
    FALSE, "pool", "pool",
    0, 268435456, NA_real_
  ))
})

test_that("F93: the dead options cuplyr.lazy / cuplyr.verbose are not set at load", {
  expect_null(getOption("cuplyr.lazy"))
  expect_null(getOption("cuplyr.verbose"))
  expect_true(is.logical(getOption("cuplyr.gpu_available")))
})

# ---------------------------------------------------------------------------
# (4) one help page

test_that("F93: ?cuplyr-options documents every option and is the only page aliasing them", {
  root <- testthat::test_path("..", "..")
  skip_if_not(file.exists(file.path(root, "DESCRIPTION")))
  rd_file <- file.path(root, "man", "cuplyr-options.Rd")
  expect_true(file.exists(rd_file))
  skip_if_not(file.exists(rd_file))
  rd <- tools::parse_Rd(rd_file)
  tags <- vapply(rd, function(x) attr(x, "Rd_tag"), "")
  aliases <- vapply(rd[tags == "\\alias"], function(x) paste(unlist(x), collapse = ""), "")
  expect_true("cuplyr-options" %in% aliases)
  expect_true("cuplyr_options" %in% aliases)
  for (o in c(f93_user_opts, f93_internal_opts)) {
    expect_true(o %in% aliases, info = o)
  }
  txt <- paste(readLines(rd_file), collapse = "\n")
  expect_no_match(txt, "\\keyword{internal}", fixed = TRUE)
  for (e in stats::na.omit(f93_env_vars)) {
    expect_match(txt, e, fixed = TRUE, info = e)
  }
  expect_match(txt, "local_options", fixed = TRUE)
  expect_match(txt, "cuplyr_warning_invalid_option", fixed = TRUE)
  for (v in c("\"lazy\"", "\"eager\"", "\"silent\"", "\"warn\"", "\"error\"",
              "\"async\"", "\"pool\"", "\"cuda\"", "\"exact\"", "\"frollmedian\"")) {
    expect_match(txt, v, fixed = TRUE, info = v)
  }

  # no other page claims an option name as an alias
  others <- setdiff(list.files(file.path(root, "man"), pattern = "\\.Rd$", full.names = TRUE),
                    normalizePath(rd_file))
  others <- others[basename(others) != "cuplyr-options.Rd"]
  for (f in others) {
    lines <- readLines(f, warn = FALSE)
    al <- sub("^\\\\alias\\{(.*)\\}\\s*$", "\\1", grep("^\\\\alias\\{", lines, value = TRUE))
    expect_length(intersect(al, c(f93_user_opts, f93_internal_opts)), 0L)
  }

  # the behaviour pages link to it
  for (page in c("cuplyr-fallback.Rd", "gpu_memory_state.Rd", "tbl_gpu.Rd")) {
    p <- file.path(root, "man", page)
    expect_true(file.exists(p), info = page)
    if (file.exists(p)) {
      expect_match(paste(readLines(p), collapse = "\n"), "cuplyr-options", fixed = TRUE,
                   info = page)
    }
  }
})

test_that("F93: help('cuplyr-options') finds the page; NEWS mentions cuplyr_options()", {
  root <- testthat::test_path("..", "..")
  skip_if_not(file.exists(file.path(root, "DESCRIPTION")))
  # unqualified: under pkgload::load_all() this is pkgload's help() shim
  h <- help("cuplyr-options")
  expect_gt(length(h), 0L)
  news <- paste(readLines(file.path(root, "NEWS.md")), collapse = "\n")
  expect_match(news, "cuplyr_options()", fixed = TRUE)
  expect_match(news, "cuplyr_warning_invalid_option", fixed = TRUE)
})
