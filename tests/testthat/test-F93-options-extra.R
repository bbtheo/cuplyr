# F93 round 2: pins for the "unset" and "blank" rules of the options registry
# that the acceptance file (test-parity-F93-options.R) does not cover, and for
# the cheap lookup paths. Invalid values are unique to this file ("-f93x").

f93x_clean <- function(env = parent.frame()) {
  withr::local_options(
    cuplyr.exec_mode = NULL, cuplyr.memory_resource = NULL,
    cuplyr.memory_pool = NULL, cuplyr.pool_initial = NULL,
    cuplyr.release_threshold = NULL, cuplyr.roll_median = NULL,
    cuplyr.roll_path = NULL, .local_envir = env
  )
  withr::local_envvar(
    CUPLYR_EXEC_MODE = NA, CUPLYR_MEMORY_RESOURCE = NA, CUPLYR_MEMORY_POOL = NA,
    CUPLYR_POOL_INITIAL = NA, CUPLYR_RELEASE_THRESHOLD = NA,
    CUPLYR_ROLL_MEDIAN = NA, CUPLYR_ROLL_PATH = NA, .local_envir = env
  )
}

# Forget what was warned about (the dedupe state is per session).
f93x_forget <- function() {
  rm(list = ls(cuplyr_opt_state, all.names = TRUE), envir = cuplyr_opt_state)
}

# gpu_mr_install() whose async creation fails (like test-F65-round3.R).
f93x_mock_install <- function(fail = "async", env = parent.frame()) {
  calls <- new.env()
  calls$kinds <- character()
  local_mocked_bindings(
    gpu_mr_exists = function() FALSE,
    gpu_mr_install = function(enabled, kind, initial, release_threshold) {
      calls$kinds <- c(calls$kinds, kind)
      if (!enabled) return("cuda")
      if (kind %in% fail) stop("could not create the ", kind, " memory resource")
      kind
    },
    .env = env
  )
  calls
}

test_that("F93x: a blank or NA cuplyr.exec_mode is unset, so the environment variable decides", {
  f93x_clean()
  withr::local_envvar(CUPLYR_EXEC_MODE = "eager")
  for (blank in list("", "  ", NA_character_, NA, character(0))) {
    withr::local_options(cuplyr.exec_mode = blank)
    expect_no_warning(mode <- resolve_exec_mode())
    expect_identical(mode, "eager", info = paste(deparse(blank), collapse = " "))
  }
  withr::local_options(cuplyr.exec_mode = NULL)
  withr::local_envvar(CUPLYR_EXEC_MODE = NA)
  expect_identical(resolve_exec_mode(), "lazy")
})

test_that("F93x: a blank cuplyr.memory_resource is the default async, not explicit, and ignores the env var", {
  f93x_clean()
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = "pool")
  for (blank in c("", "  ")) {
    withr::local_options(cuplyr.memory_resource = blank, cuplyr.pool_initial = 0)
    expect_no_warning(cfg <- memory_resource_config())
    expect_identical(cfg$kind, "async", info = blank)
    expect_false(cfg$explicit, info = blank)
    expect_true(cfg$enabled, info = blank)
  }
  # NA is unset: the env var decides, and asking for it by name is explicit
  withr::local_options(cuplyr.memory_resource = NA_character_)
  cfg <- memory_resource_config()
  expect_identical(cfg$kind, "pool")
  expect_true(cfg$explicit)
  # a typo is not explicit either (the blank is no stricter than a typo)
  withr::local_options(cuplyr.memory_resource = "bogus-f93x")
  expect_warning(cfg <- memory_resource_config(), class = "cuplyr_warning_invalid_option")
  expect_false(cfg$explicit)
})

test_that("F93x: with a blank cuplyr.memory_resource, a failing async falls back to the pool", {
  for (blank in c("", "  ")) {
    f93x_clean()
    withr::local_options(cuplyr.memory_resource = blank, cuplyr.pool_initial = 0)
    calls <- f93x_mock_install()
    expect_warning(kind <- memory_resource_install(), "default \"async\".*using \"pool\" instead")
    expect_identical(kind, "pool", info = blank)
    expect_identical(calls$kinds, c("async", "pool"), info = blank)
  }
})

test_that("F93x: cuplyr.pool_initial NA or blank is the default size silently and never reads the env var", {
  f93x_clean()
  withr::local_envvar(CUPLYR_POOL_INITIAL = "67108864")
  # NULL falls to the env var (control)
  expect_identical(cuplyr_opt("pool_initial"), 67108864)
  for (blank in list(NA, NA_real_, NA_character_, "", "  ", numeric(0))) {
    withr::local_options(cuplyr.pool_initial = blank)
    expect_no_warning(v <- cuplyr_opt("pool_initial"))
    expect_identical(v, NA_real_, info = paste(deparse(blank), collapse = " "))
  }
  withr::local_options(cuplyr.pool_initial = NA)
  tab <- cuplyr_options()
  expect_identical(tab$source[tab$option == "cuplyr.pool_initial"], "default")
  # the size memory_resource_config() computes is the default, not the env value
  skip_if_no_gpu()
  expect_false(identical(memory_resource_config()$initial, 67108864))
  # a negative one still warns and gives the default size
  withr::local_options(cuplyr.pool_initial = -7.5)
  expect_warning(v <- cuplyr_opt("pool_initial"), class = "cuplyr_warning_invalid_option")
  expect_identical(v, NA_real_)
})

test_that("F93x: cuplyr.memory_pool \"\" is silent and means on, whatever the env var says", {
  f93x_clean()
  withr::local_envvar(CUPLYR_MEMORY_POOL = "off")
  for (blank in c("", "  ")) {
    withr::local_options(cuplyr.memory_pool = blank, cuplyr.pool_initial = 0)
    expect_no_warning(cfg <- memory_resource_config())
    expect_true(cfg$enabled, info = blank)
  }
  # NA is unset, so the env var decides
  withr::local_options(cuplyr.memory_pool = NA)
  expect_false(memory_resource_config()$enabled)
  # CUPLYR_MEMORY_POOL="" is the same default
  withr::local_options(cuplyr.memory_pool = NULL)
  withr::local_envvar(CUPLYR_MEMORY_POOL = "")
  expect_no_warning(cfg <- memory_resource_config())
  expect_true(cfg$enabled)
})

test_that("F93x: the memory_pool warning reads as a sentence", {
  f93x_clean()
  withr::local_options(cuplyr.memory_pool = "nope-f93x")
  w <- tryCatch(cuplyr_opt("memory_pool"), warning = function(w) w)
  expect_s3_class(w, "cuplyr_warning_invalid_option")
  expect_match(conditionMessage(w), "Using the default (on: the resource is installed).",
               fixed = TRUE)
})

test_that("F93x: cuplyr_options() reports a function-valued option as invalid instead of failing", {
  f93x_clean()
  f93x_forget()
  for (opt in c("memory_pool", "memory_resource", "release_threshold", "pool_initial")) {
    f <- eval(substitute(function() VALUE, list(VALUE = paste0(opt, "-f93x"))))
    do.call(withr::local_options, stats::setNames(list(f), paste0("cuplyr.", opt)))
    expect_warning(tab <- cuplyr_options(), class = "cuplyr_warning_invalid_option", info = opt)
    expect_identical(tab$source[tab$option == paste0("cuplyr.", opt)], "default", info = opt)
    # once per session: the second read is silent
    expect_no_warning(cuplyr_options())
    do.call(withr::local_options, stats::setNames(list(NULL), paste0("cuplyr.", opt)))
  }
})

test_that("F93x: a failing message builder does not silence later warnings", {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.release_threshold = function() "release-f93x")
  # a function-valued threshold is described in the message without failing
  expect_warning(cuplyr_opt("release_threshold"), "<function>", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_no_warning(cuplyr_opt("release_threshold"))
  r <- list(value = NA_real_, source = "default", bad = TRUE, raw = "once-f93x",
            from = "option", option = "cuplyr.roll_median", env = "CUPLYR_ROLL_MEDIAN")
  expect_error(cuplyr_opt_warn("roll_median", r, message = function(shown, r) stop("builder")),
               "builder")
  expect_warning(cuplyr_opt_warn("roll_median", r), class = "cuplyr_warning_invalid_option")
})

test_that("F93x: the rolling readers' fast paths agree with the registry", {
  f93x_clean()
  expect_identical(roll_median_engine(), "exact")
  expect_identical(roll_path_option(), "auto")
  # a blank option is the default and the env var is not consulted
  withr::local_envvar(CUPLYR_ROLL_MEDIAN = "frollmedian", CUPLYR_ROLL_PATH = "scans")
  expect_identical(roll_median_engine(), "frollmedian")
  expect_identical(roll_path_option(), "scans")
  withr::local_options(cuplyr.roll_median = "", cuplyr.roll_path = "")
  expect_no_warning(eng <- roll_median_engine())
  expect_identical(eng, "exact")
  expect_no_warning(path <- roll_path_option())
  expect_identical(path, "auto")
  # valid options and env values come back unchanged; invalid ones warn once
  withr::local_options(cuplyr.roll_median = "frollmedian", cuplyr.roll_path = "cudf")
  expect_identical(roll_median_engine(), "frollmedian")
  expect_identical(roll_path_option(), "cudf")
  # NA is not unset for these two: it warns (whatever an earlier file warned about)
  f93x_forget()
  for (fn in list(roll_median_engine, roll_path_option)) {
    nm <- if (identical(fn, roll_median_engine)) "cuplyr.roll_median" else "cuplyr.roll_path"
    do.call(withr::local_options, stats::setNames(list(NA_character_), nm))
    expect_warning(fn(), nm, fixed = TRUE, class = "cuplyr_warning_invalid_option")
  }
})

test_that("F93x: the unset lookup of the per-group readers stays cheap", {
  # (no skip_on_cran(): the tools/ runners don't set NOT_CRAN, so it would
  # never run; cuplyr is not on CRAN)
  skip_on_covr()
  f93x_clean()
  expect_identical(cuplyr_opt_read("roll_median"),
                   cuplyr_option_registry$roll_median$default_result)
  # well under what the trimws() path cost (14 us per call); the best of five
  # runs, so one slow window on a busy machine does not fail it
  roll_median_engine()
  t <- min(vapply(1:5, function(k) {
    system.time(for (i in 1:20000) roll_median_engine())[["elapsed"]]
  }, 0))
  expect_lt(t / 20000, 1e-5)
})

test_that("F93x: the hot-path value sets are the registry's", {
  f93x_clean()
  withr::local_options(cuplyr.exec_mode = "Lazy-f93x")
  w <- tryCatch(resolve_exec_mode(), warning = function(w) w)
  expect_s3_class(w, "cuplyr_warning_invalid_option")
  for (mode in cuplyr_exec_modes) {
    expect_match(conditionMessage(w), paste0("\"", mode, "\""), fixed = TRUE)
  }
})

# --- round 3: warnings that are errors (options(warn = 2)) -------------------
# A converted warning must reach the caller with the real source and value: no
# tryCatch(error =) on the option path may catch it.

test_that("F93x: with warn = 2, cuplyr_options() raises an invalid option value, naming the option", {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.exec_mode = "Lazy-f93x-w2a", warn = 2)
  expect_error(cuplyr_options(),
               "`options(cuplyr.exec_mode = )` is \"Lazy-f93x-w2a\", which is not",
               fixed = TRUE)
  # a warning that did not return (here: raised as an error) is not recorded,
  # so every read raises it until the value is fixed
  expect_error(resolve_exec_mode(), "\"Lazy-f93x-w2a\"", fixed = TRUE)
  withr::local_options(warn = 0)
  expect_warning(resolve_exec_mode(), "\"Lazy-f93x-w2a\"", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_no_warning(resolve_exec_mode())
})

test_that("F93x: with warn = 2, a verb that swallows errors does not make an invalid value silent", {
  skip_if_no_gpu()
  f93x_clean()
  f93x_forget()
  df <- tbl_gpu(data.frame(x = 1:5), lazy = FALSE)
  withr::local_options(warn = 2)
  # a constant roll_median() is tried as a constant first, inside tryCatch(error =)
  withr::local_options(cuplyr.roll_median = "Exact-f93x-w2m")
  expect_error(collect(mutate(df, y = roll_median(c(1, 2, 3, 4, 5), 3))),
               "\"Exact-f93x-w2m\"", fixed = TRUE)
  withr::local_options(cuplyr.roll_median = "Exact-f93x-w2n")
  expect_error(collect(filter(df, roll_median(c(1, 2, 3, 4, 5), 3) > 2)),
               "\"Exact-f93x-w2n\"", fixed = TRUE)
})

test_that("F93x: with warn = 2, cuplyr_options() raises an invalid env value, naming the variable", {
  f93x_clean()
  f93x_forget()
  withr::local_envvar(CUPLYR_EXEC_MODE = "Eagre-f93x-w2b")
  withr::local_options(warn = 2)
  err <- tryCatch(cuplyr_options(), error = function(e) e)
  expect_s3_class(err, "error")
  msg <- conditionMessage(err)
  expect_match(msg, "Environment variable CUPLYR_EXEC_MODE (option `cuplyr.exec_mode`) is \"Eagre-f93x-w2b\"",
               fixed = TRUE)
  expect_no_match(msg, "<NULL>", fixed = TRUE)
})

test_that("F93x: with warn = 2, a function-valued option raises from cuplyr_options(), naming the option", {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.memory_pool = function() "pool-f93x-w2", warn = 2)
  expect_error(cuplyr_options(), "`options(cuplyr.memory_pool = )` is <function>", fixed = TRUE)
})

test_that("F93x: with warn = 2, the per-call readers raise the invalid value with its source", {
  f93x_clean()
  f93x_forget()
  withr::local_options(warn = 2)
  withr::with_options(list(cuplyr.exec_mode = "Lazy-f93x-w2c"),
    expect_error(resolve_exec_mode(), "`options(cuplyr.exec_mode = )` is \"Lazy-f93x-w2c\"",
                 fixed = TRUE))
  withr::with_envvar(c(CUPLYR_EXEC_MODE = "eagre-f93x-w2d"),
    expect_error(resolve_exec_mode(), "Environment variable CUPLYR_EXEC_MODE (option `cuplyr.exec_mode`) is \"eagre-f93x-w2d\"",
                 fixed = TRUE))
  withr::with_options(list(cuplyr.roll_median = "Exact-f93x-w2e"),
    expect_error(roll_median_engine(), "`options(cuplyr.roll_median = )` is \"Exact-f93x-w2e\"",
                 fixed = TRUE))
  withr::with_envvar(c(CUPLYR_ROLL_MEDIAN = "froll-f93x-w2f"),
    expect_error(roll_median_engine(), "Environment variable CUPLYR_ROLL_MEDIAN (option `cuplyr.roll_median`) is \"froll-f93x-w2f\"",
                 fixed = TRUE))
  withr::with_options(list(cuplyr.fallback = "loud-f93x-w2g"),
    expect_error(cuplyr_notice_mode("cuplyr.fallback"), "`options(cuplyr.fallback = )` is \"loud-f93x-w2g\"",
                 fixed = TRUE))
})

# gpu_mr_install() mock that records `enabled` too.
f93x_mock_install2 <- function(fail = character(), env = parent.frame()) {
  calls <- new.env()
  calls$kinds <- character()
  calls$enabled <- logical()
  local_mocked_bindings(
    gpu_mr_exists = function() FALSE,
    gpu_mr_install = function(enabled, kind, initial, release_threshold) {
      calls$kinds <- c(calls$kinds, kind)
      calls$enabled <- c(calls$enabled, enabled)
      if (!enabled) return("cuda")
      if (kind %in% fail) stop("could not create the ", kind, " memory resource")
      kind
    },
    .env = env
  )
  calls
}

test_that("F93x: with warn = 2, an invalid memory option still installs the default resource, then raises", {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.memory_resource = "bogus-f93x-w2h", cuplyr.pool_initial = 0, warn = 2)
  calls <- f93x_mock_install2()
  err <- tryCatch(memory_resource_install(), error = function(e) e)
  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "Unknown memory resource 'bogus-f93x-w2h' in cuplyr.memory_resource",
               fixed = TRUE)
  expect_no_match(conditionMessage(err), "plain cudaMalloc", fixed = TRUE)
  # the resource was created as without warn = 2: the default async
  expect_identical(calls$kinds, "async")
  expect_identical(calls$enabled, TRUE)
})

test_that("F93x: with warn = 2, the default-async fallback keeps the pool and raises its own warning", {
  f93x_clean()
  withr::local_options(cuplyr.pool_initial = 0, warn = 2)
  calls <- f93x_mock_install2(fail = "async")
  err <- tryCatch(memory_resource_install(), error = function(e) e)
  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "default \"async\" unavailable, using \"pool\" instead",
               fixed = TRUE)
  expect_no_match(conditionMessage(err), "plain cudaMalloc", fixed = TRUE)
  expect_identical(calls$kinds, c("async", "pool"))
})

test_that("F93x: the warnings of memory_resource_install() keep their order", {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.memory_resource = "bogus-f93x-ord", cuplyr.pool_initial = 0)
  calls <- f93x_mock_install2(fail = "async")
  w <- character()
  kind <- withCallingHandlers(memory_resource_install(), warning = function(cnd) {
    w <<- c(w, conditionMessage(cnd))
    invokeRestart("muffleWarning")
  })
  expect_identical(kind, "pool")
  expect_length(w, 2)
  expect_match(w[[1]], "bogus-f93x-ord", fixed = TRUE)
  expect_match(w[[2]], "using \"pool\" instead", fixed = TRUE)
  # (N25) a normal install gives the warnings it re-raises: none is held back
  expect_no_warning(cuplyr_options())
})

# --- round 3: N16 (one dedupe key per distinct value) ------------------------

test_that("F93x: distinct list values each warn once", {
  f93x_clean()
  withr::local_options(cuplyr.fallback = list("a-f93x-l"))
  expect_warning(cuplyr_notice_mode("cuplyr.fallback"), "list(\"a-f93x-l\")", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_no_warning(cuplyr_notice_mode("cuplyr.fallback"))
  withr::local_options(cuplyr.fallback = list("b-f93x-l"))
  expect_warning(cuplyr_notice_mode("cuplyr.fallback"), "list(\"b-f93x-l\")", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  # values that differ past the first deparsed line are distinct too
  long <- paste(rep("x", 200), collapse = "")
  withr::local_options(cuplyr.plan_break = c(long, "c-f93x-l"))
  expect_warning(cuplyr_notice_mode("cuplyr.plan_break"), class = "cuplyr_warning_invalid_option")
  withr::local_options(cuplyr.plan_break = c(long, "d-f93x-l"))
  expect_warning(cuplyr_notice_mode("cuplyr.plan_break"), class = "cuplyr_warning_invalid_option")
})

# --- round 3: N17 (a vector cuplyr.memory_resource: base's first element) ----

test_that("F93x: a vector cuplyr.memory_resource uses its first element, silently, as before F93", {
  f93x_clean()
  withr::local_options(cuplyr.memory_resource = c(" Pool ", "async"), cuplyr.pool_initial = 0)
  expect_no_warning(cfg <- memory_resource_config())
  expect_identical(cfg$kind, "pool")
  expect_true(cfg$explicit)
  tab <- cuplyr_options()
  expect_identical(tab$value[tab$option == "cuplyr.memory_resource"], "pool")
  # a blank first element is the default (as base read it)
  withr::local_options(cuplyr.memory_resource = c("", "pool"))
  expect_no_warning(cfg <- memory_resource_config())
  expect_identical(cfg$kind, "async")
  expect_false(cfg$explicit)
  # an invalid first element warns, quoting it
  withr::local_options(cuplyr.memory_resource = c("vec-f93x", "pool"))
  expect_warning(cfg <- memory_resource_config(), "Unknown memory resource 'vec-f93x'", fixed = TRUE)
  expect_identical(cfg$kind, "async")
})

# --- round 4: N19 (warn = 2 at load keeps the unraised warnings un-given) -----

test_that("F93x: with warn = 2 at load, the invalid warnings not yet raised are reported later", {
  skip_if_no_gpu()
  f93x_clean()
  f93x_forget()
  withr::defer(f93x_forget())
  calls <- f93x_mock_install(fail = character())
  withr::local_options(cuplyr.memory_pool = "pool-f93x-n19", cuplyr.pool_initial = "init-f93x-n19",
                       cuplyr.memory_resource = "pool")
  withr::with_options(list(warn = 2), {
    expect_error(memory_resource_install(), "f93x-n19")
  })
  # neither value counts as given: reading the options warns about each
  w <- character()
  withCallingHandlers(
    cuplyr_options(),
    cuplyr_warning_invalid_option = function(c) {
      w <<- c(w, conditionMessage(c))
      invokeRestart("muffleWarning")
    }
  )
  expect_true(any(grepl("pool-f93x-n19", w, fixed = TRUE)))
  expect_true(any(grepl("init-f93x-n19", w, fixed = TRUE)))
  # and once reported they are given
  expect_no_warning(cuplyr_options())
})

# --- round 4: N22 (a probe lets an invalid-option warning through) -----------

test_that("F93x: suppress_probe_warnings() mutes other warnings but not an invalid option", {
  expect_no_warning(suppress_probe_warnings(warning("plain-f93x")))
  expect_warning(
    suppress_probe_warnings(rlang::warn("bad-f93x", class = "cuplyr_warning_invalid_option")),
    "bad-f93x", class = "cuplyr_warning_invalid_option"
  )
  expect_identical(suppress_probe_warnings({ warning("x"); 3L }), 3L)
})

test_that("F93x: suppress_probe_warnings() copes with a warning that has no muffle restart", {
  # (the condition passes on, as under suppressWarnings(); it is no error)
  res <- tryCatch(
    suppress_probe_warnings({
      signalCondition(warningCondition("no-restart-f93x"))
      7L
    }),
    warning = function(w) w,
    error = function(e) e
  )
  expect_s3_class(res, "warning")
  expect_match(conditionMessage(res), "no-restart-f93x", fixed = TRUE)
})

# --- polish 2: N26 (the across/if_any probes let the warning through) --------

f93x_count_invalid <- function(expr) {
  n <- 0L
  value <- withCallingHandlers(expr, cuplyr_warning_invalid_option = function(c) {
    n <<- n + 1L
    invokeRestart("muffleWarning")
  })
  list(value = value, n = n)
}

f93x_n26 <- function(lazy, value) {
  f93x_clean()
  f93x_forget()
  withr::local_options(cuplyr.roll_median = value)
  df <- data.frame(g = c("p", "p", "q", "q"), a = c(5, NA, 3, 1), b = c(1, 4, NA, 2))
  expected <- suppressWarnings(
    dplyr::filter(dplyr::group_by(tibble::as_tibble(df), g),
                  if_any(c(a, b), ~ .x > max(roll_median(c(1, 2, 3), 3), na.rm = TRUE)))
  )
  f93x_forget()  # (the tibble run above gave the warning too)
  run <- function() {
    f93x_count_invalid(dplyr::collect(
      dplyr::filter(dplyr::group_by(tbl_gpu(df, lazy = lazy), g),
                    if_any(c(a, b), ~ .x > max(roll_median(c(1, 2, 3), 3), na.rm = TRUE)))
    ))
  }
  first <- run()
  second <- run()
  expect_identical(first$n, 1L)
  expect_identical(second$n, 0L)
  expect_equal(tibble::as_tibble(first$value), expected, ignore_attr = TRUE)
  expect_equal(tibble::as_tibble(second$value), expected, ignore_attr = TRUE)
}

test_that("F93x: an invalid option read while probing if_any() warns once (eager)", {
  skip_if_no_gpu()
  f93x_n26(FALSE, "Exact-f93x-n26-eager")
})

test_that("F93x: an invalid option read while probing if_any() warns once (lazy)", {
  skip_if_no_gpu()
  f93x_n26(TRUE, "Exact-f93x-n26-lazy")
})

# --- polish 2: N27 (retry_on_sum_overflow() under warn = 2) -------------------

test_that("F93x: with warn = 2, retry_on_sum_overflow() forgets the warnings it did not raise", {
  f93x_clean()
  f93x_forget()
  withr::defer(f93x_forget())
  withr::local_options(cuplyr.fallback = "fb-f93x-n27", cuplyr.plan_break = "pb-f93x-n27")
  attempt <- function() {
    cuplyr_notice_mode("cuplyr.fallback")
    cuplyr_notice_mode("cuplyr.plan_break")
    1L
  }
  # both are held back while the attempt runs, then re-raised; the first one
  # raised as an error ends the loop
  err <- withr::with_options(list(warn = 2), tryCatch(
    retry_on_sum_overflow(attempt, function() stop("not reached")),
    error = function(e) e
  ))
  expect_s3_class(err, "error")
  expect_match(conditionMessage(err), "fb-f93x-n27", fixed = TRUE)
  # neither counts as given: each is reported by its next read
  expect_warning(cuplyr_notice_mode("cuplyr.fallback"), "fb-f93x-n27", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  expect_warning(cuplyr_notice_mode("cuplyr.plan_break"), "pb-f93x-n27", fixed = TRUE,
                 class = "cuplyr_warning_invalid_option")
  # and a normal run gives them as it re-raises, so they stay given
  f93x_forget()
  got <- f93x_count_invalid(retry_on_sum_overflow(attempt, function() stop("not reached")))
  expect_identical(got$n, 2L)
  expect_no_warning(cuplyr_notice_mode("cuplyr.fallback"))
  expect_no_warning(cuplyr_notice_mode("cuplyr.plan_break"))
})
