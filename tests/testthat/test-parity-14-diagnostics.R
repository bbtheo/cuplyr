# Issue #14: diagnostics() reports the libcudf version, the active memory
# resource and the effective cuplyr options, and can print JSON.
#
# None of these tests depend on this machine's exact versions: they check the
# fields exist with the right type and shape, and mock the lookups where an
# exact value is needed.

d14_version_re <- "^[0-9]+\\.[0-9]{2}\\.[0-9]+$"

d14_root <- function() testthat::test_path("..", "..")

# A fake RAPIDS prefix: <prefix>/lib/libcudf.so and
# <prefix>/include/cudf/version_config.hpp holding `lines`.
d14_fake_prefix <- function(lines, env = parent.frame()) {
  prefix <- withr::local_tempdir(.local_envir = env)
  dir.create(file.path(prefix, "lib"))
  dir.create(file.path(prefix, "include", "cudf"), recursive = TRUE)
  file.create(file.path(prefix, "lib", "libcudf.so"))
  if (!is.null(lines)) {
    writeLines(lines, file.path(prefix, "include", "cudf", "version_config.hpp"))
  }
  prefix
}

d14_header <- function(major, minor, patch = NULL) {
  c("#pragma once", "",
    paste("#define CUDF_VERSION_MAJOR", major),
    paste("#define CUDF_VERSION_MINOR", minor),
    if (!is.null(patch)) paste("#define CUDF_VERSION_PATCH", patch))
}

d14_quiet <- function(expr) {
  out <- utils::capture.output(res <- expr)
  res
}

d14_json <- function(...) {
  out <- utils::capture.output(res <- diagnostics(..., format = "json"))
  list(text = paste(out, collapse = "\n"), value = res)
}

# ---------------------------------------------------------------------------
# The new fields

test_that("#14: diagnostics() takes format = c('text', 'json')", {
  skip_if_no_gpu()
  expect_identical(eval(formals(diagnostics)$format), c("text", "json"))
  expect_true("redact" %in% names(formals(diagnostics)))
  expect_error(utils::capture.output(diagnostics(format = "xml")))
})

test_that("#14: diagnostics() reports the libcudf and RMM versions it was built against", {
  skip_if_no_gpu()
  info <- d14_quiet(diagnostics(redact = FALSE))
  for (f in c("libcudf_version", "rmm_version")) {
    expect_type(info[[f]], "character")
    expect_length(info[[f]], 1L)
    expect_match(info[[f]], d14_version_re, info = f)
  }
})

test_that("#14: diagnostics() reports the loaded libcudf and its version", {
  skip_if_no_gpu()
  info <- d14_quiet(diagnostics(redact = FALSE))
  expect_type(info$libcudf_path, "character")
  expect_length(info$libcudf_path, 1L)
  # the package DLL is loaded, so libcudf is mapped into this process
  expect_false(is.na(info$libcudf_path))
  expect_match(info$libcudf_path, "libcudf\\.so")
  expect_type(info$libcudf_loaded_version, "character")
  expect_length(info$libcudf_loaded_version, 1L)
  if (!is.na(info$libcudf_loaded_version)) {
    expect_match(info$libcudf_loaded_version, d14_version_re)
  }
})

test_that("#14: the loaded version is read from the header beside the loaded library", {
  skip_if_no_gpu()
  prefix <- d14_fake_prefix(d14_header(99, "04", 1))
  lib <- file.path(prefix, "lib", "libcudf.so")
  local_mocked_bindings(loaded_libcudf_path = function(...) lib)
  info <- d14_quiet(diagnostics(redact = FALSE))
  expect_identical(info$libcudf_path, lib)
  expect_identical(info$libcudf_loaded_version, "99.04.1")
})

test_that("#14: no loaded libcudf, or no readable header, gives NA and no error", {
  skip_if_no_gpu()
  local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  expect_no_error(info <- d14_quiet(diagnostics(redact = FALSE)))
  expect_identical(info$libcudf_path, NA_character_)
  expect_identical(info$libcudf_loaded_version, NA_character_)

  prefix <- d14_fake_prefix(NULL)
  lib <- file.path(prefix, "lib", "libcudf.so")
  local_mocked_bindings(loaded_libcudf_path = function(...) lib)
  expect_no_error(info <- d14_quiet(diagnostics(redact = FALSE)))
  expect_identical(info$libcudf_path, lib)
  expect_identical(info$libcudf_loaded_version, NA_character_)
})

test_that("#14: read_cudf_version_header() formats versions as configure does", {
  # pure R: runs without a GPU too
  check <- function(lines, expected) {
    f <- withr::local_tempfile()
    writeLines(lines, f)
    expect_identical(read_cudf_version_header(f), expected)
  }
  check(d14_header(25, 12, 0), "25.12.0")
  check(d14_header(26, "02", 1), "26.02.1")
  check(d14_header(26, 2, 0), "26.02.0")
  check(d14_header(26, "08", "03"), "26.08.3")
  check(d14_header(25, 12), "25.12.0")
  check(c("#  define   CUDF_VERSION_MAJOR   24", "\t#define CUDF_VERSION_MINOR 10 // c",
          "#define CUDF_VERSION_PATCH 2"), "24.10.2")
  check(c("#pragma once", "#define CUDF_VERSION_MAJOR 25"), NA_character_)
  check(c("not a header"), NA_character_)
  check(character(), NA_character_)
  expect_identical(read_cudf_version_header(tempfile("missing")), NA_character_)
  expect_identical(read_cudf_version_header(NA_character_), NA_character_)
})

test_that("#14: diagnostics() reports the active memory resource", {
  skip_if_no_gpu()
  info <- d14_quiet(diagnostics(redact = FALSE))
  mem <- info$memory
  expect_type(mem, "list")
  for (f in c("available", "resource", "pool_bytes", "pool_used_bytes")) {
    expect_true(f %in% names(mem), info = f)
  }
  expect_true(isTRUE(mem$available))
  expect_true(mem$resource %in% c("async", "pool", "cuda"))
  expect_identical(mem$resource, gpu_memory_state()$resource)
  expect_type(mem$pool_bytes, "double")
  expect_type(mem$pool_used_bytes, "double")
  expect_length(mem$pool_bytes, 1L)
  expect_length(mem$pool_used_bytes, 1L)
  if (mem$resource == "cuda") {
    expect_true(is.na(mem$pool_bytes))
  } else {
    expect_gte(mem$pool_bytes, mem$pool_used_bytes)
  }
})

test_that("#14: failing lookups don't stop diagnostics()", {
  skip_if_no_gpu()
  local_mocked_bindings(
    cuplyr_build_versions = function(...) stop("no versions"),
    gpu_memory_state = function(...) stop("no memory state"),
    loaded_libcudf_path = function(...) stop("no maps")
  )
  expect_no_error(info <- d14_quiet(diagnostics(redact = FALSE)))
  expect_identical(info$libcudf_version, NA_character_)
  expect_identical(info$rmm_version, NA_character_)
  expect_identical(info$libcudf_path, NA_character_)
  expect_identical(info$libcudf_loaded_version, NA_character_)
  expect_false(isTRUE(info$memory$available))
  expect_identical(info$memory$resource, NA_character_)
  expect_no_error(js <- d14_json(redact = FALSE))
  expect_true(nzchar(js$text))
})

# ---------------------------------------------------------------------------
# Options

d14_option_cols <- c("option", "value", "default", "source", "env_var",
                     "internal", "description")

test_that("#14: diagnostics() lists every registered option, user and internal", {
  skip_if_no_gpu()
  opts <- d14_quiet(diagnostics(redact = FALSE))$options
  expect_s3_class(opts, "data.frame")
  expect_true(all(d14_option_cols %in% names(opts)))
  for (col in setdiff(d14_option_cols, "internal")) {
    expect_type(opts[[col]], "character")
  }
  expect_type(opts$internal, "logical")
  expect_false(anyNA(opts$internal))
  expect_false(anyDuplicated(opts$option) > 0L)
  expect_setequal(opts$option, paste0("cuplyr.", names(cuplyr_option_registry)))
  expect_setequal(opts$option, cuplyr_options(internal = TRUE)$option)
  expect_true(all(opts$source %in% c("option", "env", "default")))
  expect_true(all(nzchar(opts$description)))
  expect_false(opts$internal[opts$option == "cuplyr.exec_mode"])
  expect_true(opts$internal[opts$option == "cuplyr.roll_path"])
  expect_true(opts$internal[opts$option == "cuplyr.check_schema_cache"])
})

test_that("#14: every cuplyr option read in R/ appears in diagnostics()", {
  skip_if_no_gpu()
  r_dir <- file.path(d14_root(), "R")
  skip_if_not(dir.exists(r_dir))
  src <- unlist(lapply(list.files(r_dir, pattern = "\\.[Rr]$", full.names = TRUE),
                       readLines, warn = FALSE))
  pick <- function(re) {
    m <- regmatches(src, regexec(re, src, perl = TRUE))
    unique(vapply(Filter(length, m), `[[`, "", 2L))
  }
  read <- c(
    pick("getOption\\(\\s*[\"'](cuplyr\\.[A-Za-z0-9_.]+)[\"']"),
    paste0("cuplyr.", pick("cuplyr_opt(?:_resolve|_read)?\\(\\s*[\"']([A-Za-z0-9_.]+)[\"']"))
  )
  # the scan really finds the options (it would pass vacuously otherwise)
  expect_true(all(c("cuplyr.exec_mode", "cuplyr.fallback", "cuplyr.parquet_pushdown",
                    "cuplyr.topk_path") %in% read))
  opts <- d14_quiet(diagnostics(redact = FALSE))$options
  for (o in unique(read)) {
    expect_true(o %in% opts$option, info = o)
  }
})

test_that("#14: option values, sources and defaults are reported", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "warn", cuplyr.exec_mode = NULL,
                       cuplyr.parquet_pushdown = FALSE, cuplyr.topk_path = "sort",
                       cuplyr.pool_initial = NULL)
  withr::local_envvar(CUPLYR_EXEC_MODE = "eager", CUPLYR_POOL_INITIAL = NA)
  opts <- d14_quiet(diagnostics(redact = FALSE))$options
  row <- function(o) opts[opts$option == o, , drop = FALSE]

  expect_identical(row("cuplyr.fallback")$value, "warn")
  expect_identical(row("cuplyr.fallback")$source, "option")
  expect_identical(row("cuplyr.fallback")$default, "silent")
  expect_identical(row("cuplyr.fallback")$env_var, NA_character_)

  expect_identical(row("cuplyr.exec_mode")$value, "eager")
  expect_identical(row("cuplyr.exec_mode")$source, "env")
  expect_identical(row("cuplyr.exec_mode")$default, "lazy")
  expect_identical(row("cuplyr.exec_mode")$env_var, "CUPLYR_EXEC_MODE")

  expect_identical(row("cuplyr.parquet_pushdown")$value, "FALSE")
  expect_identical(row("cuplyr.parquet_pushdown")$source, "option")
  expect_identical(row("cuplyr.parquet_pushdown")$default, "TRUE")
  expect_false(row("cuplyr.parquet_pushdown")$internal)

  expect_identical(row("cuplyr.topk_path")$value, "sort")
  expect_identical(row("cuplyr.topk_path")$default, "auto")
  expect_true(row("cuplyr.topk_path")$internal)

  # a computed default is NA, not a made-up value
  expect_identical(row("cuplyr.pool_initial")$default, NA_character_)
  expect_identical(row("cuplyr.pool_initial")$source, "default")
  expect_identical(row("cuplyr.memory_resource")$default, "async")
})

test_that("#14: cuplyr.parquet_pushdown keeps its meaning through the registry", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.parquet_pushdown = NULL)
  expect_true(scan_pushdown_enabled())
  withr::local_options(cuplyr.parquet_pushdown = FALSE)
  expect_false(scan_pushdown_enabled())
  withr::local_options(cuplyr.parquet_pushdown = TRUE)
  expect_true(scan_pushdown_enabled())
  # anything else is invalid: one warning, and the default (on)
  withr::local_options(cuplyr.parquet_pushdown = "no-issue14")
  expect_warning(on <- scan_pushdown_enabled(), class = "cuplyr_warning_invalid_option")
  expect_true(on)
  expect_no_warning(on <- scan_pushdown_enabled())
  expect_true(on)
})

test_that("#14: cuplyr.topk_path keeps its routes; an unknown value warns and means auto", {
  skip_if_no_gpu()
  df <- data.frame(g = rep(c("a", "b"), 10), x = c(5, 3, 9, 1, 7, 2, 8, 4, 6, 10,
                                                   15, 13, 19, 11, 17, 12, 18, 14, 16, 20))
  for (p in c("auto", "topk", "sort")) {
    withr::local_options(cuplyr.topk_path = p)
    expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = 3))
    expect_same_as_dplyr_lazy(df, function(d) dplyr::slice_min(d, x, n = 3))
    expect_same_as_dplyr(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 2))
    expect_same_as_dplyr_lazy(df, function(d) dplyr::slice_max(dplyr::group_by(d, g), x, n = 2))
  }
  withr::local_options(cuplyr.topk_path = "bogus-issue14")
  expect_warning(
    res <- dplyr::collect(dplyr::slice_min(tbl_gpu(df), x, n = 3)),
    class = "cuplyr_warning_invalid_option"
  )
  expect_identical(tibble::as_tibble(res),
                   tibble::as_tibble(dplyr::slice_min(df, x, n = 3)))
  expect_no_warning(
    res <- dplyr::collect(dplyr::slice_max(dplyr::group_by(tbl_gpu(df), g), x, n = 2))
  )
  expect_identical(tibble::as_tibble(dplyr::ungroup(res)),
                   tibble::as_tibble(dplyr::ungroup(
                     dplyr::slice_max(dplyr::group_by(df, g), x, n = 2))))
})

test_that("#14: the new options are documented on ?cuplyr-options", {
  # pure R: runs without a GPU too
  rd_file <- file.path(d14_root(), "man", "cuplyr-options.Rd")
  skip_if_not(file.exists(rd_file))
  txt <- paste(readLines(rd_file), collapse = "\n")
  for (o in c("cuplyr.parquet_pushdown", "cuplyr.topk_path")) {
    expect_match(txt, paste0("\\alias{", o, "}"), fixed = TRUE, info = o)
  }
})

# ---------------------------------------------------------------------------
# Text output

test_that("#14: the text report shows the libcudf version, the resource and every option", {
  skip_if_no_gpu()
  out <- utils::capture.output(info <- diagnostics(redact = FALSE))
  txt <- paste(out, collapse = "\n")
  expect_match(txt, info$libcudf_version, fixed = TRUE)
  expect_match(txt, info$rmm_version, fixed = TRUE)
  expect_match(txt, info$libcudf_path, fixed = TRUE)
  expect_match(txt, "memory resource", ignore.case = TRUE)
  expect_match(txt, info$memory$resource, fixed = TRUE)
  for (o in info$options$option) {
    expect_match(txt, o, fixed = TRUE, info = o)
  }
  # the text format is not JSON
  expect_false(startsWith(trimws(out[1]), "{"))
})

test_that("#14: built and loaded versions are reported apart, and a mismatch is flagged", {
  skip_if_no_gpu()
  prefix <- d14_fake_prefix(d14_header(99, "04", 1))
  lib <- file.path(prefix, "lib", "libcudf.so")
  local_mocked_bindings(
    cuplyr_build_versions = function(...) c(libcudf = "25.12.0", rmm = "25.10.3"),
    loaded_libcudf_path = function(...) lib
  )
  out <- utils::capture.output(info <- diagnostics(redact = FALSE))
  expect_identical(info$libcudf_version, "25.12.0")
  expect_identical(info$rmm_version, "25.10.3")
  expect_identical(info$libcudf_loaded_version, "99.04.1")
  txt <- paste(out, collapse = "\n")
  expect_match(txt, "25.10.3", fixed = TRUE)
  expect_match(txt, "99.04.1", fixed = TRUE)
  expect_match(txt, "differs", ignore.case = TRUE)

  # matching versions: no mismatch line
  same <- d14_fake_prefix(d14_header(25, 12, 0))
  lib <- file.path(same, "lib", "libcudf.so")
  out <- utils::capture.output(info <- diagnostics(redact = FALSE))
  expect_identical(info$libcudf_loaded_version, "25.12.0")
  expect_no_match(paste(out, collapse = "\n"), "differs", ignore.case = TRUE)
})

test_that("#14: only a computed default is called computed in the text report", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.release_threshold = NULL)
  withr::local_envvar(CUPLYR_RELEASE_THRESHOLD = NA)
  out <- utils::capture.output(diagnostics(redact = FALSE))
  line <- grep("cuplyr.release_threshold", out, fixed = TRUE, value = TRUE)
  expect_length(line, 1L)
  # unset means "no automatic release", not a computed value
  expect_no_match(line, "computed", fixed = TRUE)
})

test_that("#14: with warnings as errors, one invalid option doesn't empty the options", {
  skip_if_no_gpu()
  withr::local_options(cuplyr.fallback = "bogus-issue14-warn2", warn = 2)
  expect_no_error(info <- d14_quiet(diagnostics(redact = FALSE)))
  opts <- info$options
  expect_setequal(opts$option, paste0("cuplyr.", names(cuplyr_option_registry)))
  expect_identical(opts$default[opts$option == "cuplyr.fallback"], "silent")
})

test_that("#14: no GPU memory information means an unknown resource", {
  skip_if_no_gpu()
  local_mocked_bindings(gpu_memory_state = function(...) list(
    available = FALSE, total_bytes = NA_real_, free_bytes = NA_real_,
    used_bytes = NA_real_, total_gb = NA_real_, free_gb = NA_real_,
    used_gb = NA_real_, resource = "cuda", pool_bytes = NA_real_,
    pool_used_bytes = NA_real_
  ))
  info <- d14_quiet(diagnostics(redact = FALSE))
  expect_false(info$memory$available)
  expect_identical(info$memory$resource, NA_character_)
  expect_identical(info$memory$pool_bytes, NA_real_)
})

# ---------------------------------------------------------------------------
# Redaction

test_that("#14: redaction covers the libcudf path, in both formats", {
  skip_if_no_gpu()
  home <- Sys.getenv("HOME", "")
  skip_if(!nzchar(home) || home == "/")
  lib <- file.path(home, "rapids-issue14", "lib", "libcudf.so")
  # redaction also replaces the user name anywhere in a path
  user <- Sys.info()[["user"]]
  skip_if(nzchar(user) && grepl(user, "rapids-issue14/lib/libcudf.so", fixed = TRUE))
  local_mocked_bindings(loaded_libcudf_path = function(...) lib)

  info <- d14_quiet(diagnostics(redact = TRUE))
  expect_false(grepl(home, info$libcudf_path, fixed = TRUE))
  expect_match(info$libcudf_path, "rapids-issue14/lib/libcudf.so", fixed = TRUE)
  out <- paste(utils::capture.output(diagnostics(redact = TRUE)), collapse = "\n")
  expect_false(grepl(home, out, fixed = TRUE))

  js <- d14_json(redact = TRUE)
  expect_false(grepl(home, js$text, fixed = TRUE))
  expect_match(js$text, "rapids-issue14", fixed = TRUE)

  info <- d14_quiet(diagnostics(redact = FALSE))
  expect_identical(info$libcudf_path, lib)
})

# ---------------------------------------------------------------------------
# JSON output

test_that("#14: format = 'json' prints one valid JSON object and returns the list", {
  skip_if_no_gpu()
  skip_if_not_installed("jsonlite")
  js <- d14_json(redact = FALSE)
  expect_true(startsWith(trimws(js$text), "{"))
  expect_false(grepl("cuplyr diagnostics", js$text, fixed = TRUE))
  expect_true(jsonlite::validate(js$text))
  parsed <- jsonlite::fromJSON(js$text, simplifyVector = FALSE)

  info <- js$value
  expect_type(info, "list")
  text_info <- d14_quiet(diagnostics(redact = FALSE))
  expect_identical(names(info), names(text_info))
  expect_setequal(names(parsed), names(info))

  expect_identical(parsed$libcudf_version, info$libcudf_version)
  expect_identical(parsed$rmm_version, info$rmm_version)
  expect_identical(parsed$libcudf_path, info$libcudf_path)
  expect_identical(parsed$cuplyr_version, info$cuplyr_version)
  expect_identical(parsed$r_version, info$r_version)
  expect_identical(parsed$memory$resource, info$memory$resource)
  expect_identical(parsed$memory$available, info$memory$available)
  expect_true(is.logical(parsed$gpu$available))
  expect_identical(parsed$gpu$name, info$gpu$name)
  if (!is.na(info$memory$pool_bytes)) {
    expect_equal(parsed$memory$pool_bytes, info$memory$pool_bytes)
  }

  # options: an array with one object per row
  expect_length(parsed$options, nrow(info$options))
  got <- vapply(parsed$options, function(o) o$option, "")
  expect_identical(got, info$options$option)
  for (i in seq_along(parsed$options)) {
    o <- parsed$options[[i]]
    expect_setequal(names(o), d14_option_cols)
    expect_true(is.logical(o$internal))
    # NA is null
    if (is.na(info$options$default[i])) {
      expect_null(o$default)
    } else {
      expect_identical(o$default, info$options$default[i])
    }
  }
})

test_that("#14: JSON nulls stand in for NA fields", {
  skip_if_no_gpu()
  skip_if_not_installed("jsonlite")
  local_mocked_bindings(loaded_libcudf_path = function(...) NA_character_)
  js <- d14_json(redact = FALSE)
  expect_true(jsonlite::validate(js$text))
  parsed <- jsonlite::fromJSON(js$text, simplifyVector = FALSE)
  expect_true("libcudf_path" %in% names(parsed))
  expect_null(parsed$libcudf_path)
  expect_null(parsed$libcudf_loaded_version)
})

test_that("#14: to_json() writes valid JSON for every value diagnostics() holds", {
  # pure R: runs without a GPU too
  skip_if_not_installed("jsonlite")
  rt <- function(x) {
    txt <- to_json(x)
    expect_type(txt, "character")
    expect_length(txt, 1L)
    expect_true(jsonlite::validate(txt), info = txt)
    jsonlite::fromJSON(txt, simplifyVector = FALSE)
  }
  tricky <- "a \"quoted\" back\\slash\nnew\tline \001 ctl café /path"
  expect_identical(rt(tricky), tricky)
  expect_identical(rt("plain"), "plain")
  expect_identical(rt(""), "")
  expect_null(rt(NA))
  expect_null(rt(NA_character_))
  expect_null(rt(NA_real_))
  expect_null(rt(NaN))
  expect_null(rt(Inf))
  expect_null(rt(NULL))
  expect_identical(rt(TRUE), TRUE)
  expect_identical(rt(FALSE), FALSE)
  expect_identical(rt(1.5), 1.5)
  expect_equal(rt(12e9), 12e9)
  expect_equal(rt(12345678901234), 12345678901234)
  expect_equal(rt(-0.000123), -0.000123)
  expect_identical(rt(3L), 3L)
  expect_identical(rt(character()), list())
  expect_identical(rt(c("a", "b")), list("a", "b"))
  expect_identical(rt(c(1, NA)), list(1L, NULL))
  expect_identical(rt(list()), list())
  expect_identical(rt(list(a = 1L, b = list(c = "x", d = NA))),
                   list(a = 1L, b = list(c = "x", d = NULL)))
  expect_identical(rt(list("x", TRUE)), list("x", TRUE))
  df <- data.frame(option = c("o1", "o2"), value = c("v", NA), on = c(TRUE, FALSE),
                   stringsAsFactors = FALSE)
  expect_identical(rt(df), list(list(option = "o1", value = "v", on = TRUE),
                                list(option = "o2", value = NULL, on = FALSE)))
  expect_identical(rt(df[0, ]), list())
  # names are escaped too
  expect_identical(names(rt(list(`we"ird` = 1L))), "we\"ird")
})

test_that("#14: check_deps(format = 'json') still writes valid JSON", {
  skip_if_no_gpu()
  skip_if_not_installed("jsonlite")
  out <- paste(utils::capture.output(check_deps(format = "json")), collapse = "\n")
  expect_true(jsonlite::validate(out))
  parsed <- jsonlite::fromJSON(out, simplifyVector = FALSE)
  expect_true(all(c("ok", "checks") %in% names(parsed)))
})

# ---------------------------------------------------------------------------
# Bug report template and docs

test_that("#14: the bug report template asks for diagnostics() output", {
  # pure R: runs without a GPU too
  tdir <- file.path(d14_root(), ".github", "ISSUE_TEMPLATE")
  skip_if_not(dir.exists(tdir))
  files <- list.files(tdir, pattern = "\\.(md|ya?ml)$", full.names = TRUE)
  txt <- vapply(files, function(f) paste(readLines(f, warn = FALSE), collapse = "\n"), "")
  bug <- txt[grepl("bug", txt, ignore.case = TRUE)]
  expect_gt(length(bug), 0L)
  expect_true(any(grepl("cuplyr::diagnostics()", bug, fixed = TRUE)))
  expect_true(any(grepl("memory resource", bug, ignore.case = TRUE)))
  expect_true(any(grepl("options", bug, ignore.case = TRUE)))
})

test_that("#14: ?diagnostics documents format and the new fields", {
  # pure R: runs without a GPU too
  rd_file <- file.path(d14_root(), "man", "diagnostics.Rd")
  skip_if_not(file.exists(rd_file))
  txt <- paste(readLines(rd_file), collapse = "\n")
  expect_match(txt, "\\item{format}", fixed = TRUE)
  for (w in c("json", "libcudf", "memory resource", "cuplyr_options")) {
    expect_match(txt, w, ignore.case = TRUE, info = w)
  }
})
