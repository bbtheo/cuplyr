# #138: examples and vignettes run without a GPU.
#
# - CUPLYR_DISABLE_GPU ("true" or "1") is a testing switch: has_gpu() and
#   gpu_details() report no GPU without asking the device, and loading the
#   package does not touch it. tools/check --no-gpu sets it together with an
#   empty CUDA_VISIBLE_DEVICES, so any GPU code an example or vignette runs
#   fails in that process.
# - Help-page examples that need the GPU are guarded with
#   `@examplesIf has_gpu()` (the guard sits in \dontshow{}).
# - Vignettes ship precomputed: vignettes/<name>.Rmd is knitted on the GPU
#   from vignettes/<name>.Rmd.orig by tools/vignettes-knit (functions in
#   tools/vignettes-lib.R), so the shipped files contain no code that runs at
#   build time.
#
# The source-tree tests skip when the tests run from an installed copy.

p138_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p138_skip_if_no_source_tree <- function() {
  root <- p138_root()
  skip_if_not(
    file.exists(file.path(root, "DESCRIPTION")) &&
      dir.exists(file.path(root, "man")) &&
      dir.exists(file.path(root, "vignettes")) &&
      dir.exists(file.path(root, "tools")),
    "source tree not available in this test context"
  )
}

p138_vignette_names <- c(
  "complex-analysis", "getting-started", "memory", "performance",
  "query-optimization"
)

# Run `code` (a character vector of R lines) in a fresh Rscript that loads this
# checkout with pkgload, with `env` ("NAME=value" strings) set. The script saves
# what it wants to report with saveRDS() to the path in `result`.
p138_run_rscript <- function(code, env, timeout = 900) {
  script <- tempfile("p138-", fileext = ".R")
  result <- tempfile("p138-", fileext = ".rds")
  on.exit(unlink(c(script, result)), add = TRUE)
  writeLines(c(
    sprintf("p138_result <- %s", deparse(result)),
    sprintf(
      paste0(
        "pkgload::load_all(%s, quiet = TRUE, debug = FALSE, compile = FALSE, ",
        "helpers = FALSE, export_all = FALSE)"
      ),
      deparse(p138_root())
    ),
    code
  ), script)
  out <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"), shQuote(script),
    stdout = TRUE, stderr = TRUE, env = env, timeout = timeout
  ))
  status <- attr(out, "status")
  list(
    status = if (is.null(status)) 0L else status,
    output = out,
    result = if (file.exists(result)) readRDS(result) else NULL
  )
}

p138_no_gpu_env <- c("CUPLYR_DISABLE_GPU=true", "CUDA_VISIBLE_DEVICES=")

# Text of an Rd node, leaving out nodes whose tag is in `drop`.
p138_rd_text <- function(x, drop = character()) {
  tag <- attr(x, "Rd_tag")
  if (!is.null(tag) && tag %in% drop) {
    return("")
  }
  if (is.list(x)) {
    return(paste(vapply(x, p138_rd_text, "", drop = drop), collapse = ""))
  }
  paste(as.character(x), collapse = "")
}

# Every node of an Rd tree with tag `tag`.
p138_rd_nodes <- function(x, tag) {
  if (identical(attr(x, "Rd_tag"), tag)) {
    return(list(x))
  }
  if (is.list(x)) {
    return(do.call(c, lapply(x, p138_rd_nodes, tag = tag)))
  }
  list()
}

# The examples section of an Rd file: `runs` is the code that runs and shows
# (outside \dontshow and \dontrun), `guard` the text of its \dontshow parts.
p138_rd_examples <- function(file) {
  rd <- tools::parse_Rd(file)
  tags <- vapply(rd, function(s) attr(s, "Rd_tag") %||% "", "")
  ex <- rd[tags == "\\examples"]
  if (length(ex) == 0) {
    return(NULL)
  }
  ex <- ex[[1]]
  list(
    runs = p138_rd_text(ex, drop = c("\\dontshow", "\\dontrun")),
    guard = paste(
      vapply(p138_rd_nodes(ex, "\\dontshow"), p138_rd_text, ""),
      collapse = "\n"
    )
  )
}

p138_lib <- function() {
  env <- new.env(parent = globalenv())
  sys.source(file.path(p138_root(), "tools", "vignettes-lib.R"), envir = env)
  env
}

p138_write_orig <- function(chunks, dir) {
  path <- file.path(dir, "p138.Rmd.orig")
  writeLines(c(
    "---",
    "title: \"p138\"",
    "---",
    "",
    "Some text.",
    "",
    chunks
  ), path)
  path
}

# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

test_that("CUPLYR_DISABLE_GPU makes has_gpu() FALSE without asking the device", {
  called <- FALSE
  local_mocked_bindings(gpu_is_available = function() {
    called <<- TRUE
    TRUE
  })
  for (value in c("true", "TRUE", "True", "1")) {
    withr::local_envvar(CUPLYR_DISABLE_GPU = value)
    expect_identical(has_gpu(), FALSE)
  }
  expect_false(called)
})

test_that("other values of CUPLYR_DISABLE_GPU leave the GPU check on", {
  local_mocked_bindings(gpu_is_available = function() TRUE)
  for (value in c("false", "FALSE", "0", "")) {
    withr::local_envvar(CUPLYR_DISABLE_GPU = value)
    expect_identical(has_gpu(), TRUE)
  }
  withr::local_envvar(CUPLYR_DISABLE_GPU = NA)
  expect_identical(has_gpu(), TRUE)
})

test_that("the switch is read on every call, not when the package loads", {
  local_mocked_bindings(gpu_is_available = function() TRUE)
  withr::local_envvar(CUPLYR_DISABLE_GPU = NA)
  expect_true(has_gpu())
  withr::local_envvar(CUPLYR_DISABLE_GPU = "true")
  expect_false(has_gpu())
  withr::local_envvar(CUPLYR_DISABLE_GPU = NA)
  expect_true(has_gpu())
})

test_that("has_gpu() returns FALSE, not an error, when the device query fails", {
  withr::local_envvar(CUPLYR_DISABLE_GPU = NA)
  local_mocked_bindings(gpu_is_available = function() stop("no CUDA driver"))
  expect_identical(has_gpu(), FALSE)
})

test_that("has_gpu() is TRUE on a GPU machine without the switch", {
  skip_if_no_gpu()
  withr::local_envvar(CUPLYR_DISABLE_GPU = NA)
  expect_identical(has_gpu(), TRUE)
})

test_that("gpu_details() and show_gpu() report no GPU under the switch", {
  called <- FALSE
  local_mocked_bindings(
    gpu_info = function() {
      called <<- TRUE
      list(available = TRUE, device_count = 1L)
    },
    gpu_is_available = function() {
      called <<- TRUE
      TRUE
    }
  )
  withr::local_envvar(CUPLYR_DISABLE_GPU = "true")

  info <- gpu_details()
  expect_false(info$available)
  expect_equal(info$device_count, 0)

  out <- capture.output(res <- show_gpu())
  expect_match(out, "No GPU available", fixed = TRUE, all = FALSE)
  expect_false(res$available)
  expect_false(called)
})

test_that("the startup message under the switch does not ask the device", {
  called <- FALSE
  local_mocked_bindings(
    gpu_info = function() {
      called <<- TRUE
      list(available = TRUE, device_count = 1L)
    },
    gpu_allocatable_bytes = function() {
      called <<- TRUE
      0
    }
  )
  withr::local_envvar(CUPLYR_DISABLE_GPU = "true")
  # pkgload exports no dot-names, so reach the hook through the namespace.
  expect_message(cuplyr:::.onAttach(dirname(p138_root()), "cuplyr"), "GPU")
  expect_false(called)
})

test_that("loading the package under the switch records no GPU", {
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  run <- p138_run_rscript(
    c(
      "saveRDS(list(",
      "  option = getOption('cuplyr.gpu_available'),",
      "  has_gpu = has_gpu()",
      "), p138_result)"
    ),
    env = "CUPLYR_DISABLE_GPU=true"
  )
  expect_identical(run$status, 0L, info = paste(run$output, collapse = "\n"))
  expect_identical(run$result$option, FALSE)
  expect_identical(run$result$has_gpu, FALSE)
})

test_that("with the GPU switched off and no visible device, GPU code fails", {
  # The example test below relies on this: an unguarded example that calls
  # GPU code errors (or kills the process) instead of quietly running.
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  run <- p138_run_rscript(
    c(
      "res <- tryCatch({",
      "  x <- collect(tbl_gpu(mtcars))",
      "  'ran'",
      "}, error = function(e) 'error')",
      "saveRDS(list(res = res, has_gpu = has_gpu()), p138_result)"
    ),
    env = p138_no_gpu_env
  )
  expect_true(
    run$status != 0L || identical(run$result$res, "error"),
    info = paste(run$output, collapse = "\n")
  )
})

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

test_that("every help-page example runs with the GPU switched off", {
  p138_skip_if_no_source_tree()
  run <- p138_run_rscript(
    c(
      sprintf(
        "rds <- list.files(file.path(%s, 'man'), pattern = '[.]Rd$', full.names = TRUE)",
        deparse(p138_root())
      ),
      "failed <- character()",
      "n <- 0L",
      "for (rd in rds) {",
      "  ex <- tempfile(fileext = '.R')",
      "  tools::Rd2ex(rd, ex, commentDontrun = TRUE, commentDonttest = FALSE)",
      "  if (!file.exists(ex)) next",
      "  n <- n + 1L",
      "  env <- new.env(parent = globalenv())",
      "  res <- tryCatch({",
      "    utils::capture.output(source(ex, local = env, echo = FALSE))",
      "    'ok'",
      "  }, error = function(e) conditionMessage(e))",
      "  if (!identical(res, 'ok')) failed <- c(failed, paste0(basename(rd), ': ', res))",
      "}",
      "saveRDS(list(n = n, failed = failed, has_gpu = has_gpu()), p138_result)"
    ),
    env = p138_no_gpu_env
  )
  expect_identical(run$status, 0L, info = paste(run$output, collapse = "\n"))
  expect_identical(run$result$has_gpu, FALSE)
  expect_gte(run$result$n, 40L)
  expect_identical(run$result$failed, character())
})

test_that("help pages guard GPU examples with @examplesIf, out of sight", {
  p138_skip_if_no_source_tree()
  # Pages whose examples run without a GPU by design.
  no_gpu_pages <- c(
    "has_gpu.Rd", "gpu_details.Rd", "show_gpu.Rd", "check_deps.Rd",
    "diagnostics.Rd", "cuplyr-options.Rd"
  )
  files <- list.files(file.path(p138_root(), "man"), pattern = "[.]Rd$",
                      full.names = TRUE)
  parts <- Filter(Negate(is.null), lapply(stats::setNames(files, basename(files)),
                                          p138_rd_examples))
  expect_gte(length(parts), 40L)

  runs_code <- vapply(parts, function(p) nzchar(trimws(p$runs)), logical(1))
  guarded <- vapply(parts, function(p) grepl("has_gpu()", p$guard, fixed = TRUE),
                    logical(1))

  # No visible `if (has_gpu())` wrapper; has_gpu's own page demonstrates it.
  shows_check <- vapply(parts, function(p) grepl("has_gpu(", p$runs, fixed = TRUE),
                        logical(1))
  expect_identical(
    sort(names(parts)[shows_check & names(parts) != "has_gpu.Rd"]),
    character()
  )

  # Every page that runs example code is guarded, unless it runs without a GPU.
  needs_guard <- runs_code & !names(parts) %in% no_gpu_pages
  expect_identical(sort(names(parts)[needs_guard & !guarded]), character())

  # The no-GPU pages stay unguarded, so they run (and show output) everywhere.
  present <- intersect(no_gpu_pages, names(parts))
  expect_setequal(present, no_gpu_pages)
  expect_identical(names(which(guarded[present])), character(0))

  # verify_installation() runs on the GPU instead of being \dontrun.
  expect_true(guarded[["verify_installation.Rd"]])
  expect_true(runs_code[["verify_installation.Rd"]])
  expect_true(guarded[["tbl_gpu.Rd"]])
})

# ---------------------------------------------------------------------------
# Precomputed vignettes
# ---------------------------------------------------------------------------

test_that("every vignette has a .Rmd.orig source and a precomputed .Rmd", {
  p138_skip_if_no_source_tree()
  dir <- file.path(p138_root(), "vignettes")
  origs <- sub("[.]Rmd[.]orig$", "", list.files(dir, pattern = "[.]Rmd[.]orig$"))
  rmds <- sub("[.]Rmd$", "", list.files(dir, pattern = "[.]Rmd$"))
  expect_setequal(origs, p138_vignette_names)
  expect_setequal(rmds, p138_vignette_names)
})

test_that("the shipped vignettes contain no code that runs at build time", {
  p138_skip_if_no_source_tree()
  dir <- file.path(p138_root(), "vignettes")
  for (name in p138_vignette_names) {
    path <- file.path(dir, paste0(name, ".Rmd"))
    lines <- readLines(path, warn = FALSE)
    chunk_headers <- grep("^[ \t>]*(`{3,}|~{3,})[ \t]*\\{", lines, value = TRUE)
    expect_identical(chunk_headers, character(), info = name)
    inline <- grep("`r[ #]", lines, value = TRUE)
    expect_identical(inline, character(), info = name)
    # Precomputed: the output of the code is in the file.
    expect_true(any(grepl("^#>", lines)), info = name)
    # Still a knitr::rmarkdown vignette, so R CMD build renders it.
    expect_true(
      any(grepl("%\\VignetteEngine{knitr::rmarkdown}", lines, fixed = TRUE)),
      info = name
    )
    expect_false(any(grepl("CUPLYR_VIGNETTE_EVAL", lines, fixed = TRUE)), info = name)
  }
})

test_that("each source and its shipped vignette have the same index entry", {
  p138_skip_if_no_source_tree()
  dir <- file.path(p138_root(), "vignettes")
  entry <- function(path) {
    lines <- readLines(path, warn = FALSE)
    hit <- regmatches(lines, regexpr("(?<=\\\\VignetteIndexEntry\\{).*(?=\\})",
                                     lines, perl = TRUE))
    hit
  }
  for (name in p138_vignette_names) {
    src <- entry(file.path(dir, paste0(name, ".Rmd.orig")))
    out <- entry(file.path(dir, paste0(name, ".Rmd")))
    expect_length(src, 1)
    expect_identical(out, src, info = name)
  }
})

test_that("the sources stay out of the package and nothing sets the old eval switch", {
  p138_skip_if_no_source_tree()
  root <- p138_root()
  patterns <- readLines(file.path(root, ".Rbuildignore"), warn = FALSE)
  patterns <- patterns[nzchar(patterns)]
  ignored <- function(path) {
    any(vapply(patterns, function(p) grepl(p, path, perl = TRUE, ignore.case = TRUE),
               logical(1)))
  }
  for (name in p138_vignette_names) {
    expect_true(ignored(paste0("vignettes/", name, ".Rmd.orig")), info = name)
    expect_false(ignored(paste0("vignettes/", name, ".Rmd")), info = name)
  }
  pkgdown_script <- readLines(file.path(root, "scripts", "build_pkgdown.R"), warn = FALSE)
  expect_false(any(grepl("CUPLYR_VIGNETTE_EVAL", pkgdown_script, fixed = TRUE)))
  for (name in p138_vignette_names) {
    src <- readLines(file.path(root, "vignettes", paste0(name, ".Rmd.orig")), warn = FALSE)
    expect_false(any(grepl("CUPLYR_VIGNETTE_EVAL", src, fixed = TRUE)), info = name)
  }
})

test_that("knitting a vignette needs a GPU and writes nothing without one", {
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- withr::local_tempdir("p138-")
  src <- p138_write_orig(c("```{r}", "1 + 1", "```"), dir)
  out <- file.path(dir, "p138.Rmd")
  withr::local_envvar(CUPLYR_DISABLE_GPU = "true")
  expect_error(lib$cuplyr_knit_vignette(src, out), "GPU")
  expect_false(file.exists(out))
})

test_that("a knitted vignette is plain Markdown with the output", {
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- withr::local_tempdir("p138-")
  src <- p138_write_orig(c(
    "```{r, include = FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = \"#>\")",
    "```",
    "",
    "```{r}",
    "x <- tbl_gpu(data.frame(a = 1:3)) |> dplyr::mutate(b = a * 2L)",
    "dplyr::pull(x, b)",
    "```",
    "",
    "```{r, error = TRUE}",
    "stop(\"expected138\")",
    "```",
    "",
    "```{r, eval = FALSE}",
    "not_run_138()",
    "```"
  ), dir)
  out <- file.path(dir, "p138.Rmd")
  wd <- getwd()
  lib$cuplyr_knit_vignette(src, out)
  expect_identical(getwd(), wd)
  lines <- readLines(out, warn = FALSE)
  expect_identical(grep("^[ \t]*`{3,}[ \t]*\\{", lines, value = TRUE), character())
  expect_true(any(lines == "dplyr::pull(x, b)"))
  expect_true(any(lines == "#> [1] 2 4 6"))
  # knitr shows the error as R does (the message may be on the next line).
  expect_true(any(grepl("^#> Error", lines)))
  expect_true(any(grepl("^#> .*expected138", lines)))
  expect_true(any(lines == "not_run_138()"))
  expect_true(any(lines == "Some text."))
  expect_true(any(lines == "title: \"p138\""))
})

test_that("an unexpected chunk error stops the knit and writes nothing", {
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- withr::local_tempdir("p138-")
  src <- p138_write_orig(c("```{r}", "stop(\"boom138\")", "```"), dir)
  out <- file.path(dir, "p138.Rmd")
  expect_error(lib$cuplyr_knit_vignette(src, out), "boom138")
  expect_false(file.exists(out))
})

test_that("options a vignette sets don't outlive its knit", {
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- withr::local_tempdir("p138-")
  src <- p138_write_orig(c(
    "```{r}",
    "options(cuplyr.p138_leak = 'set', cuplyr.exec_mode = 'eager')",
    "```"
  ), dir)
  mode <- getOption("cuplyr.exec_mode")
  lib$cuplyr_knit_vignette(src, file.path(dir, "p138.Rmd"))
  expect_null(getOption("cuplyr.p138_leak"))
  expect_identical(getOption("cuplyr.exec_mode"), mode)
})

test_that("cuplyr_knit_vignettes() knits each source in its own R session", {
  # Each shipped vignette shows what a fresh session prints, not state left
  # over from the vignette knitted before it.
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- withr::local_tempdir("p138-")
  setup <- c(
    "```{r, include = FALSE}",
    "knitr::opts_chunk$set(collapse = TRUE, comment = \"#>\")",
    "```",
    ""
  )
  writeLines(c(
    "---", "title: \"a\"", "---", "", setup,
    "```{r}",
    "assign(\"p138_shared\", 1, envir = globalenv())",
    "options(cuplyr.p138_shared = TRUE)",
    "```"
  ), file.path(dir, "a.Rmd.orig"))
  writeLines(c(
    "---", "title: \"b\"", "---", "", setup,
    "```{r}",
    "exists(\"p138_shared\", envir = globalenv())",
    "is.null(getOption(\"cuplyr.p138_shared\"))",
    "cuplyr::has_gpu()",
    "```"
  ), file.path(dir, "b.Rmd.orig"))

  lib$cuplyr_knit_vignettes(dir = dir, pkg = p138_root())

  expect_true(file.exists(file.path(dir, "a.Rmd")))
  b <- readLines(file.path(dir, "b.Rmd"), warn = FALSE)
  after <- function(code) b[match(code, b) + 1L]
  expect_identical(after("exists(\"p138_shared\", envir = globalenv())"), "#> [1] FALSE")
  expect_identical(after("is.null(getOption(\"cuplyr.p138_shared\"))"), "#> [1] TRUE")
  expect_identical(after("cuplyr::has_gpu()"), "#> [1] TRUE")
  # Nor did it run in this process.
  expect_false(exists("p138_shared", envir = globalenv()))
  expect_null(getOption("cuplyr.p138_shared"))
})

test_that("the shipped vignettes are up to date with their sources", {
  # Knits every source on the GPU and compares everything but the output
  # lines (memory and timings differ between runs) with the shipped file.
  skip_if_no_gpu()
  p138_skip_if_no_source_tree()
  lib <- p138_lib()
  dir <- file.path(p138_root(), "vignettes")
  tmp <- withr::local_tempdir("p138-")
  strip <- function(lines) lines[!grepl("^#>", lines)]
  for (name in p138_vignette_names) {
    out <- file.path(tmp, paste0(name, ".Rmd"))
    lib$cuplyr_knit_vignette(file.path(dir, paste0(name, ".Rmd.orig")), out)
    expect_identical(
      strip(readLines(out, warn = FALSE)),
      strip(readLines(file.path(dir, paste0(name, ".Rmd")), warn = FALSE)),
      info = name
    )
  }
})
