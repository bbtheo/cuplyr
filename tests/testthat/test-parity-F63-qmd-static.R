# F63: the rolling-window report as one Quarto document.
#
# bench/rolling-window.qmd holds the prose of the F54 report and the code
# that produces every number, table and figure in it. These checks read the
# document source (no render, no GPU): every number in the prose is an inline
# R expression, the front matter declares a self-contained HTML output and
# the run parameters, the timing chunks are cached with a cache key that
# covers the helper code, cuplyr's sources, the package versions and the run
# configuration. They also pin the cache-key helpers in bench/rolling-lib.R.
# The render itself is test-parity-F63-qmd-render.R.

test_that("the prose linter flags hand-typed numbers and exempts computed ones", {
  ok <- c(
    "---", "title: \"Rolling 1e6\"", "params:", "  n_rows: 1000000", "---", "",
    "# Results at `r fmt_rows(n_rows)` rows {#sec-results}", "",
    "Theo decided (2026-10-21) that cuplyr takes `r med('mean', 'cuplyr', 101)` ms.",
    "See F1 and Deviation 2; `RcppRoll (1 thread)` and `bench/rolling-lib.R` are names.",
    "Plots use a log10 axis. Source: <https://example.org/2024/10/03/post>.",
    "1. first step", "2. second step",
    "<!-- a comment with 2.78 ms in it",
    "spanning 3 lines -->",
    "```{r bench-main}", "#| cache: true", "x <- rnorm(1e6)", "```",
    "```", "tools/r 'source(\"bench/rolling-window.R\")' # 15 min", "```"
  )
  expect_equal(nrow(f63_prose_number_lines(ok)), 0L)

  bad <- c(
    "cuplyr takes 2.78 ms at w=11.",            # bare numbers
    "The floor is `2.0 ms`.",                    # number hidden in a code span
    "216 of 216 cells match.",
    "```{r}", "#| fig-cap: \"Rolling mean, n = 1e6\"", "plot(1)", "```",
    "```{r}", "#| fig-cap: !expr paste('n =', n_rows)", "plot(1)", "```"
  )
  res <- f63_prose_number_lines(bad)
  expect_setequal(res$line, c(1L, 2L, 3L, 4L))
})

test_that("bench/rolling-window.qmd exists and every number in its prose is computed", {
  lines <- f63_qmd_lines()
  res <- f63_prose_number_lines(lines)
  expect_equal(
    nrow(res), 0L,
    label = paste0("prose lines with hand-typed numbers:\n",
                   paste0("  ", res$line, ": ", res$text, collapse = "\n"))
  )
  # The F54 report had several hundred numbers in its prose; a document that
  # passes the linter by dropping them has not ported the report.
  expect_gte(f63_inline_r_count(lines), 150)
})

test_that("the document keeps the F54 report's sections", {
  lines <- f63_qmd_lines()
  p <- f63_parse_qmd(lines)
  heads <- trimws(sub("^#+\\s*", "", grep("^#{1,4}\\s", p$prose, value = TRUE)))
  wanted <- c(
    "Read this first: what the numbers are",
    "Method",
    "Threads actually used",
    "Hardware and versions",
    "Install results",
    "Results: median ms per call at",
    "Correctness",
    "Extension: where the GPU should help",
    "Per-call cost breakdown (overhead)",
    "Conclusions (in the post's style, updated)",
    "Findings (candidate backlog rows; nothing was fixed in this feature)"
  )
  for (w in wanted) {
    expect_true(any(startsWith(heads, w)), label = paste0("a heading starting with '", w, "'"))
  }
})

test_that("front matter: self-contained HTML and the run parameters with the F54 defaults", {
  front <- f63_front(f63_qmd_lines())
  html <- front$format$html
  expect_true(is.list(html), label = "format: html is configured")
  expect_true(isTRUE(html$`embed-resources`) || isTRUE(html$`self-contained`),
              label = "html output is self-contained (embed-resources: true)")

  defaults <- list(n_rows = 1e6, widths = c(11, 101, 1001), times = 20, times_custom = 5,
                   cell_budget = 20, probe_limit = 60, group_rows = 1e6, n_groups = 1e3,
                   group_width = 101, ext_rows = 1e7, ext_times = 10)
  for (nm in names(defaults)) {
    v <- f63_param(front, nm)
    expect_false(is.null(v), label = paste0("params$", nm, " is declared"))
    if (!is.null(v)) {
      expect_equal(as.numeric(unlist(v)), defaults[[nm]], label = paste0("default of params$", nm))
    }
  }
  for (nm in c("root", "lib", "out_dir", "cache_dir", "stats", "skip_ext", "rebuild")) {
    expect_false(is.null(front$params[[nm]]), label = paste0("params$", nm, " is declared"))
  }
  expect_setequal(unlist(f63_param(front, "stats")),
                  c("mean", "median", "min", "max", "sum", "sd", "custom"))
  expect_false(isTRUE(f63_param(front, "skip_ext")))
  expect_false(isTRUE(f63_param(front, "rebuild")))
  expect_equal(f63_param(front, "lib"), "/home/theo/cuplr/scratchpad/benchmarks/rlib")
  expect_equal(normalizePath(f63_param(front, "out_dir"), mustWork = FALSE),
               "/home/theo/cuplr/scratchpad/benchmarks/rolling")
})

test_that("timing chunks are cached, keyed on code, versions and config, and load the worktree", {
  lines <- f63_qmd_lines()
  p <- f63_parse_qmd(lines)
  labels <- vapply(p$chunks, function(ch) ch$label, "")
  bench <- p$chunks[startsWith(labels, "bench-")]
  expect_true("bench-main" %in% labels, label = "a chunk labelled bench-main runs the main timings")
  expect_gte(length(bench), 3L)
  for (ch in bench) {
    expect_true(f63_chunk_cached(ch), label = paste0("chunk ", ch$label, " has cache: true"))
  }
  # no chunk outside bench-* is cached: cached GPU objects would come back as
  # dead pointers, and cached non-timing chunks would hide code changes
  for (ch in p$chunks[!startsWith(labels, "bench-")]) {
    expect_false(f63_chunk_cached(ch), label = paste0("chunk ", if (nzchar(ch$label)) ch$label else "<unlabelled>", " is not cached"))
  }

  code <- paste(unlist(lapply(p$chunks, function(ch) c(ch$code, ch$opts))), collapse = "\n")
  front_txt <- paste(p$front, collapse = "\n")
  all_txt <- paste(code, front_txt, sep = "\n")
  expect_match(all_txt, "cache.extra", fixed = TRUE)
  expect_match(code, "rw_cache_key(", fixed = TRUE)
  expect_match(code, "rw_cuplyr_sources(", fixed = TRUE)
  expect_match(code, "rw_versions(", fixed = TRUE)
  expect_match(code, "cache.path", fixed = TRUE)
  expect_match(code, "params$cache_dir", fixed = TRUE)
  expect_match(code, "pkgload::load_all(", fixed = TRUE)
  expect_match(code, "params$root", fixed = TRUE)
  expect_match(code, ".libPaths(", fixed = TRUE)
  expect_match(code, "params$lib", fixed = TRUE)
  expect_match(code, "rolling-lib.R", fixed = TRUE)
  expect_no_match(code, "install.packages", fixed = TRUE)
  # amendment (Theo, 2026-10-22): cuplyr cells are probed for a CPU fallback
  # before timing; the forced rerun is params$rebuild (B2: --cache-refresh re-ran nothing)
  expect_match(code, "rw_native_probe(", fixed = TRUE)
  expect_match(code, "params$rebuild", fixed = TRUE)
  expect_match(code, "cache.rebuild", fixed = TRUE)
  # every chunk's code parses
  for (ch in p$chunks) {
    expect_no_error(parse(text = ch$code, keep.source = FALSE))
  }
})

test_that("install-deps.R lists the render dependencies", {
  txt <- paste(readLines(f63_require("bench/install-deps.R"), warn = FALSE), collapse = "\n")
  expect_match(txt, "\"knitr\"", fixed = TRUE)
  expect_match(txt, "\"rmarkdown\"", fixed = TRUE)
})

# ---- cache-key helpers in bench/rolling-lib.R --------------------------------------

f63_lib_env <- function() {
  path <- f63_require("bench/rolling-lib.R")
  e <- new.env(parent = globalenv())
  sys.source(path, envir = e)
  e
}

test_that("rw_versions() reports installed versions and 'not installed'", {
  rw <- f63_lib_env()
  v <- rw$rw_versions(c("stats", "f63.no.such.package"))
  expect_type(v, "character")
  expect_named(v, c("stats", "f63.no.such.package"))
  expect_equal(v[["stats"]], as.character(utils::packageVersion("stats")))
  expect_equal(v[["f63.no.such.package"]], "not installed")
})

test_that("rw_cache_key() changes with file contents, versions and config only", {
  rw <- f63_lib_env()
  f1 <- withr::local_tempfile(lines = c("a <- 1", "b <- 2"))
  f2 <- withr::local_tempfile(lines = "c <- 3")
  vers <- c(data.table = "1.18.4", cuplyr = "0.1.1.9000")
  cfg <- list(n_rows = 1e6, widths = c(11, 101, 1001))

  k <- rw$rw_cache_key(c(f1, f2), vers, cfg)
  expect_type(k, "character")
  expect_length(k, 1L)
  expect_true(nzchar(k))
  expect_identical(rw$rw_cache_key(c(f1, f2), vers, cfg), k)

  # touching a file without changing it keeps the key
  Sys.setFileTime(f1, Sys.time() + 3600)
  expect_identical(rw$rw_cache_key(c(f1, f2), vers, cfg), k)

  # a content change, a version change or a config change each move the key
  writeLines(c("a <- 1", "b <- 3"), f1)
  k_file <- rw$rw_cache_key(c(f1, f2), vers, cfg)
  expect_false(identical(k_file, k))
  writeLines(c("a <- 1", "b <- 2"), f1)
  expect_identical(rw$rw_cache_key(c(f1, f2), vers, cfg), k)

  vers2 <- vers
  vers2[["data.table"]] <- "1.18.5"
  expect_false(identical(rw$rw_cache_key(c(f1, f2), vers2, cfg), k))

  cfg2 <- cfg
  cfg2$n_rows <- 1e4
  expect_false(identical(rw$rw_cache_key(c(f1, f2), vers, cfg2), k))
})

test_that("rw_cuplyr_sources() lists cuplyr's R and C++ sources", {
  rw <- f63_lib_env()
  root <- f63_root()
  src <- rw$rw_cuplyr_sources(root)
  expect_type(src, "character")
  expect_true(all(file.exists(src)))
  expect_true(all(grepl("\\.(R|cpp|cu|cuh|h|hpp)$", src)))
  base <- basename(src)
  expect_true(all(c("roll.R", "mutate.R", "ops_window.cpp") %in% base))
  expect_false(any(grepl("\\.(o|so)$", src)))
  expect_identical(rw$rw_cuplyr_sources(root), src)
})
