# Render the rolling-window benchmark report (F54, rewritten as a Quarto
# document in F63): bench/rolling-window.qmd runs the benchmark and holds all
# the prose. This file is only the command that renders it.
#
#   tools/r 'source("bench/rolling-window.R")'
#
# The render holds the GPU for the whole benchmark (about 16 minutes on a cold
# cache), so it goes through tools/r like every other R process. Variables set
# before source():
#   FULL_RERUN   TRUE passes -P rebuild:true: rerun and re-save every timing
#                chunk (the document turns it into knitr's cache.rebuild for the
#                bench-* chunks only). Never use `quarto render --cache-refresh`:
#                it reruns no timing chunk and caches every other chunk.
#   OUT_DIR      where rolling-window.html and the CSVs are written
#   CACHE_DIR    where the knitr cache of the timing chunks lives
#   LIB          library holding the third-party packages (bench/install-deps.R)
#   PARAMS       named list of any other document parameter, e.g.
#                list(n_rows = 5000, times = 2); see the front matter of the qmd

render_rolling_report <- function() {
  root <- normalizePath(getwd(), mustWork = TRUE)
  qmd <- file.path(root, "bench", "rolling-window.qmd")
  if (!file.exists(qmd)) stop("run this from the cuplyr checkout (bench/rolling-window.qmd not found in ", root, ")")

  quarto <- Sys.getenv("QUARTO_PATH", "")
  if (!nzchar(quarto)) quarto <- unname(Sys.which("quarto"))
  if (!nzchar(quarto) && file.exists(path.expand("~/.local/bin/quarto"))) quarto <- path.expand("~/.local/bin/quarto")
  if (!nzchar(quarto)) stop("the quarto CLI was not found (put it on PATH or set QUARTO_PATH)")

  pick <- function(name) if (exists(name, envir = globalenv(), inherits = FALSE)) get(name, envir = globalenv())
  params <- c(list(root = root),
              if (!is.null(pick("LIB"))) list(lib = pick("LIB")),
              if (!is.null(pick("OUT_DIR"))) list(out_dir = pick("OUT_DIR")),
              if (!is.null(pick("CACHE_DIR"))) list(cache_dir = pick("CACHE_DIR")),
              pick("PARAMS"))
  p_args <- unlist(lapply(names(params), function(nm) {
    v <- params[[nm]]
    c("-P", paste0(nm, ":", if (is.logical(v) && length(v) == 1L) tolower(as.character(v)) else
      if (is.character(v) && length(v) == 1L) v else paste0("[", paste(v, collapse = ", "), "]")))
  }))

  # The rendered HTML ends up in the out_dir parameter (the qmd's default
  # unless overridden), never next to the source. Quarto empties its
  # --output-dir before it renders, and out_dir holds files that must survive
  # (the install logs the document reads, and by default the knitr cache), so
  # Quarto renders into a staging directory and the HTML is copied over.
  out_dir <- params$out_dir
  if (is.null(out_dir)) out_dir <- rmarkdown::yaml_front_matter(qmd)$params$out_dir
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  staging <- tempfile("rolling-html-")
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)

  args <- c("render", qmd, p_args, "--output-dir", staging)
  if (isTRUE(pick("FULL_RERUN"))) args <- c(args, "-P", "rebuild:true")

  message("quarto ", paste(args, collapse = " "))
  status <- system2(quarto, args)
  if (!identical(status, 0L)) stop("quarto render failed with exit status ", status)
  html <- file.path(staging, "rolling-window.html")
  if (!file.exists(html)) stop("quarto did not produce ", html)
  file.copy(html, file.path(out_dir, "rolling-window.html"), overwrite = TRUE)
  message("wrote ", file.path(out_dir, "rolling-window.html"))
  invisible(status)
}

render_rolling_report()
