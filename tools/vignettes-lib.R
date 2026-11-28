# Precomputed vignettes (#138).
#
# vignettes/<name>.Rmd.orig is the source; vignettes/<name>.Rmd is what ships:
# the same text with every chunk run on the GPU and its output written into the
# file as plain Markdown (no chunks, no inline code), so building the package
# runs nothing and needs no GPU. tools/vignettes-knit re-knits them for a
# commit; scripts/build_pkgdown.R re-knits them for a site build. Both source
# this file and call the functions below with cuplyr loaded.

# Knit one source to `output`. Stops before anything else, and writes nothing,
# when there is no GPU or a chunk fails (a chunk with `error = TRUE` keeps its
# error in the output). The file is written only when the knit succeeds.
cuplyr_knit_vignette <- function(input, output) {
  if (!isTRUE(cuplyr::has_gpu())) {
    stop(
      "Knitting a vignette runs its code on the GPU, but cuplyr::has_gpu() is ",
      "FALSE (no GPU, or CUPLYR_DISABLE_GPU is set).",
      call. = FALSE
    )
  }
  input <- normalizePath(input, mustWork = TRUE)
  output <- file.path(normalizePath(dirname(output), mustWork = TRUE),
                      basename(output))
  tmp <- tempfile("knit-", tmpdir = dirname(output), fileext = ".md")
  on.exit(unlink(tmp), add = TRUE)

  # The knit must not leak into the caller: options a vignette sets (and the
  # ones it adds), the working directory and knitr's own settings are put back.
  old_options <- options()
  old_wd <- setwd(dirname(input))
  old_chunk <- knitr::opts_chunk$get()
  old_knit <- knitr::opts_knit$get()
  old_patterns <- knitr::knit_patterns$get()
  old_hooks <- knitr::knit_hooks$get()
  on.exit({
    setwd(old_wd)
    added <- setdiff(names(options()), names(old_options))
    options(old_options)
    if (length(added)) options(stats::setNames(vector("list", length(added)), added))
    knitr::opts_chunk$restore(old_chunk)
    knitr::opts_knit$restore(old_knit)
    knitr::knit_patterns$restore(old_patterns)
    knitr::knit_hooks$restore(old_hooks)
  }, add = TRUE)

  # ".Rmd.orig" has no extension knitr knows, so say it is Markdown.
  knitr::knit_patterns$set(knitr::pat_md())
  knitr::render_markdown()
  # knit() defaults to error = TRUE and would bake an unexpected failure into
  # the shipped text.
  knitr::opts_chunk$set(error = FALSE)
  # knitr (through rlang) prints a shown error as "Error:" and, on the next
  # line, "! message". Put the message back on the first line, as R prints it.
  plain_error <- knitr::knit_hooks$get("error")
  knitr::knit_hooks$set(error = function(x, options) {
    lines <- strsplit(x, "\n", fixed = TRUE)[[1]]
    mark <- paste0(options$comment, " ! ")
    if (length(lines) > 1 && grepl("^(#> )?Error[^ ]*( in .*)?:$", lines[1]) &&
          startsWith(lines[2], mark)) {
      lines <- c(paste(lines[1], substring(lines[2], nchar(mark) + 1)), lines[-(1:2)])
      x <- paste0(paste(lines, collapse = "\n"), if (endsWith(x, "\n")) "\n")
    }
    plain_error(x, options)
  })

  knitr::knit(
    basename(input), output = tmp, quiet = TRUE,
    envir = new.env(parent = globalenv()), encoding = "UTF-8"
  )
  if (!file.rename(tmp, output)) {
    stop("Could not write ", output, call. = FALSE)
  }
  invisible(output)
}

# Knit every `<name>.Rmd.orig` in `dir` (or just `names`) over `<name>.Rmd`.
cuplyr_knit_vignettes <- function(dir = "vignettes", names = NULL) {
  sources <- list.files(dir, pattern = "[.]Rmd[.]orig$", full.names = TRUE)
  stems <- sub("[.]Rmd[.]orig$", "", basename(sources))
  if (!is.null(names)) {
    unknown <- setdiff(names, stems)
    if (length(unknown)) {
      stop("No ", paste0(unknown, ".Rmd.orig", collapse = ", "), " in ", dir,
           call. = FALSE)
    }
    keep <- stems %in% names
    sources <- sources[keep]
    stems <- stems[keep]
  }
  for (i in seq_along(sources)) {
    message("Knitting ", basename(sources[i]))
    cuplyr_knit_vignette(sources[i], file.path(dir, paste0(stems[i], ".Rmd")))
  }
  invisible(file.path(dir, paste0(stems, ".Rmd")))
}
