#!/usr/bin/env Rscript
# Build the pkgdown site into docs/. Run it through tools/site, which holds the
# GPU lock: the vignettes and the reference examples run GPU code.
#
# The shipped vignettes/*.Rmd are precomputed (#138): the output of their code is
# in the file. The site shows live output instead, so the articles are knitted
# from their vignettes/*.Rmd.orig sources over the .Rmd files first, on the GPU,
# and the committed .Rmd files are put back when the build ends, also on error.
#
# This is pkgdown::build_site(devel = TRUE) with one change. build_site()
# renders every article in a fresh R process, where library(cuplyr) would load
# an installed copy (or fail without one). Here the articles render in this
# process, against the checkout loaded with pkgload::load_all(), so nothing is
# installed and the site documents this checkout.

if (Sys.getenv("XDG_CACHE_HOME") == "") {
  Sys.setenv(XDG_CACHE_HOME = "/tmp")
}
pkgload::load_all(".", export_all = FALSE, helpers = FALSE, quiet = TRUE)

source("tools/vignettes-lib.R")
vignette_files <- list.files("vignettes", pattern = "[.]Rmd$", full.names = TRUE)
committed <- lapply(vignette_files, function(f) readBin(f, "raw", file.size(f)))
restore_vignettes <- function() {
  for (i in seq_along(vignette_files)) {
    writeBin(committed[[i]], vignette_files[i])
  }
}

tryCatch({
  cuplyr_knit_vignettes("vignettes")

  pkgdown::clean_site(force = TRUE)
  pkg <- pkgdown::as_pkgdown(".")
  pkgdown::init_site(pkg)
  pkgdown::build_home(pkg, preview = FALSE)
  pkgdown::build_reference(pkg, lazy = FALSE, devel = TRUE, preview = FALSE)
  for (name in pkg$vignettes$name) {
    pkgdown::build_article(name, pkg, new_process = FALSE)
  }
  pkgdown::build_articles_index(pkg)
  pkgdown::build_news(pkg, preview = FALSE)
  pkgdown:::build_sitemap(pkg)
  pkgdown::build_llm_docs(pkg)
  pkgdown::build_redirects(pkg)
  pkgdown::build_search(pkg)
  pkgdown:::check_built_site(pkg)
}, finally = restore_vignettes())

# Keep AGENTS/CLAUDE operational docs and the pull request template out of
# the published website.
unlink(
  file.path(
    "docs",
    c(
      "AGENTS.html",
      "AGENTS.md",
      "CLAUDE.html",
      "CLAUDE.md",
      "DEVELOPER_GUIDE.html",
      "DEVELOPER_GUIDE.md",
      "PULL_REQUEST_TEMPLATE.html",
      "PULL_REQUEST_TEMPLATE.md"
    )
  ),
  force = TRUE
)
