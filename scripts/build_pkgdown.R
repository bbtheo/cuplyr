#!/usr/bin/env Rscript
# Build the pkgdown site into docs/. Run it through tools/site, which holds the
# GPU lock: the vignettes and the reference examples run GPU code.
#
# This is pkgdown::build_site(devel = TRUE) with one change. build_site()
# renders every article in a fresh R process, where library(cuplyr) would load
# an installed copy (or fail without one). Here the articles render in this
# process, against the checkout loaded with pkgload::load_all(), so nothing is
# installed and the site documents this checkout.

if (Sys.getenv("XDG_CACHE_HOME") == "") {
  Sys.setenv(XDG_CACHE_HOME = "/tmp")
}
# The vignettes evaluate their code only when this is set; the site shows the
# output.
if (Sys.getenv("CUPLYR_VIGNETTE_EVAL") == "") {
  Sys.setenv(CUPLYR_VIGNETTE_EVAL = "true")
}

pkgload::load_all(".", export_all = FALSE, helpers = FALSE, quiet = TRUE)

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
