#!/usr/bin/env Rscript
# Build the pkgdown site into docs/. Run it through tools/site, which holds the
# GPU lock: the vignettes and the reference examples run GPU code.

if (Sys.getenv("XDG_CACHE_HOME") == "") {
  Sys.setenv(XDG_CACHE_HOME = "/tmp")
}

# devel = TRUE loads the package with pkgload::load_all() instead of
# installing it into a temporary library, so the site documents this checkout
# as the tests see it.
pkgdown::clean_site(force = TRUE)
pkgdown::build_site(devel = TRUE, preview = FALSE)

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
