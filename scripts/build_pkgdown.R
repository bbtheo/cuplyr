#!/usr/bin/env Rscript

if (Sys.getenv("XDG_CACHE_HOME") == "") {
  Sys.setenv(XDG_CACHE_HOME = "/tmp")
}

pkgdown::clean_site(force = TRUE)
pkgdown::build_site()

# Keep AGENTS/CLAUDE operational docs out of the published website.
unlink(
  file.path(
    "docs",
    c(
      "AGENTS.html",
      "AGENTS.md",
      "CLAUDE.html",
      "CLAUDE.md",
      "DEVELOPER_GUIDE.html",
      "DEVELOPER_GUIDE.md"
    )
  ),
  force = TRUE
)
