test_that("get_source_dir() finds cloned child cuplyr directory", {
  tmp <- tempfile("cuplyr-src-root-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  src_dir <- file.path(tmp, "cuplyr")
  dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c(
      "Package: cuplyr",
      "Version: 0.0.0.9000"
    ),
    file.path(src_dir, "DESCRIPTION")
  )
  file.create(file.path(src_dir, "configure"))

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(tmp)

  detected <- get_source_dir(
    repo = NULL,
    ref = "master",
    dry_run = FALSE,
    verbose = FALSE
  )

  expect_equal(detected, normalizePath(src_dir))
})
