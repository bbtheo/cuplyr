test_that("install.sh dry-run works outside source tree", {
  script <- normalizePath(
    file.path(testthat::test_path("..", ".."), "install.sh"),
    mustWork = FALSE
  )
  skip_if_not(
    file.exists(script),
    message = "install.sh not available in this test context"
  )

  tmp <- tempfile("cuplyr-install-test-")
  dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(tmp)

  output <- system2(
    "bash",
    c(script, "--dry-run", "--method=pixi"),
    stdout = TRUE,
    stderr = TRUE
  )
  status <- attr(output, "status")
  if (is.null(status)) {
    status <- 0L
  }

  expect_equal(
    as.integer(status),
    0L,
    info = paste(output, collapse = "\n")
  )
})
