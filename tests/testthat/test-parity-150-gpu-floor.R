# #150: the minimum GPU is stated the same way everywhere.
#
# Every supported route (install.sh, install_cuplyr(), the pixi environments
# and the conda package) installs a RAPIDS built for CUDA 13, and CUDA 13 has
# no code for Volta: the CUDA 13 libcudf ships machine code for sm_75 and
# newer only. So the floor is compute capability 7.5 (Turing).
#
# Contract, for DESCRIPTION's SystemRequirements, README.md, the
# getting-started vignette, CONTRIBUTING.md and conda/README.md:
# - each states compute capability 7.5 as the minimum;
# - none states a lower floor (7.0, "Volta or newer");
# - Volta is mentioned, if at all, only to say it isn't supported, or that
#   it needs a source build against a CUDA 12 RAPIDS.
#
# None of these tests need a GPU. Files absent from the test context (an
# installed package has no conda/ or .github/) are skipped.

p150_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p150_docs <- function() {
  files <- c(
    README = "README.md",
    vignette = file.path("vignettes", "getting-started.Rmd"),
    CONTRIBUTING = file.path(".github", "CONTRIBUTING.md"),
    conda_README = file.path("conda", "README.md")
  )
  paths <- file.path(p150_root(), files)
  names(paths) <- names(files)
  texts <- lapply(paths[file.exists(paths)], function(p) readLines(p, warn = FALSE))
  desc <- file.path(p150_root(), "DESCRIPTION")
  if (file.exists(desc)) {
    sysreq <- read.dcf(desc, fields = "SystemRequirements")[[1]]
    texts <- c(list(DESCRIPTION = strsplit(gsub("\\s+", " ", sysreq), ",\\s*")[[1]]), texts)
  }
  texts
}

p150_floor <- "compute capability\\s*(of\\s*)?(>=\\s*)?7\\.5"

test_that("every document states compute capability 7.5 as the minimum GPU", {
  docs <- p150_docs()
  skip_if(length(docs) == 0L, "no documents in this test context")
  for (nm in names(docs)) {
    text <- paste(docs[[nm]], collapse = "\n")
    expect_match(text, p150_floor, ignore.case = TRUE, info = nm)
  }
})

test_that("no document states a lower GPU floor", {
  docs <- p150_docs()
  skip_if(length(docs) == 0L, "no documents in this test context")
  for (nm in names(docs)) {
    lines <- docs[[nm]]
    low <- grep("compute capability[^\n.]{0,20}\\b(6\\.[0-9]|7\\.0)\\b", lines,
                ignore.case = TRUE, value = TRUE)
    expect_length(low, 0L)
    newer <- grep("Volta\\)? or (newer|higher|later)", lines, ignore.case = TRUE, value = TRUE)
    expect_length(newer, 0L)
    # Volta only to say it isn't supported or needs a CUDA 12 source build
    volta <- grep("Volta", lines, ignore.case = TRUE, value = TRUE)
    ok <- grepl("CUDA 12|not supported|unsupported|isn't supported|no longer", volta,
                ignore.case = TRUE)
    expect_true(all(ok), info = paste(nm, paste(volta[!ok], collapse = "\n"), sep = ": "))
  }
})

test_that("DESCRIPTION and the README requirements table give 7.5", {
  desc <- file.path(p150_root(), "DESCRIPTION")
  skip_if_not(file.exists(desc), "DESCRIPTION not available")
  sysreq <- gsub("\\s+", " ", read.dcf(desc, fields = "SystemRequirements")[[1]])
  expect_match(sysreq, "Compute Capability >= 7\\.5", ignore.case = TRUE)

  readme <- file.path(p150_root(), "README.md")
  skip_if_not(file.exists(readme), "README.md not available")
  row <- grep("^\\|\\s*NVIDIA GPU\\s*\\|", readLines(readme, warn = FALSE), value = TRUE)
  expect_length(row, 1L)
  expect_match(row, "7\\.5")
})
