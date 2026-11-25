# #135: the conda recipe (r-cuplyr) and its documentation.
#
# Contract (static checks, so the recipe can't drift from the package):
# - Two recipes describe the same package: conda/recipe.yaml (rattler-build,
#   schema v1, the preferred tool) and conda/meta.yaml (conda-build). Both are
#   named r-cuplyr and carry the version in DESCRIPTION.
# - Every non-base package in DESCRIPTION Imports is an `r-` requirement in
#   host and run with DESCRIPTION's minimum; LinkingTo packages are in host;
#   Suggests are not requirements. r-base follows Depends: R.
# - A binary is pinned by libcudf's run_exports to the RAPIDS series it was
#   built against (RAPIDS has no ABI across series), so libcudf, librmm and
#   libkvikio are pinned to the `rapids_version` variant in host and run, and
#   conda/variants.yaml lists the series to build: the tested series from
#   configure (CUDF_INSTALL_VERSION) and nothing older than configure's
#   minimum (CUDF_MIN_VERSION). CUDA follows the `cuda_version` variant, which
#   includes the CUDA major pixi's default environment uses.
# - Both recipes build from the local checkout with the shared conda/build.sh,
#   which uses no network.
# - The package test loads cuplyr and checks that libcudf and librmm resolve,
#   with no GPU call (build machines have no GPU).
# - conda/README.md gives the maintainer procedure; README.md shows the conda
#   route with a not-yet-published note; R CMD build leaves conda/ out.
#
# None of these tests need a GPU. test-parity-135-conda-build.R runs
# configure and conda/build.sh against fake prefixes.

p135_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p135_file <- function(...) file.path(p135_root(), ...)

p135_skip_if_no_conda <- function() {
  skip_if_not(
    dir.exists(p135_file("conda")) && file.exists(p135_file("DESCRIPTION")),
    message = "conda/ not available in this test context"
  )
  skip_if_not_installed("yaml")
}

# `${{ x }}` (rattler-build) and `{{ x }}` (conda-build) both become `<x>`.
p135_norm <- function(x) {
  gsub("\\$?\\{\\{\\s*(.*?)\\s*\\}\\}", "<\\1>", x, perl = TRUE)
}

p135_recipe_v1 <- function() {
  yaml::read_yaml(p135_file("conda", "recipe.yaml"))
}

# meta.yaml is Jinja over YAML: drop `{% ... %}` lines, turn `{{ x }}` into
# `<x>` and parse the rest (selectors are YAML comments).
p135_recipe_meta <- function() {
  lines <- readLines(p135_file("conda", "meta.yaml"), warn = FALSE)
  lines <- lines[!grepl("^\\s*\\{%.*%\\}\\s*$", lines)]
  yaml::yaml.load(paste(p135_norm(lines), collapse = "\n"))
}

p135_recipes <- function() {
  list(recipe.yaml = p135_recipe_v1(), meta.yaml = p135_recipe_meta())
}

# Requirement strings of one section, `if:`/`then:` blocks flattened.
p135_flatten <- function(x) {
  out <- character()
  for (item in x) {
    if (is.character(item)) {
      out <- c(out, item)
    } else if (is.list(item)) {
      out <- c(out, p135_flatten(item[["then"]]), p135_flatten(item[["else"]]))
    }
  }
  out
}

# name -> spec (spaces removed; "" when unversioned)
p135_reqs <- function(recipe, section) {
  items <- trimws(p135_norm(p135_flatten(recipe$requirements[[section]])))
  name <- sub("\\s.*$", "", items)
  spec <- gsub("\\s", "", substring(items, nchar(name) + 1L))
  stats::setNames(spec, name)
}

# DESCRIPTION field -> named vector: package -> minimum version (NA if none)
p135_desc_deps <- function(field) {
  x <- read.dcf(p135_file("DESCRIPTION"), fields = field)[[1]]
  if (is.na(x)) {
    return(stats::setNames(character(), character()))
  }
  parts <- trimws(strsplit(gsub("\\s+", " ", x), ",")[[1]])
  parts <- parts[nzchar(parts)]
  name <- trimws(sub("\\(.*$", "", parts))
  ver <- ifelse(
    grepl("\\(", parts),
    sub("^.*\\(\\s*>=\\s*([^) ]+)\\s*\\).*$", "\\1", parts),
    NA_character_
  )
  stats::setNames(ver, name)
}

p135_base_pkgs <- c(
  "base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods",
  "parallel", "splines", "stats", "stats4", "tcltk", "tools", "utils"
)

p135_pad_version <- function(v) {
  p <- as.integer(strsplit(v, ".", fixed = TRUE)[[1]])
  paste(c(p, rep(0L, max(0L, 3L - length(p)))), collapse = ".")
}

# TRUE when `spec` is exactly ">=<ver>" (1.1 and 1.1.0 are the same)
p135_floor_is <- function(spec, ver) {
  m <- regmatches(spec, regexec("^>=([0-9][0-9.]*)$", spec))[[1]]
  length(m) == 2L && identical(p135_pad_version(m[[2]]), p135_pad_version(ver))
}

p135_configure_var <- function(name) {
  lines <- readLines(p135_file("configure"), warn = FALSE)
  hit <- grep(sprintf("^%s=", name), lines, value = TRUE)
  expect_length(hit, 1L)
  gsub("\"|'", "", sub("^[^=]*=", "", hit[[1]]))
}

p135_series_key <- function(v) {
  p <- as.integer(strsplit(v, ".", fixed = TRUE)[[1]])
  p[[1]] * 1000L + p[[2]]
}

# Every string under `x`, recursively.
p135_strings <- function(x) {
  if (is.character(x)) {
    return(x)
  }
  if (is.list(x)) {
    return(unlist(lapply(x, p135_strings), use.names = FALSE))
  }
  character()
}

# The text a recipe's test section runs: its strings plus any conda/ file it
# names (a test script shipped next to the recipe).
p135_test_text <- function(section) {
  strings <- p135_strings(section)
  files <- unique(unlist(regmatches(strings, gregexpr("[A-Za-z0-9_.-]+\\.(R|r|sh)\\b", strings))))
  files <- p135_file("conda", files)
  files <- files[file.exists(files)]
  paste(c(strings, unlist(lapply(files, readLines, warn = FALSE))), collapse = "\n")
}

# ---------------------------------------------------------------------------
# Package identity
# ---------------------------------------------------------------------------

test_that("both recipes are r-cuplyr at the version in DESCRIPTION", {
  p135_skip_if_no_conda()
  version <- read.dcf(p135_file("DESCRIPTION"), fields = "Version")[[1]]

  v1 <- p135_recipe_v1()
  expect_identical(v1$package$name, "r-cuplyr")
  expect_identical(as.character(v1$context$version), version)
  expect_true(
    p135_norm(as.character(v1$package$version)) %in% c("<version>", version),
    info = paste("recipe.yaml package.version:", v1$package$version)
  )

  meta_lines <- readLines(p135_file("conda", "meta.yaml"), warn = FALSE)
  set_version <- regmatches(
    meta_lines,
    regexec("^\\s*\\{%-?\\s*set\\s+version\\s*=\\s*[\"']([^\"']+)[\"']\\s*-?%\\}", meta_lines)
  )
  set_version <- unlist(lapply(set_version, `[`, 2L))
  set_version <- set_version[!is.na(set_version)]
  expect_identical(set_version, version, info = "meta.yaml: {% set version = \"...\" %}")

  meta <- p135_recipe_meta()
  expect_identical(meta$package$name, "r-cuplyr")
  expect_identical(as.character(meta$package$version), "<version>")
})

test_that("the recipes no longer say they are not a distribution channel", {
  p135_skip_if_no_conda()
  for (f in c("recipe.yaml", "meta.yaml")) {
    text <- paste(readLines(p135_file("conda", f), warn = FALSE), collapse = "\n")
    expect_no_match(text, "not the primary end-user distribution channel", fixed = TRUE)
  }
})

# ---------------------------------------------------------------------------
# R dependencies
# ---------------------------------------------------------------------------

test_that("every Imports package is an r- requirement in host and run with DESCRIPTION's minimum", {
  p135_skip_if_no_conda()
  imports <- p135_desc_deps("Imports")
  imports <- imports[!names(imports) %in% p135_base_pkgs]
  linking <- p135_desc_deps("LinkingTo")
  expect_true(all(c("lifecycle", "pillar") %in% names(imports)))

  for (nm in names(p135_recipes())) {
    recipe <- p135_recipes()[[nm]]
    for (section in c("host", "run")) {
      reqs <- p135_reqs(recipe, section)
      r_reqs <- reqs[startsWith(names(reqs), "r-") & names(reqs) != "r-base"]
      wanted <- if (section == "host") c(imports, linking[setdiff(names(linking), names(imports))]) else imports
      expect_setequal(names(r_reqs), paste0("r-", tolower(names(wanted))))

      for (pkg in names(wanted)) {
        spec <- r_reqs[[paste0("r-", tolower(pkg))]] %||% NA_character_
        if (is.na(wanted[[pkg]])) {
          expect_false(is.na(spec), info = paste(nm, section, pkg))
        } else {
          expect_true(
            !is.na(spec) && p135_floor_is(spec, wanted[[pkg]]),
            info = sprintf("%s %s: r-%s is '%s', DESCRIPTION wants >=%s",
                           nm, section, tolower(pkg), spec, wanted[[pkg]])
          )
        }
      }
    }
  }
})

test_that("r-base follows DESCRIPTION's Depends: R", {
  p135_skip_if_no_conda()
  r_min <- p135_desc_deps("Depends")[["R"]]

  for (nm in names(p135_recipes())) {
    recipe <- p135_recipes()[[nm]]
    host <- p135_reqs(recipe, "host")
    run <- p135_reqs(recipe, "run")
    expect_true("r-base" %in% names(host), info = nm)
    expect_true(p135_floor_is(host[["r-base"]] %||% "", r_min),
                info = sprintf("%s host r-base: '%s'", nm, host[["r-base"]] %||% ""))
    # run: r-base itself or a pin_compatible() on it
    expect_true(
      "r-base" %in% names(run) || any(grepl("pin_compatible\\(\\s*[\"']r-base[\"']", names(run))),
      info = nm
    )
  }
})

# ---------------------------------------------------------------------------
# RAPIDS and CUDA
# ---------------------------------------------------------------------------

test_that("libcudf, librmm and libkvikio follow the rapids_version variant in host and run", {
  p135_skip_if_no_conda()
  series_spec <- "^(=<rapids_version>|<rapids_version>\\.\\*)$"

  for (nm in names(p135_recipes())) {
    recipe <- p135_recipes()[[nm]]
    for (section in c("host", "run")) {
      reqs <- p135_reqs(recipe, section)
      for (pkg in c("libcudf", "librmm", "libkvikio")) {
        expect_true(pkg %in% names(reqs), info = paste(nm, section, pkg))
        expect_match(reqs[[pkg]] %||% "", series_spec, info = paste(nm, section, pkg))
      }
    }
    host <- p135_reqs(recipe, "host")
    expect_match(host[["cuda-version"]] %||% "", "^(=<cuda_version>|<cuda_version>\\.\\*)$",
                 info = paste(nm, "host cuda-version"))
    # the stub libcuda that is linked, the CUDA headers and libcudart, and the
    # nvcc configure reads the CUDA version from
    expect_true("cuda-driver-dev" %in% names(host), info = nm)
    expect_true("cuda-cudart-dev" %in% names(host), info = nm)
    expect_true(any(c("cuda-nvcc", "cuda-nvcc-tools") %in% names(host)), info = nm)
    # none of them is a run requirement: the user's driver is their own
    run <- p135_reqs(recipe, "run")
    expect_false(any(c("cuda-driver-dev", "cuda-nvcc", "cuda-nvcc-tools") %in% names(run)), info = nm)
  }
})

test_that("variants.yaml pins the C standard library to libcudf's glibc floor", {
  p135_skip_if_no_conda()
  variants <- yaml::read_yaml(p135_file("conda", "variants.yaml"))

  # stdlib("c") has no default in rattler-build, and an unpinned sysroot would
  # follow the build machine's glibc and raise the package's __glibc floor
  # above libcudf's (RAPIDS 26.x builds with sysroot 2.28)
  expect_identical(variants$c_stdlib, "sysroot")
  expect_type(variants$c_stdlib_version, "character")
  expect_true(length(variants$c_stdlib_version) >= 1L)
  for (v in variants$c_stdlib_version) {
    expect_true(package_version(v) <= package_version("2.28"), info = v)
  }
  for (nm in names(p135_recipes())) {
    build <- p135_reqs(p135_recipes()[[nm]], "build")
    expect_true("<stdlib('c')>" %in% gsub("\"", "'", names(build)), info = nm)
  }
})

test_that("variants.yaml builds the tested RAPIDS series and nothing older than configure's minimum", {
  p135_skip_if_no_conda()
  variants <- yaml::read_yaml(p135_file("conda", "variants.yaml"))
  cudf_min <- p135_configure_var("CUDF_MIN_VERSION")
  cudf_tested <- p135_configure_var("CUDF_INSTALL_VERSION")

  rapids <- variants$rapids_version
  # quoted in the file: unquoted 26.10 is the number 26.1
  expect_type(rapids, "character")
  expect_true(length(rapids) >= 1L)
  expect_true(all(grepl("^[0-9]+\\.[0-9]{2}$", rapids)), info = paste(rapids, collapse = ", "))
  expect_true(cudf_tested %in% rapids, info = paste(rapids, collapse = ", "))
  for (v in rapids) {
    expect_gte(p135_series_key(v), p135_series_key(cudf_min))
  }

  cuda <- variants$cuda_version
  expect_type(cuda, "character")
  pixi <- readLines(p135_file("pixi.toml"), warn = FALSE)
  pixi_cuda <- gsub("\\s|\"|'", "", sub("^[^=]*=", "", grep("^\\s*cuda-version\\s*=", pixi, value = TRUE)))
  pixi_major <- unique(sub("\\..*$", "", pixi_cuda))
  expect_true(all(pixi_major %in% cuda), info = paste(cuda, collapse = ", "))
})

# ---------------------------------------------------------------------------
# Source, build script, platform
# ---------------------------------------------------------------------------

test_that("both recipes build the local checkout with conda/build.sh, on Linux only", {
  p135_skip_if_no_conda()
  expect_true(file.exists(p135_file("conda", "build.sh")))

  v1 <- p135_recipe_v1()
  expect_identical(v1$source$path, "..")
  expect_null(v1$source$url)
  expect_null(v1$source$git)
  script <- v1$build$script
  expect_true(
    is.null(script) || identical(script, "build.sh") || identical(script$file, "build.sh"),
    info = paste(utils::capture.output(str(script)), collapse = "\n")
  )
  expect_match(paste(p135_strings(v1$build$skip), collapse = " "), "not\\s+linux|win|osx")

  meta <- p135_recipe_meta()
  expect_identical(meta$source$path, "..")
  expect_null(meta$source$url)
  expect_null(meta$source$git_url)
  expect_null(meta$build$script)  # conda-build then runs build.sh
  meta_lines <- readLines(p135_file("conda", "meta.yaml"), warn = FALSE)
  expect_true(any(grepl("^\\s*skip:\\s*true\\s*#\\s*\\[\\s*not linux", meta_lines)))
})

test_that("conda/build.sh downloads nothing", {
  p135_skip_if_no_conda()
  lines <- readLines(p135_file("conda", "build.sh"), warn = FALSE)
  code <- sub("(^|\\s)#.*$", "", lines)
  network <- "\\b(curl|wget|git|pip|conda|mamba|micromamba|pixi|apt-get|dnf|install\\.packages|remotes::|devtools::|pak::)\\b"
  expect_false(any(grepl(network, code)), info = paste(grep(network, code, value = TRUE), collapse = "\n"))
})

# ---------------------------------------------------------------------------
# Package test
# ---------------------------------------------------------------------------

test_that("the package test loads cuplyr and checks libcudf and librmm resolve, without the GPU", {
  p135_skip_if_no_conda()
  tests <- list(
    recipe.yaml = p135_test_text(p135_recipe_v1()$tests),
    meta.yaml = p135_test_text(p135_recipe_meta()$test)
  )
  for (nm in names(tests)) {
    text <- tests[[nm]]
    expect_match(text, "library\\(\\s*[\"']?cuplyr", info = nm)
    expect_match(text, "\\bldd\\b", info = nm)
    expect_match(text, "libcudf", info = nm)
    expect_match(text, "librmm", info = nm)
    # build machines have no GPU
    expect_no_match(text, "tbl_gpu\\(|verify_installation\\(|gpu_info\\(|as_tbl_gpu\\(", info = nm)
  }
})

# ---------------------------------------------------------------------------
# Documentation and packaging
# ---------------------------------------------------------------------------

test_that("conda/README.md gives the maintainer procedure", {
  p135_skip_if_no_conda()
  path <- p135_file("conda", "README.md")
  expect_true(file.exists(path))
  text <- paste(readLines(path, warn = FALSE), collapse = "\n")

  pixi <- readLines(p135_file("pixi.toml"), warn = FALSE)
  channels <- regmatches(grep("^channels\\s*=", pixi, value = TRUE),
                         gregexpr("\"[^\"]+\"", grep("^channels\\s*=", pixi, value = TRUE)))[[1]]
  channels <- gsub("\"", "", channels)
  channel_args <- paste(sprintf("-c %s", channels), collapse = " ")

  expect_match(text, "rattler-build build", fixed = TRUE)
  expect_match(text, "conda build|conda-build")
  expect_match(text, channel_args, fixed = TRUE)
  expect_match(text, "variants.yaml", fixed = TRUE)
  expect_match(text, "create")  # a fresh environment
  expect_match(text, "r-cuplyr", fixed = TRUE)
  expect_match(text, "library(cuplyr)", fixed = TRUE)
  expect_match(text, "verify_installation()", fixed = TRUE)
  expect_match(text, "tbl_gpu(", fixed = TRUE)
  expect_match(text, "stub", ignore.case = TRUE)
})

test_that("README.md shows the conda route with a not-yet-published note", {
  p135_skip_if_no_conda()
  lines <- readLines(p135_file("README.md"), warn = FALSE)
  start <- grep("^## Installation", lines)
  expect_length(start, 1L)
  rest <- lines[seq(start + 1L, length(lines))]
  end <- c(grep("^## ", rest), length(rest) + 1L)[[1]]
  section <- paste(rest[seq_len(end - 1L)], collapse = "\n")

  expect_match(section, "r-cuplyr", fixed = TRUE)
  expect_match(
    section,
    "not (yet )?(been )?published|coming soon|not yet (available|on)|channel[^\n]{0,60}(TBD|to be decided|not yet)",
    ignore.case = TRUE
  )
})

test_that("R CMD build leaves conda/ out of the source package", {
  p135_skip_if_no_conda()
  patterns <- readLines(p135_file(".Rbuildignore"), warn = FALSE)
  patterns <- patterns[nzchar(patterns)]
  expect_true(any(vapply(patterns, function(p) grepl(p, "conda", perl = TRUE), logical(1))))
})
