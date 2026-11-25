# #135: the conda recipe (r-cuplyr) and its documentation.
#
# Contract (static checks, so the recipe can't drift from the package):
# - One recipe, conda/recipe.yaml (rattler-build, schema v1): maintainers
#   build and upload with rattler-build only, so there is no conda-build
#   meta.yaml to keep in step. It is named r-cuplyr and carries the version in
#   DESCRIPTION.
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
# - The recipe builds the local checkout with conda/build.sh, which uses no
#   network.
# - The package test loads cuplyr and checks that libcudf and librmm resolve,
#   with no GPU call (build machines have no GPU).
# - conda/README.md gives the maintainer procedure (install rattler-build,
#   build, test, upload to the `bbtheo` anaconda.org channel); README.md shows
#   the install command from that channel; R CMD build leaves conda/ out.
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

# `${{ x }}` (rattler-build's Jinja) becomes `<x>`.
p135_norm <- function(x) {
  gsub("\\$?\\{\\{\\s*(.*?)\\s*\\}\\}", "<\\1>", x, perl = TRUE)
}

p135_recipe_v1 <- function() {
  yaml::read_yaml(p135_file("conda", "recipe.yaml"))
}

p135_recipes <- function() {
  list(recipe.yaml = p135_recipe_v1())
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

test_that("the recipe is r-cuplyr at the version in DESCRIPTION", {
  p135_skip_if_no_conda()
  version <- read.dcf(p135_file("DESCRIPTION"), fields = "Version")[[1]]

  v1 <- p135_recipe_v1()
  expect_identical(v1$package$name, "r-cuplyr")
  expect_identical(as.character(v1$context$version), version)
  expect_true(
    p135_norm(as.character(v1$package$version)) %in% c("<version>", version),
    info = paste("recipe.yaml package.version:", v1$package$version)
  )
})

test_that("rattler-build is the one build tool: there is no conda-build meta.yaml", {
  p135_skip_if_no_conda()
  expect_false(file.exists(p135_file("conda", "meta.yaml")))
})

test_that("the recipe no longer says it is not a distribution channel", {
  p135_skip_if_no_conda()
  text <- paste(readLines(p135_file("conda", "recipe.yaml"), warn = FALSE), collapse = "\n")
  expect_no_match(text, "not the primary end-user distribution channel", fixed = TRUE)
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
    # no floor at the build's patch release: an R 4.5.0 env must not need an
    # R upgrade for a package built on 4.5.3 (r-base's own run_export pins
    # the minor). A pin_compatible() needs lower_bound = "x.x".
    # (p135_reqs() splits a requirement at its first space, so a Jinja pin
    # is read back whole from name and spec)
    whole <- paste(names(run), run)
    pins <- grep("pin_compatible\\(\\s*[\"']r-base[\"']", whole, value = TRUE)
    for (pin in pins) {
      expect_match(pin, "lower_bound\\s*=\\s*[\"']x\\.x[\"']", info = pin)
    }
    for (spec in run[names(run) == "r-base"]) {
      expect_identical(unname(spec), "", info = paste(nm, "run r-base spec"))
    }
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
    # cuda_runtime.h includes crt/ headers: cuda-nvcc brings them, with
    # cuda-nvcc-tools they need cuda-crt-dev for the target platform
    crt <- grepl("^cuda-crt-dev_(<target_platform>|linux-64)$", names(host))
    expect_true("cuda-nvcc" %in% names(host) || any(crt), info = paste(nm, "crt headers"))
    # none of them is a run requirement: the user's driver is their own
    run <- p135_reqs(recipe, "run")
    expect_false(any(c("cuda-driver-dev", "cuda-nvcc", "cuda-nvcc-tools") %in% names(run)), info = nm)
    expect_false(any(grepl("^cuda-crt-dev", names(run))), info = nm)
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

test_that("the recipe builds the local checkout with conda/build.sh, on Linux only", {
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
  section <- p135_recipe_v1()$tests
  tests <- list(recipe.yaml = p135_test_text(section))
  for (nm in names(tests)) {
    text <- tests[[nm]]
    expect_match(text, "library\\(\\s*[\"']?cuplyr", info = nm)
    # the test reads none of the builder's R startup files either (the
    # command itself, not a comment in the script)
    expect_true(any(grepl("Rscript\\s+--vanilla", p135_strings(section))), info = nm)
    # and checks it tests the cuplyr installed in the environment, not one
    # from a library the caller's R_LIBS points at
    expect_match(text, "find.package(", fixed = TRUE, info = nm)
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

  expect_match(text, "pixi global install rattler-build", fixed = TRUE)
  expect_match(text, "rattler-build build", fixed = TRUE)
  expect_match(text, channel_args, fixed = TRUE)
  # upload, by a maintainer, to the project's anaconda.org channel
  expect_match(text, "rattler-build upload anaconda", fixed = TRUE)
  expect_match(text, "--owner bbtheo", fixed = TRUE)
  expect_match(text, "ANACONDA_API_KEY", fixed = TRUE)
  # a one-time login that keeps the token out of the shell history
  # (stored under api.anaconda.org, the host `rattler-build upload anaconda`
  # looks the token up under)
  expect_match(text, "read -rs T && rattler-build auth login api.anaconda.org --conda-token \"$T\"; unset T",
               fixed = TRUE)
  expect_match(text, "rattler-build auth status", fixed = TRUE)  # shows the stored host
  expect_no_match(text, "auth login anaconda\\.org")
  expect_match(text, "conda-forge can follow|conda-forge[^\n]{0,80}(later|follow)", ignore.case = TRUE)
  expect_match(text, "variants.yaml", fixed = TRUE)
  expect_match(text, "create")  # a fresh environment
  expect_match(text, "r-cuplyr", fixed = TRUE)
  expect_match(text, "library(cuplyr)", fixed = TRUE)
  expect_match(text, "verify_installation()", fixed = TRUE)
  expect_match(text, "tbl_gpu(", fixed = TRUE)
  expect_match(text, "stub", ignore.case = TRUE)
})

test_that("README.md shows the conda install from the bbtheo channel", {
  p135_skip_if_no_conda()
  lines <- readLines(p135_file("README.md"), warn = FALSE)
  start <- grep("^## Installation", lines)
  expect_length(start, 1L)
  rest <- lines[seq(start + 1L, length(lines))]
  end <- c(grep("^## ", rest), length(rest) + 1L)[[1]]
  section <- paste(rest[seq_len(end - 1L)], collapse = "\n")

  expect_match(section, "conda install -c bbtheo -c rapidsai -c conda-forge -c nvidia r-cuplyr",
               fixed = TRUE)
  expect_match(section, "pixi", fixed = TRUE)
  # what the conda package needs: the CUDA 13 libcudf it installs has no code
  # for compute capability 7.0 (Volta), so its floor is 7.5 (Turing)
  conda_part <- sub("To build from source.*$", "", section)
  expect_match(conda_part, "compute capability 7\\.5", info = conda_part)
  expect_no_match(conda_part, "7\\.0", info = conda_part)
  expect_no_match(section, "TBD|to be decided", ignore.case = TRUE)
})

test_that("R CMD build leaves conda/ out of the source package", {
  p135_skip_if_no_conda()
  patterns <- readLines(p135_file(".Rbuildignore"), warn = FALSE)
  patterns <- patterns[nzchar(patterns)]
  expect_true(any(vapply(patterns, function(p) grepl(p, "conda", perl = TRUE), logical(1))))
})
