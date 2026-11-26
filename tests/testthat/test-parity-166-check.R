# #166: the package passes R CMD check (0 errors, 0 warnings) on libcudf 26.10.
#
# The check itself (tools/check) is the acceptance test. These tests reproduce
# the individual check findings with the same tools:: functions R CMD check
# uses, so each one fails fast under tools/test-file instead of after a
# 40-minute check. The ones that read the source tree (R/, man/, src/) skip
# when it isn't there (under R CMD check, which runs the real checks itself);
# the NAMESPACE ones also run against the installed package.

p166_root <- function() {
  normalizePath(testthat::test_path("..", ".."), mustWork = FALSE)
}

p166_has_source_tree <- function() {
  root <- p166_root()
  file.exists(file.path(root, "DESCRIPTION")) &&
    dir.exists(file.path(root, "R")) &&
    dir.exists(file.path(root, "man")) &&
    dir.exists(file.path(root, "src"))
}

p166_skip_if_no_source_tree <- function() {
  skip_if_not(p166_has_source_tree(), "source tree not available in this test context")
}

# The package's NAMESPACE directives, from the source tree when there is one,
# else from the installed package.
p166_ns_info <- function() {
  root <- p166_root()
  if (!file.exists(file.path(root, "NAMESPACE"))) {
    root <- system.file(package = "cuplyr")
  }
  skip_if_not(file.exists(file.path(root, "NAMESPACE")), "NAMESPACE not available")
  parseNamespaceFile(basename(root), dirname(root))
}

# Name -> package for every function the namespace imports (importFrom() and
# whole-package import()).
p166_imported <- function(info) {
  out <- character()
  for (imp in info$imports) {
    if (is.character(imp) && length(imp) == 1L) {
      ex <- getNamespaceExports(imp)
      out[ex] <- imp
    } else {
      pkg <- imp[[1L]]
      nms <- imp[[2L]]
      if (!is.null(names(nms)) && any(nzchar(names(nms)))) nms <- unname(nms)
      out[nms] <- pkg
    }
  }
  out
}

test_that("#166: every S3 method's generic is visible with only the stated imports", {
  # R CMD check loads the namespace with R_DEFAULT_PACKAGES=NULL: then only
  # cuplyr itself, its imports and base are visible when the S3 methods are
  # registered, and `S3method(head, tbl_gpu)` without an import of
  # utils::head fails with "object 'head' not found whilst loading namespace".
  info <- p166_ns_info()
  s3 <- info$S3methods
  imported <- p166_imported(info)
  own <- ls(asNamespace("cuplyr"), all.names = TRUE)
  base <- ls(.BaseNamespaceEnv, all.names = TRUE)
  delayed <- !is.na(s3[, 4L])
  generics <- unique(s3[!delayed, 1L])
  visible <- generics %in% c(own, names(imported), base)
  expect_identical(generics[!visible], character(),
                   label = "S3 generics not visible from the namespace")
})

test_that("#166: utils is a stated dependency whenever the namespace imports from it", {
  info <- p166_ns_info()
  imported <- unique(p166_imported(info))
  root <- p166_root()
  desc_path <- if (file.exists(file.path(root, "DESCRIPTION"))) {
    file.path(root, "DESCRIPTION")
  } else {
    system.file("DESCRIPTION", package = "cuplyr")
  }
  desc <- read.dcf(desc_path, fields = c("Imports", "Depends"))
  stated <- trimws(sub("\\(.*", "", unlist(strsplit(paste(desc[1, ], collapse = ","), ","))))
  stated <- stated[nzchar(stated) & !is.na(stated)]
  expect_true("head" %in% names(p166_imported(info)) &&
                identical(unname(p166_imported(info)["head"]), "utils"),
              label = "NAMESPACE imports utils::head (S3method(head, tbl_gpu))")
  expect_identical(setdiff(imported, c(stated, "base")), character(),
                   label = "packages imported in NAMESPACE but not in DESCRIPTION")
})

test_that("#166: S3 methods have the arguments of their generics (R CMD check rule)", {
  # tools::checkS3methods()'s checkArgs(): the method's arguments before `...`
  # match the generic's by name and position, it has every argument of the
  # generic (or `...`), and its own extra arguments are allowed only when the
  # generic has `...`.
  info <- p166_ns_info()
  s3 <- info$S3methods
  imported <- p166_imported(info)
  ns <- asNamespace("cuplyr")
  ops <- c("+", "-", "*", "/", "^", "%%", "%/%", "&", "|", "!", "==", "!=",
           "<", "<=", ">=", ">")

  generic_fun <- function(g, delayed_pkg) {
    if (!is.na(delayed_pkg)) {
      if (!requireNamespace(delayed_pkg, quietly = TRUE)) return(NULL)
      return(get0(g, envir = asNamespace(delayed_pkg), inherits = FALSE))
    }
    if (exists(g, envir = ns, inherits = FALSE)) return(get(g, envir = ns))
    if (!is.na(imported[g])) return(getExportedValue(imported[[g]], g))
    f <- get0(g, envir = .BaseNamespaceEnv, inherits = FALSE)
    if (is.primitive(f)) {
      f <- get0(g, envir = .GenericArgsEnv, inherits = FALSE) %||%
        get0(g, envir = .ArgsEnv, inherits = FALSE)
    }
    f
  }

  bad <- character()
  for (i in seq_len(nrow(s3))) {
    g <- s3[i, 1L]
    if (g %in% ops) next
    m <- if (is.na(s3[i, 3L])) paste(g, s3[i, 2L], sep = ".") else s3[i, 3L]
    gf <- generic_fun(g, s3[i, 4L])
    mf <- get0(m, envir = ns, inherits = FALSE)
    if (!is.function(gf) || !is.function(mf)) next
    gArgs <- names(formals(gf))
    if (g == "plot") gArgs <- gArgs[-2L]
    ogArgs <- gArgs
    mArgs <- names(formals(mf))
    if (endsWith(m, ".formula")) {
      if (gArgs[1L] != "...") gArgs <- gArgs[-1L]
      mArgs <- mArgs[-1L]
    }
    dotsPos <- which(gArgs == "...")
    ipos <- if (length(dotsPos)) seq_len(dotsPos[1L] - 1L) else seq_along(gArgs)
    dotsPos <- which(mArgs == "...")
    if (length(dotsPos)) ipos <- ipos[seq_len(dotsPos[1L] - 1L)]
    posMatchOK <- identical(gArgs[ipos], mArgs[ipos])
    argMatchOK <- all(gArgs %in% mArgs) || length(dotsPos) > 0L
    margMatchOK <- all(mArgs %in% c("...", gArgs)) || "..." %in% ogArgs
    if (!(posMatchOK && argMatchOK && margMatchOK)) {
      bad <- c(bad, sprintf("%s: generic (%s), method (%s)", m,
                            paste(ogArgs, collapse = ", "),
                            paste(names(formals(mf)), collapse = ", ")))
    }
  }
  expect_identical(bad, character())
})

test_that("#166: rowwise.tbl_gpu() takes `data` like dplyr::rowwise() and still works", {
  skip_if_no_gpu()
  expect_identical(names(formals(cuplyr:::rowwise.tbl_gpu))[1L],
                   names(formals(dplyr::rowwise))[1L])
  df <- tibble::tibble(g = c(1, 1, 2), a = c(1, 2, 3), b = c(4, 5, 6))
  expect_same_as_dplyr(df, function(d) {
    dplyr::summarise(dplyr::rowwise(d, g), t = a + b, .groups = "keep")
  })
  expect_same_as_dplyr(df, function(d) dplyr::mutate(dplyr::rowwise(d), t = max(a, b)))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(dplyr::rowwise(data = d), t = a - b))
  expect_same_as_dplyr_lazy(df, function(d) dplyr::mutate(dplyr::rowwise(d, g), t = min(a, b)))
})

test_that("#166: head() and tail() still dispatch to the tbl_gpu methods", {
  skip_if_no_gpu()
  df <- tibble::tibble(x = 1:10, y = letters[1:10])
  expect_same_as_dplyr(df, function(d) utils::head(d, 3))
  expect_same_as_dplyr(df, function(d) utils::tail(d, 4))
  expect_same_as_dplyr(df, function(d) utils::head(d, -8))
  expect_same_as_dplyr(df[0, ], function(d) utils::head(d, 3))
})

test_that("#166: R code is ASCII outside comments", {
  p166_skip_if_no_source_tree()
  expect_identical(as.character(tools:::.check_package_ASCII_code(p166_root())),
                   character())
})

test_that("#166: the code and the usage sections of the help pages agree", {
  p166_skip_if_no_source_tree()
  res <- tools::codoc(dir = p166_root())
  expect_identical(utils::capture.output(print(res)), character())
})

test_that("#166: help pages link only to topics that exist", {
  p166_skip_if_no_source_tree()
  res <- tools:::.check_Rd_xrefs(dir = p166_root())
  expect_identical(utils::capture.output(print(res)), character())
})

test_that("#166: every package the R code calls with :: is a stated dependency", {
  p166_skip_if_no_source_tree()
  res <- tools:::.check_packages_used(dir = p166_root())
  expect_identical(utils::capture.output(print(res)), character())
})

# ---------------------------------------------------------------------------
# libcudf 26.10 deprecations
# ---------------------------------------------------------------------------
#
# The source outside the two marked compatibility headers uses none of the
# APIs libcudf 26.10 deprecates. Whatever differs between 26.02 (the minimum)
# and 26.10 lives in src/cuplyr_compat.hpp (or the RMM block in
# src/cuplyr_mr.hpp). The real criterion is a build with no
# -Wdeprecated-declarations warnings; this catches a reintroduced call
# without a compile.

p166_src_files <- function() {
  src <- file.path(p166_root(), "src")
  files <- list.files(src, pattern = "\\.(cpp|hpp|h|cu|cuh)$", full.names = TRUE)
  files[!basename(files) %in% c("cuplyr_compat.hpp", "cuplyr_mr.hpp", "RcppExports.cpp")]
}

# The file's code with // and /* */ comments and #include lines removed.
p166_code <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[!grepl("^\\s*#\\s*include", lines)]
  txt <- paste(lines, collapse = "\n")
  txt <- gsub("/\\*.*?\\*/", " ", txt, perl = TRUE)
  gsub("//[^\n]*", "", txt, perl = TRUE)
}

p166_offenders <- function(pattern) {
  files <- p166_src_files()
  hit <- vapply(files, function(f) grepl(pattern, p166_code(f), perl = TRUE), logical(1))
  basename(files[hit])
}

test_that("#166: no deprecated cuda_stream_view outside the compatibility headers", {
  p166_skip_if_no_source_tree()
  expect_identical(p166_offenders("\\bcuda_stream_view\\b"), character())
})

test_that("#166: no deprecated apply_boolean_mask() outside the compatibility headers", {
  p166_skip_if_no_source_tree()
  expect_identical(p166_offenders("(?<![\\w:])apply_boolean_mask\\s*\\(|cudf::apply_boolean_mask\\b"),
                   character())
})

test_that("#166: no deprecated nans_to_nulls() outside the compatibility headers", {
  p166_skip_if_no_source_tree()
  expect_identical(p166_offenders("(?<![\\w:])nans_to_nulls\\s*\\(|cudf::nans_to_nulls\\b"),
                   character())
})

test_that("#166: no deprecated parquet reader columns() outside the compatibility headers", {
  p166_skip_if_no_source_tree()
  expect_identical(p166_offenders("\\.columns\\s*\\("), character())
})

test_that("#166: no scalar slice_strings() overload outside the compatibility headers", {
  p166_skip_if_no_source_tree()
  files <- p166_src_files()
  hit <- vapply(files, function(f) {
    code <- p166_code(f)
    calls <- regmatches(code, gregexpr("slice_strings\\s*\\([^;]*;", code, perl = TRUE))[[1]]
    any(grepl("scalar", calls))
  }, logical(1))
  expect_identical(basename(files[hit]), character())
})

test_that("#166: NaN-to-NA conversion, filters and substr() still match dplyr", {
  skip_if_no_gpu()
  # The call sites the port touches: nans_to_nulls (grouped aggregation,
  # arrange/slice ordering, min/max), the retention mask (filter, joins,
  # slice_sample/slice_*), and the substr() slice.
  df <- tibble::tibble(
    g = c(1L, 1L, 2L, 2L, 3L, 3L, NA),
    x = c(NaN, 1.5, NA, 2, NaN, 0.5, 4),
    s = c("alpha", "be", NA, "", "gamma", "e", "delta")
  )
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, x > 1 | is.na(g)))
  expect_same_as_dplyr(df, function(d) dplyr::filter(d, !is.na(x)))
  expect_same_as_dplyr(df, function(d) {
    dplyr::summarise(dplyr::group_by(d, g), m = mean(x, na.rm = TRUE),
                     mx = max(x, na.rm = TRUE), n = sum(!is.na(x)))
  }, arrange_by = "g")
  expect_same_as_dplyr(df, function(d) dplyr::arrange(d, x, g))
  expect_same_as_dplyr(df, function(d) dplyr::slice_max(d, x, n = 2))
  expect_same_as_dplyr(df, function(d) dplyr::slice_min(d, x, n = 2, with_ties = FALSE))
  expect_same_as_dplyr(df, function(d) dplyr::mutate(d, t = substr(s, 2, 4)))
  expect_same_as_dplyr_lazy(df, function(d) {
    d |> dplyr::filter(!is.na(s)) |> dplyr::mutate(t = substr(s, 1, 2))
  })
  y <- tibble::tibble(g = c(1L, 3L, NA), w = c("a", "c", "n"))
  other <- function(d) if (inherits(d, "tbl_gpu")) tbl_gpu(y, lazy = is_lazy(d)) else y
  expect_same_as_dplyr(df, function(d) dplyr::semi_join(d, other(d), by = "g"))
  expect_same_as_dplyr(df, function(d) dplyr::anti_join(d, other(d), by = "g"))
  expect_same_as_dplyr(df, function(d) dplyr::inner_join(d, other(d), by = "g"), arrange_by = c("g", "x", "s"))
  expect_same_as_dplyr(df[0, ], function(d) dplyr::filter(d, x > 1))
})

# ---------------------------------------------------------------------------
# dplyr's regroup message
# ---------------------------------------------------------------------------
#
# dplyr (summarise.R, summarise_verbose()) prints "summarise() has regrouped
# the output" only when the caller's top environment is the global
# environment (or dplyr.summarise.inform is TRUE). Under R CMD check
# testthat::test_check() evaluates the tests in a child of the package
# namespace, so neither dplyr nor cuplyr prints it there; the tests that
# expect the message must not depend on where they are evaluated.

test_that("#166: the regroup message follows dplyr's caller-environment rule", {
  skip_if_no_gpu()
  df <- tibble::tibble(a = c(1, 1, 2), b = c(1, 2, 2), x = 1:3)
  call <- quote(dplyr::summarise(dplyr::group_by(d, a, b), m = mean(x)))
  says <- function(d, parent) {
    e <- new.env(parent = parent)
    assign("d", d, envir = e)
    msgs <- character()
    withCallingHandlers(eval(call, e), message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    })
    any(grepl("regrouped", msgs))
  }
  withr::local_options(dplyr.summarise.inform = NULL)
  for (parent in list(globalenv(), asNamespace("cuplyr"))) {
    expect_identical(says(tbl_gpu(df), parent), says(df, parent))
  }
  expect_true(says(tbl_gpu(df), globalenv()))
  withr::local_options(dplyr.summarise.inform = TRUE)
  expect_true(says(tbl_gpu(df), asNamespace("cuplyr")))
  expect_true(says(df, asNamespace("cuplyr")))
})
