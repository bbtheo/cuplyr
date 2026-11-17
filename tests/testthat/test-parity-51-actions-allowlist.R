# 51: every action the GitHub workflows reference, nested ones included, must
# be allowed by the repository's Actions policy. GitHub resolves every `uses:`
# of a composite action when it sets up the job, even steps an `if:` would
# skip, so one disallowed nested action rejects the whole workflow.
#
# Static check: no GPU, no network. The policy and the nested actions of the
# known composite actions are written out below. When a workflow starts using
# a new action, classify it here (look at its action.yml: `using: node*` or
# `docker` is a leaf; `using: composite` lists its own `uses:`).

ci51_root <- function() {
  root <- testthat::test_path("..", "..")
  if (!file.exists(file.path(root, "DESCRIPTION")) ||
      !dir.exists(file.path(root, ".github", "workflows"))) {
    testthat::skip("not in a source checkout; .github/ is not shipped")
  }
  normalizePath(root)
}

ci51_workflow_files <- function() {
  dir(file.path(ci51_root(), ".github", "workflows"),
      pattern = "[.]ya?ml$", full.names = TRUE)
}

# The repository's allowed-actions policy: GitHub-owned actions, verified
# Marketplace creators, and these patterns. Verified creators can't be checked
# offline, so here they count as not allowed.
ci51_github_owners <- c("actions", "github")
ci51_allowed_patterns <- c(
  "conda-incubator/setup-miniconda@*",
  "prefix-dev/setup-pixi@*",
  "r-lib/actions/*"
)

# Actions with no nested `uses:` (node or docker actions).
ci51_leaf_actions <- c(
  "actions/checkout",
  "actions/cache",
  "actions/cache/restore",
  "actions/cache/save",
  "actions/upload-artifact",
  "actions/download-artifact",
  "r-lib/actions/setup-r",
  "r-lib/actions/setup-pandoc",
  "prefix-dev/setup-pixi",
  "conda-incubator/setup-miniconda"
)

# Composite actions and every action their steps reference.
ci51_composite_actions <- list(
  "r-lib/actions/setup-r-dependencies" = c(
    "actions/cache",
    "actions/cache/restore",
    "actions/cache/save",
    "r-lib/actions/setup-pandoc",
    "quarto-dev/quarto-actions/setup"
  )
)

# `uses:` values of one workflow file, as written (with the @ref).
ci51_uses <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[!grepl("^\\s*#", lines)]
  re <- "^\\s*(-\\s+)?uses:\\s*['\"]?([^'\"[:space:]#]+)"
  hits <- regmatches(lines, regexec(re, lines))
  vapply(Filter(length, hits), function(m) m[[3]], character(1))
}

ci51_action_name <- function(ref) sub("@.*$", "", ref)

ci51_is_allowed <- function(ref) {
  name <- ci51_action_name(ref)
  if (!grepl("@", ref)) ref <- paste0(ref, "@unpinned")
  owner <- sub("/.*$", "", name)
  if (owner %in% ci51_github_owners) return(TRUE)
  any(vapply(ci51_allowed_patterns, function(p) grepl(utils::glob2rx(p), ref),
             logical(1)))
}

# Every action GitHub resolves for the workflows: the direct ones plus the
# nested ones of known composites.
ci51_resolved_actions <- function() {
  direct <- unique(unlist(lapply(ci51_workflow_files(), ci51_uses)))
  names <- unique(ci51_action_name(direct))
  nested <- unlist(ci51_composite_actions[intersect(names, names(ci51_composite_actions))])
  list(direct = direct, nested = unique(unname(nested)))
}

test_that("the allowlist matcher follows the repository policy", {
  expect_true(ci51_is_allowed("actions/checkout@v4"))
  expect_true(ci51_is_allowed("actions/cache/restore@27d5ce7f107fe9357f9df03efb73ab90386fccae"))
  expect_true(ci51_is_allowed("r-lib/actions/setup-r@v2"))
  expect_true(ci51_is_allowed("r-lib/actions/setup-r-dependencies@v2"))
  expect_true(ci51_is_allowed("prefix-dev/setup-pixi@v0"))
  expect_true(ci51_is_allowed("conda-incubator/setup-miniconda@v3"))
  expect_false(ci51_is_allowed("quarto-dev/quarto-actions/setup@8a96df13519ee81fd526f2dfca5962811136661b"))
  expect_false(ci51_is_allowed("someone/actions-r-lib/setup@v1"))
  expect_false(ci51_is_allowed("docker://alpine:3"))
})

test_that("the workflow scanner finds the actions of the Install Matrix workflow", {
  path <- file.path(ci51_root(), ".github", "workflows", "install-matrix.yml")
  expect_true(file.exists(path))
  uses <- ci51_uses(path)
  expect_true("actions/checkout" %in% ci51_action_name(uses))
  expect_true("r-lib/actions/setup-r" %in% ci51_action_name(uses))
  expect_true("prefix-dev/setup-pixi" %in% ci51_action_name(uses))
  expect_true("conda-incubator/setup-miniconda" %in% ci51_action_name(uses))
})

test_that("every action a workflow uses is classified as leaf or composite", {
  used <- unique(ci51_action_name(ci51_resolved_actions()$direct))
  known <- c(ci51_leaf_actions, names(ci51_composite_actions))
  unknown <- setdiff(used, known)
  expect(
    length(unknown) == 0,
    paste0("unclassified action(s): ", paste(unknown, collapse = ", "),
           ". Add them to ci51_leaf_actions or ci51_composite_actions.")
  )
})

test_that("every action a workflow uses, nested ones included, is allowed", {
  acts <- ci51_resolved_actions()
  refs <- c(acts$direct, acts$nested)
  bad <- refs[!vapply(refs, ci51_is_allowed, logical(1))]
  expect(
    length(bad) == 0,
    paste0("action(s) outside the repository's allowed-actions list: ",
           paste(unique(bad), collapse = ", "))
  )
})

test_that("the pixi job still installs the package's hard dependencies with pak", {
  path <- file.path(ci51_root(), ".github", "workflows", "install-matrix.yml")
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  expect_match(txt, "pak::", fixed = TRUE)
  expect_match(txt, "\"hard\"", fixed = TRUE)
  expect_match(txt, "r-lib.github.io/p/pak", fixed = TRUE)
})
