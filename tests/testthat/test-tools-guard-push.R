# The Bash guard hook lets agents push feature branches and nothing else, keeps
# pull-request merges inside tools/merge-queue and keeps main from being
# checked out or moved by hand. Pure shell: no GPU.

guard_push_run <- function(command, hook = "guard-bash.sh") {
  path <- normalizePath(testthat::test_path("..", "..", ".claude", "hooks", hook), mustWork = FALSE)
  if (!file.exists(path)) testthat::skip(paste(hook, "is not present (build-ignored)"))
  if (!nzchar(Sys.which("jq"))) testthat::skip("jq is not available")
  json <- as.character(jsonlite::toJSON(list(tool_input = list(command = command)), auto_unbox = TRUE))
  out <- suppressWarnings(system2("bash", shQuote(path), input = json, stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (is.null(status)) 0L else as.integer(status)
}

test_that("guard-bash allows a plain push of a feature branch to origin", {
  allowed <- c(
    "git push origin feat/F86",
    "git -C /home/theo/cuplr-wt/F86 push -u origin feat/F86-a2",
    "cd x && git push -q origin feat/142 2>&1",
    "git push origin feat/142 2>&1 | tail -2"
  )
  for (cmd in allowed) expect_identical(guard_push_run(cmd), 0L, info = cmd)
})

test_that("guard-bash blocks every other push", {
  blocked <- c(
    "git push",
    "git push origin main",
    "git push origin feat/x main",
    "git push --force origin feat/x",
    "git push -f origin feat/x",
    "git push origin +feat/x",
    "git push origin feat/x:main",
    "git push origin HEAD:main",
    "git -C /x push origin dev",
    "git push origin --delete feat/x",
    "git push origin feat/a; git push origin main",
    "git push upstream feat/x",
    "git push origin v0.2.0"
  )
  for (cmd in blocked) expect_identical(guard_push_run(cmd), 2L, info = cmd)
})

test_that("guard-bash keeps merges in tools/merge-queue and main off agents' hands", {
  expect_identical(guard_push_run("gh pr merge 12 --squash"), 2L)
  expect_identical(guard_push_run("git switch main"), 2L)
  expect_identical(guard_push_run("git -C /x checkout main"), 2L)
  expect_identical(guard_push_run("git branch -D main"), 2L)
  expect_identical(guard_push_run("git update-ref refs/heads/main abc123"), 2L)

  expect_identical(guard_push_run("git switch --detach main"), 0L)
  expect_identical(guard_push_run("git merge origin/main"), 0L)
  expect_identical(guard_push_run("git log main..HEAD"), 0L)
  expect_identical(guard_push_run("gh pr create --base main --head feat/x --title t --body-file b.md"), 0L)
})

test_that("coders, testers and reviewers can't push or write to GitHub", {
  for (hook in c("coder-guard.sh", "tester-guard.sh", "reviewer-guard.sh")) {
    expect_identical(guard_push_run("git push origin feat/x", hook), 2L, info = hook)
    expect_identical(guard_push_run("gh pr create --title x", hook), 2L, info = hook)
    expect_identical(guard_push_run("gh issue comment 3 --body x", hook), 2L, info = hook)
    expect_identical(guard_push_run("gh pr view 3", hook), 0L, info = hook)
  }
})
