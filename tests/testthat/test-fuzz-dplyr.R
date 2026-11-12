# F118: the quick, fixed-seed slice of the randomized differential test.
#
# Each seed builds a random table and a random dplyr pipeline (helper-fuzz.R),
# runs it on a tibble and on a tbl_gpu, eager and lazy, and compares values,
# types, row order, grouping, errors and warnings. The seeds are fixed so this
# is deterministic; `tools/fuzz` runs the same cases by the thousand.
#
# A seed that exposes a bug is skipped with the finding it belongs to
# (fuzz_quick_skips(): seed -> finding id), so it can be turned on as each bug
# is fixed. A new bug found by a changed seed list is a finding, not a skip.

skips <- fuzz_quick_skips()

for (seed in fuzz_quick_seeds()) {
  local({
    seed <- seed
    test_that(sprintf("fuzz seed %d matches dplyr", seed), {
      skip_if_no_gpu()
      finding <- unname(skips[as.character(seed)])
      if (!is.na(finding)) skip(sprintf("known finding %s", finding))
      case <- fuzz_case(seed)
      found <- fuzz_check(case)
      info <- if (nrow(found)) {
        paste0(paste(found$mode, found$kind, found$detail, sep = ": ", collapse = "\n"),
               "\n\n", fuzz_code(case))
      } else {
        ""
      }
      expect_equal(nrow(found), 0L, info = info)
    })
  })
}
