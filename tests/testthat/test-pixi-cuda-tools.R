# Both pixi environments carry the CUDA 13 build tools, so `pixi run install`
# works on a machine with no system CUDA toolkit (the install-check workflow's
# pixi job, a new contributor following the README). Regression test for the
# move to RAPIDS 26.x, which left only the runtime libraries in the
# environments and broke that route.

test_that("each pixi feature with libcudf also lists the CUDA 13 build tools", {
  path <- testthat::test_path("..", "..", "pixi.toml")
  skip_if_not(file.exists(path), "pixi.toml is not in the package source")
  lines <- readLines(path)

  heads <- grep("^\\[", lines)
  sections <- lapply(seq_along(heads), function(i) {
    from <- heads[i]
    to <- if (i < length(heads)) heads[i + 1L] - 1L else length(lines)
    lines[from:to]
  })
  with_cudf <- Filter(function(sec) any(grepl("^\\s*libcudf\\s*=", sec)), sections)
  expect_gt(length(with_cudf), 0L)

  for (sec in with_cudf) {
    for (pkg in c("cuda-nvcc-tools", "cuda-cudart-dev", "cuda-crt-dev_linux-64", "cuda-driver-dev")) {
      hit <- grep(sprintf("^\\s*%s\\s*=", pkg), sec, value = TRUE)
      expect_length(hit, 1L)
      expect_match(hit, "\"13\\.\\*\"", info = paste(sec[1], pkg))
    }
  }
})
