# F65 round 2: edges of the new switches that the acceptance file leaves out.

test_that("gpu_gc() warns instead of failing when returning memory raises a CUDA error", {
  skip_if_no_gpu()
  local_mocked_bindings(gpu_mr_release = function() stop("CUDA error (gpu_mr_release): boom"))
  expect_warning(res <- gpu_gc(aggressive = FALSE), "could not return memory")
  expect_true(is.list(res))
  expect_true(all(c("freed_bytes", "released_bytes") %in% names(res)))
})

test_that("a logical or non-numeric cuplyr.release_threshold warns and means none; NA is unset", {
  withr::local_options(cuplyr.memory_resource = NULL, cuplyr.memory_pool = NULL,
                       cuplyr.pool_initial = NULL)
  withr::local_envvar(CUPLYR_RELEASE_THRESHOLD = NA, CUPLYR_MEMORY_RESOURCE = NA,
                      CUPLYR_MEMORY_POOL = NA)
  # NA (of any type) counts as unset, like the other switches: no warning.
  withr::with_options(list(cuplyr.release_threshold = NA_real_), {
    expect_no_warning(cfg <- memory_resource_config())
    expect_true(is.na(cfg$release_threshold))
  })
  withr::with_options(list(cuplyr.release_threshold = TRUE), {
    expect_warning(cfg <- memory_resource_config(), "cuplyr.release_threshold", fixed = TRUE)
    expect_true(is.na(cfg$release_threshold))
  })
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "1GB"), {
    expect_warning(cfg <- memory_resource_config(), "1GB", fixed = TRUE)
    expect_true(is.na(cfg$release_threshold))
  })
  # A zero-length option falls through to the env var.
  withr::with_envvar(c(CUPLYR_RELEASE_THRESHOLD = "1024"),
    withr::with_options(list(cuplyr.release_threshold = numeric(0)),
                        expect_identical(memory_resource_config()$release_threshold, 1024)))
})

test_that("cuplyr.memory_resource = 'cuda' switches off even when cuplyr.memory_pool is TRUE", {
  withr::local_options(cuplyr.memory_resource = " CUDA ", cuplyr.memory_pool = TRUE,
                       cuplyr.pool_initial = NULL, cuplyr.release_threshold = NULL)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = NA, CUPLYR_MEMORY_POOL = NA,
                      CUPLYR_RELEASE_THRESHOLD = NA)
  expect_no_warning(cfg <- memory_resource_config())
  expect_false(cfg$enabled)
})
