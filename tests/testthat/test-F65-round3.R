# F65 round 3: memory_resource_install()'s decisions, with the C++ side mocked
# (no hardware here lacks cudaMallocAsync, so the fallback can't run for real).

# Clear the switches; a fixed initial size keeps memory_resource_config() off
# the device.
f65r3_local_switches <- function(..., env = parent.frame()) {
  withr::local_options(cuplyr.memory_resource = NULL, cuplyr.memory_pool = NULL,
                       cuplyr.pool_initial = 0, cuplyr.release_threshold = NULL,
                       .local_envir = env)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = NA, CUPLYR_MEMORY_POOL = NA,
                      CUPLYR_POOL_INITIAL = NA, CUPLYR_RELEASE_THRESHOLD = NA,
                      .local_envir = env)
  if (...length()) withr::local_options(..., .local_envir = env)
}

# A gpu_mr_install() whose async (and optionally pool) creation fails; the
# kinds it was asked for are recorded in `calls`.
f65r3_mock_install <- function(fail = "async", env = parent.frame()) {
  calls <- new.env()
  calls$kinds <- character()
  calls$enabled <- logical()
  local_mocked_bindings(
    gpu_mr_exists = function() FALSE,
    gpu_mr_install = function(enabled, kind, initial, release_threshold) {
      calls$kinds <- c(calls$kinds, kind)
      calls$enabled <- c(calls$enabled, enabled)
      # Like the real one: switched off (and none to reuse) creates nothing.
      if (!enabled) return("cuda")
      if (kind %in% fail) stop("could not create the ", kind, " memory resource: not supported")
      kind
    },
    .env = env
  )
  calls
}

test_that("a default async that can't be created falls back to the pool, with a warning", {
  f65r3_local_switches()
  calls <- f65r3_mock_install()
  expect_warning(kind <- memory_resource_install(), "default \"async\".*using \"pool\" instead")
  expect_identical(kind, "pool")
  expect_identical(calls$kinds, c("async", "pool"))
})

test_that("an unknown kind (which means async) falls back like the default", {
  f65r3_local_switches(cuplyr.memory_resource = "bogus")
  calls <- f65r3_mock_install()
  w <- character()
  kind <- withCallingHandlers(memory_resource_install(), warning = function(cnd) {
    w <<- c(w, conditionMessage(cnd))
    invokeRestart("muffleWarning")
  })
  expect_identical(kind, "pool")
  expect_identical(calls$kinds, c("async", "pool"))
  expect_length(w, 2)
})

test_that("an explicit async that can't be created leaves plain cudaMalloc", {
  f65r3_local_switches(cuplyr.memory_resource = "async")
  calls <- f65r3_mock_install()
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc")
  expect_identical(kind, "cuda")
  expect_identical(calls$kinds, "async")

  withr::local_options(cuplyr.memory_resource = NULL)
  withr::local_envvar(CUPLYR_MEMORY_RESOURCE = " ASYNC ")
  calls <- f65r3_mock_install()
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc")
  expect_identical(kind, "cuda")
  expect_identical(calls$kinds, "async")
})

test_that("a pool that can't be created leaves plain cudaMalloc, explicit or as the fallback", {
  f65r3_local_switches(cuplyr.memory_resource = "pool")
  calls <- f65r3_mock_install(fail = "pool")
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc")
  expect_identical(kind, "cuda")
  expect_identical(calls$kinds, "pool")

  withr::local_options(cuplyr.memory_resource = NULL)
  calls <- f65r3_mock_install(fail = c("async", "pool"))
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc")
  expect_identical(kind, "cuda")
  expect_identical(calls$kinds, c("async", "pool"))
})

test_that("when the default async and then the pool fail, the warning names both errors", {
  f65r3_local_switches()
  calls <- f65r3_mock_install(fail = c("async", "pool"))
  w <- character()
  kind <- withCallingHandlers(memory_resource_install(), warning = function(cnd) {
    w <<- c(w, conditionMessage(cnd))
    invokeRestart("muffleWarning")
  })
  expect_identical(kind, "cuda")
  expect_length(w, 1)
  expect_match(w, "async memory resource: not supported", fixed = TRUE)
  expect_match(w, "pool memory resource: not supported", fixed = TRUE)
  # The wrapper doesn't repeat what the errors already say.
  expect_identical(lengths(regmatches(w, gregexpr("could not create", w))), 2L)
})

test_that("the fallback-to-pool warning carries the async error once", {
  f65r3_local_switches()
  calls <- f65r3_mock_install()
  w <- character()
  kind <- withCallingHandlers(memory_resource_install(), warning = function(cnd) {
    w <<- c(w, conditionMessage(cnd))
    invokeRestart("muffleWarning")
  })
  expect_identical(kind, "pool")
  expect_length(w, 1)
  expect_match(w, "async memory resource: not supported", fixed = TRUE)
  expect_identical(lengths(regmatches(w, gregexpr("could not create", w))), 1L)
})

test_that("an error reading the switches warns and still pins this copy, creating nothing", {
  # gpu_mr_install() pins the library copy; the failure path must still call it,
  # switched off so it creates nothing.
  f65r3_local_switches()
  calls <- f65r3_mock_install()
  local_mocked_bindings(memory_resource_config = function() stop("unreadable switch"))
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc.*unreadable switch")
  expect_identical(kind, "cuda")
  expect_identical(calls$enabled, FALSE)
})

test_that("an option memory_resource_config() can't read gives a warning, not a failed load", {
  f65r3_local_switches(cuplyr.memory_resource = function() "pool")
  expect_error(memory_resource_config())
  calls <- f65r3_mock_install()
  expect_warning(kind <- memory_resource_install(), "plain cudaMalloc")
  expect_identical(kind, "cuda")
  expect_identical(calls$enabled, FALSE)
})

test_that("an error after gpu_mr_install() was called does not call it again", {
  # An explicit async that fails was pinned by that call already.
  f65r3_local_switches(cuplyr.memory_resource = "async")
  calls <- f65r3_mock_install()
  expect_warning(memory_resource_install(), "plain cudaMalloc")
  expect_identical(calls$enabled, TRUE)
})

test_that("with the off switch, nothing is created and no fallback is tried", {
  f65r3_local_switches(cuplyr.memory_pool = FALSE)
  calls <- f65r3_mock_install()
  expect_no_warning(kind <- memory_resource_install())
  expect_identical(kind, "cuda")
  expect_identical(calls$enabled, FALSE)
})

test_that("when the process already has a resource, the switches are not read", {
  f65r3_local_switches(cuplyr.memory_resource = "bogus", cuplyr.release_threshold = "1GB")
  calls <- new.env()
  calls$args <- list()
  local_mocked_bindings(
    gpu_mr_exists = function() TRUE,
    gpu_mr_install = function(enabled, kind, initial, release_threshold) {
      calls$args <- c(calls$args, list(list(enabled = enabled)))
      "pool"
    },
    memory_resource_config = function() stop("the switches must not be read")
  )
  expect_no_warning(kind <- memory_resource_install())
  expect_identical(kind, "pool")
  # Reuse only: enabled = FALSE never creates a resource.
  expect_identical(calls$args, list(list(enabled = FALSE)))
})
