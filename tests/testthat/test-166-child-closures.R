# Issue 166: a function that reaches a gpu_child_r() child through `args` must
# not carry a namespace in its environment chain. test_check() roots the tests'
# environment at asNamespace("cuplyr"), and R writes a namespace as a reference:
# reading the payload in the child then runs loadNamespace("cuplyr") before the
# child function sets its options and loads the package, so .onLoad installs the
# memory resource from the environment variables alone and its warnings fire
# outside the child's handlers. Under tools/test the tests' environment is a
# child of the global environment, so a child-side test can't see it; these tests
# read what the parent sends.

# Does `f`'s environment chain reach a (non-base) namespace before the global
# environment?
c166_has_namespace <- function(f) {
  e <- environment(f)
  while (!is.null(e) && !identical(e, emptyenv())) {
    if (identical(e, globalenv()) || identical(e, baseenv()) || identical(e, .BaseNamespaceEnv)) {
      return(FALSE)
    }
    if (isNamespace(e)) return(TRUE)
    e <- parent.env(e)
  }
  FALSE
}

# Every function in `x`, looking through plain lists.
c166_functions <- function(x) {
  if (is.function(x)) return(list(x))
  if (is.list(x) && !is.object(x)) return(unlist(lapply(x, c166_functions), recursive = FALSE))
  list()
}

# What gpu_child_r() hands to callr::r() for `args`, with the caller's
# environment rooted at the cuplyr namespace.
c166_sent <- function(args_fn) {
  skip_if_not_installed("callr")
  sent <- NULL
  local_mocked_bindings(
    has_gpu = function(...) FALSE,
    .package = "cuplyr"
  )
  local_mocked_bindings(
    r = function(func, args, ...) {
      sent <<- args
      NULL
    },
    .package = "callr"
  )
  env <- new.env(parent = asNamespace("cuplyr"))
  env$offset <- 10
  args <- args_fn(env)
  gpu_child_r(function(x) x, args = args)
  sent
}

test_that("the checker sees a namespace in a closure's environment chain", {
  env <- new.env(parent = asNamespace("cuplyr"))
  inside <- eval(quote(function() 1), env)
  expect_true(c166_has_namespace(inside))
  expect_false(c166_has_namespace(`environment<-`(function() 1, globalenv())))
})

test_that("gpu_child_r() sends no function with a namespace in its environment chain", {
  sent <- c166_sent(function(env) {
    fn <- eval(quote(function(x) x + offset), env)
    list(fn = fn, nested = list(a = 1, f = fn, deeper = list(fn)), plain = 1:3)
  })
  fns <- c166_functions(sent$args)
  expect_gte(length(fns), 3L)
  for (f in fns) expect_false(c166_has_namespace(f))
  expect_identical(sent$args$plain, 1:3)
  expect_identical(sent$args$nested$a, 1)
})

test_that("a deferred function behaves as the original after a round trip", {
  sent <- c166_sent(function(env) {
    list(fn = eval(quote(function(x, ...) x + offset + sum(...)), env), n = 2)
  })
  back <- unserialize(serialize(sent$args, NULL))
  expect_identical(back$fn(1, 2, 3), 16)
  expect_identical(do.call(back$fn, list(1, 2, 3)), 16)
  expect_identical(back$n, 2)
})

test_that("the loader and the readers that travel to every child stay free of namespaces too", {
  sent <- c166_sent(function(env) list(load = cuplyr_child_loader()))
  for (f in c(c166_functions(sent$args), c166_functions(sent$readers))) {
    expect_false(c166_has_namespace(f))
  }
})
