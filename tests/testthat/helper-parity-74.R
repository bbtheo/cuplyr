# #74: expressions that used to run on the CPU. Every check runs `pipeline` on
# a tibble (dplyr, the oracle) and on a tbl_gpu, eager and lazy, through
# f124_check() (helper-parity-F124.R): values and column types exactly, group
# vars, dplyr's warnings (message and class), and the lazy schema before
# collect() against the collected types.
#
# p74_native(): no CPU fallback in either mode, and a lazy table neither warns
#   nor breaks its plan at the verb (nothing here needs the data early).
# p74_native_warn(): no CPU fallback; the lazy warning (e.g. min()/max() over
#   an all-NA group) may come at the verb or at collect().
# p74_parity(): dplyr's result; a CPU fallback is allowed.

p74_native <- function(df, pipeline, ...) {
  f124_check(df, pipeline, native = TRUE, lazy_warns_at = "collect", ...)
}

p74_native_warn <- function(df, pipeline, ...) {
  f124_check(df, pipeline, native = TRUE, lazy_warns_at = "any", ...)
}

p74_parity <- function(df, pipeline, ...) {
  f124_check(df, pipeline, native = FALSE, ...)
}

p74_i64 <- function(x) bit64::as.integer64(x)
