# Notify about a CPU-eval fallback, gated by `options(cuplyr.fallback = )`

The one channel for "this ran on the CPU". It always signals a
`cuplyr_cpu_fallback` condition (fields `verb`, `expr`, `reason`,
`rows`, `cols`, `bytes`) before any CPU work;
`options(cuplyr.fallback = )` then decides whether that also warns or
errors (default silent). See
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md)
for the options, classes and message. A lazy plan that is only computed
early on the GPU goes through
[`cuplyr_plan_break_notify()`](https://bbtheo.github.io/cuplyr/reference/cuplyr_plan_break_notify.md)
instead.

## Usage

``` r
cuplyr_fallback_notify(
  verb,
  expr = NA_character_,
  reason = NULL,
  rows = 0,
  cols = 0,
  bytes = 0,
  nothing_collected = FALSE
)
```

## Arguments

- verb:

  Character, the verb name (e.g. `"filter"`, `"summarise"`)

- expr:

  Character, the offending expression as text, or `NA` when the call as
  a whole goes

- reason:

  Character, why this runs on the CPU

- rows, cols, bytes:

  What is collected to the host (see `fallback_size()`)

- nothing_collected:

  `TRUE` when the CPU only evaluates an expression that reads no column

## Value

`invisible(NULL)`
