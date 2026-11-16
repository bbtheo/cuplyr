# Notify about a plan break, gated by `options(cuplyr.plan_break = )`

A lazy plan that had to run early, on the GPU. Always signals a
`cuplyr_plan_break` condition (fields `verb`, `expr`, `reason`,
`steps`); `options(cuplyr.plan_break = "warn")` also warns. Never an
error. Call it before the plan is computed: `steps` counts the pending
non-source nodes of `data`. See
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md).

## Usage

``` r
cuplyr_plan_break_notify(
  verb,
  reason,
  expr = NA_character_,
  data,
  own_step = FALSE
)
```

## Arguments

- verb:

  Character, the verb name

- reason:

  Character, why the plan had to run

- expr:

  Character, the expression that forced it, or `NA`

- data:

  The lazy `tbl_gpu` whose plan is about to run

- own_step:

  `TRUE` when `data` already holds the verb's own step (a join that has
  to check its rows at the verb), so the notice says the verb ran with
  its pending steps rather than that it ran them early.

## Value

`invisible(NULL)`
