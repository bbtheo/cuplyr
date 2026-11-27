# CPU fallbacks and plan breaks: what cuplyr tells you, and how to hear it

cuplyr runs a dplyr verb on the GPU whenever it can. Two things can
happen that you may want to know about. Both are silent by default, and
both are always *signalled* as conditions, so a calling handler sees
them whatever the options say.

## CPU fallbacks

A verb (or part of one) that has no GPU implementation runs in dplyr on
the host: cuplyr collects the table, runs the real dplyr verb and
uploads the result. This is correct but slow for a big table. The
condition has class `cuplyr_cpu_fallback` and the fields

- `verb`: the verb, as you wrote it (`"mutate"`);

- `expr`: the expression that forced the fallback (`"qnorm(x / 10)"`),
  or `NA` when the call as a whole goes;

- `reason`: why (`` "`qnorm()` has no GPU implementation." ``, a rowwise
  table, a function of yours that shadows a native one, an unsupported
  type, ...);

- `rows`, `cols`, `bytes`: what is collected to the host (the table's
  dimensions and an estimate of its size; a two-table verb adds both
  tables).

`options(cuplyr.fallback = )` decides what happens next: `"silent"` (the
default), `"warn"` (a warning of class `cuplyr_warning_cpu_fallback`) or
`"error"` (an error of class `cuplyr_error_fallback`, for a script that
must stay on the GPU). `"warning"` is accepted as an alias of `"warn"`.
The message reads

    cuplyr: mutate() fell back to CPU evaluation for: qnorm(x / 10)
    i Reason: `qnorm()` has no GPU implementation.
    i Collected 1,000 rows x 3 cols (about 24.4 kB) to run on the CPU.

## Plan breaks

A lazy table (the default, `tbl_gpu(df)`) collects its verbs into a plan
and runs it when you
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) or
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html). A few
steps need the result of the pending plan before it is asked for (a
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) of
an integer `sum(i + j)` or
[`max()`](https://rdrr.io/r/base/Extremes.html), a verb that leaves no
column). cuplyr then computes the plan early. This stays on the GPU and
nothing is collected, but the plan is not fused across that step. The
condition has class `cuplyr_plan_break` and the fields `verb`, `expr`,
`reason` and `steps` (the number of pending steps that ran early).

`options(cuplyr.plan_break = )` is `"silent"` (the default) or `"warn"`
(a warning of class `cuplyr_warning_plan_break`); `"warning"` is an
alias of `"warn"`. A plan break is never an error: nothing left the GPU.

## Invalid option values

Any other value of either option (including `"error"` for
`cuplyr.plan_break`) warns once per session and value, with class
`cuplyr_warning_invalid_option` (fields `option` and `value`), and the
default (`"silent"`) is used. All cuplyr options are listed on
[cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md).

## See also

[cuplyr-options](https://bbtheo.github.io/cuplyr/reference/cuplyr-options.md)
for every cuplyr option, and
[fallback-verbs](https://bbtheo.github.io/cuplyr/reference/fallback-verbs.md)
for the verbs that have no GPU implementation.

## Examples

``` r
if (FALSE) { # \dontrun{
# Count the fallbacks of a pipeline without changing its behaviour
n <- 0L
res <- withCallingHandlers(
  my_pipeline(),
  cuplyr_cpu_fallback = function(cnd) {
    message("fallback in ", cnd$verb, ": ", cnd$reason)
    n <<- n + 1L
  }
)

options(cuplyr.fallback = "warn", cuplyr.plan_break = "warn")
} # }
```
