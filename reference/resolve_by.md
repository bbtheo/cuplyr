# Resolve a verb's `.by=`/`by=` quosure into group column names

Implements `scratchpad/phase5_window_design.md` section 6: the shared
resolution logic for on-the-fly grouping across
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
(and, from task W7 onward, the
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) family).
`.by=`/`by=` is mutually exclusive with an already-grouped `.data` –
verified empirically against dplyr 1.2.1, both
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) raise
the identical message below verbatim (including the backticks) when
`.data` is grouped and `.by=` is also supplied.

## Usage

``` r
resolve_by(
  .data,
  by_quo,
  arg_name = ".by",
  call = rlang::caller_env(),
  verb = NULL
)
```

## Arguments

- .data:

  A `tbl_gpu` object

- by_quo:

  A quosure for the verb's `.by=`/`by=` argument, as returned by
  `rlang::enquo(.by)` – NOT yet evaluated

- arg_name:

  Character scalar, the argument's name as written in the verb's own
  signature (`".by"` for
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)/[`slice()`](https://dplyr.tidyverse.org/reference/slice.html),
  `"by"` for the `slice_*()` family) – interpolated into the
  grouped-`.data` error message so it names the argument the caller
  actually used

- call:

  Environment of the user-facing verb, the call the error reports

- verb:

  The verb's name for the CPU-fallback notice, when it is not the name
  of `call`'s frame

## Value

A character vector of column names:
[`character()`](https://rdrr.io/r/base/character.html) when `by_quo` is
NULL (absent or explicit `.by = NULL`, matching dplyr: both behave
identically, see the roxygen on the calling verb), otherwise the
tidyselect-resolved column names in selection order (via
[`tidyselect::eval_select()`](https://tidyselect.r-lib.org/reference/eval_select.html)'s
own [`names()`](https://rdrr.io/r/base/names.html), so `c(g, h)`,
`starts_with("g")`, etc. all work exactly as they do for real dplyr's
own `.by=`)
