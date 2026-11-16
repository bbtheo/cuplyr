# Parse `ntile(x = row_number(), n)`

Fully desugars at parse time into dplyr's own
[`ntile()`](https://dplyr.tidyverse.org/reference/ntile.html) algorithm
(verified against `dplyr:::ntile`'s actual body, not just its docs;
cross-checked numerically against a battery of cases including ties, a
non-multiple group size, all-`NA`, and a zero-row input) expressed in
plain IR plus two nested window calls – `row_number(x)` (or bare
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html)
when `x` is omitted) and, only when `x` is supplied,
`sum(if_else(is.na(x), 0L, 1L))` (the non-missing count; when `x` is
omitted, dplyr's own `len` is just the group size, i.e. plain
[`n()`](https://dplyr.tidyverse.org/reference/context.html)). Both are
ordinary window-class `ir_call` nodes at this point – their actual
extraction into `..win*..` temp columns happens later, in `R/window.R`'s
[`ir_extract_windows()`](https://bbtheo.github.io/cuplyr/reference/ir_extract_windows.md)/[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md),
not here.

## Usage

``` r
ir_parse_ntile(expr, env, schema)
```

## Details

Whether `x` was actually supplied changes the desugar (dplyr's own
source only re-ranks `x` – `x <- row_number(x)` – when it's NOT the
default), so this dispatches on
`rlang::call_match(..., defaults = FALSE)` (which omits an unsupplied
formal entirely, unlike `defaults = TRUE`) rather than merely on whether
`x`'s resolved value looks like
[`row_number()`](https://dplyr.tidyverse.org/reference/row_number.html).

`n` must be a constant, positive, whole-number scalar (dplyr's own
`check_number_whole()` requirement); `n` missing entirely (dplyr's `n`
has no default) falls back, since there's nothing sensible to build.

The division-then-floor steps in dplyr's own formula (\`(x +
(larger_size

- 1L)) /
  larger_size`, then a final `as.integer(floor(bins))`) are expressed via `%/%`(floor division) directly: every numerator/denominator pair in both branches is non-negative by construction (ranks start at 1, sizes are always >= 1), and floor division equals true-division-then-floor for non-negative operands. The sizes follow dplyr's FLOAT64`len
  /
  n`/`ceiling()`/`floor()`, and the bins are cast to INT32 once at the end (`cast_int32\`),
  matching ntile()'s own always-integer output.
