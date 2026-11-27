# Is a window spec lowerable by `src/ops_window.cpp`'s `gpu_window()`?

The non-throwing predicate half of
[`validate_window_spec()`](https://bbtheo.github.io/cuplyr/reference/validate_window_spec.md)
(below), factored out so
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)'s window
wiring (`R/mutate.R`, `mutate_window()`) can check every spec a
decomposition plan produces *before* pushing any AST node – a window
call whose spec isn't lowerable (an unrecognized `fn`, i.e. a genuine
cuplyr bug rather than a user-facing shape) must fall back to
[`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)
cleanly, never surface
[`validate_window_spec()`](https://bbtheo.github.io/cuplyr/reference/validate_window_spec.md)'s
internal-error text to a user, and never partially push GPU ops before
discovering the need to fall back (never half-lower).

## Usage

``` r
window_spec_lowerable(spec)
```

## Arguments

- spec:

  A
  [`window_spec()`](https://bbtheo.github.io/cuplyr/reference/window_spec.md)
  structure (`R/window.R`)

## Value

`TRUE`/`FALSE`

## Details

Every window `fn` whose registry `window` entry has kind `"shift"`,
`"rank"`, `"scan"` or `"agg"` is lowerable (all four have C++ support in
`gpu_window()`), and so is every `"rolling"` `fn` except `"roll_median"`
(below). Otherwise the predicate is `FALSE`: an `fn` with no registry
entry at all, or a registry entry whose `window` field is `NULL` (not a
window-class op), should be unreachable (nothing in `R/window.R`'s
decomposition machinery produces such a spec) but is checked defensively
so a bug surfaces as a clear internal error instead of silently
mis-executing.

`"roll_median"` is not lowerable: libcudf's `rolling_window()`/
`grouped_rolling_window()` have no MEDIAN aggregation (see the
file-level comment above
[`ir_parse_roll_call()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_roll_call.md),
R/ir.R). So
[`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`data.table::frollmedian()`](https://rdrr.io/pkg/data.table/man/froll.html)
mutate() calls go through `mutate_window_fallback()`, the same
whole-call CPU re-run as any other unlowerable spec, which calls
`R/roll.R`'s
[`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md) R
function.
