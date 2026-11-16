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
discovering the need to fall back (see
`scratchpad/phase5_window_design.md` section 1.3's "Never half-lower").

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

As of Phase 5 task W4, every window `fn` with a registry `window` entry
is lowerable (`"shift"`, `"rank"`, `"scan"`, and `"agg"` kinds all have
C++ support in `gpu_window()`) – this predicate now only guards against
an `fn` with NO registry entry at all, or a registry entry whose
`window` field is `NULL` (not a window-class op), which should be
unreachable in practice (nothing in `R/window.R`'s decomposition
machinery can produce such a spec) but is checked defensively so a bug
surfaces as a clear internal error instead of silently mis-executing.

The rolling-window task (scratchpad/todo.md) adds a sixth kind,
`"rolling"`, lowerable for every `fn` EXCEPT `"roll_median"`: cudf
25.12's `rolling_window()`/`grouped_rolling_window()` have no MEDIAN
aggregation instantiation at all in this environment (verified via
`nm -DC libcudf.so`, see the file-level comment above
[`ir_parse_roll_call()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_roll_call.md),
R/ir.R) – a genuine cudf capability gap, not a cuplyr bug, so
[`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md)/[`data.table::frollmedian()`](https://rdrr.io/pkg/data.table/man/froll.html)
mutate() calls route through `mutate_window_fallback()` exactly like the
not-yet-implemented-kind case below, needing `R/roll.R`'s real
[`roll_median()`](https://bbtheo.github.io/cuplyr/reference/roll.md) R
function to make that CPU re-run work.
