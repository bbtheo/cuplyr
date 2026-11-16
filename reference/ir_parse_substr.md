# Parse `substr(x, start, stop)` (Phase 11 L5)

`start`/`stop` must be constant (non-column) whole numbers – resolved
via
[`ir_eval_constant()`](https://bbtheo.github.io/cuplyr/reference/ir_eval_constant.md)
with no data mask, exactly like
[`lag()`](https://dplyr.tidyverse.org/reference/lead-lag.html)'s `n=`.
Converts R's inclusive 1-based `[start, stop]` into cuDF
`slice_strings()`'s (`strings/slice.hpp`) exclusive 0-based
`[start, stop)`, replicating R's own clamping (verified empirically
against real [`substr()`](https://rdrr.io/r/base/substr.html)): a
`start` below 1 clamps UP to 1 *before* the 0-based conversion
(`substr("hello", -2, 3)` is `"hel"`, identical to
`substr("hello", 1, 3)`), and a `stop` below the already-clamped 0-based
start clamps up to it, producing an empty string
(`substr("hello", 5, 2)` is `""`) rather than ever handing
`slice_strings()` a negative scalar. A non-numeric/`NA` `start`/`stop`,
a non-constant `start`/`stop`, or a non-STRING `x`, falls back.

## Usage

``` r
ir_parse_substr(expr, env, schema)
```
