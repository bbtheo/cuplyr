# Parse `nchar(x, type = "chars", allowNA = FALSE, keepNA = NA)`

Only the default `type = "chars"` shape is implemented – `allowNA=`/
`keepNA=` present at all, or an explicit `type=` other than `"chars"`,
falls back: cuDF's `count_characters()` (`strings/attributes.hpp`)
counts Unicode code points, matching R's own `nchar(type = "chars")`
(the default) exactly, but has no byte-count (`type = "bytes"`) or
display-width (`type = "width"`) equivalent. `x` must be a genuine
STRING-typed expression.

## Usage

``` r
ir_parse_nchar(expr, env, schema)
```
