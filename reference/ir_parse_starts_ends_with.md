# Parse `startsWith(x, prefix)` / `endsWith(x, suffix)` (Phase 11 L5)

`prefix`/`suffix` must be a constant (non-column, non-`NA`) string –
matching a per-row column-valued target
([`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html)/[`ends_with()`](https://tidyselect.r-lib.org/reference/starts_with.html)'s
`targets` column overload, `strings/find.hpp`) is not implemented this
wave. NA propagation needs no special handling in the C++ handler:
cuDF's
[`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html)/[`ends_with()`](https://tidyselect.r-lib.org/reference/starts_with.html)
already null-propagate exactly like R's own
[`startsWith()`](https://rdrr.io/r/base/startsWith.html)/[`endsWith()`](https://rdrr.io/r/base/startsWith.html)
(verified empirically: `startsWith(c("abc", NA), "a")` is `TRUE, NA`,
not `TRUE, FALSE`) – unlike
[`grepl()`](https://rdrr.io/r/base/grep.html) above, which needs an
explicit NA-to-FALSE fixup.

## Usage

``` r
ir_parse_starts_ends_with(expr, env, schema, fn_name)
```
