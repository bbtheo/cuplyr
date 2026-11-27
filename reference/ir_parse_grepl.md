# Parse `grepl(pattern, x, ignore.case = FALSE, perl = FALSE, fixed = FALSE, useBytes = FALSE)`

`pattern` must be a constant (non-column) string – resolved via
[`ir_eval_constant()`](https://bbtheo.github.io/cuplyr/reference/ir_eval_constant.md).
`ignore.case=`/`perl=`/`useBytes=` present at all falls back (cuDF's
`regex_flags` – `strings/regex/flags.hpp` – has no case-insensitive flag
to map `ignore.case=TRUE` onto, and PCRE (`perl=TRUE`) is a materially
different regex dialect from cuDF's own – only simple ASCII patterns are
expected to match R's regex results). `fixed=TRUE` routes to a literal
substring search
([`contains()`](https://tidyselect.r-lib.org/reference/starts_with.html),
`strings/find.hpp`) instead of regex (`contains_re()`,
`strings/contains.hpp`). The lowered node's 3rd argument is always a
BOOL8 literal carrying `fixed`. Result is always non-NA: R's
[`grepl()`](https://rdrr.io/r/base/grep.html) treats a `NA` element of
`x` as `FALSE` (verified empirically), enforced by the C++ handler via
`replace_nulls()` after the (null-propagating) cuDF call.

## Usage

``` r
ir_parse_grepl(expr, env, schema)
```
