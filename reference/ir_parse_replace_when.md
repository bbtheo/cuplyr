# Parse `replace_when(x, cond1 ~ val1, ..., condN ~ valN)`

Boolean-condition clauses – exactly
[`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)'s
own clause shape, see
[`ir_parse_case_when()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_case_when.md)
– but TYPE STABLE to `x`, with `x` itself as the implicit,
non-overridable final "default" (unmatched rows keep their original `x`
value; there is no `.default=`/`default=` argument at all). Lowered via
the same dedicated "replace_when" op as
[`ir_parse_replace_values()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_replace_values.md).

## Usage

``` r
ir_parse_replace_when(expr, env, schema)
```

## Details

No named arguments are supported (the real
[`replace_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)
signature is just `replace_when(x, ...)` anyway) – any named dot present
makes this return `NULL` (fall back).
