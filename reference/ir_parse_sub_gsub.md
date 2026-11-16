# Parse `sub(pattern, replacement, x, ignore.case=, perl=, fixed=, useBytes=)` / `gsub(...)` (Phase 11 L5)

Same restrictions as
[`ir_parse_grepl()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_grepl.md):
`pattern`/`replacement` must be constant strings,
`ignore.case=`/`perl=`/`useBytes=` present at all falls back,
`fixed=TRUE` routes to literal
[`replace()`](https://rdrr.io/r/base/replace.html)
(`strings/replace.hpp`) instead of `replace_re()`
(`strings/replace_re.hpp`). [`sub()`](https://rdrr.io/r/base/grep.html)
replaces only the first match per string (`max_replace_count = 1`);
[`gsub()`](https://rdrr.io/r/base/grep.html) replaces every match
(`std::nullopt`) – both share one C++ handler keyed off the op name
(`"sub"`/`"gsub"`, both registered separately below purely so
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
can look them up by their own name).

## Usage

``` r
ir_parse_sub_gsub(expr, env, schema, fn_name)
```
