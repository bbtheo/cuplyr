# Parse `paste0(..., collapse = NULL)` / `paste(..., sep = " ", collapse = NULL)`

`collapse=` (reduce-to-one-string) is not supported – its presence at
all falls back. `sep=` ([`paste()`](https://rdrr.io/r/base/paste.html)
only; [`paste0()`](https://rdrr.io/r/base/paste.html) is always
`sep = ""`) must be a constant string. Every dot must resolve to a
STRING-typed IR node – ANY non-STRING dot (numeric, logical, ...) falls
back rather than replicating R's
[`as.character()`](https://rdrr.io/r/base/character.html) formatting
rules (verified empirically: `paste0("x", 4)` is `"x4"`, NOT `"x4.0"` –
faithfully reproducing
[`as.character()`](https://rdrr.io/r/base/character.html)'s
numeric-formatting edge cases for every type is explicitly out of
scope). The lowered `ir_call("paste_cols", ...)` node's LAST argument is
always a STRING literal carrying the separator (never a per-row value) –
`apply_handler()`'s `narep = "NA"` reproduces R's own "a genuine NA
argument becomes the literal two-character string `\"NA\"`" behavior
(verified: `paste0("a", NA, "b")` is `"aNAb"`, not `NA`).

## Usage

``` r
ir_parse_paste_call(expr, env, schema, fn_name)
```
