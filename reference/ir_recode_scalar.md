# One scalar `recode()` value, by shape

Accepts only what the allowlist admits: a length-1, attribute-free
atomic of typeof character/integer/double/logical (doubles finite or
`NA_real_`; R parses `NA`, `NA_character_`, ... as such constants, so a
symbol is never accepted), or unary `-` on a numeric literal. Nothing is
evaluated.

## Usage

``` r
ir_recode_scalar(a, env, schema)
```

## Value

`list(value =, type =)` (`type` is the R typeof), or `NULL`
