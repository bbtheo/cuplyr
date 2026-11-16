# Signal a type error caused by a factor operand (F18)

The GPU types a factor as DICTIONARY32 and can't combine it with
anything else, but dplyr can (vctrs makes `factor` + `character` a
character), so a type check that fails because of a DICTIONARY32 operand
is not a user error. It signals this class instead of a plain error;
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md)
turns it into `NULL` (CPU fallback). Outside a parse the message is the
same one the plain error had.

## Usage

``` r
stop_factor_needs_cpu(...)
```

## Arguments

- ...:

  Message pieces, pasted together
