# Signal a type error dplyr resolves for a timestamp operand

The timestamp sibling of
[`stop_factor_needs_cpu()`](https://bbtheo.github.io/cuplyr/reference/stop_factor_needs_cpu.md):
`t + 3600` (a POSIXct in `t`'s zone) and a Date mixed with a POSIXct in
[`if_else()`](https://dplyr.tidyverse.org/reference/if_else.html)/[`case_when()`](https://dplyr.tidyverse.org/reference/case-and-replace-when.html)/
[`coalesce()`](https://dplyr.tidyverse.org/reference/coalesce.html) are
valid R that the GPU engine can't compute. It is a subclass, so
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md)'s
handler turns it into `NULL` as well, but only this class is raised from
the type check
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md)
runs: a factor expression has to come back as IR for
[`ir_factor_rule()`](https://bbtheo.github.io/cuplyr/reference/ir_factor_rule.md).

## Usage

``` r
stop_timestamp_needs_cpu(...)
```

## Arguments

- ...:

  Message pieces, pasted together
