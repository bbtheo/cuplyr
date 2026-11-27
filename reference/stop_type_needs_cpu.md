# Signal a type combination the GPU does not compute like dplyr

Same shape as
[`stop_timestamp_needs_cpu()`](https://bbtheo.github.io/cuplyr/reference/stop_timestamp_needs_cpu.md)
(a subclass of `cuplyr_factor_needs_cpu`, so
[`ir_parse_quo()`](https://bbtheo.github.io/cuplyr/reference/ir_parse_quo.md)
turns it into `NULL`): a STRING/BOOL8/INT32 argument of a reducer,
window or comparison that `ir_native_arg_types` does not list, or a
Period/Duration value, goes to the CPU, where dplyr gives its result,
warning or error.

## Usage

``` r
stop_type_needs_cpu(...)
```

## Arguments

- ...:

  Message pieces, pasted together
