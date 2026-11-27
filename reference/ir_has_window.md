# Does an IR (sub)tree contain a window-function call anywhere?

`TRUE` iff any `call` node in the subtree has a non-`NULL` `window`
field in its
[ir_call_registry](https://bbtheo.github.io/cuplyr/reference/ir_call_registry.md)
entry. Used by
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)'s
window guard, by the decomposition machinery in `R/window.R`
([`ir_extract_windows()`](https://bbtheo.github.io/cuplyr/reference/ir_extract_windows.md)/[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)),
and by verbs such as
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) and
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) to detect
when a parsed expression needs the window decomposition path at all.

## Usage

``` r
ir_has_window(ir)
```

## Arguments

- ir:

  An IR node, a list of IR nodes, or `NULL`

## Value

Logical scalar

## Details

Accepts either a single IR node (identified by the `cuplyr_ir` class
every
[`ir_col()`](https://bbtheo.github.io/cuplyr/reference/ir_col.md)/[`ir_lit()`](https://bbtheo.github.io/cuplyr/reference/ir_lit.md)/[`ir_call()`](https://bbtheo.github.io/cuplyr/reference/ir_call.md)
node carries) or a plain (possibly unnamed) list of IR nodes – e.g. a
call node's own `$args` – so callers can check a call's argument list
directly without unpacking it first.
