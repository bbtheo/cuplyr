# Registry of supported expression-IR operations

See the module-level comment above for the extension contract. Each
entry: `list(arity, parse, type, lower)`, plus a fifth field, `window`:
`NULL` for an ordinary entry, or
`list(kind = "shift"|"rank"|"scan"|"agg"|"rolling")` for a window-class
op. `window` non-`NULL` and `lower` non-`NULL` are mutually exclusive –
a window-class entry always has `lower = NULL` (window ops are never
`ir_call` handlers in `src/expr_eval.hpp`, see this file's own module
comment for that boundary, and are structurally prevented from reaching
lowering at all by
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)'s
guard).
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
needs no change: a window entry's `type` function is looked up and
called exactly like any other entry's.

## Usage

``` r
ir_call_registry
```
