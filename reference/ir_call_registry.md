# Registry of supported expression-IR operations

See the module-level comment above and section 6 of
`scratchpad/phase1_expression_engine.md` for the extension contract this
implements. Each entry: `list(arity, parse, type, lower)`.

## Usage

``` r
ir_call_registry
```

## Details

Phase 5 (`scratchpad/phase5_window_design.md` section 1.1) adds a fifth
field, `window`: `NULL` for every pre-Phase-5 entry (unchanged), or
`list(kind = "shift"|"rank"|"scan"|"agg")` for a window-class op.
`window` non-`NULL` and `lower` non-`NULL` are mutually exclusive – a
window-class entry always has `lower = NULL` (window ops are never
`ir_call` handlers in `src/expr_eval.hpp`, see this file's own module
comment for that boundary, and are structurally prevented from reaching
lowering at all by
[`ir_bind()`](https://bbtheo.github.io/cuplyr/reference/ir_bind.md)'s
guard).
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
needs no change: a window entry's `type` function is looked up and
called exactly like any other entry's.
