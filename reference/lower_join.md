# Lower join node

The four "mutating" join types (`"left"`/`"inner"`/`"full"`/`"right"`)
all share the same C++ calling convention –
`left_key_idx`/`right_key_idx`/ `right_drop_idx`/`nulls_equal` – and
their GPU entry points
(`gpu_left_join()`/`gpu_inner_join()`/`gpu_full_join()`/`gpu_right_join()`,
`src/ops_join.cpp`) all return `list(ptr, diag)` (Phase 7 J4's
map-pipeline restructure): `ptr` is the result table's external pointer,
`diag` is a first-offender diagnostics list consumed by
[`check_join_cardinality()`](https://bbtheo.github.io/cuplyr/reference/check_join_cardinality.md)
below (multiple=/unmatched=/relationship=, Phase 7 J5).
[`right_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
used to be lowered via a swapped `left_join(y, x)` plus a column
reorder/select (`build_right_join_via_left()`/
`resolve_right_join_select_idx()`, since removed) – `gpu_right_join()`
is now native and takes exactly the same argument shape as the other
three, so no special-casing is needed here beyond the switch itself.

## Usage

``` r
lower_join(ast, source_ptr)
```

## Details

Non-equi
[`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html) (Phase
7 J7, `scratchpad/phase7_joins_design.md` section 1.4/2): `ast$by$op`
carries each condition's real operator (not just `"=="`) whenever
[`join_by_natively_supported()`](https://bbtheo.github.io/cuplyr/reference/join_by_natively_supported.md)
(R/join.R) let the join reach this AST shape at all –
[`join_equi_spec()`](https://bbtheo.github.io/cuplyr/reference/join_equi_spec.md)/[`join_cond_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_spec.md)
split it back into an equality-key subset (routed through the same
hash-join machinery every equi join uses, via `mixed_join`'s own
equality tables) and a residual non-equi subset, folded into one AST
condition tree by
[`join_cond_ir_from_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_ir_from_spec.md) +
[`ir_bind_join()`](https://bbtheo.github.io/cuplyr/reference/ir_bind_join.md)
and handed to `gpu_cond_join()` (`src/ops_join.cpp`), which dispatches
to `mixed_*`/`conditional_*` cuDF entry points depending on whether any
equi keys survived the split. `gpu_cond_join()` reuses the exact same
`list(ptr, diag)` / bare-`ptr` return shapes as the equi paths below
(see its own C++ comment for the `join_kind == "right"` swapped-role
decision), so
[`check_join_cardinality()`](https://bbtheo.github.io/cuplyr/reference/check_join_cardinality.md)
applies identically either way.
