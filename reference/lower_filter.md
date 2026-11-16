# Lower filter node

Task T3 (Phase 1 expression-engine cutover,
`scratchpad/phase1_expression_engine.md` section 5): every predicate in
the node is a `cuplyr_ir` node (see `R/ir.R`,
`R/ast.R::make_predicate()`). All of them are folded together with `&`
into a single expression – one filter node is always exactly one fused
GPU kernel now, regardless of how many predicates it holds, so there is
no separate "fused vs sequential" branch (and no `ast$fused` flag) to
check.

## Usage

``` r
lower_filter(ast, source_ptr)
```
