# Lower filter node

Every predicate in the node is a `cuplyr_ir` node (see `R/ir.R`,
`R/ast.R::make_predicate()`). All of them are folded together with `&`
into a single expression – one filter node is always exactly one fused
GPU kernel, regardless of how many predicates it holds, so there is no
separate "fused vs sequential" branch to check.

## Usage

``` r
lower_filter(ast, source_ptr)
```
