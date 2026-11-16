# The non-equi subset of a join spec

Feeds
[`join_cond_ir_from_spec()`](https://bbtheo.github.io/cuplyr/reference/join_cond_ir_from_spec.md)
below: the residual AST condition `gpu_cond_join()` (`src/ops_join.cpp`)
evaluates via `mixed_join`'s `binary_predicate` argument (equi entries
are handled entirely by `mixed_join`'s own equality-key hashing, via
[`join_equi_spec()`](https://bbtheo.github.io/cuplyr/reference/join_equi_spec.md)
instead – re-including them in the AST condition too would be redundant,
not wrong, but this package's `gpu_cond_join()` never does).

## Usage

``` r
join_cond_spec(spec)
```
