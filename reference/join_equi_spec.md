# The equi (`==`) subset of a join spec

Every current spec is 100% equi (Phase 7 J1); this returns `spec`
unchanged in that case. Kept as a named subsetting helper (rather than
inlining `spec$op == "=="` at call sites) so the future non-equi callers
(Phase 7 J7's hash-join key selection) have one place to read the equi
subset from.

## Usage

``` r
join_equi_spec(spec)
```
