# The equi (`==`) subset of a join spec

The `==` entries of `spec` (all of it for a plain equi join): the
hash-join keys of a mixed join. Kept as a named subsetting helper
(rather than inlining `spec$op == "=="` at call sites) so the hash-join
key selection has one place to read the equi subset from.

## Usage

``` r
join_equi_spec(spec)
```
