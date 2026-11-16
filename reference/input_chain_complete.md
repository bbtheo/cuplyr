# Does an input chain reach a real leaf?

A segment of
[`optimize_with_barriers()`](https://bbtheo.github.io/cuplyr/reference/optimize_with_barriers.md)
ends in a stub source, so it is complete. "source" and "join" are the
only node types without an `$input`; a chain ending anywhere else was
cut by hand (a test).

## Usage

``` r
input_chain_complete(node)
```
