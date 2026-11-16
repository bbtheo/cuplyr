# Lower reframe node

Reshapes the wide one-row-per-group table to `k` rows per group on the
device (`gpu_reframe_expand()`, src/ops_reframe.cpp). A literal output
is uploaded as a tiny table and tiled there.

## Usage

``` r
lower_reframe(ast, source_ptr)
```
