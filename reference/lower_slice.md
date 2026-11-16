# Lower slice node

Dispatches on `ast$mode` to the matching C++ entry point
(`src/ops_slice.cpp`). `nrow`-dependent resolution (the `n=`/`prop=`
clamp, and
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html)'s
index-vector validation/negative-index handling) happens entirely in
C++, since the input's actual row count is only available once
`ast$input` has actually been lowered/executed – it can't be computed
from
[`infer_schema()`](https://bbtheo.github.io/cuplyr/reference/infer_schema.md)
(which only tracks names/types, not row counts) at R-side parse time,
and a lazy AST may not even have a concrete row count until this point
in the pipeline runs. See `src/ops_common.hpp::compute_slice_size()`.

## Usage

``` r
lower_slice(ast, source_ptr)
```
