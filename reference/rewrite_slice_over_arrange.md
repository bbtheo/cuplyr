# Turn `arrange(x) |> head(n)` into a top-k slice

`head(n)` over a one-key
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) is
`slice_min(x, n, with_ties = FALSE)` (or
[`slice_max()`](https://dplyr.tidyverse.org/reference/slice.html) for
[`desc()`](https://dplyr.tidyverse.org/reference/desc.html)): both
stable-sort with NA last and take the first `compute_slice_size()` rows.
The rank-mode slice node runs `gpu_slice_rank()`, which selects the cut
value instead of sorting the table when the cut is selective. Only a
head slice sitting directly on an ungrouped one-key arrange of a
numeric, Date or POSIXct column is rewritten; strings (collation),
factors and logicals, several keys, `.by_group` and anything between the
two nodes keep the sort. The new node is a slice, so it stays a barrier:
no filter crosses it, and
[`barrier_input_required()`](https://bbtheo.github.io/cuplyr/reference/barrier_input_required.md)
reports its key. Called on a barrier node only, and only slices are
tested at the call sites.

## Usage

``` r
rewrite_slice_over_arrange(node)
```

## Arguments

- node:

  A slice node

## Value

The rewritten node, or `node` itself
