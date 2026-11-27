# Push filters below mutates when predicates do not depend on mutate outputs

Pushing a filter across
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) is
deliberately not implemented. `filter(distinct(df, k), k > 1)` COULD
safely push the filter below
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) when
the predicate only references surviving key columns (or, when
`.keep_all = TRUE`, any column at all) – a predicate that's a pure
function of the dedup key is constant within each dedup group, so
filtering before or after the group-representative pick selects the same
groups either way. A predicate touching a NON-key column (only reachable
with `.keep_all = TRUE`) is NOT safe to push, though: it could remove
the very row
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html)
would have picked as a group's representative, changing which row (not
just which groups) survives.

## Usage

``` r
push_down_filters(ast)
```

## Arguments

- ast:

  Root AST node

## Value

AST with filters pushed down across mutates where safe

## Details

Deliberately NOT implemented here. `distinct` is a barrier
([`is_barrier()`](https://bbtheo.github.io/cuplyr/reference/is_barrier.md),
R/ast.R), so under the current
[`optimize_with_barriers()`](https://bbtheo.github.io/cuplyr/reference/optimize_with_barriers.md)
architecture a `filter` immediately above a `distinct` is split into two
INDEPENDENT segments before any pass runs (see
[`optimize_with_barriers()`](https://bbtheo.github.io/cuplyr/reference/optimize_with_barriers.md)'s
own comments) – the filter's segment ends in a stub source in place of
the `distinct`, so `distinct` is never actually present as `ast$input`
in the same `push_down_filters()` call for the ordinary linear-chain
case. Reaching across deliberately would require a barrier-aware special
case in
[`optimize_with_barriers()`](https://bbtheo.github.io/cuplyr/reference/optimize_with_barriers.md)
itself (matching on "is the barrier specifically a distinct node, and is
the segment's bottom node specifically a single filter with
only-key-column predicates"), not a `push_down_filters()` switch case –
meaningfully more invasive than the analogous mutate/ select pushdown
cases below, and with a real correctness footgun if the
key-vs-non-key-column safety check is ever gotten wrong. That risk is
why it isn't attempted.
