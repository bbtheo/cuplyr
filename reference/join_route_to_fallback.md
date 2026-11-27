# Route a join call to the transparent CPU fallback

Shared by every join verb for the shapes the native GPU path doesn't
handle, among them:

1.  A [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
    that
    [`join_by_natively_supported()`](https://bbtheo.github.io/cuplyr/reference/join_by_natively_supported.md)
    rejects (several `closest()` conditions, a string compared with a
    non-string, ...): collect the other table, then run the real dplyr
    verb on `.data`'s own materialized/grouped tibble via
    [`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)
    (the same two-table shape as
    [`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)'s
    fallback, `R/fallback.R`).

2.  The `suffix = c("", ...)` degenerate name-collision case detected by
    [`join_output_would_collide()`](https://bbtheo.github.io/cuplyr/reference/join_output_would_collide.md)
    – dplyr resolves this via a name-keyed column overwrite this
    package's fixed-column-count gather can't reproduce natively; falls
    back rather than emitting a wrong or duplicate-named result. `args`
    is the full named-argument list the corresponding `dplyr::<verb>()`
    call needs (using the ORIGINAL, not yet
    [`resolve_join_keep()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_keep.md)-resolved,
    user-facing values where relevant, e.g. `by`/`keep`, so dplyr's own
    defaulting/validation runs exactly as it would for a bare call) –
    `y_tbl` is collected once and reused across every retry-shaped
    caller.

## Usage

``` r
join_route_to_fallback(
  verb_name,
  verb_fn,
  x,
  y,
  args,
  user_env = NULL,
  reason = "this join has a shape the GPU join does not support."
)
```
