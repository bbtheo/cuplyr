# Route a join call to the transparent CPU fallback (Phase 7 J6)

Shared by every mutating/semi/anti join verb below for the two shapes
the native GPU path can't (yet, or ever) handle:

1.  A non-equi
    [`join_by()`](https://dplyr.tidyverse.org/reference/join_by.html)
    object
    ([`join_by_is_non_equi()`](https://bbtheo.github.io/cuplyr/reference/join_by_is_non_equi.md)
    is `TRUE`) – Phase 7 J7 replaces this with a native
    `mixed_join`/`conditional_join` lowering; until then this keeps user
    code working exactly like every other not-yet-native verb (mirrors
    [`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)'s
    two-table fallback shape, `R/fallback.R`: collect the OTHER table,
    then run the real dplyr verb on `.data`'s own materialized/grouped
    tibble via
    [`gpu_fallback()`](https://bbtheo.github.io/cuplyr/reference/gpu_fallback.md)).

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
