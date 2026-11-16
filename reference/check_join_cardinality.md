# Raise dplyr's exact `multiple=`/`relationship=`/`unmatched=` conditions

The Phase 7 J5 activation point: called from
[`lower_join()`](https://bbtheo.github.io/cuplyr/reference/lower_join.md)
(R/lower.R) immediately after a mutating join
(`left`/`inner`/`full`/`right`) returns its `list(ptr, diag)` result,
using the `diag` from `join_map_stats()` (`src/ops_join.cpp`), computed
after the `multiple=` filter. Never called for `semi`/`anti`/`cross` –
those join types have no `multiple=`/ `unmatched=`/`relationship=` in
dplyr's own signatures at all.

## Usage

``` r
check_join_cardinality(diag, join_type, multiple, relationship, unmatched)
```

## Arguments

- diag:

  `list(left_multi_first, right_multi_first, left_unmatched_first, right_unmatched_first)`
  from `join_map_stats()`

- join_type:

  One of `"left"`, `"inner"`, `"full"`, `"right"`

- multiple, relationship, unmatched:

  As validated by
  [`validate_join_multiple()`](https://bbtheo.github.io/cuplyr/reference/validate_join_multiple.md)/[`validate_join_relationship()`](https://bbtheo.github.io/cuplyr/reference/validate_join_relationship.md)/
  [`validate_join_unmatched()`](https://bbtheo.github.io/cuplyr/reference/validate_join_unmatched.md)

## Value

`invisible(NULL)`; called for its error/warning side effects only

## Details

Precedence (empirically verified against dplyr 1.2.1, all pairwise
combinations of "multiplicity violation present" x "unmatched violation
present"): multiplicity (`relationship=` or the auto many-to-many
warning) is ALWAYS evaluated before `unmatched=`, and within
`unmatched=`, the x-side check (`no_match`, "each row of x must have a
match") always fires before the y-side check (`remaining`, "each row of
y must be matched"). Per-join-type applicability of the unmatched checks
mirrors `dplyr:::standardise_join_no_match()`/
`standardise_join_remaining()`: x-side applies to `inner`/`right` (the
join types where a genuinely-unmatched x row can be silently dropped),
y-side applies to `inner`/`left` (where a genuinely-unmatched y row can
be silently dropped); `full` never reaches either branch in practice
since
[`full_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)
doesn't expose `unmatched=` at all (always passes `"drop"`/`"drop"` from
[`full_join.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/mutating-joins.md),
R/join.R).

The many-to-many auto-warning only fires when `relationship` is `NULL`
AND `multiple` is `"all"` (both defaults) – explicitly verified against
dplyr 1.2.1: passing any non-default `multiple=` (the user is already
handling duplicates deliberately) or any explicit `relationship=`
suppresses the auto-warning entirely (an explicit `relationship=` runs
its own check instead, which may itself pass silently, e.g.
`relationship = "many-to-many"`).
