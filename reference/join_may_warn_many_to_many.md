# Is a mutating join's default `relationship` allowed to warn?

dplyr's `compute_join_relationship()` returns "none" (no advisory
many-to-many warning) when any condition is not `==` (inequality,
overlap, mixed equi + non-equi), for `by = character()`, and when the
verb was not called directly (`is_direct(user_env)`: the caller's top
env is the global env, or the package under test). Decided at the verb
call; a lazy node carries the decision as
`relationship = "many-to-many"` (a no-op check).

## Usage

``` r
join_may_warn_many_to_many(spec, user_env)
```

## Arguments

- spec:

  Parsed `join_by` spec (`op`)

- user_env:

  The environment that called the join verb
