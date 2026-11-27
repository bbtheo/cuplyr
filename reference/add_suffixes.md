# Port of `dplyr:::add_suffixes()`

The real column-naming primitive behind every dplyr join's suffix
handling (verified via `body(dplyr:::add_suffixes)` against dplyr 1.2.1)
– not a one-shot "does this name collide with the other side" check. `x`
is the set of names to (maybe) suffix; `y` is the "cross" name set to
disambiguate against (the other side's names, plus, for the ignore-aware
caller below, this side's own ignored/key names); `suffix` is appended
to any entry of `x` that collides with `c(y, x)`, and the check is
REPEATED (each iteration re-checks the whole combined vector for
duplicates) until every entry is unique – so an "already-taken suffixed
name" (e.g. left already has a literal `val.y` column when `val` needs a
`.y` suffix from a collision with the right table) gets a SECOND suffix
(`val.y.y`), matching dplyr exactly rather than silently producing a
duplicate output column name. `suffix` equal to the literal empty string
`""` is dplyr's own escape hatch: `identical(suffix, "")` short-circuits
to `x` unchanged, even when that leaves a genuine name collision – dplyr
resolves that downstream via a name-keyed column *overwrite*
(`join_mutate()`'s `out[names(y_out)] <- vec_slice(y_out, ...)`, which
replaces same-named columns in place rather than erroring or renaming);
see
[`join_output_would_collide()`](https://bbtheo.github.io/cuplyr/reference/join_output_would_collide.md)
below for how cuplyr detects this specific degenerate case and routes
around it (CPU fallback) instead of trying to reproduce a
column-count-changing overwrite in the native gather path.

## Usage

``` r
add_suffixes(x, y, suffix)
```
