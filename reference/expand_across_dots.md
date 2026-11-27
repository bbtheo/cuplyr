# Expand `across()`/`if_any()`/`if_all()` dots at the quosure level

Walks `dots` in order; a top-level, unnamed `across(...)` dot is
replaced by `expand_one_across()`'s (possibly multi-element, possibly
empty) result, a top-level `if_any(...)`/`if_all(...)` dot is replaced
by `expand_if_any_all()`'s single folded quosure (keeping the ORIGINAL
dot's own name, if any – e.g. `mutate(flag = if_any(...))`), and every
other dot is passed through unchanged.
[`across()`](https://dplyr.tidyverse.org/reference/across.html) is
disallowed as a top-level
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) dot
(matching real dplyr 1.2.1 exactly, verified empirically – it raises at
EVALUATION time there, not parse time, but the message ("must be a
logical vector... use if_any()/if_all() instead") is the same either
way, so raising it up front here is equivalent and cheaper). A NAMED
top-level
[`across()`](https://dplyr.tidyverse.org/reference/across.html) dot
(e.g. `mutate(z = across(x, y))`) packs a data-frame column and never
gets here: `dots_need_fallback()` sends the whole call to the CPU
fallback.

## Usage

``` r
expand_across_dots(dots, schema, group_cols, verb)
```

## Arguments

- dots:

  A list of quosures (already past cur_group_id substitution, if
  applicable)

- schema:

  Current schema

- group_cols:

  Character vector, the effective group columns to exclude from
  `.cols`'s selectable universe

- verb:

  Character, `"filter"`/`"mutate"`/`"summarise"`/`"transmute"`

## Value

A new (possibly longer, possibly shorter) named list of quosures, with
the `"needs_fallback"` attribute described above, and a `"provenance"`
attribute: `list(origin, kind)`, one entry per returned quosure.
`origin` is the index of the input dot it came from; `kind` is `"dot"`
(the input dot, passed through unchanged), `"across"` (an expansion of
an unnamed
[`across()`](https://dplyr.tidyverse.org/reference/across.html) dot,
staged or not) or `"combined"` (an
[`if_any()`](https://dplyr.tidyverse.org/reference/across.html)/[`if_all()`](https://dplyr.tidyverse.org/reference/across.html)
dot). mutate()/transmute() use it to tell which original dot a value
computed from the expanded dots belongs to. When an expansion names a
column, `col` is the column each one stands for (a staged one's, not its
temp's; an
[`if_any()`](https://dplyr.tidyverse.org/reference/across.html)/[`if_all()`](https://dplyr.tidyverse.org/reference/across.html)'s
first; `""` for a plain dot), which the errors name (#30).

## Details

Expansion stops at the first dot that needs dplyr (the result is then
partial and only its `"needs_fallback"` attribute is meaningful).

The result carries a `"needs_fallback"` attribute. It is `TRUE` when the
schema or the `.fns` values show that dplyr must run the call: an
unsupported `.fns`/extra-argument shape (a non-inlinable lambda, a value
whose free variables are columns, ...), duplicate output names, or, in
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)/[`filter()`](https://dplyr.tidyverse.org/reference/filter.html),
an [`across()`](https://dplyr.tidyverse.org/reference/across.html) that
renames onto a column that a later expansion of the same
[`across()`](https://dplyr.tidyverse.org/reference/across.html) reads
(`across(c(y = x, z = y), ...)`): dplyr evaluates every function against
the pre-[`across()`](https://dplyr.tidyverse.org/reference/across.html)
data, while the expanded dots would run in sequence. That needs the
schema, so `dots_need_fallback()` can't see it; the verb calls then
re-run the ORIGINAL dots on the CPU. In
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html)/[`transmute()`](https://dplyr.tidyverse.org/reference/transmute.html)
the same hazard is staged instead: the expansions are computed into temp
columns first and assigned afterwards, and the `"staged_temps"`
attribute names the temps the caller must drop (and whose CPU fallbacks
must use the ORIGINAL dots).
