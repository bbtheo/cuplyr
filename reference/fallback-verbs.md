# CPU fallback methods for dplyr verbs without a GPU-native implementation

`cuplyr` implements every dplyr generic on `tbl_gpu`, but not every one
of them runs natively on the GPU. The methods documented here provide a
*transparent CPU fallback*: they materialize any pending lazy
operations,
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)
`.data` to a tibble (restoring
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
structure from `.data`'s grouping so grouped semantics apply exactly as
on a real `grouped_df`), run the real dplyr verb on the CPU, and
re-upload the result via
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) –
restoring grouping (read back from the verb's actual result, since some
verbs change it), execution mode (lazy stays lazy), and factor levels.

## Usage

``` r
# S3 method for class 'tbl_gpu'
do(.data, ...)

# S3 method for class 'tbl_gpu'
nest_join(
  x,
  y,
  by = NULL,
  copy = FALSE,
  keep = NULL,
  name = NULL,
  ...,
  na_matches = c("na", "never"),
  unmatched = "drop"
)

# S3 method for class 'tbl_gpu'
group_map(.data, .f, ..., .keep = FALSE)

# S3 method for class 'tbl_gpu'
group_modify(.data, .f, ..., .keep = FALSE)

# S3 method for class 'tbl_gpu'
group_nest(.tbl, ..., .key = "data", keep = FALSE)

# S3 method for class 'tbl_gpu'
nest_by(.data, ..., .key = "data", .keep = FALSE)
```

## Arguments

- .data, x, y, tbl, .tbl:

  A `tbl_gpu` object (or, for `y`/two-table verbs, optionally a plain
  `data.frame`).

- ...:

  Passed through to the corresponding dplyr verb.

## Details

Verbs whose result isn't a data frame at all (e.g.
[`dplyr::group_map()`](https://dplyr.tidyverse.org/reference/group_map.html)'s
list), or whose result is a data frame containing list-columns with no
GPU representation (e.g.
[`dplyr::nest_by()`](https://dplyr.tidyverse.org/reference/nest_by.html),
[`dplyr::group_nest()`](https://dplyr.tidyverse.org/reference/group_nest.html),
[`dplyr::nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)'s
nested match column), are returned as plain R objects rather than
re-uploaded.

A GPU key match for
[`nest_join()`](https://dplyr.tidyverse.org/reference/nest_join.html)/[`group_nest()`](https://dplyr.tidyverse.org/reference/group_nest.html)/[`nest_by()`](https://dplyr.tidyverse.org/reference/nest_by.html)
was measured and does not pay (the time is building the nested tibbles
on the host), so they stay CPU fallbacks.

Every fallback calls
[`cuplyr_fallback_notify()`](https://bbtheo.github.io/cuplyr/reference/cuplyr_fallback_notify.md),
so `options(cuplyr.fallback = "warn")` (or `"error"`) can be used to
surface (or forbid) accidental fallbacks in code that's expected to stay
fully GPU-native. The notice names the verb, the expression, the reason
and the size collected; see
[cuplyr-fallback](https://bbtheo.github.io/cuplyr/reference/cuplyr-fallback.md).
