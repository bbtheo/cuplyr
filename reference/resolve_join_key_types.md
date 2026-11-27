# Resolve every join key pair's common type

Schema-only (no GPU work). dplyr casts each key pair to its common type
([`vctrs::vec_ptype2()`](https://vctrs.r-lib.org/reference/vec_ptype2.html))
before matching and errors when a pair has none; the ladder is
[`resolve_common_col_type()`](https://bbtheo.github.io/cuplyr/reference/resolve_common_col_type.md)
(R/bind.R), shared with
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
and the set operations. Walks equi AND non-equi pairs so an incompatible
pair raises at verb time, in eager and lazy mode.

## Usage

``` r
resolve_join_key_types(
  x_schema,
  y_schema,
  join_spec,
  x_unspecified = NULL,
  y_unspecified = NULL
)
```

## Arguments

- x_schema, y_schema:

  Schemas of the two inputs

- join_spec:

  A join spec from
  [`parse_join_by()`](https://bbtheo.github.io/cuplyr/reference/parse_join_by.md)

- x_unspecified, y_unspecified:

  Names of each side's
  [`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
  columns (the tbls' `$unspecified`)

## Value

`join_spec`, with `key_casts` set when any equi pair needs a cast (and
`key_abort`, see above)

## Details

Pairs whose types already match (the common, benchmarked case) cost one
vector comparison and leave `join_spec` untouched. Equi pairs with
compatible-but-different types are recorded in `join_spec$key_casts`:
`idx` (position in the spec), `type` (the common type), `cast_left`/
`cast_right`, the source types, and a factor side's levels (needed to
decode it to `STRING`). Non-equi numeric pairs are only validated (the
expression evaluator widens their operands itself); a rolling
INT32/FLOAT64 pair is recorded (it is searched, not evaluated) and so is
a non-equi STRING pair (`string_codes = TRUE`, see below).

A BOOL8 key against a STRING, factor, Date or POSIXct key is not an
error here: dplyr (vctrs) gives an all-`NA` logical the other side's
type, but only the data can say whether the column is one. Such a pair
is recorded with `unspecified = TRUE`;
[`apply_join_key_casts()`](https://bbtheo.github.io/cuplyr/reference/apply_join_key_casts.md)
checks the values when the join runs and raises the incompatible-type
error then. A verb that records one without `marker` computes the join
at once (a plan break in lazy mode, see `join_key_check_reads_data()`),
so dplyr's error is raised at the verb. `marker` is `TRUE` for a
[`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)
column (named in the tbl's `$unspecified`): known all-`NA`, no data
check.

A pair that has no common type and comes after such an unchecked pair is
not raised at once: dplyr names the first bad pair in `by` order, and
the earlier one is only known from the data. The resolver stores it as
`join_spec$key_abort` (the arguments of the error) and stops; the join
raises it after the earlier pairs' checks.

[`lower_join()`](https://bbtheo.github.io/cuplyr/reference/lower_join.md)
applies the plan;
[`build_join_output_info()`](https://bbtheo.github.io/cuplyr/reference/build_join_output_info.md)
reads it for the merged key's output type.
