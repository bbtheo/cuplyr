# Restore `tbl_gpu`-specific attributes after a generic transformation (dplyr backend generic)

Implements dplyr's
[`dplyr_reconstruct()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
extension-contract generic for `tbl_gpu`.

## Usage

``` r
# S3 method for class 'tbl_gpu'
dplyr_reconstruct(data, template)
```

## Arguments

- data:

  The dplyr-generic's (possibly mangled, per above) `data` argument.

- template:

  A `tbl_gpu` object to restore grouping/exec_mode from.

## Value

A `tbl_gpu` object.

## Details

### An important structural gotcha (verified empirically)

dplyr's own top-level
[`dplyr_reconstruct()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
function is NOT a plain generic: its body is
`data <- dplyr_new_data_frame(data); dplyr_reconstruct_dispatch(data, template)`
– it unconditionally coerces `data` via
[`vctrs::new_data_frame()`](https://vctrs.r-lib.org/reference/new_data_frame.html)
(dispatching on `template`'s class, not `data`'s own) *before* any S3
method ever runs. For a genuinely row-oriented backend (a real data
frame, or dbplyr's lazy-query object, whose top-level list elements
really are one implicit "row" of metadata), that's harmless. For
`tbl_gpu` – a FIXED six-field metadata handle
(`ptr`/`schema`/`lazy_ops`/`groups`/`exec_mode`/`group_drop`), not a
row-oriented list – `new_data_frame()` instead produces a degenerate
`data.frame` whose "columns" are literally those six fields (verified
directly: `dplyr:::dplyr_new_data_frame(a_tbl_gpu)` returns a 0-row
`data.frame(ptr=, schema=, lazy_ops=, groups=, exec_mode=, group_drop=)`).
This method recovers the original `tbl_gpu` from that exact shape
(`new_data_frame()` doesn't deep-copy anything, so the GPU
pointer/schema survive intact) rather than trying to re-upload it as if
it were real tabular data, which would error (or silently corrupt) the
moment `df_to_gpu()` saw an external pointer as a "column".

Because of this,
[`dplyr_row_slice.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/dplyr_row_slice.tbl_gpu.md)
and
[`dplyr_col_modify.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/dplyr_col_modify.tbl_gpu.md)
(above) deliberately do NOT call
[`dplyr::dplyr_reconstruct()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
themselves – their own results are already fully-formed `tbl_gpu`
objects, so routing them through the top-level function would trigger
this exact mangling for no benefit. This method exists for the (rarer)
case where something else calls
[`dplyr::dplyr_reconstruct()`](https://dplyr.tidyverse.org/reference/dplyr_extending.html)
directly on a tbl_gpu.
