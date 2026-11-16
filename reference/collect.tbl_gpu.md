# Transfer GPU table data back to R

Copies data from GPU memory back to R as a tibble. This is typically the
final step in a GPU data manipulation pipeline, after filtering,
mutating, and selecting the data you need.

## Usage

``` r
# S3 method for class 'tbl_gpu'
collect(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object created by
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- ...:

  Additional arguments (ignored, included for compatibility).

## Value

A [tibble::tibble](https://tibble.tidyverse.org/reference/tibble.html)
containing the data from the GPU table (a `grouped_df`, with the same
`.drop`, when `x` is grouped, as for
[`dplyr::collect()`](https://dplyr.tidyverse.org/reference/compute.html)
on a database table). A table uploaded from a plain `data.frame` comes
back as a bare `data.frame` while it is ungrouped, and custom data-frame
attributes (`attr(df, "foo")`) are kept the way dplyr keeps them through
verbs. Column types are converted back to R types:

- FLOAT64/FLOAT32 -\> numeric (double)

- INT8/INT16/INT32, UINT8/UINT16 -\> integer

- UINT32 -\> double (exact)

- INT64/UINT64 -\>
  [`bit64::integer64`](https://bit64.r-lib.org/reference/bit64-package.html)
  (a double with a warning when `bit64` is absent and a value is past
  2^53); a UINT64 value past `INT64_MAX` is an error, never a wrap

- STRING -\> character

- BOOL8 -\> logical (TRUE/FALSE)

- DATE (TIMESTAMP_DAYS) -\> `Date`; TIMESTAMP_SECONDS/MILLISECONDS/
  MICROSECONDS/NANOSECONDS -\> `POSIXct` (nanoseconds finer than a
  microsecond are rounded, with a warning)

- DURATION_DAYS -\> `difftime` in days; the other DURATION\_\* -\>
  `difftime` in seconds (a warning past 2^53 ticks)

- DECIMAL32/64/128 -\> double, the nearest one (a warning when it needs
  more than 15 significant digits; an error beyond the double range)

- a dictionary with string keys -\> factor

Any other type (LIST, STRUCT, a dictionary with other keys) is an error
of class `cuplyr_error_unsupported_gpu_type` naming the column; the
warnings have class `cuplyr_warning_collect_precision` and name every
column concerned. [`print()`](https://rdrr.io/r/base/print.html) and
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) show such
a column as a placeholder.

## Details

### Grouped tables and `.drop`

The `.drop` setting of
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html)
travels with the table through the verbs that keep at least one group
column (and is reset to the default when none is left;
[`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) on a
grouped table keeps it unless given a new `.drop`), so
`group_by(f, .drop = FALSE) |> arrange(x) |> collect()` still keeps
empty factor-level groups. Empty groups of a *non-factor* column that
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) or
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html) emptied
are not kept, unlike in dplyr, except after `slice(.preserve = TRUE)`,
which keeps its input's group keys: see the `.drop` section of
[group_by()](https://bbtheo.github.io/cuplyr/reference/group_by.tbl_gpu.md).

### Memory considerations

Collecting transfers all data from GPU to CPU memory. For large
datasets, this can be slow and memory-intensive. Best practice is to:

1.  Filter rows to reduce data volume

2.  Select only needed columns

3.  Then collect the results

### Performance

Data transfer between GPU and CPU is limited by PCIe bandwidth
(typically 16-32 GB/s). For a 1 GB dataset, expect ~50-100ms transfer
time.

## See also

[`tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) for
creating GPU tables,
[`filter.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/filter.tbl_gpu.md),
[`select.tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/select.tbl_gpu.md)
for reducing data

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  gpu_mtcars <- tbl_gpu(mtcars)

  # Process on GPU, then collect
  result <- gpu_mtcars |>
    filter(mpg > 20) |>
    mutate(kpl = mpg * 0.425) |>
    select(mpg, kpl, hp) |>
    collect()

  # Result is a regular tibble
  class(result)
  print(result)
}
#>     mpg     kpl  hp
#> 1  21.0  8.9250 110
#> 2  21.0  8.9250 110
#> 3  22.8  9.6900  93
#> 4  21.4  9.0950 110
#> 5  24.4 10.3700  62
#> 6  22.8  9.6900  95
#> 7  32.4 13.7700  66
#> 8  30.4 12.9200  52
#> 9  33.9 14.4075  65
#> 10 21.5  9.1375  97
#> 11 27.3 11.6025  66
#> 12 26.0 11.0500  91
#> 13 30.4 12.9200 113
#> 14 21.4  9.0950 109
```
