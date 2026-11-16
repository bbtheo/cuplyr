# Read a parquet file straight onto the GPU

`read_parquet_gpu()` decodes a parquet file on the GPU with libcudf, so
the file's compressed bytes are what cross the PCIe bus (DECIMAL columns
are converted through the host, see Types). Like
[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) it
respects the execution mode: a lazy table (the default) holds a *scan*
that reads the file when the plan runs, an eager one reads it now. The
column names and types come from the file footer at call time without
reading any data. `write_parquet_gpu()` writes a GPU table with
libcudf's writer.

## Usage

``` r
read_parquet_gpu(path, columns = NULL, lazy = NULL)

write_parquet_gpu(.data, path, compression = "snappy", row_group_size = NULL)
```

## Arguments

- path:

  For `read_parquet_gpu()`, one path or a vector of paths, read in
  order. The files must have the same column names, in the same order,
  with the same types. libcudf also requires their footers to describe
  the columns identically, including parquet's type annotations, so
  files from different writers can be refused even when the columns
  agree (DuckDB annotates INT32 columns and arrow does not): that is a
  `cuplyr_error_parquet_schema_mismatch` when the read runs. For
  `write_parquet_gpu()`, one path (an existing file is overwritten).

- columns:

  Optional character vector (not empty): read only these columns, in
  this order. A column that is not read is never decoded, so a file with
  a column cuplyr cannot hold is readable as long as that column is left
  out.

- lazy:

  `TRUE`, `FALSE`, or `NULL` for the session default, as in
  [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md).

- .data:

  A `tbl_gpu`. A lazy plan is run first (it stays lazy: the table you
  passed is not changed), groups are not written.

- compression:

  `"snappy"` (the default), `"zstd"`, `"lz4"` or `"none"`.

- row_group_size:

  Rows per row group, or `NULL` for libcudf's default (one million).

## Value

`read_parquet_gpu()`: a `tbl_gpu`. `write_parquet_gpu()`: `.data`,
invisibly.

## Details

### Types

BOOLEAN, INT32, DOUBLE, STRING, DATE and microsecond timestamps read as
themselves. INT64 reads as `integer64` (exact, no downcast; with the
bit64 package). FLOAT reads as double, INT8/INT16/UINT8/UINT16 as
integer, UINT32 as double, UINT64 as `integer64` (an error naming the
column if a value is past `INT64_MAX`), DECIMAL as double (the nearest
one, with a `cuplyr_warning_collect_precision` warning if more than 15
digits are needed; the conversion runs on the host, so a DECIMAL column
makes a round trip over the PCIe bus), second, millisecond and
nanosecond timestamps as `POSIXct` (nanoseconds finer than a microsecond
are rounded, with that warning). LIST, STRUCT and TIME columns are an
error of class `cuplyr_error_parquet_unsupported_type` naming the column
and the file when a plan reads the column. `columns =` always keeps it
out of the read. A later
[`select()`](https://dplyr.tidyverse.org/reference/select.html) usually
does too, but a verb that has to run the plan before the
[`select()`](https://dplyr.tidyverse.org/reference/select.html) (an
integer [`sum()`](https://rdrr.io/r/base/sum.html) in
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html), for
example) reads every column, so use `columns =` if you get this error.

An INT32 or INT64 value equal to `-2147483648` or `-9223372036854775808`
is read as missing: R's `NA_integer_` and bit64's `NA` are those values,
so R (and arrow's `read_parquet()`) cannot hold them as numbers, and a
filter or a sum on the GPU has to agree with
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html).

### Factors

A factor written by `write_parquet_gpu()` comes back as the same factor
(levels, unused levels and `ordered`), because the levels are kept in
the file footer. A factor with `NA` as a level
([`addNA()`](https://rdrr.io/r/base/factor.html)) is the exception: a
file cannot tell that level from a missing value, so
`write_parquet_gpu()` warns (class
`cuplyr_warning_parquet_factor_na_level`, naming the column) and the
column reads back as strings, with the missing value as `NA`. **A factor
written by `arrow::write_parquet()` reads as a string column**: arrow
keeps the levels in the parquet dictionary pages, which libcudf does not
expose. If you need the factor, convert it afterwards, e.g.
`mutate(f = factor(f, levels = ...))` on the collected data.

### Time zones

A `POSIXct` column written by `write_parquet_gpu()` or by arrow keeps
its time zone. A timestamp with no zone information in the file (no
`ARROW:schema` entry, e.g. written by another tool) is read as UTC
instants with `tzone = "UTC"`. A timestamp arrow wrote without a zone
(`isAdjustedToUTC = false`) is read as UTC instants and has no `tzone`
attribute, as in arrow.

### If the file changes

A lazy table remembers the columns and types the file had when
`read_parquet_gpu()` was called. The plan reads the file again every
time it runs. If the columns or their types differ then, the read stops
with an error of class `cuplyr_error_parquet_schema_changed` naming the
file and the difference. If only the data changed (same columns, other
rows), the new data is read: a lazy scan sees the file as it is when the
plan runs, not as it was when the table was made. The row count
[`nrow()`](https://rdrr.io/r/base/nrow.html) of a bare scan comes from
the footer at the time you ask. Reading one lazy table twice reads the
file twice.

### Errors

All errors have class `cuplyr_error_parquet` and a more specific one,
and name the file: `cuplyr_error_parquet_not_found`, `_invalid` (not a
parquet file, corrupt footer), `_unsupported_type`, `_schema_changed`,
`_schema_mismatch` (files that differ), `_unknown_column` and `_columns`
(`columns =` names no column or is empty), `_write` and `_compression`.
Several files are read by one libcudf call that doesn't say which file a
problem is in, so those messages say "one of the parquet files". Running
out of GPU memory is `cuplyr_error_gpu_oom`, with the file in the
message.

### Writing

`write_parquet_gpu()` writes to a temporary file beside `path` and
renames it into place, so a failed write leaves an existing file as it
was. It needs write permission on the directory, not only on an existing
file.

### Pushdown

When a lazy plan over a scan runs through the optimizer
([`compute()`](https://dplyr.tidyverse.org/reference/compute.html),
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html)), the
scan reads only the columns the plan uses, and the conditions of a
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) directly
on the scan that libcudf can decide exactly are given to the reader,
which skips row groups from the file's statistics and drops the other
rows while decoding.
[`explain()`](https://dplyr.tidyverse.org/reference/explain.html) shows
the columns and the filter of the scan. A condition is pushed only when
it is a comparison of a column with a literal that converts to the
column's file type without change (`==`, `!=`, `<`, `<=`, `>`, `>=`) on
an integer, `integer64`, `Date` or `POSIXct` column, or `!is.na()` of an
`integer` or `integer64` column; the `&`s of a condition are split.
Comparisons on double and string columns are not pushed: libcudf 25.12
can skip a row group wrongly from a double column's `NaN` statistics or
a string column's deprecated min/max statistics (signed byte order),
which would lose rows. They are applied after the read, so a filter on
doubles or strings gives the same rows but skips no row groups. A factor
written by `arrow::write_parquet()` reads as a string (see above), so it
is not pushed either. Everything else (`|`, `!`,
[`is.na()`](https://rdrr.io/r/base/NA.html), `%in%`, arithmetic,
functions, a double literal against an `integer64` column, a column that
was converted on reading such as a factor, a filter on a grouped table,
with `.by`, or on a rowwise table, a filter that comes after a
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) that
changes its column, or after a
[`slice()`](https://dplyr.tidyverse.org/reference/slice.html),
[`distinct()`](https://dplyr.tidyverse.org/reference/distinct.html) or
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html))
is applied after the read, as without pushdown, and a filter with
anything left to apply is applied whole. The answer is the same either
way. `options(cuplyr.parquet_pushdown = FALSE)` turns both pushdowns
off.

A file whose footer statistics are malformed (for example a fixed-width
statistic of the wrong length; an invalid file that no common writer
produces) can crash the R session when a filter is pushed into the scan.
For files you don't trust, `options(cuplyr.parquet_pushdown = FALSE)`
avoids it.

### Previews

[`print()`](https://rdrr.io/r/base/print.html) and
[`glimpse()`](https://pillar.r-lib.org/reference/glimpse.html) of a
table that is a bare scan (or only
[`select()`](https://dplyr.tidyverse.org/reference/select.html)s and
[`rename()`](https://dplyr.tidyverse.org/reference/rename.html)s of one)
read just the first rows of the file, which libcudf decodes as whole row
groups: when the row group that holds them has more than 65,536 rows or
128 MB of column data, nothing is read and the schema is printed instead
(write files with smaller row groups, `row_group_size =`, to get a
preview). Any other plan over a file prints its schema and is not run.

### File access

libcudf's file layer (kvikio) is put in its compatibility mode, reading
through the page cache, unless the environment variable
`KVIKIO_COMPAT_MODE` is set before the first parquet file operation. Set
it to `AUTO` or `OFF` to let kvikio use GPUDirect Storage where it is
installed; that loads `libcufile`, which writes a `cufile.log` into the
working directory.

When it loads, cuplyr sets the environment variable
`LIBCUDF_USE_DEBUG_STREAM_POOL`, a libcudf debug switch that makes
libcudf use its default CUDA stream instead of forking new ones. After
parquet reads or writes, forked streams slowed later
[`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html),
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html) and
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
calls in the session by a few percent; with the switch on, reads and
writes of the files measured, up to 1e8 rows, were no slower. A value
you set before loading is left alone, but libcudf only tests whether the
variable exists, so any value, `0` included, turns the switch on. To
turn it off, call `Sys.unsetenv("LIBCUDF_USE_DEBUG_STREAM_POOL")` after
loading cuplyr and before the first `read_parquet_gpu()` or
`write_parquet_gpu()` call in the session. A lazy `read_parquet_gpu()`
counts too: it runs libcudf's reader when it is called. libcudf creates
its stream pool once, so unsetting the variable later has no effect.

## See also

[`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
[`collect.tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/collect.tbl_gpu.md)

## Examples

``` r
library(dplyr)
if (has_gpu()) {
  path <- tempfile(fileext = ".parquet")
  write_parquet_gpu(tbl_gpu(mtcars), path)
  read_parquet_gpu(path) |>
    filter(mpg > 25) |>
    collect()
  unlink(path)
}
```
