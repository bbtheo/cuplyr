# Base `rbind()` / `cbind()` for GPU tables

`rbind(gt, ...)` is
[`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
(keeping the first table's groups) and `cbind(gt, ...)` is
[`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md),
so the result is a `tbl_gpu` built on the GPU (base R would otherwise
bind the internal fields of the list). Base R picks the method from the
arguments, so this applies when a `tbl_gpu` comes first (or no argument
is a plain `data.frame`).

## Usage

``` r
# S3 method for class 'tbl_gpu'
rbind(..., deparse.level = 1)

# S3 method for class 'tbl_gpu'
cbind(..., deparse.level = 1)
```

## Arguments

- ...:

  `tbl_gpu`s, data frames and vectors, as for
  [`bind_rows()`](https://bbtheo.github.io/cuplyr/reference/bind_rows.md)
  /
  [`bind_cols()`](https://bbtheo.github.io/cuplyr/reference/bind_cols.md).

- deparse.level:

  Ignored.

## Value

A `tbl_gpu`.
