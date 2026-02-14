# Create a subquery barrier without executing

Marks the current point in the pipeline as an optimization barrier.
Operations before and after the barrier are optimized separately. Does
not execute anything until `collect()` or `compute()` is called.

## Usage

``` r
# S3 method for class 'tbl_gpu'
collapse(x, ...)
```

## Arguments

- x:

  A `tbl_gpu` object.

- ...:

  Additional arguments (unused).

## Value

A `tbl_gpu` with a barrier marker in its lazy_ops.

## Details

Use `collapse()` when you want to prevent certain optimizations from
crossing a boundary (e.g., prevent filter pushdown past a certain
point).

In eager mode, `collapse()` is a no-op.
