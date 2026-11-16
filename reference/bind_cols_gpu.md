# GPU bind_cols implementation

Sizes follow vctrs' rules
([`vctrs::vec_size_common()`](https://vctrs.r-lib.org/reference/vec_size.html)):
inputs of size 1 recycle, every other size must match, and an input with
no columns contributes its size only. A lazy input is computed once, so
its size is the post-op row count.

## Usage

``` r
bind_cols_gpu(dots, .name_repair = "unique", call = rlang::caller_env())
```

## Arguments

- dots:

  Inputs as flattened by `bind_cols_inputs()`: tbl_gpu, data frames and
  vectors (a vector becomes one column, named by its element name)

- .name_repair:

  Name repair method

- call:

  The user's call, for errors

## Value

A tbl_gpu
