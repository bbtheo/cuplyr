# Bind multiple data frames/tables by column

Combines objects horizontally by adding columns. For tbl_gpu objects,
operations are performed on the GPU. For other objects, delegates to
dplyr.

## Usage

``` r
bind_cols(
  ...,
  .name_repair = c("unique", "universal", "check_unique", "minimal")
)
```

## Arguments

- ...:

  Objects to bind (tbl_gpu, data.frame, or a list of these)

- .name_repair:

  How to handle duplicate column names

## Value

Combined data frame or tbl_gpu
