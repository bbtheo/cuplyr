# Bind multiple data frames/tables by row

Combines objects vertically by stacking rows. For tbl_gpu objects,
operations are performed on the GPU. For other objects, delegates to
dplyr.

## Usage

``` r
bind_rows(..., .id = NULL)
```

## Arguments

- ...:

  Objects to bind (tbl_gpu, data.frame, or a list of these)

- .id:

  Optional column name to identify source tables

## Value

Combined data frame or tbl_gpu
