# Add .id column to identify source tables

Add .id column to identify source tables

## Usage

``` r
add_id_column(result, id_col_name, row_counts, source_names)
```

## Arguments

- result:

  The combined tbl_gpu

- id_col_name:

  Name for the .id column

- row_counts:

  Row count of each original input (zero-column ones too)

- source_names:

  Names/identifiers for each source

## Value

A tbl_gpu with .id column prepended
