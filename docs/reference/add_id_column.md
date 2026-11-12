# Add .id column to identify source tables

Add .id column to identify source tables

## Usage

``` r
add_id_column(result, id_col_name, original_tables, source_names)
```

## Arguments

- result:

  The combined tbl_gpu

- id_col_name:

  Name for the .id column

- original_tables:

  List of original tables (for row counts)

- source_names:

  Names/identifiers for each source

## Value

A tbl_gpu with .id column prepended
