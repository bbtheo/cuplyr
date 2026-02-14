# Getting Started with cuplyr

## Introduction

cuplyr is a dplyr backend that executes data manipulation operations on
NVIDIA GPUs using the RAPIDS cuDF library. If you know dplyr, you
already know cuplyr - the same verbs work with the same syntax, just
faster on large datasets.

``` r

library(cuplyr)
library(dplyr)
```

## When to Use GPU Acceleration

GPU acceleration shines when:

- Your data has **millions of rows** (typically \>10M for clear
  benefits)
- Operations are computationally intensive (aggregations, sorting,
  filtering)
- You’re doing exploratory analysis with repeated queries

For small datasets (\<100K rows), the overhead of transferring data to
the GPU often outweighs the speedup. Stick with regular dplyr for those
cases.

## Basic Workflow

The typical cuplyr workflow has three steps:

1.  **Transfer data to GPU** with
    [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md)
2.  **Process with dplyr verbs** (filter, mutate, summarise, etc.)
3.  **Bring results back** with
    [`collect()`](https://dplyr.tidyverse.org/reference/compute.html)

### Creating a GPU Table

``` r

# Transfer a data frame to GPU memory
gpu_cars <- tbl_gpu(mtcars)

# Check that it's on the GPU
gpu_cars
```

The data is now in GPU memory. The original R data frame is unchanged.

### Using dplyr Verbs

Use familiar dplyr verbs exactly as you would normally:

``` r

# Filter rows
efficient_cars <- gpu_cars |>
  filter(mpg > 25)

# Add computed columns
with_kpl <- gpu_cars |>
  mutate(kpl = mpg * 0.425144)

# Select specific columns
subset <- gpu_cars |>
  select(mpg, cyl, hp)

# Sort rows
sorted <- gpu_cars |>
  arrange(desc(mpg))

# Group and summarize
by_cyl <- gpu_cars |>
  group_by(cyl) |>
  summarise(
    avg_mpg = mean(mpg),
    avg_hp = mean(hp),
    count = n()
  )
```

### Collecting Results

Operations build up on the GPU. Use
[`collect()`](https://dplyr.tidyverse.org/reference/compute.html) to
bring results back to R:

``` r

# Bring the grouped summary back to R
result <- by_cyl |>
  collect()

result
```

## A Complete Example

Here’s a realistic workflow analyzing the mtcars dataset:

``` r

analysis <- tbl_gpu(mtcars) |>
  # Keep only cars with reasonable fuel economy

  filter(mpg > 15) |>
  # Calculate power-to-weight ratio
  mutate(
    power_to_weight = hp / wt,
    is_efficient = mpg > 20
  ) |>
  # Group by number of cylinders
  group_by(cyl) |>
  # Calculate summary statistics
  summarise(
    n_cars = n(),
    avg_mpg = mean(mpg),
    avg_power_ratio = mean(power_to_weight),
    best_mpg = max(mpg)
  ) |>
  # Sort by average MPG descending
  arrange(desc(avg_mpg)) |>
  # Bring back to R
  collect()

analysis
```

## Chaining Multiple Operations

You can chain as many operations as needed. cuplyr handles the execution
efficiently:

``` r

# Complex pipeline
result <- tbl_gpu(mtcars) |>
  filter(cyl %in% c(4, 6)) |>
  mutate(
    efficiency_score = mpg / hp * 100,
    weight_class = wt * 1000
  ) |>
  filter(efficiency_score > 5) |>
  select(mpg, cyl, hp, efficiency_score, weight_class) |>
  arrange(desc(efficiency_score)) |>
  collect()

head(result)
```

## Supported Operations

### filter()

Filter rows based on conditions:

``` r

# Comparison operators
filter(x > 10)
filter(x >= 10)
filter(x < 10)
filter(x <= 10)
filter(x == 10)
filter(x != 10)

# Multiple conditions (combined with AND)
filter(x > 10, y < 5)

# Column-to-column comparisons
filter(col_a > col_b)
```

### select()

Select and reorder columns:

``` r

# By name
select(mpg, cyl, hp)

# Exclude columns
select(-disp, -drat)

# Rename while selecting
select(fuel_economy = mpg, cylinders = cyl)

# Helper functions
select(starts_with("d"))
select(ends_with("t"))
select(contains("p"))
```

### mutate()

Create or modify columns:

``` r

# Arithmetic operations
mutate(kpl = mpg * 0.425)
mutate(power_ratio = hp / wt)
mutate(hp_squared = hp ^ 2)

# Multiple columns at once
mutate(
  a = x + y,
  b = x - y,
  c = a * 2  # Can reference newly created columns
)

# Replace existing columns
mutate(mpg = mpg * 0.425)  # Converts to km/L in place
```

### arrange()

Sort rows:

``` r

# Ascending (default)
arrange(mpg)

# Descending
arrange(desc(mpg))

# Multiple columns
arrange(cyl, desc(mpg))
```

### group_by() + summarise()

Grouped aggregations:

``` r

# Available aggregation functions
group_by(cyl) |>
  summarise(
    total = sum(mpg),
    average = mean(mpg),
    minimum = min(mpg),
    maximum = max(mpg),
    count = n(),
    std_dev = sd(mpg),
    variance = var(mpg)
  )

# Multiple grouping columns
group_by(cyl, gear) |>
  summarise(avg_mpg = mean(mpg))
```

## Supported Column Types

cuplyr handles these R types:

| R Type             | GPU Type               | Notes           |
|--------------------|------------------------|-----------------|
| `numeric` (double) | FLOAT64                | Full precision  |
| `integer`          | INT32                  |                 |
| `character`        | STRING                 |                 |
| `logical`          | BOOL8                  |                 |
| `Date`             | TIMESTAMP_DAYS         |                 |
| `POSIXct`          | TIMESTAMP_MICROSECONDS |                 |
| `factor`           | INT32                  | Stored as codes |

## Execution Modes: Eager vs Lazy

cuplyr supports two execution modes:

### Eager Mode (Default)

Operations execute immediately. Simple and predictable:

``` r

# Default: eager execution
eager_tbl <- tbl_gpu(mtcars)
is_lazy(eager_tbl)  # FALSE

# Each operation runs immediately
result <- eager_tbl |>
  filter(mpg > 20) |>   # Executes now

  mutate(x = hp * 2) |> # Executes now
  collect()
```

### Lazy Mode

Operations are deferred and optimized before execution:

``` r

# Enable lazy execution
lazy_tbl <- tbl_gpu(mtcars, lazy = TRUE)
is_lazy(lazy_tbl)  # TRUE

# Operations build an AST without executing
pipeline <- lazy_tbl |>
  filter(mpg > 20) |>
  mutate(x = hp * 2) |>
  filter(cyl == 4)

# Check pending operations
has_pending_ops(pipeline)  # TRUE

# Execute everything at once (optimized)
result <- pipeline |> collect()
```

Lazy mode enables query optimization like filter pushdown and operation
fusion. See
[`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md)
for details.

## Controlling Execution

### compute()

Execute pending operations but keep data on GPU:

``` r

# Useful when branching into multiple analyses
base_data <- tbl_gpu(mtcars, lazy = TRUE) |>
  filter(mpg > 15) |>
  mutate(power_ratio = hp / wt) |>
  compute()  # Execute and materialize on GPU

# Now branch without re-running the filter/mutate
analysis_a <- base_data |> filter(cyl == 4) |> collect()
analysis_b <- base_data |> filter(cyl == 6) |> collect()
```

### show_query()

Inspect pending operations in lazy mode:

``` r

lazy_pipeline <- tbl_gpu(mtcars, lazy = TRUE) |>
  filter(mpg > 20) |>
  mutate(x = hp * 2)

show_query(lazy_pipeline)
```

## Memory Management

GPU memory is managed automatically through R’s garbage collector. When
a `tbl_gpu` object is no longer referenced, its GPU memory is freed.

Check GPU memory usage:

``` r

# Current memory state
gpu_memory_state()
```

Force garbage collection if needed:

``` r

# Clean up unreferenced GPU objects
gpu_gc()
```

## Best Practices

1.  **Filter early** - Reduce data size before expensive operations
2.  **Select only needed columns** - Less data to process
3.  **Use lazy mode for complex pipelines** - Enables optimization
4.  **Keep data on GPU** - Avoid repeated transfers with
    [`compute()`](https://dplyr.tidyverse.org/reference/compute.html)
5.  **Collect late** - Only transfer final results to R

``` r

# Good: filter and select early
result <- tbl_gpu(mtcars, lazy = TRUE) |>
  filter(mpg > 20) |>                    # Reduce rows first
  select(mpg, cyl, hp, wt) |>            # Reduce columns
  mutate(power_ratio = hp / wt) |>       # Now compute
  group_by(cyl) |>
  summarise(avg_ratio = mean(power_ratio)) |>
  collect()                               # Transfer only final result
```

## Error Handling

cuplyr validates operations and provides informative errors:

``` r

# Referencing non-existent column
try({
  tbl_gpu(mtcars) |>
    filter(nonexistent_column > 5)
})

# Type mismatches are caught
try({
  tbl_gpu(mtcars) |>
    filter(mpg > "not a number")
})
```

## Next Steps

- [`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md) -
  Deep dive into the AST optimizer
- `vignette("performance-tips")` - Maximizing GPU performance
- Package documentation:
  [`?tbl_gpu`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md),
  [`?compute`](https://dplyr.tidyverse.org/reference/compute.html),
  [`?gpu_memory_state`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)

## System Requirements

- NVIDIA GPU with CUDA support (Compute Capability \>= 6.0)
- CUDA Toolkit \>= 12.0
- RAPIDS cuDF \>= 25.12
- R \>= 4.3.0
