# Query Optimization in cuplyr

## Introduction

When you use cuplyr in lazy mode, operations aren’t executed
immediately. Instead, they build an Abstract Syntax Tree (AST) that
represents your query. Before execution, the AST passes through an
optimizer that rewrites the query for better performance.

This vignette explains:

1.  How the AST represents your operations
2.  The six optimization passes applied to your queries
3.  How filter pushdown works in detail
4.  When and why optimization matters

``` r

library(cuplyr)
library(dplyr)
```

## The Abstract Syntax Tree (AST)

When you write a dplyr pipeline in lazy mode, each operation creates a
node in the AST. The tree grows from bottom (source data) to top (final
operation).

``` r

# Create a lazy pipeline
pipeline <- tbl_gpu(mtcars, lazy = TRUE) |>
  filter(mpg > 20) |>
  mutate(power_ratio = hp / wt) |>
  select(mpg, cyl, power_ratio) |>
  arrange(desc(mpg))

# View the AST
show_query(pipeline)
```

### AST Node Types

Each dplyr verb creates a specific node type:

| Verb | AST Node | Description |
|----|----|----|
| [`tbl_gpu()`](https://bbtheo.github.io/cuplyr/reference/tbl_gpu.md) | `source` | Leaf node with schema metadata |
| [`filter()`](https://dplyr.tidyverse.org/reference/filter.html) | `filter` | Row filtering predicates |
| [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) | `mutate` | Column transformations |
| [`select()`](https://dplyr.tidyverse.org/reference/select.html) | `select` | Column projection |
| [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) | `arrange` | Sort specification |
| [`group_by()`](https://dplyr.tidyverse.org/reference/group_by.html) | `group_by` | Grouping metadata |
| [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) | `summarise` | Aggregation operations |
| [`head()`](https://rdrr.io/r/utils/head.html) | `head` | Row limit |
| [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) | `barrier` | Optimization fence |

### How the AST Grows

Each operation wraps the previous one:

``` r

# This pipeline:
tbl_gpu(mtcars, lazy = TRUE) |>
  filter(mpg > 20) |>
  mutate(x = hp * 2)

# Creates this AST structure:
# mutate[x]
#   filter[1 predicates]
#     source[11 cols]
```

The source node is at the bottom. Operations stack on top. When
executed, the tree is traversed from bottom to top.

## The Optimization Pipeline

Before execution, the AST passes through six optimization passes in
order:

1.  **Projection Pruning** - Remove unused columns early
2.  **Mutate Fusion** - Combine consecutive mutates
3.  **Dead Column Pruning** - Remove unused computed columns
4.  **Filter Pushdown** - Move filters closer to the source
5.  **Filter Reordering** - Cheapest filters first
6.  **Filter Fusion** - Combine simple filters into one kernel

### Why Order Matters

The passes are carefully ordered:

- Projection pruning runs first to reduce data width
- Mutate fusion happens before filter pushdown to create larger blocks
- Dead column pruning cleans up after projection changes
- Filter passes run last since earlier passes may create new
  opportunities

## Pass 1: Projection Pruning

This pass pushes column selections down the tree to reduce data width
early.

### Before Optimization

``` r

# User writes:
tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(power_ratio = hp / wt) |>
  select(mpg, power_ratio) |>
  collect()

# Unoptimized AST processes all 11 columns through mutate
```

### After Optimization

The optimizer inserts a `select` node below the `mutate` to fetch only
the columns needed (`mpg`, `hp`, `wt`):

``` r

# Optimized execution order:
# 1. Select only mpg, hp, wt from source
# 2. Compute power_ratio = hp / wt
# 3. Select mpg, power_ratio for output

# Benefit: 11 columns -> 3 columns early
```

### Impact

On wide tables, projection pruning dramatically reduces memory
bandwidth. Instead of copying all columns through every operation, only
needed columns flow through the pipeline.

## Pass 2: Mutate Fusion

Consecutive
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) calls
are combined when safe.

### Before Optimization

``` r

# User writes (or code generates):
tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(a = hp + 10) |>
  mutate(b = wt * 2) |>
  mutate(c = a + b) |>
  collect()

# Three separate mutate nodes = three intermediate tables
```

### After Optimization

``` r

# Fused into single mutate:
# mutate(a = hp + 10, b = wt * 2, c = a + b)

# One intermediate table instead of three
```

### Fusion Guards

Fusion is blocked when:

- Combined expressions exceed 8 (memory bandwidth tradeoff)
- Circular dependencies exist
- Too many intermediate columns are reused (\>4 intermediates or \>3
  uses each)

### Dependency Handling

When fusing creates dependencies, expressions are topologically sorted:

``` r

# Input expressions:
#   c = a + b
#   a = hp + 10
#   b = wt * 2

# After topological sort:
#   a = hp + 10    (no deps)
#   b = wt * 2     (no deps)
#   c = a + b      (depends on a, b)
```

## Pass 3: Dead Column Pruning

After projection pushdown and mutate fusion, some computed columns may
no longer be used. This pass removes them.

### Example

``` r

# User writes:
tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(
    a = hp + 10,
    b = wt * 2,
    c = mpg * 2
  ) |>
  select(mpg, a) |>  # Only using 'a', not 'b' or 'c'
  collect()

# After dead column pruning:
# mutate(a = hp + 10) only - 'b' and 'c' are never computed
```

This pass walks backward through the AST, tracking which columns are
actually required by downstream operations.

## Pass 4: Filter Pushdown

**Filter pushdown** is one of the most impactful optimizations. It moves
[`filter()`](https://dplyr.tidyverse.org/reference/filter.html)
operations closer to the data source, reducing the number of rows
processed by subsequent operations.

### The Core Principle

Fewer rows = faster everything. By filtering early:

- Less data flows through mutate computations
- Less data to sort in arrange
- Less memory bandwidth consumed

### When Pushdown Happens

A filter can be pushed below a `mutate` when the filter’s predicate
columns are **not** produced by that mutate.

#### Can Push

``` r

# This filter can be pushed down:
pipeline <- tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(power_ratio = hp / wt) |>
  filter(mpg > 20) |>  # mpg exists before mutate
  collect()

# Optimized order:
# 1. filter(mpg > 20) - removes rows first
# 2. mutate(power_ratio) - computed on fewer rows
```

#### Cannot Push

``` r

# This filter cannot be pushed:
pipeline <- tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(power_ratio = hp / wt) |>
  filter(power_ratio > 50) |>  # power_ratio created by mutate
  collect()

# Must stay in order:
# 1. mutate(power_ratio) - creates the column
# 2. filter(power_ratio > 50) - uses it
```

### Pushdown Across Select

Filters can also push below
[`select()`](https://dplyr.tidyverse.org/reference/select.html) if the
select keeps the predicate columns:

``` r

# Original:
tbl_gpu(data, lazy = TRUE) |>
  select(a, b, c) |>
  filter(a > 10)

# If select keeps 'a', filter pushes below:
# filter(a > 10)
# select(a, b, c)
```

### The Algorithm

The filter pushdown algorithm:

1.  Recursively process the AST from root to leaves
2.  When encountering a `filter` node:
    - Extract the columns used in predicates
    - Check if the input is a `mutate` node
    - If mutate’s output columns don’t overlap with predicate columns,
      swap them
3.  Repeat until no more pushdowns are possible

``` r

# Implementation sketch (from optimizer.R):
push_down_filters <- function(ast) {
  # Get columns used in filter predicates
  filter_cols <- get_predicate_columns(ast$predicates)

  if (ast$input$type == "mutate") {
    mutate_outputs <- get_output_columns(ast$input$expressions)

    # Can push if no overlap
    if (length(intersect(filter_cols, mutate_outputs)) == 0) {
      # Swap: filter goes below mutate
      new_filter <- ast_filter(ast$input$input, ast$predicates)
      ast$input$input <- new_filter
      return(ast$input)  # mutate is now on top
    }
  }

  ast
}
```

### Real-World Impact

Consider a 100-million row dataset where `filter(x > threshold)` keeps
10% of rows:

| Without Pushdown                    | With Pushdown                |
|-------------------------------------|------------------------------|
| mutate: 100M rows                   | filter: 100M -\> 10M rows    |
| filter: 100M -\> 10M rows           | mutate: 10M rows             |
| **Total work: 110M row-operations** | **Total work: 110M row-ops** |

Wait - same total? Not quite. The critical difference is **memory**:

- **Without pushdown**: mutate allocates 100M-row output, then filter
- **With pushdown**: filter first (minimal allocation), then mutate
  allocates 10M-row output

The memory savings compound through the pipeline. With multiple mutates
and filters, pushdown can reduce peak memory by 10x or more.

## Pass 5: Filter Reordering

When multiple filter predicates exist, they’re reordered by estimated
cost.

### Cost Model

| Predicate Type              | Estimated Cost | Rationale                 |
|-----------------------------|----------------|---------------------------|
| Scalar comparison (`x > 5`) | 1              | Single comparison per row |
| Column comparison (`x > y`) | 2              | Two memory reads per row  |

### Example

``` r

# User writes:
filter(expensive_col > other_col, cheap_col > 10)

# Reordered to:
filter(cheap_col > 10, expensive_col > other_col)
```

By running cheaper filters first, more rows are eliminated before
expensive comparisons.

### Safety Check

Non-deterministic predicates are not reordered to preserve correctness.

## Pass 6: Filter Fusion

Simple scalar predicates can be fused into a single GPU kernel call.

### Before Fusion

``` r

# Multiple filter conditions:
filter(a > 10, b < 20, c == 5)

# Without fusion: three separate filter kernel calls
```

### After Fusion

``` r

# Fused: single kernel with AND-mask

# One kernel computes:
# mask = (a > 10) & (b < 20) & (c == 5)
# apply_boolean_mask(data, mask)
```

### Fusion Criteria

Filters are fused when ALL conditions are met:

1.  All predicates are simple scalar comparisons (not column-to-column)
2.  Operators are basic comparisons: `==`, `!=`, `>`, `>=`, `<`, `<=`
3.  At most 4 predicates (GPU register limits)

Column-to-column comparisons and complex expressions remain separate.

## Optimization Barriers

Some operations act as **barriers** that prevent optimization across
them:

- [`arrange()`](https://dplyr.tidyverse.org/reference/arrange.html) -
  Sort order must be respected
- [`head()`](https://rdrr.io/r/utils/head.html) - Row limit must apply
  at specific point
- [`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html) -
  Aggregation changes row count
- [`collapse()`](https://dplyr.tidyverse.org/reference/compute.html) -
  Explicit user barrier

### Why Barriers Exist

Consider:

``` r

tbl_gpu(data, lazy = TRUE) |>
  arrange(x) |>           # Barrier: establishes order
  mutate(rank = row_number()) |>  # Depends on order
  filter(rank <= 10)      # Cannot push past arrange!
```

Pushing the filter past `arrange` would change which rows remain.

### Segment Optimization

The optimizer handles barriers by:

1.  Splitting the AST at barrier points
2.  Optimizing each segment independently
3.  Reconnecting the optimized segments

``` r

# Pipeline with barrier:
tbl_gpu(data, lazy = TRUE) |>
  filter(a > 10) |>
  mutate(b = a + 1) |>
  arrange(b) |>           # Barrier here
  filter(c > 5) |>
  mutate(d = c + 1)

# Optimized as two segments:
# Segment 1: source -> filter(a>10) -> mutate(b)
# Segment 2: arrange(b) -> filter(c>5) -> mutate(d)
```

## Inspecting Optimization

### show_query()

View the pending AST before optimization:

``` r

pipeline <- tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(a = hp + 10) |>
  filter(mpg > 20) |>
  mutate(b = a + wt)

show_query(pipeline)
```

### Understanding the Output

The [`show_query()`](https://dplyr.tidyverse.org/reference/explain.html)
output shows:

- Node type and key metadata
- Tree structure via indentation
- AST depth (number of levels)
- Node count (total operations)

## Practical Examples

### Example 1: Filter Pushdown in Action

``` r

# Without lazy mode: operations execute in written order
eager_result <- tbl_gpu(mtcars) |>
  mutate(
    a = hp + 10,
    b = wt * 2,
    c = mpg * 3
  ) |>
  filter(cyl == 4) |>  # Filters after computing all mutates
  collect()

# With lazy mode: filter pushes down
lazy_result <- tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(
    a = hp + 10,
    b = wt * 2,
    c = mpg * 3
  ) |>
  filter(cyl == 4) |>  # Optimizer moves this before mutate
  collect()

# Same result, but lazy version filters first
identical(eager_result, lazy_result)
```

### Example 2: Complex Pipeline Optimization

``` r

# A realistic analytics query
result <- tbl_gpu(mtcars, lazy = TRUE) |>
  # Step 1: Feature engineering
  mutate(
    power_to_weight = hp / wt,
    fuel_efficiency = mpg / cyl,
    is_powerful = hp > 150
  ) |>
  # Step 2: Filter to interesting subset
  filter(
    cyl %in% c(4, 6),     # Can push down (cyl exists)
    mpg > 18              # Can push down (mpg exists)
  ) |>
  # Step 3: More features using filtered data
  mutate(
    efficiency_score = fuel_efficiency * power_to_weight
  ) |>
  # Step 4: Aggregate
  group_by(cyl) |>
  summarise(
    avg_score = mean(efficiency_score),
    count = n()
  ) |>
  # Step 5: Final ordering
  arrange(desc(avg_score)) |>
  collect()

result
```

In this example, the optimizer:

1.  Pushes `filter(cyl %in% c(4,6), mpg > 18)` below the first mutate
2.  Reorders the filter predicates (scalar comparisons first)
3.  Fuses the two filter predicates into one kernel
4.  Prunes any computed columns not used downstream

### Example 3: When Pushdown Doesn’t Help

``` r

# Filter depends on computed column - cannot push
result <- tbl_gpu(mtcars, lazy = TRUE) |>
  mutate(power_ratio = hp / wt) |>
  filter(power_ratio > 50) |>  # Must stay after mutate
  collect()

# This is fine - the optimizer recognizes the dependency
nrow(result)
```

## Performance Tips

### 1. Use Lazy Mode for Complex Pipelines

``` r

# Enable lazy mode for optimization benefits
tbl_gpu(data, lazy = TRUE) |>
  # ... complex pipeline ...
  collect()
```

### 2. Filter on Source Columns When Possible

``` r

# Good: filter uses source columns
tbl_gpu(data, lazy = TRUE) |>
  filter(status == "active", date > cutoff) |>  # Can push down
  mutate(computed = complex_calculation(a, b))

# Less optimal: filter uses computed column
tbl_gpu(data, lazy = TRUE) |>
  mutate(computed = complex_calculation(a, b)) |>
  filter(computed > threshold)  # Cannot push down
```

### 3. Let the Optimizer Reorder

Write filters in logical order. The optimizer will reorder by cost:

``` r

# Write for readability
filter(
  department == "Engineering",   # May be expensive
  is_active == TRUE,             # Cheap
  salary > 50000                 # Cheap
)

# Optimizer reorders to: is_active, salary, department
```

### 4. Use collapse() for Control

If you need to prevent optimization across a boundary:

``` r

tbl_gpu(data, lazy = TRUE) |>
  filter(x > 10) |>
  mutate(y = f(x)) |>
  collapse() |>            # Barrier: optimize above and below separately
  filter(z > 5) |>
  collect()
```

## Conclusion

cuplyr’s query optimizer automatically improves your pipelines through:

- **Projection pruning**: Only process needed columns
- **Mutate fusion**: Reduce intermediate allocations
- **Dead column pruning**: Skip unnecessary computations
- **Filter pushdown**: Process fewer rows earlier
- **Filter reordering**: Cheap filters first
- **Filter fusion**: Combine filters into single kernels

For most workloads, simply use lazy mode and write natural dplyr code.
The optimizer handles the rest.

``` r

# Just write natural dplyr:
result <- tbl_gpu(big_data, lazy = TRUE) |>
  mutate(derived = complex_expr) |>
  filter(source_col > threshold) |>  # Automatically pushed down
  group_by(category) |>
  summarise(total = sum(derived)) |>
  collect()
```

## Further Reading

- [`vignette("getting-started")`](https://bbtheo.github.io/cuplyr/articles/getting-started.md) -
  Basic cuplyr usage
- `?optimize_ast` - Internal optimizer documentation
- RAPIDS cuDF documentation for GPU-specific optimizations
