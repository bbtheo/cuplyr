# Complex Data Analysis with cuplyr

## Introduction

This vignette demonstrates how to use cuplyr for real-world data
analysis workflows. We’ll cover multi-step analytical pipelines,
branching analyses, working with different data types, and patterns for
handling large datasets efficiently.

``` r

library(cuplyr)
library(dplyr)
```

## Building Analytical Workflows

Real analysis rarely consists of a single operation. Let’s walk through
progressively more complex workflows.

### Example Dataset

We’ll create a synthetic sales dataset to demonstrate these patterns:

``` r

set.seed(42)
n <- 10000

sales_data <- data.frame(
 transaction_id = seq_len(n),
 date = as.Date("2023-01-01") + sample(0:364, n, replace = TRUE),
 customer_id = sample(1:500, n, replace = TRUE),
 product_category = sample(c("Electronics", "Clothing", "Food", "Home", "Sports"),
                           n, replace = TRUE),
 region = sample(c("North", "South", "East", "West"), n, replace = TRUE),
 quantity = sample(1:10, n, replace = TRUE),
 unit_price = round(runif(n, 5, 500), 2),
 discount_pct = sample(c(0, 5, 10, 15, 20), n, replace = TRUE,
                       prob = c(0.5, 0.2, 0.15, 0.1, 0.05))
)

# Transfer to GPU
gpu_sales <- tbl_gpu(sales_data, lazy = TRUE)
```

### Multi-Step Revenue Analysis

Let’s compute revenue with discounts applied, then analyze by multiple
dimensions:

``` r

revenue_analysis <- gpu_sales |>
 # Step 1: Calculate derived metrics
 mutate(
   gross_amount = quantity * unit_price,
   discount_amount = gross_amount * discount_pct / 100,
   net_revenue = gross_amount - discount_amount
 ) |>
 # Step 2: Filter to significant transactions
 filter(net_revenue > 50) |>
 # Step 3: Aggregate by category and region
 group_by(product_category, region) |>
 summarise(
   total_revenue = sum(net_revenue),
   total_transactions = n(),
   avg_transaction = mean(net_revenue),
   total_quantity = sum(quantity)
 ) |>
 # Step 4: Sort by revenue
 arrange(desc(total_revenue)) |>
 collect()

head(revenue_analysis, 10)
```

### Calculating Metrics Across Groups

Compute group-level statistics and compare to overall metrics:

``` r

# Overall metrics
overall_stats <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 summarise(
   total_revenue = sum(revenue),
   avg_revenue = mean(revenue),
   total_orders = n()
 ) |>
 collect()

# Per-category metrics
category_stats <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(product_category) |>
 summarise(
   category_revenue = sum(revenue),
   category_avg = mean(revenue),
   category_orders = n()
 ) |>
 arrange(desc(category_revenue)) |>
 collect()

# Display results
overall_stats
category_stats
```

## Branching Analyses

Often you need to run multiple analyses from a common base. Use
[`compute()`](https://dplyr.tidyverse.org/reference/compute.html) to
materialize intermediate results and branch efficiently.

### Creating a Shared Base

``` r

# Prepare base data (filter and add computed columns)
base_analysis <- gpu_sales |>
 mutate(
   revenue = quantity * unit_price * (1 - discount_pct/100),
   is_high_value = unit_price > 200,
   is_bulk = quantity >= 5
 ) |>
 filter(revenue > 0) |>
 compute()  # Materialize on GPU

# Now branch into different analyses
```

### Branch 1: Regional Performance

``` r

regional_performance <- base_analysis |>
 group_by(region) |>
 summarise(
   revenue = sum(revenue),
   orders = n(),
   high_value_orders = sum(is_high_value),
   bulk_orders = sum(is_bulk)
 ) |>
 mutate(
   high_value_pct = high_value_orders * 100 / orders,
   bulk_pct = bulk_orders * 100 / orders
 ) |>
 arrange(desc(revenue)) |>
 collect()

regional_performance
```

### Branch 2: Category Deep-Dive

``` r

category_analysis <- base_analysis |>
 group_by(product_category) |>
 summarise(
   revenue = sum(revenue),
   avg_price = mean(unit_price),
   avg_quantity = mean(quantity),
   orders = n()
 ) |>
 mutate(
   revenue_per_order = revenue / orders
 ) |>
 arrange(desc(revenue_per_order)) |>
 collect()

category_analysis
```

### Branch 3: Discount Effectiveness

``` r

discount_analysis <- base_analysis |>
 group_by(discount_pct) |>
 summarise(
   orders = n(),
   total_revenue = sum(revenue),
   avg_quantity = mean(quantity)
 ) |>
 mutate(
   revenue_share = total_revenue * 100 / sum(total_revenue)
 ) |>
 arrange(discount_pct) |>
 collect()

discount_analysis
```

## Working with Dates

cuplyr supports Date and POSIXct columns. Here’s how to work with
temporal data.

### Date-Based Filtering

``` r

# Filter to Q1 2023
q1_sales <- gpu_sales |>
 filter(
   date >= as.Date("2023-01-01"),
   date < as.Date("2023-04-01")
 ) |>
 collect()

nrow(q1_sales)
```

### Extracting Date Components

For date component extraction (year, month, day), compute these in R
before transferring or use integer arithmetic:

``` r

# Add date components in R, then transfer
sales_with_dates <- sales_data |>
 mutate(
   month = as.integer(format(date, "%m")),
   quarter = ceiling(month / 3),
   day_of_week = as.integer(format(date, "%u"))  # 1=Monday, 7=Sunday
 )

# Now analyze on GPU
monthly_trend <- tbl_gpu(sales_with_dates, lazy = TRUE) |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(month) |>
 summarise(
   monthly_revenue = sum(revenue),
   orders = n()
 ) |>
 arrange(month) |>
 collect()

monthly_trend
```

### Quarterly Analysis

``` r

quarterly_analysis <- tbl_gpu(sales_with_dates, lazy = TRUE) |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(quarter, region) |>
 summarise(
   revenue = sum(revenue),
   orders = n()
 ) |>
 arrange(quarter, desc(revenue)) |>
 collect()

quarterly_analysis
```

## Segmentation Analysis

### Customer Segmentation by Purchase Behavior

``` r

customer_segments <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(customer_id) |>
 summarise(
   total_spent = sum(revenue),
   order_count = n(),
   avg_order_value = mean(revenue),
   total_items = sum(quantity)
 ) |>
 collect()

# Segment in R (for complex logic)
customer_segments <- customer_segments |>
 mutate(
   segment = case_when(
     total_spent > 5000 & order_count > 20 ~ "VIP",
     total_spent > 2000 | order_count > 10 ~ "Regular",
     order_count > 5 ~ "Occasional",
     TRUE ~ "New"
   )
 )

# Summary by segment
table(customer_segments$segment)
```

### Product Performance Tiers

``` r

category_performance <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(product_category) |>
 summarise(
   revenue = sum(revenue),
   units_sold = sum(quantity),
   avg_price = mean(unit_price),
   transactions = n()
 ) |>
 mutate(
   revenue_per_unit = revenue / units_sold
 ) |>
 arrange(desc(revenue)) |>
 collect()

category_performance
```

## Comparative Analysis

### Before/After Comparison

Compare metrics across time periods:

``` r

# Add period indicator in R
sales_with_period <- sales_data |>
 mutate(
   period = ifelse(date < as.Date("2023-07-01"), "H1", "H2")
 )

period_comparison <- tbl_gpu(sales_with_period, lazy = TRUE) |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(period, product_category) |>
 summarise(
   revenue = sum(revenue),
   orders = n(),
   avg_order = mean(revenue)
 ) |>
 arrange(product_category, period) |>
 collect()

period_comparison
```

### Region-to-Region Comparison

``` r

region_metrics <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(region) |>
 summarise(
   total_revenue = sum(revenue),
   total_orders = n(),
   avg_order_value = mean(revenue),
   avg_quantity = mean(quantity),
   avg_discount = mean(discount_pct)
 ) |>
 collect()

# Calculate indices relative to mean
overall_avg <- mean(region_metrics$total_revenue)
region_metrics$revenue_index <- region_metrics$total_revenue / overall_avg * 100

region_metrics
```

## Handling Large Datasets

### Memory-Efficient Patterns

When working with very large data, minimize memory usage:

``` r

# Pattern 1: Filter early, select only needed columns
result <- tbl_gpu(huge_data, lazy = TRUE) |>
 filter(status == "active") |>      # Reduce rows first
 select(id, date, amount, category) |>  # Then reduce columns
 mutate(amount_adj = amount * 1.1) |>
 group_by(category) |>
 summarise(total = sum(amount_adj)) |>
 collect()

# Pattern 2: Process in logical segments
base <- tbl_gpu(huge_data, lazy = TRUE) |>
 filter(date >= start_date, date < end_date) |>
 select(required_columns) |>
 compute()  # Checkpoint: clear intermediate memory

result <- base |>
 # Continue analysis on reduced data
 group_by(segment) |>
 summarise(metrics) |>
 collect()
```

### Monitoring Memory Usage

``` r

# Check GPU memory state
gpu_memory_state()

# After large operations, clean up
gpu_gc()
```

## Multi-Pass Analysis

Some analyses require multiple passes over data. Structure these
efficiently:

### Pass 1: Compute Aggregates

``` r

# First pass: get category totals
category_totals <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(product_category) |>
 summarise(category_total = sum(revenue)) |>
 collect()

category_totals
```

### Pass 2: Detailed Analysis with Context

``` r

# Add category totals back to data (in R)
sales_enriched <- sales_data |>
 left_join(category_totals, by = "product_category")

# Second pass: analyze with category context
regional_share <- tbl_gpu(sales_enriched, lazy = TRUE) |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(product_category, region) |>
 summarise(
   region_revenue = sum(revenue),
   category_total = max(category_total)  # Same value per group
 ) |>
 mutate(
   region_share = region_revenue * 100 / category_total
 ) |>
 arrange(product_category, desc(region_share)) |>
 collect()

regional_share
```

## Building Reusable Analysis Functions

Encapsulate common patterns in functions:

``` r

#' Calculate revenue metrics by grouping columns
#'
#' @param gpu_data A tbl_gpu object with sales data
#' @param ... Grouping columns
#' @return Collected data frame with revenue metrics
calculate_revenue_metrics <- function(gpu_data, ...) {
 gpu_data |>
   mutate(
     gross = quantity * unit_price,
     net = gross * (1 - discount_pct/100)
   ) |>
   group_by(...) |>
   summarise(
     gross_revenue = sum(gross),
     net_revenue = sum(net),
     total_discount = sum(gross) - sum(net),
     order_count = n(),
     avg_order = mean(net)
   ) |>
   mutate(
     discount_rate = total_discount * 100 / gross_revenue
   ) |>
   arrange(desc(net_revenue)) |>
   collect()
}

# Use the function
by_category <- calculate_revenue_metrics(gpu_sales, product_category)
by_region <- calculate_revenue_metrics(gpu_sales, region)
by_both <- calculate_revenue_metrics(gpu_sales, product_category, region)

head(by_both, 10)
```

### Parameterized Analysis

``` r

#' Analyze high-value transactions
#'
#' @param gpu_data A tbl_gpu object
#' @param min_value Minimum transaction value
#' @param group_col Column to group by
analyze_high_value <- function(gpu_data, min_value = 100, group_col) {
 group_col <- rlang::enquo(group_col)

 gpu_data |>
   mutate(transaction_value = quantity * unit_price * (1 - discount_pct/100)) |>
   filter(transaction_value >= min_value) |>
   group_by(!!group_col) |>
   summarise(
     high_value_revenue = sum(transaction_value),
     high_value_count = n(),
     avg_high_value = mean(transaction_value)
   ) |>
   arrange(desc(high_value_revenue)) |>
   collect()
}

# Different thresholds
high_value_100 <- analyze_high_value(gpu_sales, 100, product_category)
high_value_500 <- analyze_high_value(gpu_sales, 500, product_category)

high_value_100
high_value_500
```

## Combining GPU and CPU Processing

Some operations are better suited for R. Use a hybrid approach:

### GPU for Heavy Lifting

``` r

# Heavy aggregation on GPU
aggregated <- gpu_sales |>
 mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
 group_by(product_category, region, discount_pct) |>
 summarise(
   revenue = sum(revenue),
   orders = n(),
   quantity = sum(quantity)
 ) |>
 collect()

nrow(aggregated)  # Much smaller than original
```

### R for Complex Logic

``` r

# Complex categorization in R
final_analysis <- aggregated |>
 mutate(
   performance = case_when(
     revenue > quantile(revenue, 0.9) ~ "Top 10%",
     revenue > quantile(revenue, 0.5) ~ "Above Average",
     revenue > quantile(revenue, 0.25) ~ "Below Average",
     TRUE ~ "Bottom 25%"
   ),
   discount_tier = case_when(
     discount_pct == 0 ~ "Full Price",
     discount_pct <= 10 ~ "Light Discount",
     discount_pct <= 15 ~ "Moderate Discount",
     TRUE ~ "Heavy Discount"
   )
 ) |>
 group_by(performance, discount_tier) |>
 summarise(
   total_revenue = sum(revenue),
   segments = n(),
   .groups = "drop"
 )

final_analysis
```

## Error Handling in Analysis Pipelines

Robust pipelines handle edge cases:

``` r

safe_analysis <- function(data, min_rows = 100) {
 # Validate input
 if (!is.data.frame(data)) {
   stop("Input must be a data frame")
 }

 if (nrow(data) < min_rows) {
   warning("Dataset has fewer than ", min_rows, " rows")
 }

 # Check required columns
 required <- c("quantity", "unit_price", "discount_pct")
 missing <- setdiff(required, names(data))
 if (length(missing) > 0) {
   stop("Missing required columns: ", paste(missing, collapse = ", "))
 }

 # Perform analysis
 tryCatch({
   tbl_gpu(data, lazy = TRUE) |>
     mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
     filter(revenue > 0) |>
     summarise(
       total = sum(revenue),
       avg = mean(revenue),
       n = n()
     ) |>
     collect()
 }, error = function(e) {
   message("Analysis failed: ", e$message)
   data.frame(total = NA, avg = NA, n = 0)
 })
}

# Test with valid data
safe_analysis(sales_data)
```

## Summary Statistics Dashboard

Create a comprehensive summary:

``` r

create_sales_dashboard <- function(gpu_data) {
 # Overall metrics
 overall <- gpu_data |>
   mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
   summarise(
     total_revenue = sum(revenue),
     total_orders = n(),
     total_units = sum(quantity),
     avg_order_value = mean(revenue),
     avg_discount = mean(discount_pct)
   ) |>
   collect()

 # Top categories
 top_categories <- gpu_data |>
   mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
   group_by(product_category) |>
   summarise(revenue = sum(revenue), orders = n()) |>
   arrange(desc(revenue)) |>
   collect()

 # Regional breakdown
 regional <- gpu_data |>
   mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
   group_by(region) |>
   summarise(revenue = sum(revenue), orders = n()) |>
   arrange(desc(revenue)) |>
   collect()

 list(
   overall = overall,
   by_category = top_categories,
   by_region = regional,
   generated_at = Sys.time()
 )
}

dashboard <- create_sales_dashboard(gpu_sales)

cat("=== Sales Dashboard ===\n\n")
cat("Overall Metrics:\n")
print(dashboard$overall)
cat("\nBy Category:\n")
print(dashboard$by_category)
cat("\nBy Region:\n")
print(dashboard$by_region)
```

## Performance Comparison

Compare GPU vs CPU performance on your data:

``` r

library(bench)

# Benchmark: GPU vs CPU for aggregation
comparison <- bench::mark(
 gpu = {
   tbl_gpu(sales_data, lazy = TRUE) |>
     mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
     group_by(product_category, region) |>
     summarise(
       total = sum(revenue),
       avg = mean(revenue),
       n = n()
     ) |>
     collect()
 },
 cpu = {
   sales_data |>
     mutate(revenue = quantity * unit_price * (1 - discount_pct/100)) |>
     group_by(product_category, region) |>
     summarise(
       total = sum(revenue),
       avg = mean(revenue),
       n = n(),
       .groups = "drop"
     )
 },
 check = FALSE,
 min_iterations = 5
)

comparison
```

For small datasets like our example, CPU may be faster due to transfer
overhead. The GPU advantage becomes clear with millions of rows.

## Next Steps

- [`vignette("getting-started")`](https://bbtheo.github.io/cuplyr/articles/getting-started.md) -
  Basic cuplyr usage
- [`vignette("query-optimization")`](https://bbtheo.github.io/cuplyr/articles/query-optimization.md) -
  Understanding the optimizer
- Package documentation for specific functions

## Best Practices Summary

1.  **Filter and select early** to reduce data size
2.  **Use lazy mode** for complex multi-step pipelines
3.  **Use compute()** to checkpoint before branching
4.  **Prepare date components in R** before transfer
5.  **Use hybrid GPU/CPU** - GPU for aggregation, R for complex logic
6.  **Monitor memory** with
    [`gpu_memory_state()`](https://bbtheo.github.io/cuplyr/reference/gpu_memory_state.md)
    and
    [`gpu_gc()`](https://bbtheo.github.io/cuplyr/reference/gpu_gc.md)
7.  **Encapsulate patterns** in reusable functions
8.  **Handle errors gracefully** in production pipelines
