---
name: implement-dplyr-verb
description: >
  Guide for implementing dplyr verbs (filter, mutate, select, etc.) for custom

  backends. Use when building dplyr-compatible table classes like database

  connections (dbplyr), data.table wrappers (dtplyr), Arrow tables, Spark
  DataFrames, or GPU tables. Covers S3 method dispatch, rlang expression
  parsing, NAMESPACE exports, Rcpp integration, and testing patterns.
---

# Implementing dplyr Verbs for Custom Backends

Build dplyr-compatible table classes by implementing S3 methods for dplyr generics.

## How dplyr Dispatch Works

dplyr verbs are S3 generics. When you call `filter(x, ...)`, R dispatches to:
- `filter.tbl_df` for tibbles
- `filter.tbl_lazy` for dbplyr
- `filter.your_class` for your custom class

Your job: implement `<verb>.your_class` methods.

## Minimal Backend Structure

A dplyr backend needs:

```r
# Constructor
new_my_tbl <- function(data, ...) {

structure(
  list(
    data = data,
    # ... backend-specific fields
  ),
  class = c("my_tbl", "list")
)
}

# Coercion from data.frame
my_tbl <- function(x) {
new_my_tbl(data = x)
}

# Coercion back to data.frame
#' @export
#' @importFrom dplyr collect
collect.my_tbl <- function(x, ...) {
as.data.frame(x$data)
}
```

## Implementing a Verb (Pure R)

### Step 1: Write the S3 Method

```r
#' @export
#' @importFrom dplyr filter
filter.my_tbl <- function(.data, ..., .preserve = FALSE) {
# 1. Capture expressions
dots <- rlang::enquos(...)
if (length(dots) == 0) return(.data)

# 2. Evaluate predicates against data
mask <- rlang::new_data_mask(rlang::as_environment(.data$data))
for (expr in dots) {
  result <- rlang::eval_tidy(expr, data = mask)
  # Combine predicates with AND
}

# 3. Apply filter
filtered_data <- .data$data[result, , drop = FALSE]

# 4. Return new object (preserving class)
new_my_tbl(data = filtered_data)
}
```

### Step 2: Update NAMESPACE

Add to `NAMESPACE` (or use roxygen2 tags):

```
# S3 method registration
S3method(filter,my_tbl)

# Import the generic
importFrom(dplyr,filter)
```

With roxygen2, these lines are generated from:
```r
#' @export
#' @importFrom dplyr filter
```

### Step 3: Write Tests

```r
test_that("filter() subsets rows", {
df <- data.frame(x = 1:5, y = letters[1:5])
my_df <- my_tbl(df)

result <- my_df |>
  dplyr::filter(x > 2) |>
  collect()

expect_equal(nrow(result), 3)
expect_equal(result$x, 3:5)
})

test_that("filter() handles multiple predicates", {
df <- data.frame(x = 1:10, y = rep(c("a", "b"), 5))
my_df <- my_tbl(df)

result <- my_df |>
  dplyr::filter(x > 3, y == "a") |>
  collect()

expect_equal(result$x, c(5, 7, 9))
})
```

## Implementing a Verb (with Rcpp/C++)

For performance-critical backends (GPU, databases with custom drivers), implement the core logic in C++.

### Step 1: C++ Implementation

Create `src/ops_filter.cpp`:

```cpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
SEXP backend_filter(SEXP data_ptr, IntegerVector mask) {
// 1. Get data from external pointer (if using XPtr)
// Rcpp::XPtr<YourDataType> ptr(data_ptr);

// 2. Validate inputs
if (mask.size() == 0) {
  return data_ptr;
}

// 3. Perform filter operation
// ... backend-specific logic ...

// 4. Return result (new XPtr or wrapped data)
return result;
}
```

### Step 2: R Wrapper

```r
#' @export
#' @importFrom dplyr filter
filter.my_tbl <- function(.data, ..., .preserve = FALSE
) {
dots <- rlang::enquos(...)
if (length(dots) == 0) return(.data)

# Parse expressions to backend representation
filter_spec <- parse_filter_exprs(dots, .data$schema)

# Call C++ implementation
new_ptr <- backend_filter(.data$ptr, filter_spec)

# Return new object
new_my_tbl(
  ptr = new_ptr,
  schema = .data$schema
)
}
```

### Step 3: Generate Rcpp Exports

After adding C++ functions, regenerate exports:

```r
Rcpp::compileAttributes()
# Or via devtools
devtools::document()
```
This updates:
- `src/RcppExports.cpp` - C++ wrapper functions
- `R/RcppExports.R` - R function declarations

## Expression Parsing with rlang

Most verbs need to parse user expressions. Common patterns:

### Capturing Expressions

```r
# Capture as quosures (preserves environment)
dots <- rlang::enquos(...)

# Get expression text (for error messages)
expr_text <- rlang::quo_text(expr)

# Get raw expression (for inspection)
raw_expr <- rlang::quo_get_expr(expr)
```

### Detecting Expression Types

```r
# Bare column name: filter(df, x)
if (is.symbol(raw_expr)) {
col_name <- as.character(raw_expr)
}

# Function call: filter(df, x > 5)
if (is.call(raw_expr)) {
fn_name <- as.character(raw_expr[[1]])  # ">"
lhs <- raw_expr[[2]]                     # x
rhs <- raw_expr[[3]]                     # 5
}

# Literal value
if (is.numeric(raw_expr) || is.character(raw_expr) || is.logical(raw_expr)) {
value <- raw_expr
}
```

### Evaluating Expressions

```r
# Against a data mask (standard tidyverse evaluation)
mask <- rlang::new_data_mask(rlang::as_environment(data))
result <- rlang::eval_tidy(expr, data = mask)

# With a custom pronoun for .data
mask$.data <- rlang::as_data_pronoun(data)
```

### Walking Expression Trees

```r
# Recursively process an expression
walk_expr <- function(expr, env) {
if (is.call(expr)) {
  fn <- as.character(expr[[1]])
  args <- lapply(expr[-1], walk_expr, env = env)
  # ... process function call
} else if (is.symbol(expr)) {
  # ... handle column reference
} else {
  # ... handle literal
}
}
```

## Column Index Handling

R uses 1-based indices; C/C++ uses 0-based:

```r
# R column name to 0-based index (for C++)
col_idx_cpp <- match(col_name, .data$schema$names) - 1L

# Validate column exists
if (is.na(col_idx_cpp)) {
rlang::abort(
  paste0("Column '", col_name, "' not found"),
  class = "my_pkg_column_error"
)
}
```

## Verb-Specific Patterns

### mutate()

Key considerations:
- New columns added to schema
- Existing columns can be overwritten
- Expression order matters (later expressions can reference earlier ones)

```r
#' @export
#' @importFrom dplyr mutate
mutate.my_tbl <- function(.data, ...) {
dots <- rlang::enquos(...)
if (length(dots) == 0) return(.data)

# Get or generate column names
names <- names(dots)
names <- ifelse(names == "", vapply(dots, rlang::quo_text, ""), names)

# Process each expression
new_schema <- .data$schema
for (i in seq_along(dots)) {
  col_name <- names[[i]]
  expr <- dots[[i]]

  # Evaluate or translate expression
  # Update schema
}

new_my_tbl(ptr = new_ptr, schema = new_schema)
}
```

### select()

Key considerations:
- Supports tidyselect helpers (starts_with, everything, etc.)
- Can rename columns: `select(df, new_name = old_name)`
- Column order in output matches selection order

```r
#' @export
#' @importFrom dplyr select
select.my_tbl <- function(.data, ...) {
# Use tidyselect for column selection
cols <- tidyselect::eval_select(
  rlang::expr(c(...)),
  data = rlang::set_names(seq_along(.data$schema$names), .data$schema$names)
)

col_indices <- unname(cols) - 1L  # 0-based for C++
new_names <- names(cols)

new_ptr <- backend_select(.data$ptr, col_indices)
new_my_tbl(
  ptr = new_ptr,
  schema = list(names = new_names, types = .data$schema$types[cols])
)
}
```

### arrange()
Key considerations:
- `desc()` for descending order
- Multiple columns = hierarchical sort
- NA handling (first or last)

```r
#' @export
#' @importFrom dplyr arrange
arrange.my_tbl <- function(.data, ..., .by_group = FALSE) {
dots <- rlang::enquos(...)
if (length(dots) == 0) return(.data)

# Parse each expression for column and direction
sort_spec <- lapply(dots, function(expr) {
  raw <- rlang::quo_get_expr(expr)
  if (is.call(raw) && as.character(raw[[1]]) == "desc") {
    list(col = as.character(raw[[2]]), desc = TRUE)
  } else {
    list(col = as.character(raw), desc = FALSE)
  }
})

# ... apply sort
}
```

### group_by() / summarise()

`group_by()` typically stores metadata; `summarise()` uses it:

```r
#' @export
#' @importFrom dplyr group_by
group_by.my_tbl <- function(.data, ..., .add = FALSE) {
dots <- rlang::enquos(...)
group_cols <- vapply(dots, function(q) {
  as.character(rlang::quo_get_expr(q))
}, character(1))

if (.add) {
  group_cols <- union(.data$groups, group_cols)
}

new_my_tbl(
  ptr = .data$ptr,
  schema = .data$schema,
  groups = group_cols
)
}

#' @export
#' @importFrom dplyr summarise
summarise.my_tbl <- function(.data, ..., .groups = NULL) {
dots <- rlang::enquos(...)
agg_names <- names(dots)

# Parse aggregation functions: mean(x), sum(y), n()
agg_spec <- lapply(dots, parse_aggregation)

# Apply groupby + aggregation
new_ptr <- backend_summarise(
  .data$ptr,
  group_cols = .data$groups,
  agg_spec = agg_spec
)

# Determine output grouping based on .groups
new_groups <- switch(.groups %||% "drop_last",
  drop_last = head(.data$groups, -1),
  drop = character(),
  keep = .data$groups,
  rowwise = stop(".groups = 'rowwise' not supported")
)

new_my_tbl(ptr = new_ptr, schema = new_schema, groups = new_groups)
}
```

## Lazy Evaluation Pattern

For backends that build query plans (SQL, Spark), defer execution:

```r
# Store operations as AST nodes
filter.my_lazy_tbl <- function(.data, ...) {
dots <- rlang::enquos(...)

new_my_lazy_tbl(
  ops = c(.data$ops, list(
    type = "filter",
    predicates = dots
  ))
)
}

# Execute on collect()
collect.my_lazy_tbl <- function(x, ...) {
plan <- optimize(x$ops)
execute(plan)
}
```

## Complete Verb Checklist

When implementing a verb:

- [ ] S3 method signature matches generic (`?dplyr::<verb>`)
- [ ] `@export` and `@importFrom dplyr <verb>` in roxygen
- [ ] Empty input returns `.data` unchanged
- [ ] Schema/metadata updated correctly
- [ ] Groups preserved (unless verb modifies grouping)
- [ ] Class preserved in return value
- [ ] Tests cover: basic case, edge cases, interaction with other verbs

## Common Mistakes

1. **Forgetting to import the generic**
 ```r
 # Wrong: creates a new generic instead of extending dplyr's
 filter.my_tbl <- function(.data, ...) { }

 # Right: import first
 #' @importFrom dplyr filter
 filter.my_tbl <- function(.data, ...) { }
 ```

2. **Losing class on return**
 ```r
 # Wrong: returns plain list
 filter.my_tbl <- function(.data, ...) {
   list(data = filtered)
 }

 # Right: use constructor
 filter.my_tbl <- function(.data, ...) {
   new_my_tbl(data = filtered)
 }
 ```

3. **Modifying input in place**
 ```r
 # Wrong: side effects
 filter.my_tbl <- function(.data, ...) {
   .data$data <- filtered  # Mutates input!
   .data
 }

 # Right: return new object
 filter.my_tbl <- function(.data, ...) {
   new_my_tbl(data = filtered)
 }
 ```

4. **Ignoring groups**
 ```r
 # Wrong: loses grouping information
 filter.my_tbl <- function(.data, ...) {
   new_my_tbl(data = filtered)
 }

 # Right: preserve groups
 filter.my_tbl <- function(.data, ...) {
   new_my_tbl(data = filtered, groups = .data$groups)
 }
 ```

## Testing Strategy

```r
# Test verb in isolation
test_that("filter() basic case", { ... })

# Test verb preserves groups
test_that("filter() preserves groups", {
result <- my_tbl(df) |>
  group_by(g) |>
  filter(x > 1)

expect_equal(group_vars(result), "g")
})

# Test verb chains
test_that("filter + select + mutate pipeline works", {
result <- my_tbl(df) |>
  filter(x > 0) |>
  select(x, y) |>
  mutate(z = x + 1) |>
  collect()

expect_named(result, c("x", "y", "z"))
})

# Test equivalence with dplyr
test_that("filter() matches dplyr behavior", {
df <- data.frame(x = 1:10, y = letters[1:10])

dplyr_result <- df |> dplyr::filter(x > 5)
my_result <- my_tbl(df) |> dplyr::filter(x > 5) |> collect()

expect_equal(my_result, dplyr_result, ignore_attr = TRUE)
})
```

## Reference

- [dplyr vignette: programming](https://dplyr.tidyverse.org/articles/programming.html)
- [rlang tidy evaluation](https://rlang.r-lib.org/reference/topic-data-mask.html)
- [dbplyr source](https://github.com/tidyverse/dbplyr) - SQL backend example
- [dtplyr source](https://github.com/tidyverse/dtplyr) - data.table backend example
