# cuplr: GPU-Accelerated dplyr Backend via libcudf

## Complete Developer Reference and Implementation Blueprint

**Version**: 1.0.0
**Target RAPIDS Version**: 25.12+ (stable), with guidance for 26.x
**Target CUDA Toolkit**: 12.x
**Target R Version**: 4.3+

---

## Table of Contents

1. [Scope & Audience](#1-scope--audience)
2. [Goals & Acceptance Criteria](#2-goals--acceptance-criteria)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Detailed Mapping Table](#4-detailed-mapping-table)
5. [Representations & Types](#5-representations--types)
6. [Binding Strategy & C++ Glue](#6-binding-strategy--c-glue)
7. [Build System & Packaging](#7-build-system--packaging)
8. [Minimal Working Prototype](#8-minimal-working-prototype)
9. [Lazy Translation & AST Approach](#9-lazy-translation--ast-approach)
10. [Testing & Validation](#10-testing--validation)
11. [Debugging, Logging & Observability](#11-debugging-logging--observability)
12. [Performance & Optimization](#12-performance--optimization)
13. [Interoperability](#13-interoperability)
14. [Packaging, Distribution & Licensing](#14-packaging-distribution--licensing)
15. [Example User Workflows](#15-example-user-workflows)
16. [Security & Safety](#16-security--safety)
17. [Maintenance & Migration](#17-maintenance--migration)
18. [Deliverables Checklist](#18-deliverables-checklist)
19. [Search Keywords & Resources](#19-search-keywords--resources)

---

## 1. Scope & Audience

### Target Audience

This guide is written for:
- **Experienced R package authors** familiar with S3/S4 OOP, `Rcpp` or `cpp11`, and R's build system
- **C++ developers** with CUDA toolchain experience
- **Data engineers** familiar with dplyr internals and backend implementations (e.g., dbplyr, dtplyr)

### Platform Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| OS | Ubuntu 22.04+ / Rocky Linux 8+ | RAPIDS officially supports these |
| GPU | NVIDIA Pascal+ (Compute Capability 6.0+) | Volta+ recommended for production |
| CUDA Toolkit | 12.0 - 12.6 | Must match RAPIDS build |
| Driver | 525.60.13+ | Check `nvidia-smi` |
| R | 4.3.0+ | For vctrs 0.6+ compatibility |
| GCC | 11.x - 13.x | Must match RAPIDS ABI |

### Version Detection Script

```bash
#!/bin/bash
# detect_rapids_version.sh - Query current RAPIDS/libcudf version

echo "=== System Check ==="
echo "CUDA Toolkit:"
nvcc --version 2>/dev/null || echo "nvcc not found"

echo -e "\nDriver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"

echo -e "\nGPU Compute Capability:"
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null

echo -e "\nRAIDS libcudf (conda):"
conda list libcudf 2>/dev/null | grep libcudf || echo "Not installed via conda"

echo -e "\nRAIDS libcudf (pip):"
pip show libcudf-cu12 2>/dev/null | grep Version || echo "Not installed via pip"

echo -e "\nR Version:"
R --version | head -1
```

### Compatibility Matrix

| RAPIDS Version | CUDA Toolkit | Python | Minimum Driver |
|---------------|--------------|--------|----------------|
| 25.12 (stable) | 12.0-12.5 | 3.10-3.12 | 525.60.13 |
| 25.10 (legacy) | 12.0-12.5 | 3.10-3.11 | 525.60.13 |
| 26.02 (nightly) | 12.0-12.6 | 3.10-3.12 | 535.54.03 |

### Migration Steps When Versions Change

1. **Check RAPIDS release notes** at https://docs.rapids.ai/notices/rsn/
2. **Update Dockerfile** base image tag
3. **Run configure script** to detect new include/lib paths
4. **Rebuild package** with `R CMD INSTALL --preclean`
5. **Run full test suite** to catch ABI breaks
6. **Update `DESCRIPTION`** SystemRequirements field

---

## 2. Goals & Acceptance Criteria

### Functional Goals

| Goal | Acceptance Criteria |
|------|---------------------|
| Idiomatic dplyr syntax | Users can write `tbl_gpu %>% filter(x > 10) %>% mutate(y = x * 2)` |
| GPU execution | Core operations execute entirely on GPU (verify with `nvprof`) |
| Correctness | Results match `dplyr` on reference datasets (tolerance: 1e-10 for floats) |
| Memory efficiency | No unnecessary CPU round-trips for chained operations |
| Reproducible builds | Dockerfile produces identical binaries across builds |

### Supported Verbs (MVP)

| Verb | Priority | libcudf Backing |
|------|----------|-----------------|
| `filter()` | P0 | `cudf::apply_boolean_mask` |
| `select()` | P0 | Column subsetting |
| `mutate()` | P0 | `cudf::transform` / binary ops |
| `arrange()` | P0 | `cudf::sort` |
| `group_by()` | P0 | `cudf::groupby::groupby` |
| `summarise()` | P0 | `cudf::groupby::aggregate` |
| `left_join()` | P1 | `cudf::left_join` |
| `inner_join()` | P1 | `cudf::inner_join` |
| `right_join()` | P2 | `cudf::left_join` (swap) |
| `distinct()` | P2 | `cudf::distinct` |
| `slice()` | P2 | `cudf::slice` |

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Filter 100M rows | < 50ms | `bench::mark()` |
| Group-by aggregate 10M rows, 1K groups | < 100ms | `bench::mark()` |
| Memory overhead vs raw cudf | < 10% | `nvidia-smi` monitoring |

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         R User Code                              │
│  tbl_gpu(df) %>% filter(x > 10) %>% group_by(g) %>% summarise() │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      R API Layer (S3)                            │
│  • tbl_gpu class constructor                                     │
│  • dplyr verb S3 methods (filter.tbl_gpu, mutate.tbl_gpu, ...)  │
│  • print/format methods via pillar                               │
│  • collect() / compute() materialization                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Translation Layer (AST)                        │
│  • Capture quosures from dplyr verbs                            │
│  • Build operation AST (lazy pipeline)                          │
│  • Query optimization (predicate pushdown, projection pruning)  │
│  • Lower AST nodes to libcudf operation sequence                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Native Layer (C++/Rcpp)                       │
│  • Rcpp::XPtr wrappers for cudf::table                          │
│  • GPU memory management (RAII via unique_ptr)                  │
│  • libcudf API calls (sort, filter, groupby, join)              │
│  • Arrow C Data Interface for zero-copy interop                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      libcudf (RAPIDS)                            │
│  • cudf::column, cudf::table, cudf::table_view                  │
│  • GPU kernels for data operations                              │
│  • RMM (RAPIDS Memory Manager) for allocations                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CUDA Runtime                             │
│                         NVIDIA GPU                               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### R Layer
- Export user-facing API: `tbl_gpu()`, `as_tbl_gpu()`, `collect()`, `compute()`
- Implement dplyr S3 generics for `tbl_gpu` class
- Use `vctrs` for type coercion and `pillar` for pretty printing
- Capture expressions as quosures using `rlang`

#### Translation Layer
- Convert R expressions to an internal AST representation
- Map R functions to libcudf equivalents
- Perform optimizations before execution
- Support both eager and lazy execution modes

#### Native Layer
- Wrap `std::unique_ptr<cudf::table>` in `Rcpp::XPtr`
- Implement C++ functions callable from R via `.Call()`
- Handle GPU memory lifecycle with RAII
- Provide Arrow C Data Interface export/import

---

## 4. Detailed Mapping Table

### dplyr Verb → libcudf Operation Mapping

| dplyr Verb | R Expression | libcudf Header | libcudf Function | Notes |
|------------|--------------|----------------|------------------|-------|
| `filter()` | `filter(x > 10)` | `cudf/stream_compaction.hpp` | `cudf::apply_boolean_mask()` | Create mask with `cudf::binary_operation` |
| `select()` | `select(a, b)` | `cudf/table/table_view.hpp` | `table_view::select()` | Column index subsetting |
| `mutate()` | `mutate(y = x * 2)` | `cudf/binaryop.hpp` | `cudf::binary_operation()` | Element-wise operations |
| `mutate()` | `mutate(y = sqrt(x))` | `cudf/unary.hpp` | `cudf::unary_operation()` | Unary transforms |
| `arrange()` | `arrange(x)` | `cudf/sorting.hpp` | `cudf::sort()` | Returns sorted table |
| `arrange()` | `arrange(desc(x))` | `cudf/sorting.hpp` | `cudf::sort()` with `order::DESCENDING` | |
| `group_by()` | `group_by(g)` | `cudf/groupby.hpp` | `cudf::groupby::groupby` | Store grouping columns |
| `summarise()` | `summarise(mean(x))` | `cudf/groupby.hpp` | `groupby.aggregate()` | With `make_mean_aggregation` |
| `left_join()` | `left_join(y, by="k")` | `cudf/join.hpp` | `cudf::left_join()` | Returns gather maps |
| `inner_join()` | `inner_join(y, by="k")` | `cudf/join.hpp` | `cudf::inner_join()` | |
| `distinct()` | `distinct()` | `cudf/stream_compaction.hpp` | `cudf::distinct()` | |
| `slice()` | `slice(1:100)` | `cudf/copying.hpp` | `cudf::slice()` | |
| `head()` | `head(n)` | `cudf/copying.hpp` | `cudf::slice()` | `{0, n}` |
| `rename()` | `rename(new = old)` | N/A | Metadata only | Update column names in R |

### Aggregation Function Mapping

| R Function | libcudf Aggregation | Header |
|------------|---------------------|--------|
| `sum()` | `make_sum_aggregation<>()` | `cudf/aggregation.hpp` |
| `mean()` | `make_mean_aggregation<>()` | `cudf/aggregation.hpp` |
| `min()` | `make_min_aggregation<>()` | `cudf/aggregation.hpp` |
| `max()` | `make_max_aggregation<>()` | `cudf/aggregation.hpp` |
| `n()` | `make_count_aggregation<>()` | `cudf/aggregation.hpp` |
| `sd()` | `make_std_aggregation<>()` | `cudf/aggregation.hpp` |
| `var()` | `make_variance_aggregation<>()` | `cudf/aggregation.hpp` |
| `median()` | `make_median_aggregation<>()` | `cudf/aggregation.hpp` |
| `first()` | `make_nth_element_aggregation<>(0)` | `cudf/aggregation.hpp` |
| `last()` | `make_nth_element_aggregation<>(-1)` | `cudf/aggregation.hpp` |

### Numeric Binary Operations

| R Operator | libcudf binary_operator |
|------------|-------------------------|
| `+` | `binary_operator::ADD` |
| `-` | `binary_operator::SUB` |
| `*` | `binary_operator::MUL` |
| `/` | `binary_operator::DIV` |
| `^` | `binary_operator::POW` |
| `%%` | `binary_operator::MOD` |
| `%/%` | `binary_operator::FLOOR_DIV` |

### Comparison Operations

| R Operator | libcudf binary_operator |
|------------|-------------------------|
| `==` | `binary_operator::EQUAL` |
| `!=` | `binary_operator::NOT_EQUAL` |
| `<` | `binary_operator::LESS` |
| `<=` | `binary_operator::LESS_EQUAL` |
| `>` | `binary_operator::GREATER` |
| `>=` | `binary_operator::GREATER_EQUAL` |

### Logical Operations

| R Operator | libcudf binary_operator |
|------------|-------------------------|
| `&` | `binary_operator::BITWISE_AND` (bool) |
| `\|` | `binary_operator::BITWISE_OR` (bool) |
| `!` | `cudf::unary_operation` with `unary_operator::NOT` |

### String Operations

| R Function | libcudf Header | libcudf Function |
|------------|----------------|------------------|
| `str_length()` | `cudf/strings/attributes.hpp` | `cudf::strings::count_characters()` |
| `str_sub()` | `cudf/strings/substring.hpp` | `cudf::strings::slice_strings()` |
| `str_to_lower()` | `cudf/strings/case.hpp` | `cudf::strings::to_lower()` |
| `str_to_upper()` | `cudf/strings/case.hpp` | `cudf::strings::to_upper()` |
| `str_detect()` | `cudf/strings/contains.hpp` | `cudf::strings::contains_re()` |
| `str_replace()` | `cudf/strings/replace.hpp` | `cudf::strings::replace()` |
| `str_c()` | `cudf/strings/combine.hpp` | `cudf::strings::concatenate()` |

### Window Functions (Future)

| R Function | libcudf Function | Notes |
|------------|------------------|-------|
| `lag()` | `cudf::groupby::shift()` | Negative offset |
| `lead()` | `cudf::groupby::shift()` | Positive offset |
| `cumsum()` | `cudf::groupby::scan()` | `make_sum_aggregation` |
| `row_number()` | `cudf::rank()` | `rank_method::FIRST` |

---

## 5. Representations & Types

### The `tbl_gpu` S3 Class

```r
# Class structure
# A tbl_gpu object contains:
# - ptr: Rcpp::XPtr to cudf::table (or NULL if lazy)
# - schema: list(names = character(), types = character())
# - lazy_ops: list of unevaluated operations (AST nodes)
# - groups: character vector of grouping column names

new_tbl_gpu <- function(ptr = NULL, schema = list(), lazy_ops = list(), groups = character()) {
structure(
list(
  ptr = ptr,
  schema = schema,
  lazy_ops = lazy_ops,
  groups = groups
),
class = c("tbl_gpu", "tbl_lazy", "tbl")
)
}

# Validator
validate_tbl_gpu <- function(x) {
stopifnot(
is.list(x),
inherits(x, "tbl_gpu"),
is.list(x$schema),
is.character(x$schema$names),
is.character(x$schema$types),
is.list(x$lazy_ops),
is.character(x$groups)
)
x
}

# Constructor (user-facing)
tbl_gpu <- function(data, ...) {
UseMethod("tbl_gpu")
}

tbl_gpu.data.frame <- function(data, ...) {
# Transfer to GPU
ptr <- .Call(`_cuplr_df_to_gpu`, data)
schema <- list(
names = names(data),
types = vapply(data, function(col) gpu_type_from_r(col), character(1))
)
new_tbl_gpu(ptr = ptr, schema = schema)
}

tbl_gpu.tbl_gpu <- function(data, ...) {
data
}
```

### Type Mapping: R ↔ cudf

| R Type | vctrs Type | cudf::type_id | Notes |
|--------|------------|---------------|-------|
| `logical` | `logical()` | `BOOL8` | |
| `integer` | `integer()` | `INT32` | |
| `double` | `double()` | `FLOAT64` | |
| `character` | `character()` | `STRING` | UTF-8 encoded |
| `Date` | `new_date()` | `TIMESTAMP_DAYS` | |
| `POSIXct` | `new_datetime()` | `TIMESTAMP_MICROSECONDS` | |
| `factor` | `factor()` | `DICTIONARY32` | |
| `integer64` | `bit64::integer64()` | `INT64` | Requires bit64 |

### GPU Type Detection

```r
gpu_type_from_r <- function(x) {
if (is.logical(x)) return("BOOL8")
if (is.integer(x)) return("INT32")
if (is.double(x)) {
if (inherits(x, "Date")) return("TIMESTAMP_DAYS")
if (inherits(x, "POSIXct")) return("TIMESTAMP_MICROSECONDS")
return("FLOAT64")
}
if (is.character(x)) return("STRING")
if (is.factor(x)) return("DICTIONARY32")
if (inherits(x, "integer64")) return("INT64")
stop("Unsupported type: ", typeof(x))
}

r_type_from_gpu <- function(gpu_type) {
switch(gpu_type,
"BOOL8" = logical(),
"INT32" = integer(),
"INT64" = bit64::integer64(),
"FLOAT32" = double(),
"FLOAT64" = double(),
"STRING" = character(),
"TIMESTAMP_DAYS" = vctrs::new_date(),
"TIMESTAMP_MICROSECONDS" = vctrs::new_datetime(),
"DICTIONARY32" = factor(),
stop("Unsupported GPU type: ", gpu_type)
)
}
```

### NA/NULL Handling

libcudf uses validity bitmasks (Arrow-style) where a 0 bit indicates NULL.

```cpp
// C++ side: Check for nulls
bool has_nulls = column.has_nulls();
cudf::size_type null_count = column.null_count();

// R side: Convert NA handling
// R's NA is automatically mapped to cudf NULL via our transfer functions
```

**Semantic differences to document:**
- R's `NA` in logical → cudf NULL (not a third value)
- R's `NA_integer_` → cudf NULL in INT32 column
- R's `NA_real_` (NaN) → **Preserved as NaN**, separate from NULL
- R's `NA_character_` → cudf NULL in STRING column

### Type Promotion Rules

Follow R's type promotion, implemented at transfer time:

```r
# Promotion hierarchy (lowest to highest)
# logical < integer < double
# integer64 is separate branch

promote_types <- function(type1, type2) {
hierarchy <- c("BOOL8" = 1, "INT32" = 2, "INT64" = 3, "FLOAT64" = 4)
if (hierarchy[type1] >= hierarchy[type2]) type1 else type2
}
```

