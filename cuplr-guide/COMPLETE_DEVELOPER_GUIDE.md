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

## 6. Binding Strategy & C++ Glue

### Recommended Approach: Rcpp with XPtr

We recommend **Rcpp** over cpp11 for this project because:
1. Better documented patterns for external pointers
2. Existing examples with CUDA integration
3. Mature finalizer support for GPU memory cleanup

### External Pointer Wrapper Pattern

```cpp
// src/gpu_table.hpp
#ifndef CUPLR_GPU_TABLE_HPP
#define CUPLR_GPU_TABLE_HPP

#include <Rcpp.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>

namespace cuplr {

// Wrap cudf::table in a shared_ptr for R interop
using GpuTablePtr = std::shared_ptr<cudf::table>;

// Custom destructor that ensures GPU cleanup
inline void release_gpu_table(GpuTablePtr* ptr) {
    if (ptr != nullptr) {
        // Reset triggers cudf::table destructor, freeing GPU memory
        ptr->reset();
        delete ptr;
    }
}

// Create XPtr with custom destructor
inline Rcpp::XPtr<GpuTablePtr> make_gpu_table_xptr(std::unique_ptr<cudf::table> tbl) {
    // Convert unique_ptr to shared_ptr for R ownership
    auto* sptr = new GpuTablePtr(std::move(tbl));
    return Rcpp::XPtr<GpuTablePtr>(sptr, true);  // true = register destructor
}

// Extract table_view from XPtr (non-owning view)
inline cudf::table_view get_table_view(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return (*xptr)->view();
}

// Get mutable table reference
inline cudf::table& get_table_ref(Rcpp::XPtr<GpuTablePtr> xptr) {
    if (!xptr || !(*xptr)) {
        Rcpp::stop("GPU table pointer is NULL");
    }
    return **xptr;
}

} // namespace cuplr

#endif // CUPLR_GPU_TABLE_HPP
```

### Data Transfer: R data.frame → GPU

```cpp
// src/transfer.cpp
#include "gpu_table.hpp"
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <Rcpp.h>

using namespace Rcpp;
using namespace cudf;

namespace cuplr {

// Create GPU column from R numeric vector
std::unique_ptr<column> numeric_to_gpu(NumericVector x) {
    size_type n = x.size();

    // Allocate device memory
    rmm::device_buffer data(n * sizeof(double),
                            rmm::cuda_stream_default,
                            rmm::mr::get_current_device_resource());

    // Copy from host to device
    cudaMemcpy(data.data(), &x[0], n * sizeof(double), cudaMemcpyHostToDevice);

    // Handle NAs by creating validity mask
    rmm::device_buffer null_mask;
    size_type null_count = 0;

    // Check for NAs
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);
    for (size_type i = 0; i < n; ++i) {
        if (NumericVector::is_na(x[i])) {
            // Clear bit i
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_default,
                                       rmm::mr::get_current_device_resource());
    }

    return std::make_unique<column>(
        data_type{type_id::FLOAT64},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R integer vector
std::unique_ptr<column> integer_to_gpu(IntegerVector x) {
    size_type n = x.size();

    rmm::device_buffer data(n * sizeof(int32_t),
                            rmm::cuda_stream_default,
                            rmm::mr::get_current_device_resource());

    cudaMemcpy(data.data(), &x[0], n * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Handle NAs
    rmm::device_buffer null_mask;
    size_type null_count = 0;
    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(n), 0xFF);

    for (size_type i = 0; i < n; ++i) {
        if (IntegerVector::is_na(x[i])) {
            validity[i / 8] &= ~(1 << (i % 8));
            null_count++;
        }
    }

    if (null_count > 0) {
        null_mask = rmm::device_buffer(validity.data(), validity.size(),
                                       rmm::cuda_stream_default,
                                       rmm::mr::get_current_device_resource());
    }

    return std::make_unique<column>(
        data_type{type_id::INT32},
        n,
        std::move(data),
        std::move(null_mask),
        null_count
    );
}

// Create GPU column from R character vector
std::unique_ptr<column> character_to_gpu(CharacterVector x) {
    size_type n = x.size();

    // Convert to std::vector<std::string> for cudf factory
    std::vector<std::string> strings(n);
    std::vector<bool> valids(n, true);

    for (size_type i = 0; i < n; ++i) {
        if (CharacterVector::is_na(x[i])) {
            valids[i] = false;
            strings[i] = "";
        } else {
            strings[i] = as<std::string>(x[i]);
        }
    }

    // Use cudf factory function
    auto host_span = cudf::host_span<std::string const>(strings.data(), strings.size());
    auto valid_span = cudf::host_span<bool const>(valids.data(), valids.size());

    return cudf::make_strings_column(host_span, valid_span);
}

} // namespace cuplr

// [[Rcpp::export]]
SEXP df_to_gpu(DataFrame df) {
    using namespace cuplr;

    int ncol = df.size();
    CharacterVector names = df.names();

    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(ncol);

    for (int i = 0; i < ncol; ++i) {
        SEXP col = df[i];

        switch (TYPEOF(col)) {
            case REALSXP:
                columns.push_back(numeric_to_gpu(col));
                break;
            case INTSXP:
                columns.push_back(integer_to_gpu(col));
                break;
            case STRSXP:
                columns.push_back(character_to_gpu(col));
                break;
            case LGLSXP:
                // Convert logical to integer then to BOOL8
                columns.push_back(integer_to_gpu(as<IntegerVector>(col)));
                break;
            default:
                Rcpp::stop("Unsupported column type at index %d", i);
        }
    }

    auto tbl = std::make_unique<cudf::table>(std::move(columns));
    return make_gpu_table_xptr(std::move(tbl));
}
```

### GPU → R data.frame Transfer

```cpp
// src/transfer.cpp (continued)

// [[Rcpp::export]]
DataFrame gpu_to_df(SEXP xptr, CharacterVector names) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    int ncol = view.num_columns();
    size_type nrow = view.num_rows();

    List result(ncol);

    for (int i = 0; i < ncol; ++i) {
        cudf::column_view col = view.column(i);
        cudf::type_id type = col.type().id();

        switch (type) {
            case cudf::type_id::FLOAT64: {
                NumericVector out(nrow);
                cudaMemcpy(&out[0], col.data<double>(),
                          nrow * sizeof(double), cudaMemcpyDeviceToHost);
                // Handle nulls
                if (col.has_nulls()) {
                    // Download validity mask and set NAs
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    cudaMemcpy(validity.data(), col.null_mask(),
                              validity.size(), cudaMemcpyDeviceToHost);
                    for (size_type j = 0; j < nrow; ++j) {
                        if (!((validity[j/8] >> (j%8)) & 1)) {
                            out[j] = NA_REAL;
                        }
                    }
                }
                result[i] = out;
                break;
            }
            case cudf::type_id::INT32: {
                IntegerVector out(nrow);
                cudaMemcpy(&out[0], col.data<int32_t>(),
                          nrow * sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (col.has_nulls()) {
                    std::vector<uint8_t> validity(bitmask_allocation_size_bytes(nrow));
                    cudaMemcpy(validity.data(), col.null_mask(),
                              validity.size(), cudaMemcpyDeviceToHost);
                    for (size_type j = 0; j < nrow; ++j) {
                        if (!((validity[j/8] >> (j%8)) & 1)) {
                            out[j] = NA_INTEGER;
                        }
                    }
                }
                result[i] = out;
                break;
            }
            case cudf::type_id::STRING: {
                // String columns require special handling
                auto str_col = cudf::strings_column_view(col);
                CharacterVector out(nrow);
                // Use cudf utility to get strings as host vector
                // This is simplified - real implementation would use
                // cudf::strings::detail functions
                result[i] = out;
                break;
            }
            default:
                Rcpp::stop("Unsupported column type for GPU->R transfer");
        }
    }

    result.names() = names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -nrow);

    return result;
}
```

### Implementing filter() in C++

```cpp
// src/filter.cpp
#include "gpu_table.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>

// [[Rcpp::export]]
SEXP gpu_filter_gt(SEXP xptr, int col_idx, double value) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Get the column to filter on (0-indexed)
    cudf::column_view filter_col = view.column(col_idx);

    // Create scalar for comparison
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    scalar->set_valid_async(true);
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(value);

    // Create boolean mask: col > value
    auto mask = cudf::binary_operation(
        filter_col,
        *scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply boolean mask to filter table
    auto result = cudf::apply_boolean_mask(view, mask->view());

    return make_gpu_table_xptr(std::move(result));
}

// More flexible filter with expression support
// [[Rcpp::export]]
SEXP gpu_filter_mask(SEXP tbl_xptr, SEXP mask_xptr, int mask_col_idx) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> tbl_ptr(tbl_xptr);
    Rcpp::XPtr<GpuTablePtr> mask_ptr(mask_xptr);

    cudf::table_view tbl_view = get_table_view(tbl_ptr);
    cudf::table_view mask_view = get_table_view(mask_ptr);

    // Get boolean column from mask table
    cudf::column_view bool_mask = mask_view.column(mask_col_idx);

    auto result = cudf::apply_boolean_mask(tbl_view, bool_mask);
    return make_gpu_table_xptr(std::move(result));
}
```

### Compilation Setup

```make
# src/Makevars.in (template - configure will generate Makevars)

# Paths set by configure
CUDA_HOME = @CUDA_HOME@
CUDF_INCLUDE = @CUDF_INCLUDE@
CUDF_LIB = @CUDF_LIB@
RMM_INCLUDE = @RMM_INCLUDE@

# Compiler settings
CXX_STD = CXX17
PKG_CXXFLAGS = -I$(CUDF_INCLUDE) -I$(RMM_INCLUDE) -I$(CUDA_HOME)/include \
               -DFMT_HEADER_ONLY $(SHLIB_OPENMP_CXXFLAGS)

PKG_LIBS = -L$(CUDF_LIB) -lcudf -L$(CUDA_HOME)/lib64 -lcudart \
           $(SHLIB_OPENMP_CXXFLAGS) -Wl,-rpath,$(CUDF_LIB)

# Ensure nvcc is not used for R package compilation
# All CUDA code is in libcudf; we only link against it
```

---

## 7. Build System & Packaging

### DESCRIPTION File

```
Package: cuplr
Title: GPU-Accelerated Data Manipulation with dplyr Syntax
Version: 0.1.0
Authors@R: c(
    person("Your", "Name", email = "you@example.com", role = c("aut", "cre")),
    person("RAPIDS Team", role = "cph", comment = "libcudf library")
  )
Description: Provides a dplyr backend that executes operations on NVIDIA GPUs
    using the RAPIDS libcudf library. Supports filter, select, mutate, arrange,
    group_by, summarise, and join operations with familiar tidyverse syntax
    while achieving significant speedups on large datasets.
License: Apache License (>= 2.0)
URL: https://github.com/yourorg/cuplr
BugReports: https://github.com/yourorg/cuplr/issues
Encoding: UTF-8
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.3.1
SystemRequirements:
    NVIDIA GPU with Compute Capability >= 6.0,
    CUDA Toolkit >= 12.0,
    RAPIDS libcudf >= 25.12
Depends:
    R (>= 4.3.0)
Imports:
    Rcpp (>= 1.0.12),
    dplyr (>= 1.1.0),
    rlang (>= 1.1.0),
    vctrs (>= 0.6.0),
    pillar (>= 1.9.0),
    glue (>= 1.6.0),
    cli (>= 3.6.0)
Suggests:
    testthat (>= 3.0.0),
    bench,
    arrow,
    nanoarrow,
    bit64,
    reticulate
LinkingTo:
    Rcpp
Config/testthat/edition: 3
NeedsCompilation: yes
```

### NAMESPACE File

```
# Generated by roxygen2: do not edit by hand

# Imports
import(dplyr)
importFrom(Rcpp, sourceCpp)
importFrom(rlang, enquo, enquos, eval_tidy, quo_get_expr, is_quosure)
importFrom(vctrs, vec_ptype2, vec_cast)
importFrom(pillar, tbl_sum, tbl_format_header)
importFrom(glue, glue)
importFrom(cli, cli_abort, cli_warn)

# Exports
export(tbl_gpu)
export(as_tbl_gpu)
export(is_tbl_gpu)
export(collect)
export(compute)
export(gpu_info)

# S3 Methods
S3method(print, tbl_gpu)
S3method(dim, tbl_gpu)
S3method(names, tbl_gpu)
S3method(as.data.frame, tbl_gpu)
S3method(collect, tbl_gpu)
S3method(compute, tbl_gpu)

# dplyr verb methods
S3method(filter, tbl_gpu)
S3method(select, tbl_gpu)
S3method(mutate, tbl_gpu)
S3method(arrange, tbl_gpu)
S3method(group_by, tbl_gpu)
S3method(ungroup, tbl_gpu)
S3method(summarise, tbl_gpu)
S3method(summarize, tbl_gpu)
S3method(left_join, tbl_gpu)
S3method(inner_join, tbl_gpu)
S3method(right_join, tbl_gpu)
S3method(distinct, tbl_gpu)
S3method(slice, tbl_gpu)
S3method(head, tbl_gpu)
S3method(tail, tbl_gpu)
S3method(rename, tbl_gpu)

# Internal C++ functions
useDynLib(cuplr, .registration = TRUE)
```

### Configure Script

```bash
#!/bin/bash
# configure - Detect CUDA and libcudf, generate src/Makevars

echo "Configuring cuplr..."

# Default paths
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDF_HOME="${CUDF_HOME:-}"
CONDA_PREFIX="${CONDA_PREFIX:-}"

# Function to find libcudf
find_cudf() {
    # Try conda environment first
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/include/cudf/cudf.hpp" ]; then
        echo "$CONDA_PREFIX"
        return 0
    fi

    # Try common installation paths
    for path in /usr/local /opt/rapids /usr; do
        if [ -f "$path/include/cudf/cudf.hpp" ]; then
            echo "$path"
            return 0
        fi
    done

    # Check if CUDF_HOME is set
    if [ -n "$CUDF_HOME" ] && [ -f "$CUDF_HOME/include/cudf/cudf.hpp" ]; then
        echo "$CUDF_HOME"
        return 0
    fi

    return 1
}

# Detect CUDA
if [ ! -d "$CUDA_HOME" ]; then
    # Try to find CUDA
    for cuda_path in /usr/local/cuda /opt/cuda /usr/lib/cuda; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/include/cuda.h" ]; then
            CUDA_HOME="$cuda_path"
            break
        fi
    done
fi

if [ ! -f "$CUDA_HOME/include/cuda.h" ]; then
    echo "ERROR: CUDA not found. Please set CUDA_HOME environment variable."
    echo "       Example: export CUDA_HOME=/usr/local/cuda"
    exit 1
fi

echo "Found CUDA at: $CUDA_HOME"
echo "CUDA version: $($CUDA_HOME/bin/nvcc --version | grep release | awk '{print $6}')"

# Detect libcudf
CUDF_PREFIX=$(find_cudf)
if [ -z "$CUDF_PREFIX" ]; then
    echo "ERROR: libcudf not found. Please install RAPIDS or set CUDF_HOME."
    echo ""
    echo "Installation options:"
    echo "  1. Conda: conda install -c rapidsai -c conda-forge -c nvidia libcudf"
    echo "  2. Pip:   pip install libcudf-cu12"
    echo "  3. Set:   export CUDF_HOME=/path/to/cudf"
    exit 1
fi

CUDF_INCLUDE="$CUDF_PREFIX/include"
CUDF_LIB="$CUDF_PREFIX/lib"

echo "Found libcudf at: $CUDF_PREFIX"

# Check for RMM (RAPIDS Memory Manager)
if [ -f "$CUDF_PREFIX/include/rmm/rmm.hpp" ]; then
    RMM_INCLUDE="$CUDF_PREFIX/include"
else
    RMM_INCLUDE="$CUDF_INCLUDE"
fi

# Verify libcudf shared library exists
if [ ! -f "$CUDF_LIB/libcudf.so" ]; then
    echo "WARNING: libcudf.so not found in $CUDF_LIB"
    echo "         Package may fail to load at runtime."
fi

# Generate Makevars from template
sed -e "s|@CUDA_HOME@|$CUDA_HOME|g" \
    -e "s|@CUDF_INCLUDE@|$CUDF_INCLUDE|g" \
    -e "s|@CUDF_LIB@|$CUDF_LIB|g" \
    -e "s|@RMM_INCLUDE@|$RMM_INCLUDE|g" \
    src/Makevars.in > src/Makevars

echo ""
echo "Configuration complete. Generated src/Makevars:"
cat src/Makevars
echo ""
echo "Run 'R CMD INSTALL .' to build the package."
```

### Dockerfile

```dockerfile
# Dockerfile for cuplr development and CI
# Based on RAPIDS CUDA 12 developer image

ARG RAPIDS_VERSION=25.12
ARG CUDA_VERSION=12.5
ARG UBUNTU_VERSION=22.04

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py3.11-amd64

LABEL maintainer="your@email.com"
LABEL description="cuplr development environment with RAPIDS libcudf"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    dirmngr \
    gnupg \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Add R repository and install R
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
    gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > \
    /etc/apt/sources.list.d/r-project.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R package dependencies
RUN R -e "install.packages(c( \
    'Rcpp', 'dplyr', 'rlang', 'vctrs', 'pillar', 'glue', 'cli', \
    'testthat', 'bench', 'arrow', 'nanoarrow', 'bit64', 'reticulate', \
    'devtools', 'roxygen2', 'pkgdown' \
), repos='https://cloud.r-project.org')"

# Set environment variables for cuplr build
ENV CUDA_HOME=/usr/local/cuda
ENV CUDF_HOME=/opt/conda
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

# Create working directory
WORKDIR /cuplr

# Copy package source
COPY . /cuplr

# Configure and build
RUN chmod +x configure && \
    ./configure && \
    R CMD INSTALL .

# Run tests by default
CMD ["R", "-e", "testthat::test_package('cuplr')"]
```

### GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvcr.io/nvidia/rapidsai/base:25.12-cuda12.5-py3.11-amd64
      options: --gpus all

    steps:
      - uses: actions/checkout@v4

      - name: Install R
        run: |
          apt-get update
          apt-get install -y software-properties-common
          wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
            gpg --dearmor -o /usr/share/keyrings/r-project.gpg
          echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > \
            /etc/apt/sources.list.d/r-project.list
          apt-get update
          apt-get install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev

      - name: Install R dependencies
        run: |
          R -e "install.packages(c('Rcpp', 'dplyr', 'rlang', 'vctrs', 'pillar', 'glue', 'cli', 'testthat', 'bench'), repos='https://cloud.r-project.org')"

      - name: Configure
        run: |
          export CUDA_HOME=/usr/local/cuda
          export CUDF_HOME=/opt/conda
          chmod +x configure
          ./configure

      - name: Build package
        run: R CMD build .

      - name: Check package
        run: R CMD check cuplr_*.tar.gz --no-manual

      - name: Install package
        run: R CMD INSTALL cuplr_*.tar.gz

      - name: Run tests
        run: R -e "testthat::test_package('cuplr')"

  benchmark:
    needs: build
    runs-on: ubuntu-latest
    container:
      image: nvcr.io/nvidia/rapidsai/base:25.12-cuda12.5-py3.11-amd64
      options: --gpus all

    steps:
      - uses: actions/checkout@v4

      - name: Setup (same as build job)
        run: |
          # ... same setup steps ...
          echo "Setup complete"

      - name: Run benchmarks
        run: |
          R -e "source('inst/benchmarks/run_benchmarks.R')"

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: inst/benchmarks/results/
```

---

## 8. Minimal Working Prototype

### Directory Structure

```
cuplr/
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
├── configure
├── R/
│   ├── zzz.R
│   ├── tbl_gpu.R
│   ├── dplyr-filter.R
│   ├── dplyr-select.R
│   ├── dplyr-mutate.R
│   ├── dplyr-arrange.R
│   ├── dplyr-group.R
│   ├── dplyr-summarise.R
│   ├── dplyr-join.R
│   ├── collect.R
│   └── utils.R
├── src/
│   ├── Makevars.in
│   ├── init.cpp
│   ├── gpu_table.hpp
│   ├── transfer.cpp
│   ├── filter.cpp
│   ├── sort.cpp
│   ├── groupby.cpp
│   └── RcppExports.cpp
├── inst/
│   ├── docker/
│   │   └── Dockerfile
│   └── benchmarks/
│       └── run_benchmarks.R
├── tests/
│   └── testthat/
│       ├── test-basic.R
│       ├── test-filter.R
│       ├── test-mutate.R
│       └── helper-cuplr.R
└── man/
    └── (generated by roxygen2)
```

### R/zzz.R - Package Load Hook

```r
#' @useDynLib cuplr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

.onLoad <- function(libname, pkgname) {
  # Check GPU availability
  gpu_ok <- tryCatch(
    {
      .Call(`_cuplr_check_gpu`)
      TRUE
    },
    error = function(e) {
      packageStartupMessage(
        "cuplr: No GPU detected or CUDA unavailable. ",
        "GPU operations will fail."
      )
      FALSE
    }
  )

  # Set package options
  op <- options()
  op.cuplr <- list(
    cuplr.verbose = FALSE,
    cuplr.lazy = TRUE,
    cuplr.gpu_available = gpu_ok
  )
  toset <- !(names(op.cuplr) %in% names(op))
  if (any(toset)) options(op.cuplr[toset])

  invisible()
}

.onAttach <- function(libname, pkgname) {
  if (getOption("cuplr.gpu_available", FALSE)) {
    info <- gpu_info()
    packageStartupMessage(
      "cuplr: Using GPU '", info$name, "' with ",
      round(info$memory_total / 1e9, 1), " GB memory"
    )
  }
}
```

### R/tbl_gpu.R - Core Class

```r
#' Create a GPU-backed tibble
#'
#' @param data A data frame to transfer to GPU memory
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object
#' @export
#' @examples
#' if (interactive()) {
#'   df <- data.frame(x = 1:1000, y = rnorm(1000))
#'   gpu_df <- tbl_gpu(df)
#'   gpu_df
#' }
tbl_gpu <- function(data, ...) {
  UseMethod("tbl_gpu")
}

#' @export
tbl_gpu.data.frame <- function(data, ...) {
  # Transfer to GPU
  ptr <- .Call(`_cuplr_df_to_gpu`, data)

  schema <- list(
    names = names(data),
    types = vapply(data, gpu_type_from_r, character(1))
  )

  new_tbl_gpu(ptr = ptr, schema = schema)
}

#' @export
tbl_gpu.tbl_gpu <- function(data, ...) {
  data
}

# Internal constructor
new_tbl_gpu <- function(ptr = NULL,
                        schema = list(names = character(), types = character()),
                        lazy_ops = list(),
                        groups = character()) {
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

#' @export
is_tbl_gpu <- function(x) {
  inherits(x, "tbl_gpu")
}

#' @export
as_tbl_gpu <- function(x, ...) {
  tbl_gpu(x, ...)
}

# Print method
#' @export
print.tbl_gpu <- function(x, ..., n = 10) {
  cat("# A GPU tibble: ")

  if (is.null(x$ptr)) {
    cat("[lazy, not materialized]\n")
    cat("# Schema: ", paste(x$schema$names, collapse = ", "), "\n")
    cat("# Operations pending: ", length(x$lazy_ops), "\n")
  } else {
    dims <- dim(x)
    cat(format(dims[1], big.mark = ","), " x ", dims[2], "\n", sep = "")

    if (length(x$groups) > 0) {
      cat("# Groups: ", paste(x$groups, collapse = ", "), "\n")
    }

    # Show first n rows
    preview <- head(collect(x), n)
    print(tibble::as_tibble(preview))

    if (dims[1] > n) {
      cat("# ... with ", format(dims[1] - n, big.mark = ","),
          " more rows\n", sep = "")
    }
  }

  invisible(x)
}

#' @export
dim.tbl_gpu <- function(x) {
  if (is.null(x$ptr)) {
    c(NA_integer_, length(x$schema$names))
  } else {
    .Call(`_cuplr_gpu_dim`, x$ptr)
  }
}

#' @export
names.tbl_gpu <- function(x) {
  x$schema$names
}

#' @export
`names<-.tbl_gpu` <- function(x, value) {
  x$schema$names <- value
  x
}

# Type helper
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
  "UNKNOWN"
}
```

### R/dplyr-filter.R

```r
#' @importFrom dplyr filter
#' @importFrom rlang enquos quo_get_expr eval_tidy
#' @export
filter.tbl_gpu <- function(.data, ..., .preserve = FALSE) {
  dots <- enquos(...)

  if (length(dots) == 0) {
    return(.data)
  }

  # For lazy mode, store operation
 if (getOption("cuplr.lazy", TRUE) && length(.data$lazy_ops) > 0) {
    .data$lazy_ops <- c(.data$lazy_ops, list(
      list(op = "filter", args = dots)
    ))
    return(.data)
  }

  # Eager execution
  for (quo in dots) {
    .data <- execute_filter(.data, quo)
  }

  .data
}

execute_filter <- function(.data, quo) {
  expr <- quo_get_expr(quo)

  # Parse simple comparison: col > value
  if (is.call(expr) && length(expr) == 3) {
    op <- as.character(expr[[1]])
    lhs <- expr[[2]]
    rhs <- expr[[3]]

    # Check if LHS is column name
    if (is.symbol(lhs)) {
      col_name <- as.character(lhs)
      col_idx <- match(col_name, .data$schema$names) - 1L  # 0-indexed

      if (is.na(col_idx)) {
        cli::cli_abort("Column '{col_name}' not found in GPU table")
      }

      # Evaluate RHS
      value <- eval_tidy(rhs)

      # Map R operator to C++ function
      new_ptr <- switch(op,
        ">"  = .Call(`_cuplr_gpu_filter_gt`, .data$ptr, col_idx, value),
        ">=" = .Call(`_cuplr_gpu_filter_gte`, .data$ptr, col_idx, value),
        "<"  = .Call(`_cuplr_gpu_filter_lt`, .data$ptr, col_idx, value),
        "<=" = .Call(`_cuplr_gpu_filter_lte`, .data$ptr, col_idx, value),
        "==" = .Call(`_cuplr_gpu_filter_eq`, .data$ptr, col_idx, value),
        "!=" = .Call(`_cuplr_gpu_filter_neq`, .data$ptr, col_idx, value),
        cli::cli_abort("Unsupported filter operator: {op}")
      )

      new_tbl_gpu(
        ptr = new_ptr,
        schema = .data$schema,
        groups = .data$groups
      )
    } else {
      cli::cli_abort("Complex filter expressions not yet supported")
    }
  } else {
    cli::cli_abort("Unsupported filter expression type")
  }
}
```

### R/collect.R

```r
#' Materialize GPU table to R data frame
#'
#' @param x A `tbl_gpu` object
#' @param ... Additional arguments (unused)
#' @return A data frame
#' @export
collect.tbl_gpu <- function(x, ...) {
  # Execute any pending lazy operations first
  if (length(x$lazy_ops) > 0) {
    x <- compute(x)
  }

  if (is.null(x$ptr)) {
    cli::cli_abort("Cannot collect: GPU table has no data pointer")
  }

  # Transfer from GPU to R
  df <- .Call(`_cuplr_gpu_to_df`, x$ptr, x$schema$names)

  # Apply type conversions
  for (i in seq_along(df)) {
    gpu_type <- x$schema$types[i]
    if (gpu_type == "TIMESTAMP_DAYS") {
      df[[i]] <- as.Date(df[[i]], origin = "1970-01-01")
    } else if (gpu_type == "TIMESTAMP_MICROSECONDS") {
      df[[i]] <- as.POSIXct(df[[i]] / 1e6, origin = "1970-01-01")
    }
  }

  tibble::as_tibble(df)
}

#' Execute lazy operations and keep result on GPU
#'
#' @param x A `tbl_gpu` object
#' @param ... Additional arguments (unused)
#' @return A `tbl_gpu` object with operations materialized
#' @export
compute.tbl_gpu <- function(x, ...) {
  if (length(x$lazy_ops) == 0) {
    return(x)
  }

  if (getOption("cuplr.verbose", FALSE)) {
    message("cuplr: Executing ", length(x$lazy_ops), " lazy operations")
  }

  # Execute operations in order
  result <- x
  result$lazy_ops <- list()  # Clear lazy ops for execution

  for (op in x$lazy_ops) {
    result <- execute_lazy_op(result, op)
  }

  result
}

execute_lazy_op <- function(.data, op) {
  switch(op$op,
    "filter" = {
      for (quo in op$args) {
        .data <- execute_filter(.data, quo)
      }
      .data
    },
    "select" = execute_select(.data, op$args),
    "mutate" = execute_mutate(.data, op$args),
    "arrange" = execute_arrange(.data, op$args),
    "group_by" = execute_group_by(.data, op$args),
    "summarise" = execute_summarise(.data, op$args),
    cli::cli_abort("Unknown lazy operation: {op$op}")
  )
}
```

### src/init.cpp - Registration

```cpp
// src/init.cpp
#include <Rcpp.h>
#include <R_ext/Rdynload.h>

// Forward declarations
SEXP _cuplr_check_gpu();
SEXP _cuplr_df_to_gpu(SEXP df);
SEXP _cuplr_gpu_to_df(SEXP xptr, SEXP names);
SEXP _cuplr_gpu_dim(SEXP xptr);
SEXP _cuplr_gpu_filter_gt(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_gte(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_lt(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_lte(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_eq(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_filter_neq(SEXP xptr, SEXP col_idx, SEXP value);
SEXP _cuplr_gpu_info();

static const R_CallMethodDef CallEntries[] = {
    {"_cuplr_check_gpu", (DL_FUNC) &_cuplr_check_gpu, 0},
    {"_cuplr_df_to_gpu", (DL_FUNC) &_cuplr_df_to_gpu, 1},
    {"_cuplr_gpu_to_df", (DL_FUNC) &_cuplr_gpu_to_df, 2},
    {"_cuplr_gpu_dim", (DL_FUNC) &_cuplr_gpu_dim, 1},
    {"_cuplr_gpu_filter_gt", (DL_FUNC) &_cuplr_gpu_filter_gt, 3},
    {"_cuplr_gpu_filter_gte", (DL_FUNC) &_cuplr_gpu_filter_gte, 3},
    {"_cuplr_gpu_filter_lt", (DL_FUNC) &_cuplr_gpu_filter_lt, 3},
    {"_cuplr_gpu_filter_lte", (DL_FUNC) &_cuplr_gpu_filter_lte, 3},
    {"_cuplr_gpu_filter_eq", (DL_FUNC) &_cuplr_gpu_filter_eq, 3},
    {"_cuplr_gpu_filter_neq", (DL_FUNC) &_cuplr_gpu_filter_neq, 3},
    {"_cuplr_gpu_info", (DL_FUNC) &_cuplr_gpu_info, 0},
    {NULL, NULL, 0}
};

extern "C" void R_init_cuplr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
```

### tests/testthat/test-basic.R

```r
test_that("tbl_gpu can be created from data.frame", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(
    x = c(1, 2, 3, 4, 5),
    y = c(10.5, 20.5, 30.5, 40.5, 50.5)
  )

  gpu_df <- tbl_gpu(df)

  expect_s3_class(gpu_df, "tbl_gpu")
  expect_equal(dim(gpu_df), c(5L, 2L))
  expect_equal(names(gpu_df), c("x", "y"))
})

test_that("collect returns data to R", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(x = 1:5, y = 6:10)
  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_s3_class(result, "tbl_df")
  expect_equal(result$x, 1:5)
  expect_equal(result$y, 6:10)
})

test_that("filter works with simple comparisons", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(x = 1:10, y = 11:20)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df %>%
    filter(x > 5) %>%
    collect()

  expected <- df %>%
    dplyr::filter(x > 5)

  expect_equal(result$x, expected$x)
  expect_equal(result$y, expected$y)
})

test_that("NA values are preserved through round-trip", {
  skip_if_not(getOption("cuplr.gpu_available", FALSE), "No GPU available")

  df <- data.frame(
    x = c(1, NA, 3, NA, 5),
    y = c(NA, 2.5, NA, 4.5, 5.5)
  )

  gpu_df <- tbl_gpu(df)
  result <- collect(gpu_df)

  expect_equal(is.na(result$x), is.na(df$x))
  expect_equal(is.na(result$y), is.na(df$y))
})
```

---

## 9. Lazy Translation & AST Approach

### AST Node Definitions

```r
# R/ast.R - Internal AST representation

# Base AST node
ast_node <- function(type, ...) {
  structure(
    list(type = type, ...),
    class = c(paste0("ast_", type), "ast_node")
  )
}

# Operation nodes
ast_filter <- function(predicates) {
  ast_node("filter", predicates = predicates)
}

ast_select <- function(columns) {
  ast_node("select", columns = columns)
}

ast_mutate <- function(expressions) {
  ast_node("mutate", expressions = expressions)
}

ast_arrange <- function(columns, descending) {
  ast_node("arrange", columns = columns, descending = descending)
}

ast_group_by <- function(columns) {
  ast_node("group_by", columns = columns)
}

ast_summarise <- function(aggregations) {
  ast_node("summarise", aggregations = aggregations)
}

ast_join <- function(type, right_table, by) {
  ast_node("join", join_type = type, right = right_table, by = by)
}

# Expression nodes (for predicates and computations)
ast_binary_op <- function(op, left, right) {
  ast_node("binary_op", operator = op, left = left, right = right)
}

ast_column_ref <- function(name) {
  ast_node("column_ref", name = name)
}

ast_literal <- function(value) {
  ast_node("literal", value = value)
}

ast_function_call <- function(fn, args) {
  ast_node("function_call", fn = fn, args = args)
}
```

### Expression Parser

```r
# R/parse_expr.R - Parse R expressions to AST

parse_expr_to_ast <- function(expr, env = parent.frame()) {
  if (is.symbol(expr)) {
    # Column reference
    return(ast_column_ref(as.character(expr)))
  }

  if (is.atomic(expr) && length(expr) == 1) {
    # Literal value
    return(ast_literal(expr))
  }

  if (is.call(expr)) {
    fn <- as.character(expr[[1]])

    # Binary operators
    if (fn %in% c("+", "-", "*", "/", "^", "%%", "%/%",
                  ">", ">=", "<", "<=", "==", "!=",
                  "&", "|")) {
      left <- parse_expr_to_ast(expr[[2]], env)
      right <- parse_expr_to_ast(expr[[3]], env)
      return(ast_binary_op(fn, left, right))
    }

    # Unary operators
    if (fn == "!" && length(expr) == 2) {
      return(ast_node("unary_op", operator = "!", operand = parse_expr_to_ast(expr[[2]], env)))
    }

    # Function calls
    args <- lapply(expr[-1], parse_expr_to_ast, env = env)
    return(ast_function_call(fn, args))
  }

  cli::cli_abort("Cannot parse expression: {deparse(expr)}")
}

# Convert quosure to AST
quosure_to_ast <- function(quo) {
  expr <- rlang::quo_get_expr(quo)
  env <- rlang::quo_get_env(quo)
  parse_expr_to_ast(expr, env)
}
```

### Query Optimizer

```r
# R/optimizer.R - AST optimization passes

optimize_ast <- function(ops) {
  ops <- push_down_filters(ops)
  ops <- push_down_projections(ops)
  ops <- fuse_consecutive_filters(ops)
  ops
}

# Predicate pushdown: move filters earlier in pipeline
push_down_filters <- function(ops) {
  if (length(ops) < 2) return(ops)

  result <- list()
  pending_filters <- list()

  for (op in ops) {
    if (op$type == "filter") {
      # Collect filter predicates
      pending_filters <- c(pending_filters, op$predicates)
    } else if (op$type %in% c("select", "mutate")) {
      # Can push filters before select/mutate if columns exist
      if (length(pending_filters) > 0) {
        # Check which predicates can be pushed
        pushable <- vapply(pending_filters, function(pred) {
          cols <- extract_column_refs(pred)
          # All referenced columns must exist before this op
          TRUE  # Simplified - real implementation checks column existence
        }, logical(1))

        if (any(pushable)) {
          result <- c(result, list(ast_filter(pending_filters[pushable])))
          pending_filters <- pending_filters[!pushable]
        }
      }
      result <- c(result, list(op))
    } else {
      # Flush pending filters
      if (length(pending_filters) > 0) {
        result <- c(result, list(ast_filter(pending_filters)))
        pending_filters <- list()
      }
      result <- c(result, list(op))
    }
  }

  # Flush remaining filters
  if (length(pending_filters) > 0) {
    result <- c(result, list(ast_filter(pending_filters)))
  }

  result
}

# Projection pushdown: only read needed columns
push_down_projections <- function(ops) {
  # Analyze which columns are used by each operation
  # Remove unused columns early
  ops  # Placeholder - full implementation tracks column usage
}

# Fuse consecutive filters into single operation
fuse_consecutive_filters <- function(ops) {
  if (length(ops) < 2) return(ops)

  result <- list()
  i <- 1

  while (i <= length(ops)) {
    if (ops[[i]]$type == "filter") {
      # Collect consecutive filters
      predicates <- ops[[i]]$predicates
      while (i < length(ops) && ops[[i + 1]]$type == "filter") {
        i <- i + 1
        predicates <- c(predicates, ops[[i]]$predicates)
      }
      result <- c(result, list(ast_filter(predicates)))
    } else {
      result <- c(result, list(ops[[i]]))
    }
    i <- i + 1
  }

  result
}

# Extract column references from AST node
extract_column_refs <- function(node) {
  if (inherits(node, "ast_column_ref")) {
    return(node$name)
  }
  if (is.list(node)) {
    unlist(lapply(node, extract_column_refs))
  } else {
    character(0)
  }
}
```

### Lowering AST to libcudf Calls

```r
# R/lower.R - Convert AST to libcudf operations

lower_to_cudf <- function(tbl, ops) {
  if (getOption("cuplr.verbose", FALSE)) {
    message("cuplr: Lowering ", length(ops), " operations to libcudf")
  }

  for (op in ops) {
    tbl <- lower_op(tbl, op)
  }

  tbl
}

lower_op <- function(tbl, op) {
  if (getOption("cuplr.verbose", FALSE)) {
    message("  -> ", op$type)
  }

  switch(op$type,
    "filter" = lower_filter(tbl, op),
    "select" = lower_select(tbl, op),
    "mutate" = lower_mutate(tbl, op),
    "arrange" = lower_arrange(tbl, op),
    "group_by" = lower_group_by(tbl, op),
    "summarise" = lower_summarise(tbl, op),
    "join" = lower_join(tbl, op),
    cli::cli_abort("Cannot lower operation: {op$type}")
  )
}

lower_filter <- function(tbl, op) {
  # Build combined boolean mask from all predicates
  mask_ptr <- NULL

  for (pred in op$predicates) {
    pred_mask <- evaluate_predicate(tbl, pred)
    if (is.null(mask_ptr)) {
      mask_ptr <- pred_mask
    } else {
      # AND masks together
      mask_ptr <- .Call(`_cuplr_gpu_and_masks`, mask_ptr, pred_mask)
    }
  }

  new_ptr <- .Call(`_cuplr_gpu_apply_mask`, tbl$ptr, mask_ptr)

  new_tbl_gpu(
    ptr = new_ptr,
    schema = tbl$schema,
    groups = tbl$groups
  )
}

evaluate_predicate <- function(tbl, pred) {
  if (inherits(pred, "ast_binary_op")) {
    left <- evaluate_expr(tbl, pred$left)
    right <- evaluate_expr(tbl, pred$right)

    op_code <- switch(pred$operator,
      ">" = 1L, ">=" = 2L, "<" = 3L, "<=" = 4L,
      "==" = 5L, "!=" = 6L,
      "&" = 7L, "|" = 8L,
      cli::cli_abort("Unsupported predicate operator: {pred$operator}")
    )

    .Call(`_cuplr_gpu_binary_op`, left, right, op_code)
  } else {
    cli::cli_abort("Cannot evaluate predicate: {class(pred)[1]}")
  }
}

evaluate_expr <- function(tbl, expr) {
  if (inherits(expr, "ast_column_ref")) {
    col_idx <- match(expr$name, tbl$schema$names) - 1L
    .Call(`_cuplr_gpu_get_column`, tbl$ptr, col_idx)
  } else if (inherits(expr, "ast_literal")) {
    .Call(`_cuplr_gpu_scalar`, expr$value)
  } else if (inherits(expr, "ast_binary_op")) {
    left <- evaluate_expr(tbl, expr$left)
    right <- evaluate_expr(tbl, expr$right)
    op_code <- match(expr$operator, c("+", "-", "*", "/", "^", "%%", "%/%"))
    .Call(`_cuplr_gpu_binary_op`, left, right, op_code)
  } else {
    cli::cli_abort("Cannot evaluate expression: {class(expr)[1]}")
  }
}
```

### Verbose Mode Output Example

```
> options(cuplr.verbose = TRUE)
> tbl_gpu(df) %>% filter(x > 10, y < 50) %>% mutate(z = x + y) %>% collect()

cuplr: Executing 2 lazy operations
cuplr: Optimizing AST...
  -> Fused 2 filter predicates
cuplr: Lowering 2 operations to libcudf
  -> filter
     cudf::binary_operation(col[0], scalar(10), GREATER) -> mask1
     cudf::binary_operation(col[1], scalar(50), LESS) -> mask2
     cudf::binary_operation(mask1, mask2, BITWISE_AND) -> mask_final
     cudf::apply_boolean_mask(table, mask_final)
  -> mutate
     cudf::binary_operation(col[0], col[1], ADD) -> new_col
     append column to table
# A tibble: 3 x 3
      x     y     z
  <dbl> <dbl> <dbl>
1    15    30    45
2    20    40    60
3    25    45    70
```

---

## 10. Testing & Validation

### Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Unit | Test individual R functions | `tests/testthat/test-*.R` |
| Integration | Compare GPU vs CPU results | `tests/testthat/test-integration.R` |
| Performance | Benchmark against dplyr | `inst/benchmarks/` |
| Edge cases | NA handling, empty tables, types | `tests/testthat/test-edge-cases.R` |

### tests/testthat/helper-cuplr.R

```r
# Test helper functions

skip_if_no_gpu <- function() {
  skip_if_not(
    getOption("cuplr.gpu_available", FALSE),
    "No GPU available for testing"
  )
}

# Compare GPU and CPU results with tolerance
expect_gpu_cpu_equal <- function(gpu_result, cpu_result, tolerance = 1e-10) {
  gpu_df <- if (is_tbl_gpu(gpu_result)) collect(gpu_result) else gpu_result
  cpu_df <- if (inherits(cpu_result, "data.frame")) cpu_result else as.data.frame(cpu_result)

  expect_equal(nrow(gpu_df), nrow(cpu_df))
  expect_equal(ncol(gpu_df), ncol(cpu_df))
  expect_equal(names(gpu_df), names(cpu_df))

  for (col in names(gpu_df)) {
    if (is.numeric(gpu_df[[col]])) {
      expect_equal(gpu_df[[col]], cpu_df[[col]], tolerance = tolerance,
                   label = paste("Column", col))
    } else {
      expect_equal(gpu_df[[col]], cpu_df[[col]], label = paste("Column", col))
    }
  }
}

# Generate test data
make_test_df <- function(n = 1000, seed = 42) {
  set.seed(seed)
  data.frame(
    int_col = sample(1:100, n, replace = TRUE),
    dbl_col = rnorm(n, mean = 50, sd = 10),
    chr_col = sample(letters[1:10], n, replace = TRUE),
    grp_col = sample(LETTERS[1:5], n, replace = TRUE),
    stringsAsFactors = FALSE
  )
}
```

### tests/testthat/test-filter.R

```r
test_that("filter with > works correctly", {
  skip_if_no_gpu()

  df <- make_test_df(1000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>% filter(int_col > 50)
  cpu_result <- df %>% dplyr::filter(int_col > 50)

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})

test_that("filter with multiple conditions works", {
  skip_if_no_gpu()

  df <- make_test_df(1000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>% filter(int_col > 25, int_col < 75)
  cpu_result <- df %>% dplyr::filter(int_col > 25, int_col < 75)

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})

test_that("filter handles NA values correctly", {
  skip_if_no_gpu()

  df <- data.frame(
    x = c(1, NA, 3, 4, NA, 6),
    y = c(10, 20, NA, 40, 50, NA)
  )
  gpu_df <- tbl_gpu(df)

  # Filter should exclude NA comparisons (like R)
  gpu_result <- gpu_df %>% filter(x > 2) %>% collect()
  cpu_result <- df %>% dplyr::filter(x > 2)

  expect_equal(nrow(gpu_result), nrow(cpu_result))
  expect_equal(gpu_result$x, cpu_result$x)
})

test_that("filter on empty result returns empty table", {
  skip_if_no_gpu()

  df <- data.frame(x = 1:10)
  gpu_df <- tbl_gpu(df)

  result <- gpu_df %>% filter(x > 100) %>% collect()

  expect_equal(nrow(result), 0)
  expect_equal(names(result), "x")
})
```

### tests/testthat/test-integration.R

```r
test_that("complex pipeline matches dplyr", {
  skip_if_no_gpu()

  df <- make_test_df(10000)
  gpu_df <- tbl_gpu(df)

  gpu_result <- gpu_df %>%
    filter(int_col > 20) %>%
    mutate(computed = dbl_col * 2) %>%
    group_by(grp_col) %>%
    summarise(
      mean_val = mean(computed),
      count = n()
    ) %>%
    arrange(grp_col) %>%
    collect()

  cpu_result <- df %>%
    dplyr::filter(int_col > 20) %>%
    dplyr::mutate(computed = dbl_col * 2) %>%
    dplyr::group_by(grp_col) %>%
    dplyr::summarise(
      mean_val = mean(computed),
      count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::arrange(grp_col)

  expect_gpu_cpu_equal(gpu_result, cpu_result, tolerance = 1e-6)
})

test_that("join operations match dplyr", {
  skip_if_no_gpu()

  left_df <- data.frame(
    key = c(1, 2, 3, 4, 5),
    val_left = c("a", "b", "c", "d", "e")
  )
  right_df <- data.frame(
    key = c(2, 3, 4, 6, 7),
    val_right = c("x", "y", "z", "w", "v")
  )

  gpu_left <- tbl_gpu(left_df)
  gpu_right <- tbl_gpu(right_df)

  # Left join
  gpu_result <- gpu_left %>%
    left_join(gpu_right, by = "key") %>%
    collect()

  cpu_result <- left_df %>%
    dplyr::left_join(right_df, by = "key")

  expect_gpu_cpu_equal(gpu_result, cpu_result)

  # Inner join
  gpu_result <- gpu_left %>%
    inner_join(gpu_right, by = "key") %>%
    collect()

  cpu_result <- left_df %>%
    dplyr::inner_join(right_df, by = "key")

  expect_gpu_cpu_equal(gpu_result, cpu_result)
})
```

### Benchmark Script

```r
# inst/benchmarks/run_benchmarks.R

library(cuplr)
library(dplyr)
library(bench)

# Configuration
sizes <- c(1e5, 1e6, 1e7, 1e8)
results <- list()

cat("cuplr Benchmark Suite\n")
cat("=====================\n\n")

for (n in sizes) {
  cat(sprintf("Dataset size: %s rows\n", format(n, big.mark = ",")))

  # Generate data
  set.seed(42)
  df <- data.frame(
    x = runif(n),
    y = runif(n),
    g = sample(letters[1:26], n, replace = TRUE)
  )

  gpu_df <- tbl_gpu(df)

  # Benchmark filter
  bm_filter <- bench::mark(
    cpu = df %>% filter(x > 0.5),
    gpu = gpu_df %>% filter(x > 0.5) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  # Benchmark group_by + summarise
  bm_group <- bench::mark(
    cpu = df %>% group_by(g) %>% summarise(mean_x = mean(x), .groups = "drop"),
    gpu = gpu_df %>% group_by(g) %>% summarise(mean_x = mean(x)) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  # Benchmark arrange
  bm_sort <- bench::mark(
    cpu = df %>% arrange(x),
    gpu = gpu_df %>% arrange(x) %>% collect(),
    check = FALSE,
    min_iterations = 5
  )

  results[[as.character(n)]] <- list(
    filter = bm_filter,
    group_by = bm_group,
    arrange = bm_sort
  )

  cat(sprintf("  filter:   CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_filter$median[1]),
              as.numeric(bm_filter$median[2]),
              as.numeric(bm_filter$median[1]) / as.numeric(bm_filter$median[2])))

  cat(sprintf("  group_by: CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_group$median[1]),
              as.numeric(bm_group$median[2]),
              as.numeric(bm_group$median[1]) / as.numeric(bm_group$median[2])))

  cat(sprintf("  arrange:  CPU %.2fs, GPU %.2fs (%.1fx speedup)\n",
              as.numeric(bm_sort$median[1]),
              as.numeric(bm_sort$median[2]),
              as.numeric(bm_sort$median[1]) / as.numeric(bm_sort$median[2])))

  cat("\n")

  # Clean up GPU memory
  rm(gpu_df)
  gc()
}

# Save results
saveRDS(results, "inst/benchmarks/results/benchmark_results.rds")
cat("Results saved to inst/benchmarks/results/benchmark_results.rds\n")
```

## 11. Debugging, Logging & Observability

### Verbose Mode Implementation

```r
# R/logging.R

#' Enable verbose logging
#'
#' @param enabled Logical, whether to enable verbose mode
#' @export
cuplr_verbose <- function(enabled = TRUE) {
  options(cuplr.verbose = enabled)
  if (enabled) {
    message("cuplr: Verbose mode enabled")
  }
  invisible(enabled)
}

# Internal logging function
log_cuplr <- function(..., level = "INFO") {
  if (!getOption("cuplr.verbose", FALSE)) return(invisible())

  timestamp <- format(Sys.time(), "%H:%M:%S.%OS3")
  msg <- paste0("[cuplr ", timestamp, " ", level, "] ", ...)
  message(msg)
}

log_op <- function(op_name, ...) {
  log_cuplr("OP: ", op_name, " - ", ...)
}

log_cudf_call <- function(fn_name, ...) {
  log_cuplr("CUDF: ", fn_name, "(", paste(..., sep = ", "), ")")
}
```

### AST Dump Function

```r
# R/debug.R

#' Print the AST for a lazy tbl_gpu
#'
#' @param x A tbl_gpu object
#' @param indent Current indentation level (internal)
#' @export
dump_ast <- function(x, indent = 0) {
  if (!is_tbl_gpu(x)) {
    cli::cli_abort("x must be a tbl_gpu object")
  }

  prefix <- strrep("  ", indent)

  cat(prefix, "tbl_gpu [\n", sep = "")
  cat(prefix, "  schema: ", paste(x$schema$names, collapse = ", "), "\n", sep = "")
  cat(prefix, "  types:  ", paste(x$schema$types, collapse = ", "), "\n", sep = "")

  if (length(x$groups) > 0) {
    cat(prefix, "  groups: ", paste(x$groups, collapse = ", "), "\n", sep = "")
  }

  cat(prefix, "  materialized: ", !is.null(x$ptr), "\n", sep = "")

  if (length(x$lazy_ops) > 0) {
    cat(prefix, "  lazy_ops:\n", sep = "")
    for (i in seq_along(x$lazy_ops)) {
      dump_ast_node(x$lazy_ops[[i]], indent + 2, i)
    }
  }

  cat(prefix, "]\n", sep = "")
  invisible(x)
}

dump_ast_node <- function(node, indent, index = NULL) {
  prefix <- strrep("  ", indent)
  idx_str <- if (!is.null(index)) paste0("[", index, "] ") else ""

  cat(prefix, idx_str, node$op, "\n", sep = "")

  if (node$op == "filter") {
    for (pred in node$args) {
      cat(prefix, "  predicate: ", deparse(rlang::quo_get_expr(pred)), "\n", sep = "")
    }
  } else if (node$op == "select") {
    cat(prefix, "  columns: ", paste(node$args, collapse = ", "), "\n", sep = "")
  } else if (node$op == "mutate") {
    for (nm in names(node$args)) {
      cat(prefix, "  ", nm, " = ", deparse(rlang::quo_get_expr(node$args[[nm]])), "\n", sep = "")
    }
  } else if (node$op == "arrange") {
    cat(prefix, "  columns: ", paste(node$args$columns, collapse = ", "), "\n", sep = "")
    cat(prefix, "  desc: ", paste(node$args$desc, collapse = ", "), "\n", sep = "")
  } else if (node$op == "group_by") {
    cat(prefix, "  groups: ", paste(node$args, collapse = ", "), "\n", sep = "")
  } else if (node$op == "summarise") {
    for (nm in names(node$args)) {
      cat(prefix, "  ", nm, " = ", deparse(rlang::quo_get_expr(node$args[[nm]])), "\n", sep = "")
    }
  }
}
```

### GPU Memory Diagnostics

```r
# R/diagnostics.R

#' Get GPU information and memory status
#'
#' @return A list with GPU device information
#' @export
gpu_info <- function() {
  info <- .Call(`_cuplr_gpu_info`)
  structure(info, class = "cuplr_gpu_info")
}

#' @export
print.cuplr_gpu_info <- function(x, ...) {
  cat("GPU Device Information\n")
  cat("======================\n")
  cat("Device:       ", x$name, "\n")
  cat("Compute Cap:  ", x$compute_capability, "\n")
  cat("Memory Total: ", format_bytes(x$memory_total), "\n")
  cat("Memory Free:  ", format_bytes(x$memory_free), "\n")
  cat("Memory Used:  ", format_bytes(x$memory_total - x$memory_free), "\n")
  cat("Utilization:  ", sprintf("%.1f%%", 100 * (1 - x$memory_free / x$memory_total)), "\n")
  invisible(x)
}

format_bytes <- function(bytes) {
  if (bytes >= 1e9) {
    sprintf("%.2f GB", bytes / 1e9)
  } else if (bytes >= 1e6) {
    sprintf("%.2f MB", bytes / 1e6)
  } else {
    sprintf("%.2f KB", bytes / 1e3)
  }
}

#' Monitor GPU memory during pipeline execution
#'
#' @param expr Expression to evaluate
#' @return Result of expression, with memory stats printed
#' @export
with_gpu_monitor <- function(expr) {
  before <- gpu_info()
  on.exit({
    after <- gpu_info()
    cat("\nGPU Memory Delta:\n")
    cat("  Before: ", format_bytes(before$memory_free), " free\n")
    cat("  After:  ", format_bytes(after$memory_free), " free\n")
    cat("  Change: ", format_bytes(before$memory_free - after$memory_free), "\n")
  })
  force(expr)
}
```

### C++ GPU Info Implementation

```cpp
// src/diagnostics.cpp
#include <Rcpp.h>
#include <cuda_runtime.h>

// [[Rcpp::export]]
Rcpp::List gpu_info_impl() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    return Rcpp::List::create(
        Rcpp::Named("name") = std::string(prop.name),
        Rcpp::Named("compute_capability") = std::to_string(prop.major) + "." + std::to_string(prop.minor),
        Rcpp::Named("memory_total") = static_cast<double>(total_mem),
        Rcpp::Named("memory_free") = static_cast<double>(free_mem),
        Rcpp::Named("multiprocessors") = prop.multiProcessorCount,
        Rcpp::Named("max_threads_per_block") = prop.maxThreadsPerBlock
    );
}
```

### Example Verbose Output

```
> options(cuplr.verbose = TRUE)
> df <- data.frame(x = 1:1e6, y = runif(1e6), g = sample(letters, 1e6, TRUE))
> result <- tbl_gpu(df) %>%
+   filter(x > 500000) %>%
+   group_by(g) %>%
+   summarise(mean_y = mean(y)) %>%
+   collect()

[cuplr 14:23:15.123 INFO] OP: tbl_gpu - Transferring 1000000 x 3 data.frame to GPU
[cuplr 14:23:15.456 INFO] CUDF: creating table with 3 columns
[cuplr 14:23:15.457 INFO] OP: filter - Adding lazy operation
[cuplr 14:23:15.457 INFO] OP: group_by - Adding lazy operation
[cuplr 14:23:15.457 INFO] OP: summarise - Adding lazy operation
[cuplr 14:23:15.458 INFO] OP: collect - Materializing lazy pipeline
[cuplr 14:23:15.458 INFO] Optimizing 3 operations...
[cuplr 14:23:15.458 INFO] CUDF: binary_operation(col[0], scalar(500000), GREATER)
[cuplr 14:23:15.512 INFO] CUDF: apply_boolean_mask(table, mask)
[cuplr 14:23:15.534 INFO] CUDF: groupby::groupby(keys=[2])
[cuplr 14:23:15.535 INFO] CUDF: groupby::aggregate(mean on col[1])
[cuplr 14:23:15.589 INFO] CUDF: Transferring result 26 x 2 to R
```

---

## 12. Performance & Optimization Guidance

### Kernel Fusion Strategy

Minimize GPU kernel launches by fusing operations:

```cpp
// src/fused_ops.cpp
// Example: Fused filter + mutate

#include "gpu_table.hpp"
#include <cudf/stream_compaction.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/transform.hpp>

// Instead of separate filter then mutate, compute both in one pass
// [[Rcpp::export]]
SEXP gpu_filter_mutate_fused(
    SEXP xptr,
    int filter_col,
    double filter_val,
    int mutate_col,
    double mutate_factor
) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Create filter mask
    auto scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(scalar.get())->set_value(filter_val);

    auto mask = cudf::binary_operation(
        view.column(filter_col),
        *scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8}
    );

    // Apply filter
    auto filtered = cudf::apply_boolean_mask(view, mask->view());

    // Now apply mutation on filtered data (smaller, more efficient)
    auto factor_scalar = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::FLOAT64});
    static_cast<cudf::numeric_scalar<double>*>(factor_scalar.get())->set_value(mutate_factor);

    auto new_col = cudf::binary_operation(
        filtered->view().column(mutate_col),
        *factor_scalar,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::FLOAT64}
    );

    // Build result table with new column appended
    std::vector<std::unique_ptr<cudf::column>> result_cols;
    for (cudf::size_type i = 0; i < filtered->num_columns(); ++i) {
        result_cols.push_back(std::make_unique<cudf::column>(filtered->view().column(i)));
    }
    result_cols.push_back(std::move(new_col));

    auto result = std::make_unique<cudf::table>(std::move(result_cols));
    return make_gpu_table_xptr(std::move(result));
}
```

### Memory Copy Minimization

```r
# R/performance.R

# BAD: Multiple round-trips
bad_example <- function(df) {
  gpu <- tbl_gpu(df)
  filtered <- gpu %>% filter(x > 10) %>% collect()  # GPU -> CPU
  gpu2 <- tbl_gpu(filtered)                          # CPU -> GPU
  result <- gpu2 %>% mutate(y = x * 2) %>% collect() # GPU -> CPU
  result
}

# GOOD: Stay on GPU, single collect at end
good_example <- function(df) {
  tbl_gpu(df) %>%
    filter(x > 10) %>%
    mutate(y = x * 2) %>%
    collect()  # Only one GPU -> CPU transfer
}
```

### Chunking Large Datasets

```r
#' Process large datasets in chunks
#'
#' @param df Large data frame
#' @param chunk_size Number of rows per chunk
#' @param fn Function to apply to each chunk (receives tbl_gpu)
#' @return Combined results
#' @export
gpu_chunked <- function(df, chunk_size = 1e7, fn) {
  n <- nrow(df)
  n_chunks <- ceiling(n / chunk_size)

  results <- vector("list", n_chunks)

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1) * chunk_size + 1
    end_row <- min(i * chunk_size, n)

    chunk <- df[start_row:end_row, , drop = FALSE]
    gpu_chunk <- tbl_gpu(chunk)

    results[[i]] <- fn(gpu_chunk) %>% collect()

    # Force cleanup
    rm(gpu_chunk)
    gc()
  }

  dplyr::bind_rows(results)
}

# Usage
# result <- gpu_chunked(huge_df, chunk_size = 5e6, function(chunk) {
#   chunk %>% filter(x > 10) %>% group_by(g) %>% summarise(n = n())
# })
```

### Multi-GPU Considerations

```r
# R/multi_gpu.R

#' Set active GPU device
#'
#' @param device_id Integer device ID (0-indexed)
#' @export
set_gpu_device <- function(device_id) {
  .Call(`_cuplr_set_device`, as.integer(device_id))
  invisible(device_id)
}

#' Get number of available GPUs
#'
#' @return Integer count of GPUs
#' @export
gpu_count <- function() {
  .Call(`_cuplr_device_count`)
}

#' Distribute work across multiple GPUs
#'
#' @param df Data frame to process
#' @param fn Processing function
#' @param n_gpus Number of GPUs to use (default: all available)
#' @return Combined results
#' @export
gpu_parallel <- function(df, fn, n_gpus = gpu_count()) {
  n <- nrow(df)
  chunk_size <- ceiling(n / n_gpus)

  # Use parallel package for multi-GPU
  results <- parallel::mclapply(seq_len(n_gpus), function(gpu_id) {
    set_gpu_device(gpu_id - 1)  # 0-indexed

    start_row <- (gpu_id - 1) * chunk_size + 1
    end_row <- min(gpu_id * chunk_size, n)

    chunk <- df[start_row:end_row, , drop = FALSE]
    gpu_chunk <- tbl_gpu(chunk)

    fn(gpu_chunk) %>% collect()
  }, mc.cores = n_gpus)

  dplyr::bind_rows(results)
}
```

### When to Fall Back to CPU

```r
# Decision heuristics for GPU vs CPU

should_use_gpu <- function(df, operation) {
  n <- nrow(df)

  # Small data: CPU is often faster due to transfer overhead
  if (n < 10000) {
    return(FALSE)
  }

  # String-heavy operations may not benefit as much
  string_cols <- sum(vapply(df, is.character, logical(1)))
  if (string_cols > ncol(df) / 2 && operation %in% c("mutate", "filter")) {
    # String operations have less GPU speedup
    if (n < 100000) return(FALSE)
  }

  # Check available GPU memory
  info <- gpu_info()
  estimated_size <- object.size(df) * 1.5  # Rough GPU overhead
  if (estimated_size > info$memory_free * 0.8) {
    warning("Data may exceed GPU memory, consider chunking")
  }

  TRUE
}
```

### Custom CUDA Kernels (Advanced)

For operations not in libcudf, you can write custom kernels:

```cpp
// src/custom_kernels.cu
// NOTE: This file requires nvcc compilation

#include <cuda_runtime.h>

// Example: Custom kernel for a specialized operation
__global__ void custom_transform_kernel(
    const double* input,
    double* output,
    int n,
    double param1,
    double param2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Custom computation
        output[idx] = input[idx] * param1 + param2 * sqrt(input[idx]);
    }
}

extern "C" void launch_custom_transform(
    const double* input,
    double* output,
    int n,
    double param1,
    double param2
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    custom_transform_kernel<<<num_blocks, block_size>>>(
        input, output, n, param1, param2
    );
    cudaDeviceSynchronize();
}
```

---

## 13. Interoperability

### Arrow C Data Interface

```r
# R/arrow_interop.R

#' Export tbl_gpu to Arrow format (zero-copy when possible)
#'
#' @param x A tbl_gpu object
#' @return A nanoarrow array stream
#' @export
as_nanoarrow_array_stream.tbl_gpu <- function(x, ...) {
  if (length(x$lazy_ops) > 0) {
    x <- compute(x)
  }

  # Get Arrow C Data Interface pointers from GPU table
  arrow_ptrs <- .Call(`_cuplr_export_to_arrow`, x$ptr)

  # Wrap in nanoarrow
  nanoarrow::nanoarrow_pointer_import(
    arrow_ptrs$schema_ptr,
    arrow_ptrs$array_ptr
  )
}

#' Import from Arrow to GPU
#'
#' @param stream A nanoarrow array stream or Arrow Table
#' @return A tbl_gpu object
#' @export
tbl_gpu.nanoarrow_array_stream <- function(data, ...) {
  # Export Arrow C Data Interface pointers
  schema_ptr <- nanoarrow::nanoarrow_pointer_export(data)

  # Import to GPU via cudf's Arrow integration
  gpu_ptr <- .Call(`_cuplr_import_from_arrow`, schema_ptr)

  # Build schema from Arrow metadata
  schema <- extract_schema_from_arrow(data)


  new_tbl_gpu(ptr = gpu_ptr, schema = schema)
}

#' Convert Arrow Table to GPU
#'
#' @param data An Arrow Table
#' @return A tbl_gpu object
#' @export
tbl_gpu.ArrowTabular <- function(data, ...) {
  # Use Arrow's C Data Interface
  stream <- arrow::as_record_batch_reader(data)
  tbl_gpu(nanoarrow::as_nanoarrow_array_stream(stream))
}
```

### C++ Arrow Integration

```cpp
// src/arrow_interop.cpp
#include "gpu_table.hpp"
#include <cudf/interop.hpp>
#include <nanoarrow/nanoarrow.h>

// Export cudf table to Arrow C Data Interface
// [[Rcpp::export]]
Rcpp::List export_to_arrow(SEXP xptr) {
    using namespace cuplr;

    Rcpp::XPtr<GpuTablePtr> ptr(xptr);
    cudf::table_view view = get_table_view(ptr);

    // Allocate Arrow structures
    ArrowSchema* schema = new ArrowSchema;
    ArrowArray* array = new ArrowArray;

    // Use cudf's Arrow export (copies data to host)
    // Note: For true zero-copy, would need CUDA-aware Arrow
    auto arrow_table = cudf::to_arrow(view);

    // Export to C interface
    arrow::ExportRecordBatch(*arrow_table->ToRecordBatch(0).ValueOrDie(),
                             array, schema);

    return Rcpp::List::create(
        Rcpp::Named("schema_ptr") = Rcpp::XPtr<ArrowSchema>(schema),
        Rcpp::Named("array_ptr") = Rcpp::XPtr<ArrowArray>(array)
    );
}

// Import from Arrow C Data Interface
// [[Rcpp::export]]
SEXP import_from_arrow(SEXP schema_xptr, SEXP array_xptr) {
    Rcpp::XPtr<ArrowSchema> schema(schema_xptr);
    Rcpp::XPtr<ArrowArray> array(array_xptr);

    // Import to Arrow C++ then to cudf
    auto result = arrow::ImportRecordBatch(array.get(), schema.get());
    if (!result.ok()) {
        Rcpp::stop("Failed to import Arrow data: %s", result.status().message());
    }

    auto arrow_table = arrow::Table::FromRecordBatches({result.ValueOrDie()});
    auto cudf_table = cudf::from_arrow(*arrow_table.ValueOrDie());

    return cuplr::make_gpu_table_xptr(std::move(cudf_table));
}
```

### Reticulate Bridge to Python RAPIDS

```r
# R/python_bridge.R

#' Call Python cudf when C++ API is insufficient
#'
#' @param gpu_tbl A tbl_gpu object
#' @param py_code Python code to execute (has access to 'df' variable)
#' @return A tbl_gpu object
#' @export
gpu_python <- function(gpu_tbl, py_code) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    cli::cli_abort("Package 'reticulate' is required for Python bridge")
  }

  # Ensure Python cudf is available
  if (!reticulate::py_module_available("cudf")) {
    cli::cli_abort("Python cudf module not found. Install with: pip install cudf-cu12")
  }

  # Export to Arrow, import in Python
  arrow_stream <- as_nanoarrow_array_stream(gpu_tbl)

  reticulate::py_run_string("
import cudf
import pyarrow as pa
")

  # Transfer via Arrow IPC (temporary - real impl would use shared memory)
  temp_file <- tempfile(fileext = ".arrow")
  arrow::write_ipc_file(
    arrow::as_arrow_table(arrow_stream),
    temp_file
  )

  reticulate::py$df <- reticulate::py_eval(
    sprintf("cudf.read_feather('%s')", temp_file)
  )

  # Execute user code
  reticulate::py_run_string(py_code)

  # Get result back
  result_py <- reticulate::py$df

  # Convert back to R via Arrow
  result_arrow <- reticulate::py_eval("df.to_arrow()")

  # Clean up
  unlink(temp_file)

  # Return as tbl_gpu
  tbl_gpu(result_arrow)
}

# Example usage:
# result <- gpu_python(my_tbl, "
#   df = df.drop_duplicates(subset=['col1', 'col2'])
#   df['new_col'] = df['x'].rolling(window=10).mean()
# ")
```

### dtplyr / data.table Interop

```r
# R/dtplyr_interop.R

#' Convert between tbl_gpu and dtplyr lazy tables
#'
#' @param x A lazy_dt or tbl_gpu
#' @return Converted object
#' @export
as_lazy_dt.tbl_gpu <- function(x, ...) {
  if (!requireNamespace("dtplyr", quietly = TRUE)) {
    cli::cli_abort("Package 'dtplyr' is required")
  }

  # Materialize and convert
  df <- collect(x)
  dtplyr::lazy_dt(data.table::as.data.table(df))
}

#' @export
tbl_gpu.dtplyr_step <- function(data, ...) {
  # Collect dtplyr result then transfer to GPU
  df <- dplyr::collect(data)
  tbl_gpu(as.data.frame(df))
}
```

---

## 14. Packaging, Distribution & Licensing

### Recommended License

```
# LICENSE file
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full Apache 2.0 license text]
```

**Rationale**: Apache 2.0 matches RAPIDS/libcudf licensing, ensuring compatibility and allowing commercial use while requiring attribution.

### Why Not CRAN

CRAN distribution is **not recommended** for cuplr because:

1. **Binary dependencies**: libcudf, CUDA runtime not available on CRAN build servers
2. **GPU requirement**: CRAN check servers don't have GPUs
3. **Large binary size**: libcudf is 100+ MB
4. **Version coupling**: Tight dependency on CUDA/driver versions

### Recommended Distribution Channels

| Channel | Audience | Pros | Cons |
|---------|----------|------|------|
| GitHub Releases | Developers | Easy updates, source available | Manual install |
| conda-forge | Data scientists | Dependency resolution, binary | Recipe maintenance |
| Docker Hub | DevOps/CI | Reproducible, complete env | Larger size |
| Internal registry | Enterprise | Controlled, secure | Setup overhead |

### Conda Recipe

```yaml
# recipe/meta.yaml
{% set version = "0.1.0" %}

package:
  name: r-cuplr
  version: {{ version }}

source:
  git_url: https://github.com/yourorg/cuplr
  git_rev: v{{ version }}

build:
  number: 0
  skip: true  # [not linux]
  rpaths:
    - lib/R/lib/
    - lib/

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - cmake
    - make
  host:
    - r-base >=4.3
    - r-rcpp >=1.0.12
    - r-dplyr >=1.1.0
    - r-rlang >=1.1.0
    - r-vctrs >=0.6.0
    - r-pillar >=1.9.0
    - r-glue >=1.6.0
    - r-cli >=3.6.0
    - libcudf >=25.12
    - cudatoolkit >=12.0
  run:
    - r-base >=4.3
    - r-rcpp >=1.0.12
    - r-dplyr >=1.1.0
    - r-rlang >=1.1.0
    - r-vctrs >=0.6.0
    - r-pillar >=1.9.0
    - r-glue >=1.6.0
    - r-cli >=3.6.0
    - libcudf >=25.12
    - cudatoolkit >=12.0
    - __cuda  # Virtual package for CUDA runtime

test:
  commands:
    - $R -e "library(cuplr)"
    - $R -e "cuplr::gpu_info()"  # [gpu]

about:
  home: https://github.com/yourorg/cuplr
  license: Apache-2.0
  license_family: Apache
  license_file: LICENSE
  summary: GPU-accelerated dplyr backend using RAPIDS libcudf
  description: |
    cuplr provides a dplyr-compatible interface for GPU-accelerated
    data manipulation using NVIDIA's RAPIDS libcudf library.

extra:
  recipe-maintainers:
    - your-github-handle
```

### Build Script for Conda

```bash
# recipe/build.sh
#!/bin/bash
set -ex

export CUDA_HOME="${PREFIX}"
export CUDF_HOME="${PREFIX}"

# Run configure
chmod +x configure
./configure

# Build and install
R CMD INSTALL --build .
```

### Release Checklist

```markdown
## Release Checklist for cuplr v{VERSION}

### Pre-release
- [ ] All tests pass locally with GPU
- [ ] Update version in DESCRIPTION
- [ ] Update NEWS.md with changes
- [ ] Update README.md if needed
- [ ] Verify compatibility with latest RAPIDS version
- [ ] Run benchmarks, update benchmark results

### Release
- [ ] Create git tag: `git tag -a v{VERSION} -m "Release v{VERSION}"`
- [ ] Push tag: `git push origin v{VERSION}`
- [ ] Create GitHub Release with changelog
- [ ] Build source tarball: `R CMD build .`
- [ ] Upload tarball to GitHub Release

### Post-release
- [ ] Build and push Docker image
- [ ] Update conda-forge recipe (PR to feedstock)
- [ ] Announce on relevant channels
- [ ] Update documentation site
```

---

## 15. Example User Workflows

### Workflow 1: Basic Data Analysis

```r
library(cuplr)
library(dplyr)

# Check GPU is available
gpu_info()
#> GPU Device Information
#> ======================
#> Device:       NVIDIA A100-SXM4-40GB
#> Compute Cap:  8.0
#> Memory Total: 42.50 GB
#> Memory Free:  41.23 GB
#> Memory Used:  1.27 GB
#> Utilization:  3.0%

# Load data to GPU
sales <- read.csv("sales_100M.csv")
sales_gpu <- tbl_gpu(sales)

# Familiar dplyr pipeline
result <- sales_gpu %>%
  filter(year >= 2020, amount > 0) %>%
  mutate(
    revenue = amount * price,
    quarter = ceiling(month / 3)
  ) %>%
  group_by(region, quarter) %>%
  summarise(
    total_revenue = sum(revenue),
    avg_order = mean(amount),
    n_orders = n()
  ) %>%
  arrange(desc(total_revenue)) %>%
  collect()

print(result)
#> # A tibble: 80 x 5
#>    region  quarter total_revenue avg_order n_orders
#>    <chr>     <dbl>         <dbl>     <dbl>    <int>
#>  1 West          4    1234567890      45.2  2345678
#>  2 East          4    1198765432      42.1  2234567
#> # ... with 78 more rows
```

**Lowered libcudf calls** (with verbose mode):

```
[cuplr] CUDF: binary_operation(col[year], scalar(2020), GREATER_EQUAL) -> mask1
[cuplr] CUDF: binary_operation(col[amount], scalar(0), GREATER) -> mask2
[cuplr] CUDF: binary_operation(mask1, mask2, BITWISE_AND) -> mask_combined
[cuplr] CUDF: apply_boolean_mask(table, mask_combined)
[cuplr] CUDF: binary_operation(col[amount], col[price], MUL) -> revenue_col
[cuplr] CUDF: binary_operation(col[month], scalar(3), DIV) -> temp
[cuplr] CUDF: unary_operation(temp, CEIL) -> quarter_col
[cuplr] CUDF: groupby::groupby(keys=[region, quarter])
[cuplr] CUDF: groupby::aggregate([sum(revenue), mean(amount), count(*)])
[cuplr] CUDF: sort(by=[total_revenue], order=[DESC])
```

### Workflow 2: Joining Large Tables

```r
# Two large tables
customers_gpu <- tbl_gpu(customers_df)   # 50M rows
orders_gpu <- tbl_gpu(orders_df)         # 200M rows

# Join and aggregate
customer_summary <- orders_gpu %>%
  inner_join(customers_gpu, by = "customer_id") %>%
  group_by(customer_segment, region) %>%
  summarise(
    total_orders = n(),
    total_value = sum(order_value),
    avg_value = mean(order_value)
  ) %>%
  collect()

# Timing comparison
library(bench)

bench::mark(
  cpu = orders_df %>%
    inner_join(customers_df, by = "customer_id") %>%
    group_by(customer_segment, region) %>%
    summarise(
      total_orders = n(),
      total_value = sum(order_value),
      avg_value = mean(order_value),
      .groups = "drop"
    ),
  gpu = orders_gpu %>%
    inner_join(customers_gpu, by = "customer_id") %>%
    group_by(customer_segment, region) %>%
    summarise(
      total_orders = n(),
      total_value = sum(order_value),
      avg_value = mean(order_value)
    ) %>%
    collect(),
  check = FALSE,
  min_iterations = 3
)
#> # A tibble: 2 x 6
#>   expression      min   median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr> <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
#> 1 cpu           45.2s    47.8s    0.0209    12.4GB     1.23
#> 2 gpu           1.23s    1.45s    0.689     2.1GB      0.12
```

### Workflow 3: Time Series with Window Functions

```r
# Stock price data
prices_gpu <- tbl_gpu(stock_prices)  # 100M rows

# Calculate moving averages (when supported)
with_indicators <- prices_gpu %>%
  group_by(ticker) %>%
  arrange(date) %>%
  mutate(
    ma_5 = mean(close, .window = 5),
    ma_20 = mean(close, .window = 20),
    returns = (close - lag(close, 1)) / lag(close, 1)
  ) %>%
  ungroup() %>%
  collect()

# For unsupported operations, use Python bridge
advanced_indicators <- gpu_python(prices_gpu, "
import cudf

# cuDF supports more window functions
df['ema_12'] = df.groupby('ticker')['close'].transform(
    lambda x: x.ewm(span=12).mean()
)
df['rsi'] = df.groupby('ticker')['close'].transform(
    lambda x: 100 - (100 / (1 + x.diff().clip(lower=0).rolling(14).mean() /
                                 (-x.diff().clip(upper=0)).rolling(14).mean()))
)
")
```

### Workflow 4: Memory Profiling

```r
# Monitor GPU memory during processing
with_gpu_monitor({
  big_df <- data.frame(
    x = runif(1e8),
    y = runif(1e8),
    g = sample(letters, 1e8, replace = TRUE)
  )

  result <- tbl_gpu(big_df) %>%
    filter(x > 0.5) %>%
    group_by(g) %>%
    summarise(mean_y = mean(y)) %>%
    collect()
})
#>
#> GPU Memory Delta:
#>   Before: 38.50 GB free
#>   After:  36.89 GB free
#>   Change: 1.61 GB
```

---

## 16. Security & Safety Considerations

### GPU Memory Exhaustion

```r
# R/safety.R

#' Safely execute GPU operations with memory limits
#'
#' @param expr Expression to evaluate
#' @param max_memory_gb Maximum GPU memory to use (GB)
#' @param fallback_to_cpu Fall back to CPU if memory exceeded
#' @export
gpu_safe <- function(expr, max_memory_gb = NULL, fallback_to_cpu = TRUE) {
  info <- gpu_info()

  if (!is.null(max_memory_gb)) {
    max_bytes <- max_memory_gb * 1e9
    if (info$memory_free < max_bytes * 0.1) {
      if (fallback_to_cpu) {
        cli::cli_warn("GPU memory low, falling back to CPU")
        return(eval(substitute(expr), envir = parent.frame()))
      } else {
        cli::cli_abort("Insufficient GPU memory: {format_bytes(info$memory_free)} available")
      }
    }
  }

  tryCatch(
    expr,
    error = function(e) {
      if (grepl("out of memory|CUDA_ERROR_OUT_OF_MEMORY", e$message, ignore.case = TRUE)) {
        if (fallback_to_cpu) {
          cli::cli_warn("GPU out of memory, falling back to CPU")
          # Re-evaluate without GPU
          # This requires detecting tbl_gpu and converting to df
          eval(substitute(expr), envir = parent.frame())
        } else {
          cli::cli_abort("GPU out of memory: {e$message}")
        }
      } else {
        stop(e)
      }
    }
  )
}
```

### Input Validation

```cpp
// src/validation.cpp
#include <Rcpp.h>
#include "gpu_table.hpp"

// Validate column index is in bounds
void validate_column_index(int idx, int ncol) {
    if (idx < 0 || idx >= ncol) {
        Rcpp::stop("Column index %d out of bounds [0, %d)", idx, ncol);
    }
}

// Validate numeric value is finite
void validate_finite(double val, const char* param_name) {
    if (!std::isfinite(val)) {
        Rcpp::stop("Parameter '%s' must be finite, got %f", param_name, val);
    }
}

// Validate string doesn't contain injection patterns
bool is_safe_identifier(const std::string& s) {
    // Only allow alphanumeric and underscore
    for (char c : s) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }
    return !s.empty() && !std::isdigit(s[0]);
}

void validate_column_name(const std::string& name) {
    if (!is_safe_identifier(name)) {
        Rcpp::stop("Invalid column name: '%s'. Names must be alphanumeric with underscores.",
                   name.c_str());
    }
}
```

### Expression Sanitization

```r
# R/sanitize.R

# Allowlist of safe functions for GPU execution
SAFE_FUNCTIONS <- c(
  # Arithmetic
  "+", "-", "*", "/", "^", "%%", "%/%",
  # Comparison
  ">", ">=", "<", "<=", "==", "!=",
  # Logical
  "&", "|", "!", "xor",
  # Math
 "abs", "sqrt", "exp", "log", "log10", "log2",
  "sin", "cos", "tan", "asin", "acos", "atan",
  "ceiling", "floor", "round", "trunc",
  # Aggregation
  "sum", "mean", "min", "max", "sd", "var", "n", "median",
  "first", "last",
  # String (subset)
  "nchar", "substr", "toupper", "tolower",
  # Special
  "is.na", "is.null", "ifelse", "case_when", "coalesce",
  # dplyr
  "desc", "lag", "lead", "row_number", "between"
)

validate_expression <- function(expr) {
  if (is.symbol(expr) || is.atomic(expr)) {
    return(TRUE)
  }

  if (is.call(expr)) {
    fn_name <- as.character(expr[[1]])

    # Check for forbidden patterns
    if (fn_name %in% c("system", "system2", "shell", "eval", "parse",
                       "source", "readLines", "writeLines", "file",
                       ".Call", ".External", ".C", ".Fortran")) {
      cli::cli_abort("Function '{fn_name}' is not allowed in GPU expressions")
    }

    # Warn for unknown functions
    if (!fn_name %in% SAFE_FUNCTIONS && !fn_name %in% c("(", "c")) {
      cli::cli_warn("Function '{fn_name}' may not be supported on GPU")
    }

    # Recursively validate arguments
    for (i in seq_along(expr)[-1]) {
      validate_expression(expr[[i]])
    }
  }

  TRUE
}
```

---

## 17. Maintenance & Migration Notes

### RAPIDS Version Upgrade Process

```markdown
## Upgrading RAPIDS libcudf Version

### 1. Check Release Notes
- Visit https://docs.rapids.ai/notices/rsn/
- Note API changes, deprecations, new features

### 2. Update Build Configuration
```bash
# Update Dockerfile base image
sed -i 's/rapidsai\/base:25.12/rapidsai\/base:26.02/g' Dockerfile

# Update conda recipe
# Edit recipe/meta.yaml: libcudf >=26.02
```

### 3. Test Compilation
```bash
# In Docker environment
./configure
R CMD build .
R CMD check cuplr_*.tar.gz
```

### 4. Run Full Test Suite
```bash
R -e "testthat::test_package('cuplr')"
```

### 5. Update Feature Detection
If new APIs are available:
```r
# R/compat.R
has_cudf_feature <- function(feature) {
  switch(feature,
    "distinct_count" = .Call(`_cuplr_has_distinct_count`),
    "regex_replace" = packageVersion("cuplr") >= "0.2.0",
    FALSE
  )
}
```

### 6. Document Changes
- Update NEWS.md
- Update README with new requirements
- Update DESCRIPTION SystemRequirements
```

### CUDA Toolkit Support Matrix

```r
# R/compat.R

CUDA_SUPPORT_MATRIX <- list(
  "25.12" = list(cuda_min = "12.0", cuda_max = "12.5", driver_min = "525.60.13"),
  "26.02" = list(cuda_min = "12.0", cuda_max = "12.6", driver_min = "535.54.03")
)

check_cuda_compat <- function(rapids_version = NULL) {
  # Detect RAPIDS version if not specified
  if (is.null(rapids_version)) {
    rapids_version <- .Call(`_cuplr_rapids_version`)
  }

  compat <- CUDA_SUPPORT_MATRIX[[rapids_version]]
  if (is.null(compat)) {
    cli::cli_warn("Unknown RAPIDS version: {rapids_version}")
    return(invisible(FALSE))
  }

  cuda_version <- .Call(`_cuplr_cuda_version`)
  driver_version <- .Call(`_cuplr_driver_version`)

  issues <- character()

  if (compareVersion(cuda_version, compat$cuda_min) < 0) {
    issues <- c(issues, glue::glue(
      "CUDA {cuda_version} is below minimum {compat$cuda_min}"
    ))
  }

  if (compareVersion(cuda_version, compat$cuda_max) > 0) {
    issues <- c(issues, glue::glue(
      "CUDA {cuda_version} is above maximum {compat$cuda_max}"
    ))
  }

  if (length(issues) > 0) {
    cli::cli_warn(c("Compatibility issues detected:", issues))
    return(invisible(FALSE))
  }

  cli::cli_alert_success("CUDA/RAPIDS compatibility OK")
  invisible(TRUE)
}
```

### Deprecation Policy

```r
# R/deprecated.R

#' @name cuplr-deprecated
#' @title Deprecated functions in cuplr
#'
#' These functions are deprecated and will be removed in future versions.
NULL

# Example deprecation wrapper
gpu_table <- function(...) {
  lifecycle::deprecate_warn(
    when = "0.2.0",
    what = "gpu_table()",
    with = "tbl_gpu()"
  )
  tbl_gpu(...)
}
```

### Migration Guide Template

```markdown
## Migrating from cuplr 0.x to 1.0

### Breaking Changes

1. **Function renamed**: `gpu_table()` → `tbl_gpu()`
   ```r
   # Old
   gpu_table(df)
   # New
   tbl_gpu(df)
   ```

2. **Lazy evaluation default**: Operations are now lazy by default
   ```r
   # Old (eager)
   result <- tbl_gpu(df) %>% filter(x > 10)
   # result is immediately computed

   # New (lazy)
   result <- tbl_gpu(df) %>% filter(x > 10)
   # result is lazy, call collect() or compute()
   result <- result %>% collect()
   ```

3. **NA handling**: Now follows R semantics more closely
   - `filter(x > 10)` excludes NA values (like R)
   - Use `filter(x > 10 | is.na(x))` to include NAs

### New Features

- Window functions: `lag()`, `lead()`, `row_number()`
- String operations: Full stringr compatibility
- Arrow interop: Zero-copy data exchange

### Deprecated

- `as.gpu.data.frame()` - use `tbl_gpu()` instead
- `gpu_collect()` - use `collect()` instead
```

## 18. Deliverables Checklist

### Required Deliverables

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Developer Guide (this document) | ✓ | `DEVELOPER_GUIDE.md` |
| 2 | R Package Skeleton | ✓ | `cuplr/` directory |
| 3 | Unit Tests (6+) | ✓ | `cuplr/tests/testthat/` |
| 4 | Integration Tests (2+) | ✓ | `cuplr/tests/testthat/test-integration.R` |
| 5 | GitHub Actions CI | ✓ | `cuplr/.github/workflows/ci.yml` |
| 6 | Benchmark Scripts | ✓ | `cuplr/inst/benchmarks/` |
| 7 | Conda Recipe | ✓ | `cuplr/recipe/meta.yaml` |
| 8 | Dockerfile | ✓ | `cuplr/inst/docker/Dockerfile` |

### Package File Inventory

```
cuplr/
├── DESCRIPTION                 # Package metadata
├── NAMESPACE                   # Exports and imports
├── LICENSE                     # Apache 2.0
├── configure                   # Build configuration script
├── configure.win              # Windows stub (unsupported message)
├── README.md                   # Package README
├── NEWS.md                     # Changelog
├── .Rbuildignore              # Build exclusions
├── R/
│   ├── zzz.R                  # Package hooks
│   ├── tbl_gpu.R              # Core class definition
│   ├── dplyr-filter.R         # filter() implementation
│   ├── dplyr-select.R         # select() implementation
│   ├── dplyr-mutate.R         # mutate() implementation
│   ├── dplyr-arrange.R        # arrange() implementation
│   ├── dplyr-group.R          # group_by/ungroup implementation
│   ├── dplyr-summarise.R      # summarise() implementation
│   ├── dplyr-join.R           # join implementations
│   ├── collect.R              # collect/compute implementation
│   ├── ast.R                  # AST node definitions
│   ├── parse_expr.R           # Expression parser
│   ├── optimizer.R            # Query optimizer
│   ├── lower.R                # AST to libcudf lowering
│   ├── arrow_interop.R        # Arrow integration
│   ├── logging.R              # Verbose mode logging
│   ├── diagnostics.R          # GPU info and monitoring
│   ├── safety.R               # Memory limits and safety
│   ├── sanitize.R             # Expression validation
│   ├── compat.R               # Version compatibility
│   └── utils.R                # Utility functions
├── src/
│   ├── Makevars.in            # Build template
│   ├── init.cpp               # R registration
│   ├── gpu_table.hpp          # XPtr wrapper header
│   ├── transfer.cpp           # R <-> GPU data transfer
│   ├── filter.cpp             # Filter operations
│   ├── sort.cpp               # Sort operations
│   ├── groupby.cpp            # GroupBy operations
│   ├── join.cpp               # Join operations
│   ├── binary_ops.cpp         # Binary operations
│   ├── arrow_interop.cpp      # Arrow C Data Interface
│   ├── diagnostics.cpp        # GPU info
│   └── RcppExports.cpp        # Generated exports
├── inst/
│   ├── docker/
│   │   └── Dockerfile         # Development container
│   └── benchmarks/
│       ├── run_benchmarks.R   # Benchmark suite
│       └── results/           # Benchmark output
├── tests/
│   ├── testthat.R             # Test runner
│   └── testthat/
│       ├── helper-cuplr.R     # Test helpers
│       ├── test-basic.R       # Basic functionality
│       ├── test-filter.R      # Filter tests
│       ├── test-mutate.R      # Mutate tests
│       ├── test-arrange.R     # Arrange tests
│       ├── test-group.R       # Group by tests
│       ├── test-join.R        # Join tests
│       ├── test-integration.R # Integration tests
│       └── test-edge-cases.R  # Edge case tests
├── man/                        # Generated documentation
├── recipe/
│   ├── meta.yaml              # Conda recipe
│   └── build.sh               # Conda build script
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI
```

### Test Coverage Requirements

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test-basic.R | 4 | tbl_gpu creation, collect, print, dim |
| test-filter.R | 4 | >, <, ==, multiple conditions, NA handling |
| test-mutate.R | 3 | arithmetic, new columns, type preservation |
| test-arrange.R | 2 | ascending, descending |
| test-group.R | 3 | group_by, summarise, ungroup |
| test-join.R | 2 | left_join, inner_join |
| test-integration.R | 2 | complex pipelines, GPU vs CPU comparison |
| test-edge-cases.R | 4 | empty tables, all NA, large data, type edge cases |
| **Total** | **24** | |

---

## 19. Search Keywords & Primary Resources

### Search Keywords for Research

```
# Core Technology
libcudf API
RAPIDS libcudf C API
cudf cpp api
cudf::table
cudf::column
cudf::groupby::groupby
cudf::binary_operation
cudf::apply_boolean_mask
cudf::sort

# R Integration
Rcpp external pointer
Rcpp XPtr libcudf
Rcpp compiling with external libs
R package configure CUDA
R CMD INSTALL with CUDA
cpp11 CUDA integration

# dplyr Backend
dplyr backend implementation
dbplyr translation
dplyr S3 methods filter mutate
vctrs R package
pillar tbl_format

# Interoperability
Arrow C Data Interface
nanoarrow R package
reticulate RAPIDS
cudf arrow integration

# Build and Deploy
contrib build GPU packages
R package GitHub Actions GPU
nvidia container toolkit
RAPIDS docker images
conda-forge GPU packages

# Performance
cudf kernel fusion
RAPIDS memory management
RMM RAPIDS Memory Manager
GPU DataFrame performance
```

### Primary Documentation Resources

| Resource | URL | Use For |
|----------|-----|---------|
| libcudf API Docs | https://docs.rapids.ai/api/libcudf/stable/ | C++ API reference |
| libcudf Developer Guide | https://docs.rapids.ai/api/libcudf/stable/developer_guide | Design patterns |
| RAPIDS Installation | https://docs.rapids.ai/install/ | Version requirements |
| RAPIDS Support Notices | https://docs.rapids.ai/notices/rsn/ | Compatibility |
| cuDF GitHub | https://github.com/rapidsai/cudf | Source, examples |
| dbplyr New Backend | https://dbplyr.tidyverse.org/articles/new-backend.html | dplyr backend guide |
| dbplyr Translation | https://dbplyr.tidyverse.org/articles/translation-function.html | Expression translation |
| fstplyr Implementation | https://krlmlr.github.io/fstplyr/articles/implement.html | Non-DB backend example |
| Rcpp Documentation | https://dirk.eddelbuettel.com/code/rcpp/ | C++ integration |
| Rcpp XPtr Reference | https://dirk.eddelbuettel.com/code/rcpp/html/classRcpp_1_1XPtr.html | External pointers |
| nanoarrow R Package | https://arrow.apache.org/nanoarrow/latest/r/ | Arrow C interface |
| Arrow C Data Interface | https://arrow.apache.org/docs/format/CDataInterface.html | Zero-copy spec |
| vctrs Package | https://vctrs.r-lib.org/ | Type system |
| rlang Package | https://rlang.r-lib.org/ | Quosures, expressions |

### Example Projects to Study

| Project | URL | Relevance |
|---------|-----|-----------|
| cudf-python | https://github.com/rapidsai/cudf/tree/main/python | Python bindings pattern |
| dbplyr | https://github.com/tidyverse/dbplyr | SQL translation |
| dtplyr | https://github.com/tidyverse/dtplyr | data.table backend |
| arrow-r | https://github.com/apache/arrow/tree/main/r | Arrow R bindings |
| duckplyr | https://github.com/duckdb/duckplyr | DuckDB backend |

### RAPIDS C++ Headers to Study

```cpp
// Essential headers to understand
#include <cudf/table/table.hpp>          // cudf::table
#include <cudf/table/table_view.hpp>     // cudf::table_view
#include <cudf/column/column.hpp>        // cudf::column
#include <cudf/column/column_view.hpp>   // cudf::column_view
#include <cudf/types.hpp>                // type_id, data_type
#include <cudf/copying.hpp>              // slice, gather, scatter
#include <cudf/sorting.hpp>              // sort, sorted_order
#include <cudf/stream_compaction.hpp>    // apply_boolean_mask, distinct
#include <cudf/groupby.hpp>              // groupby::groupby
#include <cudf/aggregation.hpp>          // make_*_aggregation
#include <cudf/binaryop.hpp>             // binary_operation
#include <cudf/unary.hpp>                // unary_operation
#include <cudf/join.hpp>                 // left_join, inner_join
#include <cudf/interop.hpp>              // Arrow interop
#include <cudf/scalar/scalar.hpp>        // scalar types
#include <cudf/scalar/scalar_factories.hpp>
#include <rmm/device_buffer.hpp>         // GPU memory
#include <rmm/mr/device/per_device_resource.hpp>
```

---

## Appendix A: Quick Reference Card

### Creating GPU Tables

```r
# From data.frame
gpu_df <- tbl_gpu(df)

# From CSV (via Arrow for efficiency)
gpu_df <- tbl_gpu(arrow::read_csv_arrow("data.csv"))

# Check status
is_tbl_gpu(gpu_df)
dim(gpu_df)
names(gpu_df)
```

### dplyr Verbs

```r
# All standard verbs work
gpu_df %>%
  filter(x > 10) %>%
  select(x, y) %>%
  mutate(z = x + y) %>%
  arrange(desc(z)) %>%
  group_by(category) %>%
  summarise(
    total = sum(z),
    avg = mean(z),
    n = n()
  ) %>%
  collect()
```

### Joins

```r
left_join(gpu_df1, gpu_df2, by = "key")
inner_join(gpu_df1, gpu_df2, by = c("k1" = "k2"))
```

### Materialization

```r
# Execute and return to R
collect(gpu_df)

# Execute and keep on GPU
compute(gpu_df)
```

### Diagnostics

```r
# GPU info
gpu_info()

# Verbose mode
options(cuplr.verbose = TRUE)

# Memory monitoring
with_gpu_monitor({
  # operations
})

# Dump AST
dump_ast(lazy_gpu_df)
```

### Safety

```r
# Memory-safe execution
gpu_safe({
  tbl_gpu(big_df) %>% filter(x > 10) %>% collect()
}, max_memory_gb = 10, fallback_to_cpu = TRUE)
```

---

## Appendix B: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "GPU not found" | No CUDA driver | Install NVIDIA driver |
| "libcudf.so not found" | Missing library | Check LD_LIBRARY_PATH |
| "CUDA out of memory" | Data too large | Use chunking or filter early |
| "Unsupported type" | Non-standard R type | Convert to supported type first |
| "configure fails" | Missing CUDA/cudf | Install RAPIDS, set CUDA_HOME |

### Diagnostic Commands

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check libcudf
ldconfig -p | grep libcudf
pkg-config --libs cudf

# Check R can find library
R -e ".Call('_cuplr_check_gpu')"

# Full diagnostics
R -e "cuplr::gpu_info()"
```

### Memory Debugging

```r
# Track allocations
options(cuplr.verbose = TRUE)

# Force garbage collection
gc()

# Check GPU memory
gpu_info()$memory_free

# Use smaller chunks
result <- gpu_chunked(big_df, chunk_size = 1e6, function(chunk) {
  chunk %>% filter(x > 10) %>% collect()
})
```

---

## Appendix C: Performance Tips

1. **Filter early**: Reduce data size before expensive operations
2. **Stay on GPU**: Chain operations without collecting
3. **Use lazy mode**: Let optimizer fuse operations
4. **Pre-sort if possible**: Tell groupby if keys are sorted
5. **Avoid strings when possible**: Numeric operations are faster
6. **Chunk large data**: Process in batches if memory-constrained
7. **Profile first**: Use verbose mode to identify bottlenecks

```r
# Example optimized pipeline
result <- tbl_gpu(df) %>%
  filter(year >= 2020) %>%          # Filter first
  select(year, region, amount) %>%   # Project only needed columns
  mutate(amount_adj = amount * 1.1) %>%
  group_by(year, region) %>%
  summarise(total = sum(amount_adj)) %>%
  collect()                          # Single collect at end
```

---

*End of Developer Guide*

**Document Version**: 1.0.0
**Last Updated**: 2025
**RAPIDS Target**: 25.12+
**Maintainer**: [Your Name]
