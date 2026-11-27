# Physical GPU storage type for a logical schema type

cuplyr keeps factor columns as plain INT32 codes physically on the GPU
rather than adopting cudf's real `DICTIONARY32` column type (a
dictionary-encoded keys+indices structure). `"DICTIONARY32"` is
therefore a LOGICAL type only, in `gpu_type_from_r()` and every schema's
`$types` – confirmed by `df_to_gpu()`: a factor's `TYPEOF()` is
`INTSXP`, so it uploads through the exact same `integer_to_gpu()` path
as a real integer column (`src/transfer_io.cpp`).

## Usage

``` r
gpu_physical_type(type)
```

## Arguments

- type:

  A GPU type string (e.g. from `schema$types`), or a vector of them

## Value

`"INT32"` for `"DICTIONARY32"`, `type` unchanged otherwise

## Details

Call sites split into two camps:

- **Physical-column sites** (talk to the actual `cudf::column`, which is
  INT32 for a factor): C++ casting primitives (`gpu_cast_column()`,
  `gpu_make_null_column()`), transfer-layer assertions. These must route
  the type string through this helper first, or hand the C++ side a
  `"DICTIONARY32"` it has no case for.

- **Logical-semantics sites** (care whether a column IS a factor, not
  what bytes back it): the mutate arithmetic guard
  (`R/ast.R::infer_mutate_output_type()`), join key type-compatibility
  checks
  ([`resolve_join_key_types()`](https://bbtheo.github.io/cuplyr/reference/resolve_join_key_types.md),
  `R/join.R`), memory accounting (`R/gpu-memory.R`). These must keep
  seeing `"DICTIONARY32"` verbatim – routing them through this helper
  would make a factor column silently indistinguishable from a real
  integer column (e.g. letting arithmetic on a factor slip through, or
  joining a factor key against a plain integer key as if they were the
  same domain).
