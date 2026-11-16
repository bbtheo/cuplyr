# Resolve a call's function symbol to the actual function object it invokes

Resolve a call's function symbol to the actual function object it
invokes

## Usage

``` r
resolve_call_fn(head, env)
```

## Arguments

- head:

  The call's own head (`expr[[1]]`, as derived by `ir_parse_expr()`'s
  call-handling branch) – a bare symbol, or a `pkg::fn`/`pkg:::fn` call

- env:

  The quosure's environment (used only for a bare symbol)

## Value

The resolved function, or `NULL` if it can't be resolved (an undefined
symbol, or a non-function binding)
