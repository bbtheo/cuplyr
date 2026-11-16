# Extract innermost window calls from an IR tree (one decomposition pass)

Implements section 1.2's decomposition algorithm: a single bottom-up
walk that replaces every window-class call node whose OWN arguments
contain no (nested) window call anywhere – i.e. every *innermost* window
call – with an
[`ir_col()`](https://bbtheo.github.io/cuplyr/reference/ir_col.md)
reference to a freshly materialized (or, on a cache hit, previously
materialized) temp column, leaving any window call that is NOT yet
innermost (its arguments still contain a deeper window call) untouched
under its own op name for a later pass (see
[`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md)'s
driving loop) once that deeper call has been resolved.

## Usage

``` r
ir_extract_windows(ir, schema, state)
```

## Arguments

- ir:

  An IR node (typically an entire mutate/filter expression)

- schema:

  The schema this pass starts from (list(names=, types=)) – used to
  resolve column references inside window-call arguments and pre-mutate
  steps; NOT updated mid-pass (a single call's own arguments can only
  ever reference genuinely pre-existing columns, never a sibling temp
  created earlier in the SAME pass, since window-bearing expressions
  within one pass are independent of each other by construction –
  callers advance to a new schema between passes, see
  [`plan_window_stages()`](https://bbtheo.github.io/cuplyr/reference/plan_window_stages.md))

- state:

  A mutable environment as described in `extract_one_window_call()`'s
  docs; the caller creates a fresh one per pass and inspects
  `state$pre`/`state$specs` afterward

## Value

The IR tree with every innermost window call (this pass) replaced by an
[`ir_col()`](https://bbtheo.github.io/cuplyr/reference/ir_col.md)
reference; identical (by structure) to `ir` if it contained no window
calls at all

## Details

Critically, the innermost-ness test for a given call node uses that
node's ORIGINAL (pre-walk) arguments, not the
post-walk/post-substitution ones – this is what makes `lag(cumsum(x))`
take two passes rather than collapsing into one: on pass 1, `lag`'s own
argument (`cumsum(x)`) does contain a window call, so `lag` is left
as-is (its argument is still recursed into, extracting `cumsum(x)` –
which IS innermost on pass 1 – into a temp column reference); only on
pass 2, once `lag`'s argument has become a plain
[`ir_col()`](https://bbtheo.github.io/cuplyr/reference/ir_col.md), does
`lag` itself qualify as innermost.

CSE (common subexpression elimination): two structurally
[`identical()`](https://rdrr.io/r/base/identical.html) window calls
(same op, same args, same meta) anywhere in the tree(s) processed
against the same `state` resolve to the SAME temp column, checked via a
hash key
([`rlang::hash()`](https://rlang.r-lib.org/reference/hash.html), already
an indirect dependency via rlang) rather than pairwise
[`identical()`](https://rdrr.io/r/base/identical.html) comparisons, for
a cheap single-pass lookup. `state$seen` is the cache, keyed by that
hash.

Any window call argument that isn't already a bare column reference
(e.g. `mean(x + y)`) is hoisted into its own pre-mutate step first (an
`..winarg{k}..` temp column, added to `state$pre`) – see
`extract_one_window_call()`.
