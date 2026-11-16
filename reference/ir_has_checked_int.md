# Does an IR subtree hold an F42 checked integer node?

A checked node is a binary `+`, `-` or `*` whose operands are all
INT32/BOOL8 (a bare untyped NA literal adopts its sibling's type): the
rule of `checked_int_kind()` kind 1 in `src/expr_eval.hpp`. Such a node
can warn "NAs produced by integer overflow", and dplyr's warning depends
on the rows its verb receives, so the optimizer must not change them
(F70). `%/%` and `%%` never warn and are not checked, nor is a node with
a null literal operand (`a + NA`; its other operand is still walked).
Asked through `record_checked_int()`, once per record and only by the
optimizer (F110).

## Usage

``` r
ir_has_checked_int(ir, schema)
```

## Arguments

- ir:

  An IR node (or `NULL`)

- schema:

  List with `names`/`types` the IR is bound against

## Value

A single logical

## Details

One walk visits each node once and stops at the first checked node.
Column, literal and vector operands are typed off the schema/IR (both
first, so a double leaf settles a node without typing a call operand). A
call operand is typed bottom-up as the walk descends into it
(`ir_call_type()`, the step
[`ir_infer_type()`](https://bbtheo.github.io/cuplyr/reference/ir_infer_type.md)
takes per node), so a chain of `+` nodes is typed once, not once per
level. A type that cannot be inferred counts as checked. The walk is
`ir_checked_walk()`, a top-level function: it runs when the optimizer
asks, not on every eager verb call.
