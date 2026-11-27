# Create an IR vector-literal node (mutate()/transmute() only)

A whole-dot constant vector (`mutate(y = 1:5)`): its length is the
table's row count (or the call is an error), so it is uploaded as a
column at lowering time
([`lower_mutate()`](https://bbtheo.github.io/cuplyr/reference/lower_mutate.md)),
never evaluated by the GPU expression engine. Only ever the top node of
a mutate expression record.

## Usage

``` r
ir_vec(value, type, quo, name, loss = NULL)
```

## Arguments

- value:

  Atomic vector (names stripped), length != 1

- type:

  GPU type string

- quo:

  The named quosure of the dot, used to raise dplyr's own error when the
  length doesn't fit

- name:

  Output column name

- loss:

  What the upload would change in a Date/POSIXct `value`
  (`timestamp_vector_loss()`), or `NULL` (any other vector). The caller
  that records it also warns for it when the call is made, so it is
  marked `warned`:
  [`lower_mutate()`](https://bbtheo.github.io/cuplyr/reference/lower_mutate.md)
  mutes the upload's own warning for such a vector only (a plan would
  repeat it at every run).

## Value

A `cuplyr_ir_vec` node
