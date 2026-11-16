# Create a window AST node

One node per window-decomposition "stage"
(`scratchpad/phase5_window_design.md` section 2): evaluates every
[`window_spec()`](https://bbtheo.github.io/cuplyr/reference/window_spec.md)
in `specs` (`R/window.R`) against `input`'s current row set/order and
APPENDS one new column per spec (never replaces an existing column –
temp names guarantee this). Built by a verb's own window-wiring
(`R/mutate.R`/`R/filter.R`/`R/slice.R`, from W3/W5/W6/W7 onward) or, in
W2, by hand for `test-window-native.R`.

## Usage

``` r
ast_window(input, specs, group_cols = character(), on_sum_overflow = "error")
```

## Arguments

- input:

  Input AST node

- specs:

  List of
  [`window_spec()`](https://bbtheo.github.io/cuplyr/reference/window_spec.md)
  structures (`R/window.R`)

- group_cols:

  Character vector of group-by column names –
  [`character()`](https://rdrr.io/r/base/character.html) for an
  ungrouped window (section 3: lowering substitutes a constant key
  column and skips the sort when there's also no `order_cols` on every
  spec)

- on_sum_overflow:

  (F42) What to do when an integer/logical window
  [`sum()`](https://rdrr.io/r/base/sum.html) (declared INT32) leaves R's
  integer range at execution. `"error"`: raise
  `cuplyr_error_sum_overflow` (mutate() and eager filter(): eager reruns
  the call on the CPU; a lazy
  [`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) only
  plans a sum of a logical, an INT32 sum runs at the verb, F80).
  `"double"`: the column comes back FLOAT64, as in dplyr, for a consumer
  that reads the real column type (lazy
  [`filter()`](https://dplyr.tidyverse.org/reference/filter.html)'s
  predicate). A spec's own `on_sum_overflow` field, when set, overrides
  this for that spec
  ([`filter()`](https://dplyr.tidyverse.org/reference/filter.html) sets
  `"error"` on a sum a later stage consumes).

## Value

An ast_window node
