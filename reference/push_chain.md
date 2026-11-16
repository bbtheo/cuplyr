# Run a verb's node chain as ONE eager execution

A few eager verbs push several nodes (a window mutate is window -\>
mutate -\> select). Pushed one by one, each node builds a full table;
here `build()` gets a lazy view of `.data` (so its
[`push_op()`](https://bbtheo.github.io/cuplyr/reference/push_op.md)s
only do schema work), and the chain it returns is lowered once, with no
optimizer (as for any eager node), into one result table. Fields and
groups come from the lazy pushes, which derive them exactly as the eager
ones do; the frame metadata (attributes, bare class) is carried once
from `.data` at the end, which is what carrying it through each step
comes to. A lazy `.data` is just handed to `build()`.

## Usage

``` r
push_chain(.data, build)
```

## Arguments

- .data:

  A `tbl_gpu`

- build:

  `function(x)` pushing nodes on `x` with `push_op(carry = FALSE)`;
  returns the resulting `tbl_gpu`

## Value

A `tbl_gpu` in `.data`'s exec_mode
