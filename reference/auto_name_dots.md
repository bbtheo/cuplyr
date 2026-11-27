# Auto-name unnamed dots, warning as dplyr's verbs do

Verbs such as
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html)
accept unnamed expressions (e.g. `mutate(mpg * 2)`) and, matching dplyr,
fall back to the expression's deparsed text as the output column name –
while warning that an explicit name would be clearer. This is the single
shared implementation for
[`mutate()`](https://dplyr.tidyverse.org/reference/mutate.html) and
[`summarise()`](https://dplyr.tidyverse.org/reference/summarise.html),
parameterized by `verb` so each caller's warning names itself correctly.

## Usage

``` r
auto_name_dots(dots, verb)
```

## Arguments

- dots:

  A list of quosures, as returned by `rlang::enquos(...)`

- verb:

  Character, the verb name to use in the warning text (e.g. `"mutate"`,
  `"summarise"`)

## Value

`dots`, with `names(dots)` filled in for any unnamed entries
