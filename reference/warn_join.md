# Raise a `dplyr_warning_join*` condition with dplyr's own class-vector shape

Mirrors `dplyr:::warn_join()`/`dplyr:::warn_dplyr()`: the final class
vector is `c(class, "dplyr_warning_join", "dplyr_warning")`, with
[`rlang::warn()`](https://rlang.r-lib.org/reference/abort.html)
appending `"rlang_warning"`/`"warning"`/`"condition"`.
[`rlang::warn()`](https://rlang.r-lib.org/reference/abort.html) has no
`call` parameter at all (unlike
[`rlang::abort()`](https://rlang.r-lib.org/reference/abort.html)), so
there's nothing to suppress here.

## Usage

``` r
warn_join(message, class)
```
