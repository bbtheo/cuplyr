# Raise a `dplyr_error_join*` condition with dplyr's own class-vector shape

Mirrors `dplyr:::stop_join()`/`dplyr:::stop_dplyr()`: the final class
vector is `c(class, "dplyr_error_join", "dplyr_error")`, with
[`rlang::abort()`](https://rlang.r-lib.org/reference/abort.html)
appending `"rlang_error"`/`"error"`/`"condition"`. No `call` is threaded
through (unlike dplyr's own `error_call` – the user's
[`left_join()`](https://dplyr.tidyverse.org/reference/mutate-joins.html)/etc.
call) – this package's
[`lower_join()`](https://bbtheo.github.io/cuplyr/reference/lower_join.md)
call chain doesn't preserve that frame the way dplyr's own
`error_call = caller_env()` threading does, and
[`rlang::abort()`](https://rlang.r-lib.org/reference/abort.html)'s own
default (`call = NULL`, no call shown) is fine here:
[`conditionMessage()`](https://rdrr.io/r/base/conditions.html)/`cnd_message()`
(what every test in test-dplyr-join-rows.R compares) are unaffected
either way – the call only ever shows in the *printed* "Error in ...:"
header, never the message text itself.

## Usage

``` r
abort_join(message, class)
```
