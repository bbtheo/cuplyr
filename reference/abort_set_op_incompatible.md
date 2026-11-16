# Raise dplyr's own "`x` and `y` are not compatible." error

`bullets` is pre-formatted, final display text (already `"✖ "`- prefixed
where dplyr itself would show a bullet, plain otherwise – see the "`y`
must be a data frame." case, which dplyr renders with NO bullet marker
at all). Joined into a single pre-formatted string rather than handed to
[`rlang::abort()`](https://rlang.r-lib.org/reference/abort.html) as a
named (`x = ...`) character vector: verified empirically that
[`rlang::abort()`](https://rlang.r-lib.org/reference/abort.html)'s
automatic cli bullet decoration for an UNNAMED continuation line depends
on the calling package/frame context in ways this package can't reliably
control, so bullets are embedded literally instead of relying on that
formatting – this reproduces dplyr's exact
[`conditionMessage()`](https://rdrr.io/r/base/conditions.html)
byte-for-byte regardless of context.

## Usage

``` r
abort_set_op_incompatible(bullets)
```
