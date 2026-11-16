# The Phase 11 L6 namespace guard: TRUE iff a call's function symbol resolves to exactly `lubridate::<name>`

`FALSE` (never an error) whenever lubridate isn't installed, `name`
isn't one of its exports, or the call's own function symbol resolves to
anything else (including simply not resolving at all) – every one of
those cases must fall back to L1's CPU path, never hard-error here.

## Usage

``` r
ir_require_lubridate_fn(head, env, name)
```
