# Reject any window spec `gpu_window()` doesn't implement

See
[`window_spec_lowerable()`](https://bbtheo.github.io/cuplyr/reference/window_spec_lowerable.md)'s
docs for the shared predicate this wraps with an internal-error
[`stop()`](https://rdrr.io/r/base/stop.html) call.

## Usage

``` r
validate_window_spec(spec)
```
