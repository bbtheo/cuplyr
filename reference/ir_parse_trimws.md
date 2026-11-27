# Parse `trimws(x, which = "both", whitespace = "[ \\t\\r\\n]")`

Only the default `whitespace=` (space/tab/CR/LF – cuDF `strip()`'s
(`strings/strip.hpp`) own "empty `to_strip` means strip whitespace"
behavior) is implemented – an explicit `whitespace=` argument at all
falls back. `which=` maps to `cudf::strings::side_type`
(`"both"`/`"left"`/`"right"`), matched via
[`pmatch()`](https://rdrr.io/r/base/pmatch.html) against the 3 full
names (real [`trimws()`](https://rdrr.io/r/base/trimws.html) uses
[`match.arg()`](https://rdrr.io/r/base/match.arg.html) internally, which
also accepts an unambiguous partial match like `"l"`).

## Usage

``` r
ir_parse_trimws(expr, env, schema)
```
