# Register library paths via ldconfig for runtime dlopen()

Writes library paths to /etc/ld.so.conf.d/00-cuplyr-rapids.conf and runs
ldconfig. This ensures the dynamic linker finds conda's newer libstdc++
even when LD_LIBRARY_PATH changes after process startup don't affect
dlopen().

## Usage

``` r
register_library_paths(paths)
```

## Arguments

- paths:

  Character vector of library directories to register.

## Value

Logical: TRUE if ldconfig succeeded, FALSE otherwise.
