# Extract physical array bindings from a frame

Storage codecs use this helper to pair each source-free FDS array
declaration with its current physical or in-memory `array_source`.

## Usage

``` r
fds_frame_bindings(x)
```

## Arguments

- x:

  An `fmri_frame`.

## Value

A named list of `array_source` descriptors keyed exactly like the
manifest `arrays` registry.
