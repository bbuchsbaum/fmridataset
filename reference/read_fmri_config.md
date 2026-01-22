# read a basic fMRI configuration file

Reads a fMRI configuration file in YAML or JSON format. This replaces
the previous implementation that used source() for security reasons.

## Usage

``` r
read_fmri_config(file_name, base_path = NULL)
```

## Arguments

- file_name:

  name of configuration file (YAML or JSON format)

- base_path:

  the file path to be prepended to relative file names

## Value

a `fmri_config` instance
