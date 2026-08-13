# Lazily mask invalid observation-feature cells with missing values

Lazily mask invalid observation-feature cells with missing values

## Usage

``` r
validity_masked_source(source, observation_mask_id, bank)
```

## Arguments

- source:

  Observation-by-feature array source.

- observation_mask_id:

  One mask-bank ID per source row.

- bank:

  Compatible `mask_bank`.

## Value

A serializable `validity_masked_source`.
