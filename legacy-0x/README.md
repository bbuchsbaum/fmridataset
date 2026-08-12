# Frozen fmridataset 0.x architecture

This directory preserves the source and user guides for the pre-frame 0.x
architecture. It is excluded from R package builds and is not loaded into the
1.0 namespace.

The frozen boundary commit is
`21de2ed441939a789d3704df52b959c7264e7b0c`. Applications that still depend on
`fmri_dataset`, storage backends, sampling frames, `fmri_series`, latent
datasets, BIDS archive wrappers, chunk iterators, or `fmri_group` should pin
that final 0.x line while migrating.

The supported 1.0 migration entry point is `upgrade_dataset()`. See
`inst/architecture/ADR-003-legacy-compatibility-boundary.md` for the decision
and explicit supported/unsupported cases.
