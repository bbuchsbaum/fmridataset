# Test tiers

The computational contracts are split by the evidence they provide.

## Pull requests

Run `devtools::test()` and `R CMD check` with small deterministic fixtures.
This tier includes contract, differential, metamorphic, randomized property,
adversarial, serialization, and zero-read instrumentation tests. Tests may use
virtual large shapes but must not rely on timing thresholds.

## Nightly

Run optional-backend conformance against real NIfTI and HDF5 files, randomized
source/view cases, interrupted-I/O tests, and performance guardrails. The
controlled performance runner records source bytes, read amplification, peak
RSS, file handles, package revisions, and host configuration.

## Release

Run the pull-request and nightly tiers on all supported R and operating-system
combinations, then run the physical 40 GB reference workload. A release report
must distinguish passed, failed, skipped, and unrun gates. Zarr results are
reported as experimental and do not certify the 1.0 HDF5 store.

Floating-point tests declare tolerances at the test or algorithm level. Exact
metadata, IDs, factor levels, relations, and digests never use numerical
tolerances.
