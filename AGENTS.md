# fmridataset agent guide

`fmridataset` is an R package for spatially typed, annotated fMRI data. Its
core representation is an observation-by-feature array linked to explicit
observation metadata, feature metadata, entities, relations, and spatial
identity.

## Start here

1. Read `DESCRIPTION`, the relevant source under `R/`, and its tests under
   `tests/testthat/`. Use the vignettes for public workflows.
2. Run `git status --short` and inspect existing changes. This is often a
   shared, dirty checkout; preserve work you did not create.
3. Inspect the Mote tracker and reserve exact paths before editing:

   ```sh
   mote doctor
   mote actor show
   mote ready
   mote show <id>
   mote preflight --issue <id> --paths <path> [<path> ...]
   mote begin <id> --paths <path> [<path> ...] --note "starting"
   ```

## Contracts to preserve

- Arrays are always observations by features. Keep axis IDs, order, metadata,
  entities, and relations aligned through every subset, reorder, bind, and
  serialization round trip.
- Spatial identity is semantic. Use `compatible_space()` or
  `assert_compatible_space()`; matching dimensions or feature counts is not
  evidence that two spaces align.
- Keep feature IDs stable and explicit. Composite-space IDs are part-qualified,
  and routing order is authoritative.
- Preserve lazy and chunked execution. Do not materialize a full assay unless
  the public API explicitly requires it. Respect memory budgets and backend
  pushdown.
- A new backend or array source must match existing selection, ordering,
  empty-selection, lifecycle, and error behavior. Validate it through the
  package's backend/source contracts.
- Reject ambiguous inputs early with the package's structured errors. Do not
  silently recycle, reshape, reorder, or align by position.
- Treat FDS and HDF5 manifests as compatibility contracts. Update validation,
  digests, readers, writers, and round-trip tests together.

## Make changes

- Keep public behavior in exported functions and backend/source generics;
  isolate representation details behind constructors and validators.
- Add behavioral tests for every contract change, including reordered,
  duplicated, empty, malformed, and lazy inputs where relevant.
- Update roxygen comments and run `devtools::document()` when exports or public
  documentation change. Do not hand-edit `NAMESPACE` or `man/` files.
- Update `NEWS.md` for user-visible changes. Keep examples and vignettes
  executable and consistent with the current API.
- Do not hand-edit generated pkgdown files under `docs/`.

## Verify

Run the narrowest useful check first, then widen it in proportion to the
change:

```sh
Rscript -e 'testthat::test_file("tests/testthat/test-feature-space.R")'
Rscript -e 'devtools::test()'
Rscript -e 'lints <- lintr::lint_package(); print(lints); quit(status = length(lints) > 0L)'
Rscript -e 'devtools::document()'
R CMD build .
R CMD check --as-cran fmridataset_*.tar.gz
```

For documentation-only edits, `git diff --check` plus direct inspection is
usually sufficient. Do not claim optional-dependency, hosted, or cross-platform
coverage unless those gates actually ran.

## Track work with Mote

Mote is the repository's daemonless, append-only tracker and path coordinator.
Never hand-edit `.mote/ops/*.json`.

```sh
mote ready
mote show <id>
mote new "Short concrete title" -p 2 --tag <area>
mote note <id> --kind progress "what changed"
mote done <id> --note "verification completed"
```

Use an existing issue when one fits; create one only when needed. Keep path
reservations narrow. If `mote preflight` or `mote begin` exits 2, inspect
`mote who-has <path>` and do not edit the conflicting path. Record meaningful
progress and decisions with `mote note`.

Mote issue IDs may retain a historical `bd-...` prefix. The prefix is only an
identifier: use it with `mote` commands, never `bd`. Finish with `mote done`, or
use `mote handoff <id> --to <actor> --note "..." --release` when work remains.

## Finish

1. Run the relevant quality gates and inspect `git diff --check`.
2. File genuine follow-up work, then run `mote done <id>` with concrete
   verification evidence. Hand off or release unfinished work.
3. Stage and commit only task-owned paths. Never use broad staging in a dirty
   checkout.
4. Fetch and integrate upstream changes safely, then push without force.
5. Verify that the local commit, upstream tracking ref, and live remote ref
   agree. Report any unrelated pre-existing changes instead of cleaning them.
6. Leave a concise handoff: outcome, tests run, unrun gates, and remaining work.

Work is complete only when the requested changes are pushed successfully.
