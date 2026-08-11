# Project state

Date: 2026-08-11
Development version: 0.10.0.9000
Active milestone: 0.10 core frame and walking skeleton

The implementation backlog is maintained in Beads and mirrored by the Mote
coordination log. The dependency-ordered work starts with repository and test
contracts, then IDs and axes, base spaces and sources, frames and views, and an
end-to-end memory/HDF5 walking skeleton.

Current baseline qualifications:

- the local `fmridataset` test suite passes with optional-backend skips;
- persistent HDF5 and Zarr paths are not yet certified by those skipped tests;
- the current `fmristore` checkout must be green before it becomes the
  certified writer;
- companion repository changes are preserved through isolated worktrees.

Historical 0.9 planning material remains under `.planning/milestones/` and
`.planning/phases/` and must not be presented as current release evidence.
