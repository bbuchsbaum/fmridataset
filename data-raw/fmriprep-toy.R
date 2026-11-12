# Builds inst/extdata/fmriprep-toy.tar.gz: a minimal fMRIPrep-style derivative
# tree (one subject, two 3-volume runs on a 2 x 2 x 2 grid, with run masks that
# differ and one events.tsv per run). Used by examples and vignettes, which
# unpack it into tempdir(). It ships as an archive because BIDS file names are
# long: unpacked, its paths exceed the 100 bytes a portable tarball allows.
# Run from the package root: Rscript data-raw/fmriprep-toy.R
staging <- tempfile("fmriprep-toy-")
root <- file.path(staging, "fmriprep-toy")
func_dir <- file.path(root, "derivatives", "fmriprep", "sub-01", "func")
raw_dir <- file.path(root, "sub-01", "func")
dir.create(func_dir, recursive = TRUE)
dir.create(raw_dir, recursive = TRUE)
writeLines('{"Name": "toy", "BIDSVersion": "1.10.0"}', file.path(root, "dataset_description.json"))
writeLines(
  paste0(
    '{"Name": "fMRIPrep", "BIDSVersion": "1.10.0", "DatasetType": "derivative", ',
    '"GeneratedBy": [{"Name": "fMRIPrep"}]}'
  ),
  file.path(root, "derivatives", "fmriprep", "dataset_description.json")
)
write.table(
  data.frame(participant_id = "sub-01"), file.path(root, "participants.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)
grid <- c(2L, 2L, 2L)
masks <- list(
  array(c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE), grid),
  array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE), grid)
)
for (run in 1:2) {
  stem <- sprintf("sub-01_task-memory_run-%02d_space-MNI152NLin6Asym", run)
  bold <- array(seq_len(prod(grid) * 3L) + 100L * (run - 1L), c(grid, 3L))
  neuroim2::write_vec(
    neuroim2::NeuroVec(bold, neuroim2::NeuroSpace(c(grid, 3L))),
    file.path(func_dir, paste0(stem, "_desc-preproc_bold.nii"))
  )
  writeLines('{"RepetitionTime": 2}', file.path(func_dir, paste0(stem, "_desc-preproc_bold.json")))
  neuroim2::write_vol(
    neuroim2::LogicalNeuroVol(masks[[run]], neuroim2::NeuroSpace(grid)),
    file.path(func_dir, paste0(stem, "_desc-brain_mask.nii"))
  )
  write.table(
    data.frame(onset = 2 * (run - 1), duration = 1, trial_type = c("old", "new")[run]),
    file.path(raw_dir, sprintf("sub-01_task-memory_run-%02d_events.tsv", run)),
    sep = "\t", row.names = FALSE, quote = FALSE
  )
}

archive <- normalizePath(file.path("inst", "extdata"), mustWork = TRUE)
old <- setwd(staging)
# Long member names are stored with the GNU extension that untar() reads on
# every platform; the archive's own path is short, which is what CRAN checks.
suppressWarnings(utils::tar(
  file.path(archive, "fmriprep-toy.tar.gz"), "fmriprep-toy",
  compression = "gzip", tar = "internal"
))
setwd(old)
unlink(staging, recursive = TRUE)
