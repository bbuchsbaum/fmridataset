# A complete observation-by-feature analysis using optional companion packages.
required <- c("fmridataset", "fmristore", "multidesign", "fmrigds")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install the walking-skeleton packages: ", paste(missing, collapse = ", "))
}

subject_id <- factor(rep(c("sub-01", "sub-02", "sub-03"), c(6L, 5L, 7L)))
observations <- data.frame(
  .obs_id = sprintf("estimate-%03d", seq_along(subject_id)),
  subject_id = subject_id,
  stimulus_id = paste0(
    "stim-",
    c(1L, 2L, 3L, 4L, 5L, 6L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 1L, 3L, 5L, 7L, 8L)
  ),
  Fac1 = factor(
    c("A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A",
      "A", "B", "A", "B", "A", "B", "A"),
    levels = c("A", "B")
  ),
  Fac2 = factor(
    c("old", "old", "new", "new", "old", "new", "new", "old", "old",
      "new", "new", "old", "new", "old", "new", "new", "old", "old"),
    levels = c("old", "new")
  ),
  age = unname(c("sub-01" = 62, "sub-02" = 70, "sub-03" = 67)[subject_id])
)

set.seed(20260811)
visual_pca <- matrix(stats::rnorm(8L * 3L), nrow = 8L, ncol = 3L)
stimulus <- fmridataset::entity_frame(
  key = "stimulus_id",
  data = data.frame(stimulus_id = paste0("stim-", seq_len(8L))),
  blocks = list(
    visual_pca = fmridataset::axis_block(
      visual_pca,
      components = data.frame(.component_id = c("PC01", "PC02", "PC03"))
    )
  )
)
space <- fmridataset::volume_space(
  dim = c(2L, 2L, 2L),
  support = 1:6,
  template = "walking-skeleton"
)
beta <- matrix(stats::rnorm(nrow(observations) * 6L), ncol = 6L)
variance <- matrix(
  stats::runif(nrow(observations) * 6L, min = .01, max = .04),
  ncol = 6L
)
frame <- fmridataset::fmri_frame(
  assays = list(beta = beta, variance = variance),
  observations = observations,
  features = fmridataset::feature_axis(
    data.frame(
      .feature_id = fmridataset::feature_ids(space),
      parcel = c("hippocampus", "hippocampus", "visual", "visual", "motor", "motor")
    ),
    space = space
  ),
  entities = list(stimulus = stimulus),
  relations = list(
    observation_stimulus = fmridataset::key_relation(
      "stimulus_id",
      target = "stimulus"
    )
  ),
  active_assay = "beta"
)

# These operations inspect aligned metadata but read no assay values.
selected <- frame |>
  fmridataset::filter_obs(Fac1 == "A" & age >= 60) |>
  fmridataset::select_features(parcel == "hippocampus")
stopifnot(identical(dim(selected), c(10L, 2L)))

spec <- multidesign::design_spec(
  fixed = ~ Fac1 * Fac2 + age + mv(stimulus.visual_pca, 1:3),
  random = ~ 1 | subject_id
)
fit <- fmrigds::fit_group(
  frame,
  estimate = "beta",
  variance = "variance",
  design = spec,
  memory_budget = 256 * 1024^2,
  block_size = 2L
)
map <- fmridataset::spatial_map(
  fit$result,
  observation = "Fac1B",
  assay = "estimate"
)
stopifnot(methods::is(map, "NeuroVol"))

path <- tempfile(fileext = ".fds.h5")
on.exit(unlink(path), add = TRUE)
fmridataset::write_frame(fit$result, path)
reopened <- fmridataset::open_frame(path)
stopifnot(
  identical(fmridataset::feature_ids(reopened), fmridataset::feature_ids(fit$result)),
  identical(
    fmridataset::space_digest(fmridataset::space(reopened)),
    fmridataset::space_digest(fmridataset::space(fit$result))
  ),
  isTRUE(all.equal(
    fmridataset::collect_assay(reopened, "estimate"),
    fmridataset::collect_assay(fit$result, "estimate"),
    tolerance = 0
  ))
)
