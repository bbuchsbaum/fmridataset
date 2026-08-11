make_frame_fixture <- function(instrument = FALSE) {
  observations_data <- tibble::tibble(
    subject_id = c("sub-01", "sub-01", "sub-02", "sub-02", "sub-02", "sub-03", "sub-03"),
    run_id = c("run-1", "run-1", "run-1", "run-1", "run-2", "run-1", "run-2"),
    stimulus_id = c("stim-1", "stim-2", "stim-1", "stim-3", "stim-2", "stim-1", "stim-3"),
    Fac1 = factor(c("A", "B", "A", "B", "A", "A", "B"), levels = c("A", "B")),
    Fac2 = factor(c("old", "new", "new", "old", "old", "new", "new"), levels = c("old", "new")),
    accuracy = c(.91, .83, .88, .79, .84, .93, .81)
  )

  motion <- axis_block(
    matrix(seq_len(14) / 10, nrow = 7, ncol = 2),
    components = tibble::tibble(
      .component_id = c("translation", "rotation"),
      units = c("mm", "radian")
    ),
    role = "confound"
  )

  stimulus <- list(
    data = tibble::tibble(
      stimulus_id = c("stim-1", "stim-2", "stim-3"),
      category = c("face", "scene", "object")
    ),
    blocks = list(
      visual_pca = axis_block(
        matrix(c(.2, .8, .5, .1, .6, .9, .7, .4, .3), nrow = 3),
        components = tibble::tibble(
          .component_id = c("PC01", "PC02", "PC03"),
          explained_variance = c(.4, .25, .12)
        ),
        role = "continuous"
      )
    )
  )

  feature_space <- volume_space(
    dim = c(2L, 2L, 2L),
    affine = diag(4),
    support = 1:6,
    template = "fixture"
  )
  feature_data <- tibble::tibble(
    parcel = c("hippocampus", "hippocampus", "visual", "visual", "motor", "motor")
  )

  beta <- matrix(seq_len(42), nrow = 7, ncol = 6) / 10
  variance <- matrix(seq_len(42), nrow = 7, ncol = 6) / 100 + .1
  sources <- list(beta = memory_source(beta), variance = memory_source(variance))
  if (instrument) {
    sources <- lapply(sources, counting_source)
  }

  frame <- fmri_frame(
    assays = sources,
    observations = axis_frame(observations_data, blocks = list(motion = motion)),
    features = feature_axis(feature_data, space = feature_space),
    entities = list(stimulus = stimulus),
    active_assay = "beta"
  )

  list(
    frame = frame,
    beta = beta,
    variance = variance,
    stimulus = stimulus,
    feature_space = feature_space
  )
}
