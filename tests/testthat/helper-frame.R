make_frame_fixture <- function(instrument = FALSE) {
  observations_data <- tibble::tibble(
    .obs_id = sprintf("estimate-%03d", seq_len(7L)),
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

  stimulus <- entity_frame(
    data = tibble::tibble(
      stimulus_id = c("stim-1", "stim-2", "stim-3"),
      category = c("face", "scene", "object")
    ),
    key = "stimulus_id",
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

make_composite_space_fixture <- function() {
  left <- surface_space(
    vertex_ids = paste0("L-", 1:3),
    hemisphere = rep("left", 3L),
    topology = matrix(c(1, 2, 3), nrow = 1L),
    geometry = cbind(-seq_len(3L), c(0, 1, 0), 0),
    template = "fsLR-toy"
  )
  right <- surface_space(
    vertex_ids = paste0("R-", 1:3),
    hemisphere = rep("right", 3L),
    topology = matrix(c(1, 2, 3), nrow = 1L),
    geometry = cbind(seq_len(3L), c(0, 1, 0), 0),
    template = "fsLR-toy"
  )
  subcortical <- volume_space(
    c(2, 1, 1), support = 1:2, template = "MNI-toy"
  )
  composite_space(
    list(
      left_cortex = left,
      right_cortex = right,
      subcortical = subcortical
    ),
    composite_type = "grayordinate_like",
    metadata = list(convention = "surface-plus-volume")
  )
}

make_walking_skeleton_fixture <- function(instrument = FALSE) {
  subject_id <- factor(rep(c("sub-01", "sub-02", "sub-03"), c(6L, 5L, 7L)))
  Fac1 <- factor(
    c("A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A",
      "A", "B", "A", "B", "A", "B", "A"),
    levels = c("A", "B")
  )
  Fac2 <- factor(
    c("old", "old", "new", "new", "old", "new", "new", "old", "old",
      "new", "new", "old", "new", "old", "new", "new", "old", "old"),
    levels = c("old", "new")
  )
  stimulus_id <- paste0(
    "stim-",
    c(1L, 2L, 3L, 4L, 5L, 6L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 1L, 3L, 5L, 7L, 8L)
  )
  age <- unname(c("sub-01" = 62, "sub-02" = 70, "sub-03" = 67)[subject_id])
  observation_data <- tibble::tibble(
    .obs_id = sprintf("estimate-%03d", seq_along(subject_id)),
    subject_id = subject_id,
    stimulus_id = stimulus_id,
    Fac1 = Fac1,
    Fac2 = Fac2,
    age = age
  )

  set.seed(20260811)
  visual_pca <- matrix(stats::rnorm(8L * 3L), nrow = 8L, ncol = 3L)
  visual_components <- tibble::tibble(
    .component_id = c("PC01", "PC02", "PC03"),
    explained_variance = c(.41, .23, .14)
  )
  stimulus <- entity_frame(
    key = "stimulus_id",
    data = tibble::tibble(stimulus_id = paste0("stim-", seq_len(8L))),
    blocks = list(
      visual_pca = axis_block(
        visual_pca,
        components = visual_components,
        role = "continuous"
      )
    )
  )

  lifted_pca <- visual_pca[
    match(observation_data$stimulus_id, stimulus$data$stimulus_id),
    ,
    drop = FALSE
  ]
  dense_data <- cbind(observation_data, as.data.frame(lifted_pca))
  names(dense_data)[(ncol(dense_data) - 2L):ncol(dense_data)] <- visual_components$.component_id
  dense_design <- stats::model.matrix(
    ~ Fac1 * Fac2 + age + PC01 + PC02 + PC03,
    data = dense_data
  )

  set.seed(20260812)
  coefficient <- matrix(
    stats::rnorm(ncol(dense_design) * 6L, sd = .12),
    nrow = ncol(dense_design),
    ncol = 6L
  )
  coefficient[1L, ] <- seq(.3, .8, length.out = 6L)
  variance <- matrix(
    stats::runif(nrow(dense_design) * 6L, min = .01, max = .04),
    nrow = nrow(dense_design),
    ncol = 6L
  )
  random_intercept <- c("sub-01" = -.2, "sub-02" = .15, "sub-03" = .05)
  beta <- dense_design %*% coefficient +
    random_intercept[as.character(subject_id)] +
    matrix(
      stats::rnorm(length(variance), sd = sqrt(variance + .03)),
      nrow = nrow(variance),
      ncol = ncol(variance)
    )

  sources <- list(beta = memory_source(beta), variance = memory_source(variance))
  if (instrument) sources <- lapply(sources, counting_source)
  feature_space <- volume_space(
    dim = c(2L, 2L, 2L),
    affine = diag(4),
    support = 1:6,
    template = "walking-skeleton"
  )
  frame <- fmri_frame(
    assays = sources,
    observations = observation_data,
    features = feature_axis(
      tibble::tibble(
        .feature_id = feature_ids(feature_space),
        parcel = c("hippocampus", "hippocampus", "visual", "visual", "motor", "motor")
      ),
      space = feature_space
    ),
    entities = list(stimulus = stimulus),
    relations = list(
      observation_stimulus = key_relation("stimulus_id", target = "stimulus")
    ),
    active_assay = "beta"
  )

  list(
    frame = frame,
    beta = beta,
    variance = variance,
    observation_data = observation_data,
    visual_pca = visual_pca,
    lifted_pca = lifted_pca,
    dense_design = dense_design,
    feature_space = feature_space
  )
}
